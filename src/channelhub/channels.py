"""Adapters to unmodified upstream Sionna 2.2 channel generators."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import math

import numpy as np

from .config import Array, RunConfig

C = 299792458.0


@dataclass
class Channel:
    h: np.ndarray  # time, rx, tx, subcarrier
    taps: np.ndarray  # time, rx, tx, path
    delays_s: np.ndarray  # rx, tx, path; negative padding for RT


def frequencies(cfg):
    return (np.arange(cfg.num_subcarriers) - cfg.num_subcarriers // 2) * cfg.scs_khz * 1000


def taps_to_frequency(taps, delays, freq):
    """H(f,t)=sum_p a_p(t) exp(-j 2π f τ_p); no power normalization."""
    # Native RT delays can be float32. Promote before multiplying by 2πj:
    # NumPy otherwise rounds that intermediate even with float64 frequencies.
    delays = np.asarray(delays, dtype=np.float64)
    if taps.ndim != 4 or delays.shape != taps.shape[1:]:
        raise ValueError("Expected taps[time,rx,tx,path] and delays[rx,tx,path]")
    out = np.empty((*taps.shape[:3], len(freq)), dtype=np.complex128)
    valid = delays >= 0
    safe_delays = np.where(valid, delays, 0)
    for k in range(0, len(freq), 48):
        phase = np.exp(-2j * np.pi * safe_delays[..., None] * freq[k : k + 48]) * valid[..., None]
        out[..., k : k + 48] = np.einsum("trqp,rqpk->trqk", taps, phase, optimize=True)
    return out


def _phy_array(spec: Array, carrier_hz, device="cpu"):
    from sionna.phy.channel.tr38901 import AntennaArray

    return AntennaArray(
        num_rows=spec.rows,
        num_cols=spec.cols,
        polarization="dual" if spec.polarization == "VH" else "single",
        polarization_type=spec.polarization,
        antenna_pattern="omni" if spec.pattern == "iso" else "38.901",
        carrier_frequency=carrier_hz,
        precision="double",
        vertical_spacing=spec.spacing_lambda,
        horizontal_spacing=spec.spacing_lambda,
        device=device,
    )


def statistical(cfg: RunConfig) -> list[Channel]:
    try:
        import torch
        from sionna.phy import config as sconfig
        from sionna.phy.channel.tr38901 import CDL, TDL
    except ImportError as exc:
        raise RuntimeError('Install the statistical engine: pip install ".[statistical]"') from exc
    result = []
    speed = float(np.linalg.norm(cfg.rx.velocity_m_s))
    for index, tx in enumerate([cfg.tx, *cfg.interferers]):
        sconfig.seed = int(np.random.SeedSequence([cfg.seed, 10, index]).generate_state(1)[0])
        common = dict(
            model=cfg.profile[-1],
            delay_spread=cfg.delay_spread_ns * 1e-9,
            carrier_frequency=cfg.carrier_ghz * 1e9,
            device="cpu",
            precision="double",
        )
        if cfg.profile.startswith("CDL"):
            generator = CDL(
                **common,
                bs_array=_phy_array(tx.array, cfg.carrier_ghz * 1e9),
                ut_array=_phy_array(cfg.rx.array, cfg.carrier_ghz * 1e9),
                direction="downlink",
                bs_orientation=torch.tensor([math.radians(tx.array.yaw_deg), 0.0, 0.0], dtype=torch.float64),
                ut_orientation=torch.tensor(
                    [math.radians(cfg.rx.array.yaw_deg), 0.0, 0.0], dtype=torch.float64
                ),
                ut_velocity=torch.tensor(cfg.rx.velocity_m_s, dtype=torch.float64),
            )
        else:
            generator = TDL(
                **common,
                num_rx_ant=cfg.rx.array.ports,
                num_tx_ant=tx.array.ports,
                min_speed=speed,
                max_speed=speed,
            )
        a, tau = generator(
            batch_size=1, num_time_steps=cfg.num_times, sampling_frequency=1 / cfg.symbol_interval_s
        )
        taps = a.detach().cpu().numpy()[0, 0, :, 0].transpose(3, 0, 1, 2).astype(np.complex128)
        delays = np.broadcast_to(tau.detach().cpu().numpy()[0, 0, 0], taps.shape[1:]).copy()
        distance = float(np.linalg.norm(np.asarray(tx.position_m) - cfg.rx.position_m))
        loss = tx.pathloss_db
        if loss is None:
            loss = max(0.0, 20 * math.log10(4 * math.pi * distance * cfg.carrier_ghz * 1e9 / C))
        taps *= 10 ** (-loss / 20)
        result.append(Channel(taps_to_frequency(taps, delays, frequencies(cfg)), taps, delays))
    return result


def ray_tracing(cfg: RunConfig, scene_path: Path | None = None) -> list[Channel]:
    try:
        from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
    except ImportError as exc:
        raise RuntimeError('Install a working Sionna RT backend: pip install ".[rt]"') from exc
    if scene_path is None and cfg.rt.scene == "reflector":
        scene_path = Path(str(files("channelhub").joinpath("scenes/reflector.xml")))
    scene = load_scene(str(scene_path) if scene_path else None)
    scene.frequency = cfg.carrier_ghz * 1e9

    def array(a):
        return PlanarArray(
            num_rows=a.rows,
            num_cols=a.cols,
            vertical_spacing=a.spacing_lambda,
            horizontal_spacing=a.spacing_lambda,
            pattern="iso" if a.pattern == "iso" else "tr38901",
            polarization=a.polarization,
        )

    scene.tx_array = array(cfg.tx.array)
    scene.rx_array = array(cfg.rx.array)
    for i, tx in enumerate([cfg.tx, *cfg.interferers]):
        scene.add(
            Transmitter(
                name=f"tx{i}", position=tx.position_m, orientation=[math.radians(tx.array.yaw_deg), 0, 0]
            )
        )
    scene.add(
        Receiver(
            name="rx",
            position=cfg.rx.position_m,
            velocity=cfg.rx.velocity_m_s,
            orientation=[math.radians(cfg.rx.array.yaw_deg), 0, 0],
        )
    )
    paths = PathSolver()(
        scene,
        max_depth=cfg.rt.max_depth,
        samples_per_src=cfg.rt.samples_per_source,
        synthetic_array=True,
        los=cfg.rt.line_of_sight,
        specular_reflection=cfg.rt.specular_reflection,
        diffuse_reflection=cfg.rt.diffuse_reflection,
        refraction=cfg.rt.refraction,
        seed=cfg.seed,
    )
    a, tau = paths.cir(
        sampling_frequency=1 / cfg.symbol_interval_s,
        num_time_steps=cfg.num_times,
        normalize_delays=False,
        out_type="numpy",
    )
    a, tau = np.asarray(a), np.asarray(tau)
    result = []
    for i in range(1 + len(cfg.interferers)):
        taps = a[0, :, i].transpose(3, 0, 1, 2).astype(np.complex128)
        # Synthetic arrays share each link's path delays across antenna ports;
        # tau is [rx, tx, path], unlike the antenna-expanded CIR coefficients.
        if tau.ndim == 3:
            link_delays = tau[0, i]
        elif tau.ndim == 5:
            link_delays = tau[0, :, i]
        else:
            raise RuntimeError(f"Unexpected upstream RT delay shape: {tau.shape}")
        delays = np.broadcast_to(link_delays, taps.shape[1:]).copy()
        result.append(Channel(taps_to_frequency(taps, delays, frequencies(cfg)), taps, delays))
    return result
