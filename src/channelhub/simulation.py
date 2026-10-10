"""Pure forward simulation and an explicit, receiver-independent export contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .config import RunConfig
from .channels import statistical, ray_tracing, frequencies
from .signals import make_grid, apply_channel


def simulate(
    cfg: RunConfig,
    *,
    custom_x: np.ndarray | None = None,
    scene_path: Path | None = None,
    progress: Callable[[int, str], None] = lambda *_: None,
):
    progress(5, "Preparing transmit resource grids")
    transmissions = []
    for i, tx in enumerate([cfg.tx, *cfg.interferers]):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 20, i]))
        transmissions.append(make_grid(cfg, tx, rng, custom_x if i == 0 else None))
    progress(15, "Generating propagation channels")
    channels = statistical(cfg) if cfg.engine == "statistical" else ray_tracing(cfg, scene_path)
    progress(70, "Composing signal, interference and noise")
    arrays = {}
    signals = []
    for i, (channel, (x, mask)) in enumerate(zip(channels, transmissions, strict=True)):
        prefix = "desired" if i == 0 else f"interferer_{i - 1}"
        arrays.update(
            {
                f"{prefix}_h": channel.h,
                f"{prefix}_cir": channel.taps,
                f"{prefix}_delays_s": channel.delays_s,
                f"{prefix}_x": x,
                f"{prefix}_active_re": mask,
            }
        )
        signals.append(apply_channel(channel.h, x))
    desired = signals[0]
    interference = sum(signals[1:], np.zeros_like(desired))
    noise_variance = (
        10 ** ((cfg.noise_psd_dbm_hz + cfg.noise_figure_db - 30) / 10) * cfg.scs_khz * 1000
        if cfg.noise_enabled
        else 0.0
    )
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 30]))
    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal(desired.shape) + 1j * rng.standard_normal(desired.shape)
    )
    arrays.update(
        y=desired + interference + noise,
        signal=desired,
        interference=interference,
        noise=noise,
        frequency_offset_hz=frequencies(cfg),
        time_s=np.arange(cfg.num_times) * cfg.symbol_interval_s,
        receiver_position_m=np.asarray(cfg.rx.position_m)[None, :]
        + np.arange(cfg.num_times)[:, None] * cfg.symbol_interval_s * np.asarray(cfg.rx.velocity_m_s),
    )
    config_data = cfg.model_dump(mode="json")
    canonical = json.dumps(config_data, sort_keys=True, separators=(",", ":"))
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "sionna" if cfg.engine == "statistical" else "sionna-rt")
    }
    metadata = {
        "schema": "channelhub-observation-v1",
        "version": __version__,
        "config": config_data,
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "dependencies": versions,
        "equation": "y = H_desired X_desired + sum(H_interferer X_interferer) + noise",
        "axes": {
            "h": ["time", "rx_port", "tx_port", "subcarrier"],
            "cir": ["time", "rx_port", "tx_port", "path"],
            "delays_s": ["rx_port", "tx_port", "path"],
            "x": ["time", "tx_port", "subcarrier"],
            "y": ["time", "rx_port", "subcarrier"],
            "active_re": ["time", "subcarrier"],
        },
        "units": {
            "h": "dimensionless",
            "cir": "dimensionless",
            "x": "sqrt(W) per RE",
            "y": "sqrt(W) per RE",
            "delays_s": "s",
        },
        "noise_variance_w_per_re": noise_variance,
        "time_grid": "slot-average normal-CP symbol intervals; no time-domain OFDM or ICI simulation",
        "delay_reference": "excess delay" if cfg.engine == "statistical" else "absolute propagation delay",
        "statistical_geometry": "independent links; initial-distance free-space loss or user pathloss; no network spatial consistency"
        if cfg.engine == "statistical"
        else None,
        "rt_motion": "fixed ray geometry with Doppler evolution over observation window"
        if cfg.engine == "sionna_rt"
        else None,
        "custom_x_sha256": hashlib.sha256(np.ascontiguousarray(custom_x).tobytes()).hexdigest()
        if custom_x is not None
        else None,
        "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest() if scene_path else None,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }
    progress(90, "Writing observation package")
    return arrays, metadata


def save_result(directory: Path, arrays: dict, metadata: dict):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "observations.npz").exists() or (directory / "metadata.json").exists():
        raise FileExistsError("Result already exists; choose a new output directory")
    np.savez_compressed(directory / "observations.npz", **arrays)
    digest = hashlib.sha256((directory / "observations.npz").read_bytes()).hexdigest()
    metadata = {**metadata, "observations_sha256": digest}
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
