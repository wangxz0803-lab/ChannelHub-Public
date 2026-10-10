"""Transmit resource grids, expressed in sqrt(watt) per resource element."""

from __future__ import annotations

import numpy as np

from .config import RunConfig, Transmitter
from .ref_signals.srs import SRSResourceConfig, srs_sequence, srs_subcarrier_indices
from .ref_signals.csi_rs import csi_rs_sequence
from .ref_signals.ssb import pss, sss, pbch_dmrs


def dft_weights(ports: int, beam: int) -> np.ndarray:
    """Unit-norm DFT weights in physical port order, w[n]=exp(2πjnb/N)/sqrt(N)."""
    if not 0 <= beam < ports:
        raise ValueError("Beam index outside port range")
    return np.exp(2j * np.pi * np.arange(ports) * beam / ports) / np.sqrt(ports)


def make_grid(
    cfg: RunConfig, tx: Transmitter, rng: np.random.Generator, custom_x: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    shape = (cfg.num_times, tx.array.ports, cfg.num_subcarriers)
    wave = tx.waveform
    if wave.kind == "custom":
        if custom_x is None:
            raise ValueError("Custom waveform requires X as a complex .npy array")
        x = np.asarray(custom_x)
        if x.shape != shape or x.dtype.kind != "c" or not np.isfinite(x).all():
            raise ValueError(f"X must be a finite complex array with shape {shape}")
        # Caller-supplied physical amplitudes are never renormalized.
        return x.astype(np.complex128, copy=True), np.any(x != 0, axis=1)
    x = np.zeros(shape, dtype=np.complex128)
    mu = int(np.log2(cfg.scs_khz / 15))
    weights = dft_weights(tx.array.ports, wave.beam_index)
    for slot in range(cfg.num_slots):
        if rng.random() >= tx.activity:
            continue
        offset = slot * 14
        if wave.kind == "srs":
            s = wave.srs
            resource = SRSResourceConfig(
                C_SRS=s.c_srs,
                B_SRS=s.b_srs,
                K_TC=s.comb,
                n_RRC=s.n_rrc,
                b_hop=s.b_hop,
                n_SRS_ID=wave.identity,
                T_SRS=s.period_slots,
                T_offset=s.offset_slots,
                N_ap=s.ports,
                n_cs=s.cyclic_shift,
                group_hopping=s.group_hopping,
                sequence_hopping=s.sequence_hopping,
                R=s.repetition,
                comb_offset=s.comb_offset,
                n_shift=s.shift_rb,
                N_symb=s.symbols,
                start_position=s.start_position,
                numerology=mu,
            )
            if not resource.is_slot_active(slot):
                continue
            for symbol in resource.symbol_indices:
                for port in range(s.ports):
                    tones = srs_subcarrier_indices(resource, slot, symbol, cfg.num_rb, port)
                    values = srs_sequence(
                        wave.identity,
                        s.comb,
                        s.cyclic_shift,
                        s.ports,
                        len(tones),
                        slot,
                        symbol,
                        port,
                        s.group_hopping,
                        s.sequence_hopping,
                        numerology=mu,
                    )
                    x[offset + symbol, port, tones] = values / np.sqrt(s.ports)
        elif wave.kind == "csi_rs":
            # TS 38.211 Table 7.4.1.5.3-1, row 2: one port, density 1, no CDM.
            tones = 12 * np.arange(cfg.num_rb) + wave.csi_subcarrier
            values = csi_rs_sequence(wave.identity, slot % (10 * 2**mu), wave.csi_symbol, 1, cfg.num_rb)
            x[offset + wave.csi_symbol][:, tones] = weights[:, None] * values[None, :]
        elif wave.kind == "ssb":
            # SS/PBCH reference-only grid. PBCH payload/encoding is not synthesized.
            block = np.zeros((4, 240), dtype=np.complex128)
            block[0, 56:183] = pss(wave.identity % 3)
            block[2, 56:183] = sss(wave.identity)
            v = wave.identity % 4
            indices = [
                np.arange(v, 240, 4),
                np.r_[np.arange(v, 48, 4), np.arange(192 + v, 240, 4)],
                np.arange(v, 240, 4),
            ]
            dmrs = pbch_dmrs(wave.identity, wave.ssb_index)
            start = 0
            for symbol, tones in zip((1, 2, 3), indices):
                block[symbol, tones] = dmrs[start : start + len(tones)]
                start += len(tones)
            k0 = (cfg.num_subcarriers - 240) // 2
            for symbol in range(4):
                x[offset + wave.ssb_start_symbol + symbol, :, k0 : k0 + 240] = (
                    weights[:, None] * block[symbol]
                )
        elif wave.kind == "qpsk":
            bits = rng.integers(0, 2, size=(14, tx.array.ports, cfg.num_subcarriers, 2))
            x[offset : offset + 14] = ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / np.sqrt(
                2 * tx.array.ports
            )
    energy = np.sum(np.abs(x) ** 2, axis=1, keepdims=True)
    x /= np.sqrt(np.where(energy > 0, energy, 1))
    x *= np.sqrt(10 ** ((tx.power_dbm - 30) / 10))
    return x, np.any(x != 0, axis=1)


def apply_channel(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """H[t,rx,tx,k] X[t,tx,k] -> Y[t,rx,k], with no conjugation."""
    if h.ndim != 4 or x.ndim != 3 or (h.shape[0], h.shape[2], h.shape[3]) != x.shape:
        raise ValueError("H and X have incompatible time, transmitter or frequency axes")
    return np.einsum("trpk,tpk->trk", h, x, optimize=True)
