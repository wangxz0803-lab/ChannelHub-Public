"""NZP-CSI-RS sequence generation per 3GPP TS 38.211 §7.4.1.5.

The CSI-RS reference signal is a QPSK sequence

    r(m) = (1/√2)·[1 - 2·c(2m)] + j·(1/√2)·[1 - 2·c(2m+1)]

with the Gold PRBS initialised per 38.211 §7.4.1.5.2:

    c_init = (2^10·(N_symb_slot·n_s,f^μ + l + 1)·(2·N_ID + 1) + N_ID) mod 2^31

This module generates the *sequence* values; the RE-mapping (which depends on
``row`` in Table 7.4.1.5.3-1, CDM type, density ρ and frequency-domain location
``k̄_p``) is orthogonal to the sequence itself and only lightly covered here.

Supported CDM types — Table 7.4.1.5.3-1:
    * noCDM (CDM-1)
    * fd-CDM2 (CDM-2, two REs in freq)
    * cdm4-FD2-TD2 (CDM-4, 2×2 in freq/time)
    * cdm8-FD2-TD4 (CDM-8, 2×4 in freq/time)

Supported densities: ρ ∈ {0.5, 1, 3} (ρ=0.5 maps to every other PRB, ρ=3 to
three tones per PRB in the density-3 rows used by TRS).

Port counts covered in :func:`csi_rs_port_info`: 1, 2, 4, 8, 12, 16, 24, 32.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .gold import pseudo_random

__all__ = [
    "CSIRSPortInfo",
    "CDM_WF",
    "CDM_WT",
    "csi_rs_sequence",
    "csi_rs_port_info",
    "N_SC_PER_RB",
]

N_SC_PER_RB: int = 12


# -----------------------------------------------------------------------------
# CDM cover codes per 38.211 Table 7.4.1.5.3-1 (simplified form).
# Each entry maps (cdm_type, index_within_cdm_group) → (w_f, w_t) with w_f of
# length {1, 2} and w_t of length {1, 2, 4}.
# -----------------------------------------------------------------------------
CDM_WF: dict[str, list[list[int]]] = {
    "noCDM": [[1]],
    "fd-CDM2": [[1, 1], [1, -1]],
    "cdm4-FD2-TD2": [[1, 1], [1, -1], [1, 1], [1, -1]],
    "cdm8-FD2-TD4": [[1, 1], [1, -1], [1, 1], [1, -1], [1, 1], [1, -1], [1, 1], [1, -1]],
}
CDM_WT: dict[str, list[list[int]]] = {
    "noCDM": [[1]],
    "fd-CDM2": [[1], [1]],
    "cdm4-FD2-TD2": [[1, 1], [1, 1], [1, -1], [1, -1]],
    "cdm8-FD2-TD4": [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
        [1, -1, -1, 1],
    ],
}


@dataclass(frozen=True)
class CSIRSPortInfo:
    """Lightweight descriptor for a CSI-RS configuration.

    The intent is to expose the row-driven parameters of Table 7.4.1.5.3-1
    without fully re-implementing every row; callers can use ``cdm_type`` and
    ``ports_per_cdm`` to apply cover codes themselves.
    """

    n_ports: int
    cdm_type: str
    ports_per_cdm: int  # 1 / 2 / 4 / 8
    density_options: tuple[float, ...]


_PORT_INFO: dict[int, CSIRSPortInfo] = {
    1: CSIRSPortInfo(1, "noCDM", 1, (0.5, 1.0, 3.0)),
    2: CSIRSPortInfo(2, "fd-CDM2", 2, (0.5, 1.0)),
    4: CSIRSPortInfo(4, "fd-CDM2", 2, (1.0,)),
    8: CSIRSPortInfo(8, "fd-CDM2", 2, (1.0,)),
    12: CSIRSPortInfo(12, "fd-CDM2", 2, (1.0,)),
    16: CSIRSPortInfo(16, "fd-CDM2", 2, (0.5, 1.0)),
    24: CSIRSPortInfo(24, "cdm4-FD2-TD2", 4, (0.5, 1.0)),
    32: CSIRSPortInfo(32, "cdm4-FD2-TD2", 4, (0.5, 1.0)),
}


def csi_rs_port_info(n_ports: int) -> CSIRSPortInfo:
    """Return :class:`CSIRSPortInfo` for a supported port count.

    Raises :class:`ValueError` for unsupported ``n_ports`` values.
    """
    if n_ports not in _PORT_INFO:
        raise ValueError(f"Unsupported CSI-RS n_ports={n_ports}; supported: {sorted(_PORT_INFO)}")
    return _PORT_INFO[n_ports]


def _csi_rs_c_init(slot: int, symbol: int, n_ID: int) -> int:
    """38.211 §7.4.1.5.2 eq (7.4.1.5.2-2):

    c_init = (2^10·(N_symb_slot·n_s,f^μ + l + 1)·(2·N_ID + 1) + N_ID) mod 2^31
    """
    N_symb_slot = 14  # Normal CP.
    term = (1 << 10) * (N_symb_slot * int(slot) + int(symbol) + 1) * (2 * int(n_ID) + 1)
    return (term + int(n_ID)) & 0x7FFFFFFF


def csi_rs_sequence(
    n_ID: int,
    slot: int,
    symbol: int,
    density: float,
    n_rb: int,
    n_start_rb: int = 0,
) -> np.ndarray:
    """Return the QPSK CSI-RS symbols for one OFDM symbol of one CDM pair.

    The number of output symbols is ``max(1, round(density)) · n_rb`` for
    ρ ∈ {1, 3} and ``n_rb / 2`` for ρ = 0.5 (even-PRB mask). This matches the
    effective tone count after PRB-density masking; the exact RE-mapping is left
    to the caller.

    Parameters
    ----------
    n_ID:
        Scrambling identity (0..1023).
    slot:
        Slot index within the radio frame (0..N_slot_frame-1).
    symbol:
        OFDM symbol index (0..13 normal CP).
    density:
        Density ρ ∈ {0.5, 1, 3}.
    n_rb:
        Number of allocated PRBs.
    n_start_rb:
        Starting PRB offset — advances the PRBS by 2 bits per CSI-RS RE per
        38.211 §7.4.1.5.3 (``m'`` counter across PRBs of the CSI-RS bandwidth).
    """
    if density not in (0.5, 1.0, 3.0):
        raise ValueError("density must be one of {0.5, 1.0, 3.0}")
    if n_rb <= 0:
        raise ValueError("n_rb must be positive")
    # Effective number of CSI-RS tones in the OFDM symbol per CDM pair.
    if density == 0.5:
        m_max = n_rb // 2
    elif density == 1.0:
        m_max = n_rb
    else:  # density == 3.0
        m_max = 3 * n_rb
    if m_max == 0:
        return np.zeros(0, dtype=np.complex128)

    # Advance the PRBS start so different PRB offsets yield consistent sequences.
    start = int(n_start_rb) * (2 if density < 3.0 else 6)
    c_init = _csi_rs_c_init(slot, symbol, n_ID)
    # A deterministic transmitted sequence is shared across UE observations;
    # never cache received noise or a measured channel. Bound both count and
    # item bytes (<=64 MiB) and preserve the old caller-owned writable result.
    if m_max <= 2048:
        return _cached_csi_rs_qpsk(c_init, start, m_max).copy()
    return _csi_rs_qpsk(c_init, start, m_max)


def _csi_rs_qpsk(c_init, start, m_max):
    c = pseudo_random(c_init, start + 2 * m_max)
    c = c[start:]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    re = inv_sqrt2 * (1.0 - 2.0 * c[0::2].astype(np.float64))
    im = inv_sqrt2 * (1.0 - 2.0 * c[1::2].astype(np.float64))
    return (re + 1j * im).astype(np.complex128)


@lru_cache(maxsize=2048)
def _cached_csi_rs_qpsk(c_init, start, m_max):
    values = _csi_rs_qpsk(c_init, start, m_max)
    values.flags.writeable = False
    return values
