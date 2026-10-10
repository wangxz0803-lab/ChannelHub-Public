"""SS/PBCH block reference signals per 3GPP TS 38.211 §7.4.2 / §7.4.3.

Implements:
    * PSS (§7.4.2.2) — BPSK m-sequence of length 127 with three cyclic shifts.
    * SSS (§7.4.2.3) — product of two length-127 m-sequences, parameterised by
      the PCI (N_ID_1, N_ID_2).
    * PCI aggregation helper: N_cell_ID = 3·N_ID_1 + N_ID_2.
    * PBCH-DMRS (§7.4.1.4) — built on the shared Gold sequence generator using
      ``c_init = 2^11 · (i_SSB + 1) · ((N_ID / 4) + 1) + 2^6 · (i_SSB + 1)
               + (N_ID mod 4)``.

All outputs are numpy arrays with dtype ``complex128`` (for DMRS) or ``float64``
(for PSS/SSS ±1 BPSK symbols).
"""

from __future__ import annotations

import numpy as np

from .gold import pseudo_random

__all__ = ["pss", "sss", "pci", "pbch_dmrs", "PSS_LEN", "SSS_LEN"]

PSS_LEN: int = 127
SSS_LEN: int = 127


def pci(N_ID_1: int, N_ID_2: int) -> int:
    """Physical Cell ID aggregation per 38.211 §7.4.2.1: 3·N_ID_1 + N_ID_2."""
    if not (0 <= N_ID_1 < 336):
        raise ValueError("N_ID_1 must be in [0, 336)")
    if not (0 <= N_ID_2 < 3):
        raise ValueError("N_ID_2 must be in [0, 3)")
    return 3 * int(N_ID_1) + int(N_ID_2)


def _pss_m_sequence() -> np.ndarray:
    """Generate the length-127 m-sequence x(i+7) = (x(i+4) + x(i)) mod 2.

    Initial condition [x(0..6)] = [0, 1, 1, 0, 1, 1, 1] per 38.211 §7.4.2.2.
    Returns an int8 vector of length 127 with values in {0, 1}.
    """
    x = np.zeros(127 + 7, dtype=np.int8)
    x[0:7] = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.int8)
    for i in range(127):
        x[i + 7] = (x[i + 4] + x[i]) & 1
    return x[:127]


def pss(N_ID_2: int) -> np.ndarray:
    """Primary Synchronization Signal d_PSS per 38.211 §7.4.2.2.

        d_PSS(n) = 1 - 2·x((n + 43·N_ID_2) mod 127),     n = 0..126

    Returns a ``float64`` vector of ±1 BPSK symbols (length 127).
    """
    if not (0 <= N_ID_2 < 3):
        raise ValueError("N_ID_2 must be in [0, 3)")
    x = _pss_m_sequence()
    n = np.arange(PSS_LEN)
    idx = (n + 43 * int(N_ID_2)) % 127
    return (1.0 - 2.0 * x[idx]).astype(np.float64)


def _sss_m_sequences() -> tuple[np.ndarray, np.ndarray]:
    """Return the two length-127 m-sequences x0, x1 used by SSS.

    x0 recurrence: x0(i+7) = (x0(i+4) + x0(i)) mod 2, init [1,0,0,0,0,0,0]
    x1 recurrence: x1(i+7) = (x1(i+1) + x1(i)) mod 2, init [1,0,0,0,0,0,0]
    """
    x0 = np.zeros(127 + 7, dtype=np.int8)
    x0[0:7] = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    for i in range(127):
        x0[i + 7] = (x0[i + 4] + x0[i]) & 1

    x1 = np.zeros(127 + 7, dtype=np.int8)
    x1[0:7] = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    for i in range(127):
        x1[i + 7] = (x1[i + 1] + x1[i]) & 1
    return x0[:127], x1[:127]


def sss(N_ID: int) -> np.ndarray:
    """Secondary Synchronization Signal d_SSS per 38.211 §7.4.2.3.

    With ``N_ID = 3·N_ID_1 + N_ID_2`` the SSS is

        m0 = 15·floor(N_ID_1/112) + 5·N_ID_2
        m1 = N_ID_1 mod 112
        d_SSS(n) = [1 - 2·x0((n + m0) mod 127)] · [1 - 2·x1((n + m1) mod 127)]

    Returns a ``float64`` vector of ±1 BPSK symbols (length 127).
    """
    if not (0 <= N_ID < 1008):
        raise ValueError("N_ID must be in [0, 1008)")
    N_ID_2 = N_ID % 3
    N_ID_1 = N_ID // 3
    m0 = 15 * (N_ID_1 // 112) + 5 * N_ID_2
    m1 = N_ID_1 % 112
    x0, x1 = _sss_m_sequences()
    n = np.arange(SSS_LEN)
    a = 1.0 - 2.0 * x0[(n + m0) % 127]
    b = 1.0 - 2.0 * x1[(n + m1) % 127]
    return (a * b).astype(np.float64)


def pbch_dmrs(N_ID: int, i_SSB: int, length: int = 144) -> np.ndarray:
    """PBCH-DMRS sequence per 38.211 §7.4.1.4.1 as QPSK complex symbols.

        r(m) = (1/√2)·[1 - 2·c(2m)] + j·(1/√2)·[1 - 2·c(2m+1)]

    with c(·) the Gold sequence initialised by

        c_init = 2^11·(i_SSB + 1)·(floor(N_ID/4) + 1) + 2^6·(i_SSB + 1)
                 + (N_ID mod 4)

    Parameters
    ----------
    N_ID:
        Physical cell ID (0..1007).
    i_SSB:
        SSB candidate index (0..7 for L_max ≤ 8 or derived for larger L_max).
    length:
        Number of complex PBCH-DMRS symbols (default 144 REs per SS/PBCH block).
    """
    if not (0 <= N_ID < 1008):
        raise ValueError("N_ID must be in [0, 1008)")
    if i_SSB < 0:
        raise ValueError("i_SSB must be non-negative")
    c_init = (
        (1 << 11) * ((int(i_SSB) + 1) * ((int(N_ID) // 4) + 1))
        + (1 << 6) * (int(i_SSB) + 1)
        + (int(N_ID) % 4)
    ) & 0x7FFFFFFF
    c = pseudo_random(c_init, 2 * length)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    real = inv_sqrt2 * (1.0 - 2.0 * c[0::2].astype(np.float64))
    imag = inv_sqrt2 * (1.0 - 2.0 * c[1::2].astype(np.float64))
    return (real + 1j * imag).astype(np.complex128)
