"""Gold pseudo-random sequence generator per 3GPP TS 38.211 §5.2.1.

The NR pseudo-random sequence c(n) is defined as the length-31 Gold sequence:

    c(n) = (x1(n + Nc) + x2(n + Nc)) mod 2
    x1(n+31) = (x1(n+3) + x1(n)) mod 2
    x2(n+31) = (x2(n+3) + x2(n+2) + x2(n+1) + x2(n)) mod 2

with Nc = 1600, the initial x1 fixed as x1(0)=1, x1(1..30)=0, and x2 initialized
from the 31-bit integer ``c_init`` such that ``c_init = sum_{i=0..30} x2(i)*2^i``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pseudo_random", "NC"]

# 3GPP constant: the Gold sequence pre-advance length.
NC: int = 1600


def _advance_x1(state: int, steps: int) -> int:
    """Advance the 31-bit x1 LFSR by ``steps`` ticks using 32-bit bitwise maths.

    x1 recurrence: new_bit = x1[0] XOR x1[3]. We keep the state as a 31-bit int
    where bit i holds x1(i+offset).
    """
    s = state & 0x7FFFFFFF
    for _ in range(steps):
        new_bit = ((s >> 0) ^ (s >> 3)) & 1
        s = (s >> 1) | (new_bit << 30)
    return s & 0x7FFFFFFF


def _advance_x2(state: int, steps: int) -> int:
    """Advance the 31-bit x2 LFSR by ``steps`` ticks.

    x2 recurrence: new_bit = x2[0] XOR x2[1] XOR x2[2] XOR x2[3].
    """
    s = state & 0x7FFFFFFF
    for _ in range(steps):
        new_bit = ((s >> 0) ^ (s >> 1) ^ (s >> 2) ^ (s >> 3)) & 1
        s = (s >> 1) | (new_bit << 30)
    return s & 0x7FFFFFFF


def pseudo_random(c_init: int, length: int) -> np.ndarray:
    """Generate the NR length-31 Gold sequence c(n) for n = 0..length-1.

    Parameters
    ----------
    c_init:
        31-bit initial value used to seed x2.
    length:
        Number of output bits to generate.

    Returns
    -------
    np.ndarray[int8]
        Vector of shape ``(length,)`` with values in {0, 1}.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if not (0 <= int(c_init) < (1 << 31)):
        raise ValueError("c_init must fit in 31 bits (0..2**31-1)")

    # x1(0)=1, x1(1..30)=0 per §5.2.1.
    x1 = 1 & 0x7FFFFFFF
    # x2 initialised from c_init: bit i of c_init = x2(i).
    x2 = int(c_init) & 0x7FFFFFFF

    # Advance both by Nc=1600 samples.
    x1 = _advance_x1(x1, NC)
    x2 = _advance_x2(x2, NC)

    out = np.empty(length, dtype=np.int8)
    for n in range(length):
        # c(n) = x1(n+Nc) XOR x2(n+Nc) — the lowest bit is the current output.
        out[n] = (x1 ^ x2) & 1
        # Step LFSRs by one.
        b1 = ((x1 >> 0) ^ (x1 >> 3)) & 1
        x1 = ((x1 >> 1) | (b1 << 30)) & 0x7FFFFFFF
        b2 = ((x2 >> 0) ^ (x2 >> 1) ^ (x2 >> 2) ^ (x2 >> 3)) & 1
        x2 = ((x2 >> 1) | (b2 << 30)) & 0x7FFFFFFF
    return out
