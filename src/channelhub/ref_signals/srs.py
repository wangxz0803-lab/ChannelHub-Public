"""Sounding Reference Signal (SRS) per 3GPP TS 38.211 §6.4.1.4.

Covers:
    * Transmission combs K_TC ∈ {2, 4, 8} (§6.4.1.4.2).
    * Cyclic shift α_i for antenna ports 1000 + i (6, 8, or 12 shifts,
      depending on the transmission comb).
    * Complete C_SRS / B_SRS bandwidth configuration table
      (Table 6.4.1.4.3-1, all 64 rows).
    * Sequence-group / sequence hopping (§6.4.1.4.2) driven by the NR Gold PRBS.
    * Base-sequence selection: long (ZC) for Msc ≥ 36, short (CG tables) for
      Msc ∈ {6, 12, 18, 24}.

The main API ``srs_sequence`` returns the complex ``r̃_{u,v}^{α}(n)`` of
length ``Msc`` with the per-port cyclic shift already applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gold import pseudo_random
from .srs_rel15_tables import (
    REL15_TABLE_SEMANTICS,
    SRS_BW_TABLE,
    SRSBandwidthRow,
)
from .zc import _next_prime_below, r_uv_long, r_uv_short


__all__ = [
    "SRSBandwidthRow",
    "SRS_BW_TABLE",
    "SRS_PERIODICITY_TABLE",
    "SRSResourceConfig",
    "srs_cyclic_shift",
    "srs_base_sequence",
    "srs_sequence",
    "srs_group_number",
    "srs_freq_position",
    "srs_rb_indices",
    "srs_hopping_cycle_length",
    "srs_subcarrier_indices",
    "srs_frequency_position_indices",
    "srs_transmission_counter",
    "srs_port_comb_offset",
    "REL15_TABLE_SEMANTICS",
]


# -----------------------------------------------------------------------------
# 38.211 Table 6.4.1.4.3-1: (C_SRS, B_SRS=0..3 → m_SRS, N)
# Each row stores m_SRS,b and N_b for b=0..3.
# -----------------------------------------------------------------------------


def srs_cyclic_shift(n_cs: int, n_ap_index: int, N_ap: int, K_TC: int) -> float:
    """Compute α_i for port 1000 + i  per 38.211 §6.4.1.4.2 eq (6.4.1.4.2-2).

        α_i = 2·π · (n_cs + (n_cs^max · (p_i - 1000)) / N_ap) / n_cs^max  (mod 2π)

    where Table 6.4.1.4.2-1 gives n_cs^max = 8, 12, and 6 for
    K_TC = 2, 4, and 8, respectively.
    """
    if K_TC not in (2, 4, 8):
        raise ValueError("K_TC must be in {2, 4, 8}")
    n_cs_max = {2: 8, 4: 12, 8: 6}[K_TC]
    if not (0 <= n_cs < n_cs_max):
        raise ValueError(f"n_cs must be in [0, {n_cs_max})")
    if N_ap not in (1, 2, 4):
        raise ValueError("N_ap must be in {1, 2, 4}")
    if not (0 <= n_ap_index < N_ap):
        raise ValueError(f"n_ap_index must be in [0, {N_ap})")
    n_cs_i = (n_cs + (n_cs_max * n_ap_index) // N_ap) % n_cs_max
    return 2.0 * np.pi * n_cs_i / n_cs_max


def srs_group_number(
    n_SRS_ID: int,
    slot: int,
    symbol: int,
    group_hopping: bool,
    sequence_hopping: bool,
    *,
    numerology: int = 1,
    symbols_per_slot: int = 14,
) -> tuple[int, int]:
    """Compute (u, v) for SRS per 38.211 §6.4.1.4.2 / §5.2.2.

    * No hopping:  u = n_SRS_ID mod 30,  v = 0.
    * Group hopping: u = (f_gh + n_SRS_ID) mod 30 with f_gh from the PRBS.
    * Sequence hopping: v = c(n_s·N_symb_slot + l)  if Msc ≥ 72 (see caller).
    """
    if group_hopping and sequence_hopping:
        raise ValueError("group_hopping and sequence_hopping are mutually exclusive")

    if numerology not in (0, 1, 2, 3, 4):
        raise ValueError("numerology must be in [0, 4]")
    if symbols_per_slot not in (12, 14) or (symbols_per_slot == 12 and numerology != 2):
        raise ValueError("12-symbol extended CP requires numerology=2")
    if not 0 <= int(symbol) < symbols_per_slot:
        raise ValueError("symbol is outside the physical slot")
    if not 0 <= int(n_SRS_ID) <= 1023:
        raise ValueError("SRS sequenceId must be in [0, 1023]")
    # PRBS resets every 10-ms radio frame; frequency hopping uses a separate
    # absolute-slot counter. Negative history queries retain their frame phase.
    _slot = int(slot) % (10 * 2**numerology)

    if not group_hopping:
        f_gh = 0
    else:
        N_symb_slot = symbols_per_slot
        offset = 8 * (N_symb_slot * _slot + int(symbol))
        c_init = n_SRS_ID
        c = pseudo_random(c_init, offset + 8)
        f_gh = 0
        for i in range(8):
            f_gh = (f_gh + int(c[offset + i]) * (1 << i)) % 30
    u = (f_gh + n_SRS_ID) % 30

    if not sequence_hopping:
        v = 0
    else:
        N_symb_slot = symbols_per_slot
        c_init = n_SRS_ID & 0x7FFFFFFF
        idx = N_symb_slot * _slot + int(symbol)
        c = pseudo_random(c_init, idx + 1)
        v = int(c[idx])
    return u, v


def srs_base_sequence(u: int, v: int, Msc: int) -> np.ndarray:
    """Return r̄_{u,v}(n) of length ``Msc`` per 38.211 §5.2.2."""
    if Msc in (6, 12, 18, 24):
        return r_uv_short(u, v, Msc)
    if Msc < 36:
        raise ValueError(f"Msc {Msc} not supported (need Msc ≥ 36 for long sequence)")
    Nzc = _next_prime_below(Msc)
    r = r_uv_long(u, v, Nzc)
    if Msc == Nzc:
        return r
    # Cyclic extension to Msc: r̄(n) = x_q(n mod Nzc).
    n = np.arange(Msc)
    return r[n % Nzc]


def srs_sequence(
    n_SRS_ID: int,
    K_TC: int,
    n_cs: int,
    N_ap: int,
    Msc: int,
    slot: int,
    symbol: int,
    n_ap_index: int = 0,
    group_hopping: bool = False,
    sequence_hopping: bool = False,
    u_override: int | None = None,
    v_override: int | None = None,
    *,
    numerology: int = 1,
    symbols_per_slot: int = 14,
) -> np.ndarray:
    """Return r̃_{u,v}^{α_i}(n) of length ``Msc`` per 38.211 §6.4.1.4.2.

    The output is

        r̃(n) = exp(j · α_i · n) · r̄_{u,v}(n),   n = 0..Msc-1

    with α_i the port-specific cyclic shift and (u, v) the SRS group/sequence
    indices from :func:`srs_group_number`.

    Parameters
    ----------
    n_SRS_ID:
        SRS scrambling identity (0..1023).
    K_TC:
        Transmission comb in {2, 4, 8}.
    n_cs:
        Cyclic shift index (0..7 for K_TC=2, 0..11 for K_TC=4, and
        0..5 for K_TC=8).
    N_ap:
        Number of SRS ports (1, 2 or 4).
    Msc:
        Sub-carrier count per symbol (must equal m_SRS·N_sc/K_TC for the
        configured bandwidth row — not enforced here).
    slot, symbol:
        Absolute physical slot and OFDM symbol within the slot. PRBS uses
        the corresponding frame-local slot at the supplied numerology.
    n_ap_index:
        0..N_ap-1, picks the per-port cyclic shift.
    group_hopping, sequence_hopping:
        Mutually exclusive. Default: both off (ID hopping disabled).
    u_override, v_override:
        If provided, force (u, v) and skip the hopping computation (used for
        unit testing against known ZC / CG vectors).
    """
    if u_override is None or v_override is None:
        u, v = srs_group_number(
            n_SRS_ID,
            slot,
            symbol,
            group_hopping,
            sequence_hopping and Msc >= 72,
            numerology=numerology,
            symbols_per_slot=symbols_per_slot,
        )
    else:
        u, v = int(u_override), int(v_override)
    r_bar = srs_base_sequence(u, v, Msc)
    alpha = srs_cyclic_shift(n_cs, n_ap_index, N_ap, K_TC)
    n = np.arange(Msc, dtype=np.int64)
    return (np.exp(1j * alpha * n) * r_bar).astype(np.complex128)


# ---------------------------------------------------------------------------
# 38.211 Table 6.4.1.4.4-1 — SRS periodicity and slot offset
# T_SRS (slots) and the valid offset range [0, T_SRS-1].
# ---------------------------------------------------------------------------
SRS_PERIODICITY_TABLE: tuple[int, ...] = (
    1,
    2,
    4,
    5,
    8,
    10,
    16,
    20,
    32,
    40,
    64,
    80,
    160,
    320,
    640,
    1280,
    2560,
)


@dataclass(frozen=True)
class SRSResourceConfig:
    """Configuration for one SRS resource per 38.211 §6.4.1.4.

    Captures the parameters needed for frequency-domain position (hopping)
    and time-domain periodicity.
    """

    C_SRS: int  # Bandwidth configuration index (row in Table 6.4.1.4.3-1)
    B_SRS: int  # Bandwidth index 0..3 (column level in the BW tree)
    K_TC: int  # Transmission comb {2, 4, 8}
    n_RRC: int  # RRC-configured frequency-domain starting position
    b_hop: int  # Frequency hopping parameter: ≥ B_SRS means no hopping
    n_SRS_ID: int  # SRS scrambling identity (0..1023)
    T_SRS: int  # Periodicity in slots (from SRS_PERIODICITY_TABLE)
    T_offset: int  # Slot offset within period [0, T_SRS-1]
    N_ap: int = 1  # Number of SRS antenna ports {1, 2, 4}
    n_cs: int = 0  # Cyclic shift index
    group_hopping: bool = False
    sequence_hopping: bool = False
    R: int = 1  # Repetition factor {1, 2, 4}
    comb_offset: int = 0  # k_TC_offset: absolute comb phase within the grid
    n_shift: int = 0  # Independent frequency-domain shift, in RBs.
    N_symb: int = 1
    start_position: int = 0
    resource_type: str = "periodic"
    resource_id: int = 0
    resource_set_id: int = 0
    usage: str = "antennaSwitching"
    table_semantics: str = REL15_TABLE_SEMANTICS
    numerology: int = 1

    def __post_init__(self):
        if self.T_SRS not in SRS_PERIODICITY_TABLE:
            raise ValueError(f"T_SRS={self.T_SRS} not in SRS_PERIODICITY_TABLE")
        if not (0 <= self.T_offset < self.T_SRS):
            raise ValueError(f"T_offset must be in [0, {self.T_SRS})")
        if self.K_TC not in (2, 4, 8):
            raise ValueError("K_TC must be in {2, 4, 8}")
        if not (0 <= int(self.comb_offset) < self.K_TC):
            raise ValueError(f"comb_offset must be in [0, {self.K_TC}), got {self.comb_offset}")
        if not 0 <= int(self.b_hop) <= 3:
            raise ValueError("b_hop must be in [0, 3]")
        if not 0 <= int(self.B_SRS) <= 3:
            raise ValueError("B_SRS must be in [0, 3]")
        if self.table_semantics not in (REL15_TABLE_SEMANTICS,):
            raise ValueError("unsupported SRS table semantics")
        if not 0 <= self.C_SRS <= 63 or not 0 <= self.n_RRC <= 67 or not 0 <= self.n_shift <= 268:
            raise ValueError("C_SRS/n_RRC/n_shift outside standard resource ranges")
        if self.N_ap not in (1, 2, 4) or self.N_symb not in (1, 2, 4):
            raise ValueError("SRS ports and symbols must be one of 1, 2, 4")
        if self.R not in (1, 2, 4) or self.N_symb < self.R:
            raise ValueError("R must be one of 1, 2, 4 and no greater than N_symb")
        if not 0 <= self.start_position <= 5 or self.start_symbol + self.N_symb > 14:
            raise ValueError("SRS symbols must fit in the last six normal-CP symbols")
        if self.resource_type not in ("periodic", "semi-persistent", "aperiodic"):
            raise ValueError("unsupported SRS resource_type")
        if not 0 <= self.n_SRS_ID <= 1023 or self.numerology not in (0, 1, 2, 3, 4):
            raise ValueError("invalid SRS sequenceId or numerology")
        if self.group_hopping and self.sequence_hopping:
            raise ValueError("group_hopping and sequence_hopping are mutually exclusive")
        srs_cyclic_shift(self.n_cs, 0, self.N_ap, self.K_TC)

    def is_slot_active(self, slot: int) -> bool:
        """Check if SRS is transmitted in this slot per the periodicity."""
        return (slot - self.T_offset) % self.T_SRS == 0

    @property
    def hopping_enabled(self) -> bool:
        return self.b_hop < self.B_SRS

    @property
    def start_symbol(self) -> int:
        return 13 - self.start_position

    @property
    def symbol_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start_symbol, self.start_symbol + self.N_symb))


def _get_bw_row(C_SRS: int, table_semantics: str = REL15_TABLE_SEMANTICS) -> SRSBandwidthRow:
    """Look up bandwidth config row by C_SRS index."""
    if table_semantics not in (REL15_TABLE_SEMANTICS,):
        raise ValueError("unsupported SRS table semantics")
    for row in SRS_BW_TABLE:
        if row.c_srs == C_SRS:
            return row
    raise ValueError(f"C_SRS={C_SRS} not found in SRS_BW_TABLE (have 0..{SRS_BW_TABLE[-1].c_srs})")


def srs_transmission_counter(cfg: SRSResourceConfig, slot: int, symbol: int) -> int:
    """Absolute-occasion frequency counter, never reset at radio-frame boundaries.

    An l' in [0, N_symb) is retained for existing RB-only callers. Grid mapping
    validates real physical symbols before calling this low-level function.
    """
    if symbol in cfg.symbol_indices:
        l_prime = symbol - cfg.start_symbol
    elif 0 <= symbol < cfg.N_symb:
        l_prime = symbol
    else:
        raise ValueError("symbol is outside the configured SRS resource")
    if cfg.resource_type == "aperiodic":
        return l_prime // cfg.R
    return ((int(slot) - cfg.T_offset) // cfg.T_SRS) * (cfg.N_symb // cfg.R) + l_prime // cfg.R


def srs_frequency_position_indices(cfg: SRSResourceConfig, slot: int, symbol: int) -> tuple[int, ...]:
    return _srs_frequency_position_indices(
        cfg,
        _get_bw_row(cfg.C_SRS, cfg.table_semantics),
        srs_transmission_counter(cfg, slot, symbol),
    )


def _srs_frequency_position_indices(
    cfg: SRSResourceConfig,
    row: SRSBandwidthRow,
    n_srs: int,
) -> tuple[int, ...]:
    """Return ``n_b`` for levels 0 through ``B_SRS``.

    Implements the frequency-position and ``F_b(n_SRS)`` definitions in
    3GPP TS 38.211 V18.7.0, clause 6.4.1.4.3.  In the products used by
    ``F_b``, ``N[b_hop]`` is one regardless of the bandwidth-table value.
    """
    indices: list[int] = []
    hopping_enabled = cfg.hopping_enabled

    for b in range(cfg.B_SRS + 1):
        n_b_count = row.n[b]
        fixed_offset = ((4 * cfg.n_RRC) // row.m_srs[b]) % n_b_count

        if not hopping_enabled or b <= cfg.b_hop:
            indices.append(fixed_offset)
            continue

        # Product from b_hop through b-1 with N[b_hop] defined as one.
        product_before = 1
        for b_prime in range(cfg.b_hop + 1, b):
            product_before *= row.n[b_prime]

        if n_b_count % 2 == 0:
            wrapped_counter = n_srs % (product_before * n_b_count)
            hopping_offset = (n_b_count // 2) * (wrapped_counter // product_before)
            hopping_offset += wrapped_counter // (2 * product_before)
        else:
            hopping_offset = (n_b_count // 2) * (n_srs // product_before)

        indices.append((hopping_offset + fixed_offset) % n_b_count)

    return tuple(indices)


def srs_port_comb_offset(cfg: SRSResourceConfig, n_ap_index: int = 0) -> int:
    """Return k_TC^(p_i), including the Release-15 four-port rule."""

    if not 0 <= n_ap_index < cfg.N_ap:
        raise ValueError(f"n_ap_index must be in [0, {cfg.N_ap})")
    n_cs_max = {2: 8, 4: 12, 8: 6}[cfg.K_TC]
    if cfg.N_ap == 4 and n_ap_index in (1, 3) and cfg.n_cs >= n_cs_max // 2:
        return (cfg.comb_offset + cfg.K_TC // 2) % cfg.K_TC
    return cfg.comb_offset


def srs_freq_position(
    cfg: SRSResourceConfig,
    slot: int,
    symbol: int,
    n_ap_index: int = 0,
) -> tuple[int, int]:
    """Return ``(n_B_SRS, k0)`` with k0 in subcarriers."""

    row = _get_bw_row(cfg.C_SRS, cfg.table_semantics)
    n_b = srs_frequency_position_indices(cfg, slot, symbol)
    k0 = cfg.n_shift * 12 + srs_port_comb_offset(cfg, n_ap_index)
    for b, position in enumerate(n_b):
        # K_TC * M_sc,b = K_TC * (m_SRS,b*12/K_TC) = 12*m_SRS,b.
        k0 += 12 * row.m_srs[b] * position
    return n_b[cfg.B_SRS], k0


def srs_subcarrier_indices(
    cfg: SRSResourceConfig,
    slot: int,
    symbol: int,
    total_rb: int,
    n_ap_index: int = 0,
) -> np.ndarray:
    """Return the exact comb-spaced RE indices for one port and SRS symbol."""

    row = _get_bw_row(cfg.C_SRS, cfg.table_semantics)
    m_srs = row.m_srs[cfg.B_SRS]
    m_sc = m_srs * 12 // cfg.K_TC
    _, k0 = srs_freq_position(cfg, slot, symbol, n_ap_index)
    indices = k0 + cfg.K_TC * np.arange(m_sc, dtype=np.int64)
    if indices.size and (indices[0] < 0 or indices[-1] >= int(total_rb) * 12):
        raise ValueError(
            "SRS allocation exceeds the BWP: "
            f"RE range {indices[0]}..{indices[-1]}, BWP REs 0..{int(total_rb) * 12 - 1}. "
            "Adjust C_SRS/B_SRS, n_RRC or n_shift."
        )
    return indices


def srs_rb_indices(
    cfg: SRSResourceConfig,
    slot: int,
    symbol: int,
    total_rb: int,
) -> np.ndarray:
    """Return the exact RB projection of the SRS RE allocation.

    Allocations outside the configured BWP are rejected.
    """

    subcarriers = srs_subcarrier_indices(cfg, slot, symbol, total_rb)
    return np.unique(subcarriers // 12).astype(np.int64, copy=False)


def srs_hopping_cycle_length(cfg: SRSResourceConfig) -> int:
    """Number of SRS counters needed to sweep all hopping positions once.

    Per 38.211 §6.4.1.4.3, the cycle equals the product of N[b] for
    b = b_hop+1 .. B_SRS.  Returns 1 when hopping is disabled.
    """
    if not cfg.hopping_enabled:
        return 1
    row = _get_bw_row(cfg.C_SRS)
    cycle = 1
    for b in range(cfg.b_hop + 1, cfg.B_SRS + 1):
        cycle *= row.n[b]
    return max(cycle, 1)
