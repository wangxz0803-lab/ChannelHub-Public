"""Explicit public experiment configuration; unknown fields are rejected."""

from __future__ import annotations

from typing import Literal
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Array(Settings):
    rows: int = Field(1, ge=1, le=8)
    cols: int = Field(1, ge=1, le=8)
    polarization: Literal["V", "H", "VH"] = "V"
    pattern: Literal["iso", "3gpp"] = "iso"
    spacing_lambda: float = Field(0.5, gt=0, le=4)
    yaw_deg: float = Field(0, ge=-180, le=180)

    @property
    def ports(self) -> int:
        return self.rows * self.cols * (2 if self.polarization == "VH" else 1)

    @model_validator(mode="after")
    def check_ports(self):
        if self.ports > 64:
            raise ValueError("An array may have at most 64 physical antenna ports")
        return self


class SRS(Settings):
    c_srs: int = Field(0, ge=0, le=63)
    b_srs: int = Field(0, ge=0, le=3)
    b_hop: int = Field(3, ge=0, le=3)
    comb: Literal[2, 4] = 2
    comb_offset: int = Field(0, ge=0, le=3)
    cyclic_shift: int = Field(0, ge=0, le=11)
    n_rrc: int = Field(0, ge=0, le=67)
    shift_rb: int = Field(0, ge=0, le=268)
    ports: Literal[1, 2, 4] = 1
    period_slots: int = Field(1, ge=1, le=2560)
    offset_slots: int = Field(0, ge=0, le=2559)
    symbols: Literal[1, 2, 4] = 1
    start_position: int = Field(0, ge=0, le=5)
    repetition: Literal[1, 2, 4] = 1
    group_hopping: bool = False
    sequence_hopping: bool = False


class Waveform(Settings):
    kind: Literal["srs", "csi_rs", "ssb", "qpsk", "custom"] = "srs"
    identity: int = Field(0, ge=0, le=1007)
    srs: SRS = Field(default_factory=SRS)
    csi_symbol: int = Field(4, ge=0, le=13)
    csi_subcarrier: int = Field(0, ge=0, le=11)
    ssb_start_symbol: int = Field(2, ge=0, le=10)
    ssb_index: int = Field(0, ge=0, le=7)
    beam_index: int = Field(0, ge=0, le=63)


class Transmitter(Settings):
    position_m: tuple[float, float, float] = (0, 0, 10)
    power_dbm: float = Field(0, ge=-100, le=60, description="Total power per active RE across TX ports")
    array: Array = Field(default_factory=Array)
    waveform: Waveform = Field(default_factory=Waveform)
    activity: float = Field(1, ge=0, le=1, description="Independent per-slot transmit probability")
    pathloss_db: float | None = Field(
        None, ge=0, le=250, description="Statistical engine only; null uses free-space loss"
    )


class Receiver(Settings):
    position_m: tuple[float, float, float] = (50, 0, 1.5)
    velocity_m_s: tuple[float, float, float] = (0, 0, 0)
    array: Array = Field(default_factory=lambda: Array(cols=2, yaw_deg=180))


class RayTracing(Settings):
    scene: Literal["empty", "reflector"] = "empty"
    max_depth: int = Field(2, ge=0, le=6)
    samples_per_source: int = Field(10000, ge=100, le=1000000)
    line_of_sight: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False


class RunConfig(Settings):
    engine: Literal["statistical", "sionna_rt"] = "statistical"
    seed: int = Field(42, ge=0, le=2**31 - 1)
    carrier_ghz: float = Field(3.5, ge=0.1, le=100)
    scs_khz: Literal[15, 30, 60, 120] = 30
    num_rb: int = Field(24, ge=4, le=100)
    num_slots: int = Field(2, ge=1, le=32)
    profile: Literal[
        "CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E", "TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"
    ] = "CDL-C"
    delay_spread_ns: float = Field(100, gt=0, le=10000)
    tx: Transmitter = Field(default_factory=Transmitter)
    rx: Receiver = Field(default_factory=Receiver)
    interferers: list[Transmitter] = Field(default_factory=list, max_length=8)
    noise_psd_dbm_hz: float = Field(-174, ge=-240, le=-80)
    noise_figure_db: float = Field(5, ge=0, le=40)
    noise_enabled: bool = True
    rt: RayTracing = Field(default_factory=RayTracing)

    @property
    def num_times(self):
        return 14 * self.num_slots

    @property
    def num_subcarriers(self):
        return 12 * self.num_rb

    @property
    def symbol_interval_s(self):
        # Slot-average normal-CP OFDM symbol spacing. Exported explicitly.
        return (1e-3 * 15 / self.scs_khz) / 14

    @model_validator(mode="after")
    def check_experiment(self):
        points = [self.rx.position_m, self.rx.velocity_m_s]
        points += [tx.position_m for tx in [self.tx, *self.interferers]]
        if any(not math.isfinite(x) or abs(x) > 1e6 for p in points for x in p):
            raise ValueError("Coordinates and velocities must be finite and within ±1e6")
        if math.sqrt(sum(v * v for v in self.rx.velocity_m_s)) > 200:
            raise ValueError("Receiver speed must not exceed 200 m/s")
        for index, tx in enumerate([self.tx, *self.interferers]):
            if tx.position_m == self.rx.position_m:
                raise ValueError("Transmitter and receiver positions must differ")
            if index and tx.waveform.kind == "custom":
                raise ValueError(
                    "Custom X is supported for the desired transmitter; use the Python composition API for arbitrary interferers"
                )
            if tx.waveform.kind == "srs" and tx.waveform.srs.ports > tx.array.ports:
                raise ValueError("SRS ports cannot exceed transmitter antenna ports")
            if tx.waveform.kind == "ssb" and self.num_rb < 20:
                raise ValueError("SSB requires at least 240 subcarriers (20 RB)")
            if tx.waveform.beam_index >= tx.array.ports:
                raise ValueError("Beam index must be smaller than the transmitter port count")
            if self.engine == "sionna_rt" and tx.pathloss_db is not None:
                raise ValueError("RT computes path loss from geometry; remove pathloss_db")
        if self.engine == "sionna_rt" and any(tx.array != self.tx.array for tx in self.interferers):
            raise ValueError("The RT engine requires the same array configuration for every transmitter")
        elements = (
            self.num_times
            * self.num_subcarriers
            * self.rx.array.ports
            * sum(t.array.ports for t in [self.tx, *self.interferers])
        )
        if elements > 12_000_000:
            raise ValueError(
                "Experiment exceeds the 12 million channel-coefficient limit; reduce grid, slots, antennas or interferers"
            )
        return self
