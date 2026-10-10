import numpy as np
import pytest

from channelhub.channels import taps_to_frequency
from channelhub.config import RunConfig, Transmitter, Waveform, Array, SRS
from channelhub.signals import apply_channel, make_grid, dft_weights
from channelhub.ref_signals.gold import pseudo_random


def test_single_path_amplitude_delay_phase():
    delay = np.array([[[175e-9]]])
    coefficient = 0.1 * np.exp(0.3j)
    taps = np.full((3, 1, 1, 1), coefficient)
    f = np.linspace(-2e6, 2e6, 33)
    h = taps_to_frequency(taps, delay, f)
    np.testing.assert_allclose(h[0, 0, 0], coefficient * np.exp(-2j * np.pi * f * 175e-9), rtol=1e-14)
    np.testing.assert_allclose(abs(h), 0.1)


def test_padded_invalid_path_does_not_contribute():
    h = taps_to_frequency(np.array([[[[2.0, 999.0]]]]), np.array([[[0.0, -1.0]]]), np.arange(9))
    np.testing.assert_array_equal(h, np.full((1, 1, 1, 9), 2))


def test_mimo_complex_superposition_without_conjugation():
    h = np.array([[[[1 + 2j], [3 - 4j]], [[5j], [2]]]])
    x = np.array([[[2 + 1j], [1 - 3j]]])
    y = apply_channel(h, x)
    np.testing.assert_allclose(
        y[0, :, 0], [(1 + 2j) * (2 + 1j) + (3 - 4j) * (1 - 3j), 5j * (2 + 1j) + 2 * (1 - 3j)]
    )
    with pytest.raises(ValueError):
        apply_channel(h, x[:, :, :0])


def test_gold_against_independent_bit_array_recurrence():
    count = 1650
    for seed in [0, 1, 0x12345678, 2**31 - 1]:
        a, b = np.zeros(count + 31, dtype=int), np.zeros(count + 31, dtype=int)
        a[0] = 1
        b[:31] = [(seed >> i) & 1 for i in range(31)]
        for i in range(count):
            a[i + 31] = (a[i + 3] + a[i]) % 2
            b[i + 31] = (b[i + 3] + b[i + 2] + b[i + 1] + b[i]) % 2
        np.testing.assert_array_equal(pseudo_random(seed, 50), (a[1600:1650] + b[1600:1650]) % 2)


@pytest.mark.parametrize("kind,count", [("srs", 24), ("csi_rs", 24), ("ssb", 398), ("qpsk", 14 * 288)])
def test_reference_resource_counts_and_total_re_power(kind, count):
    cfg = RunConfig(
        num_slots=1, tx=Transmitter(array=Array(cols=2), waveform=Waveform(kind=kind), power_dbm=7)
    )
    x, mask = make_grid(cfg, cfg.tx, np.random.default_rng(1))
    assert mask.sum() == count
    np.testing.assert_allclose(np.sum(abs(x) ** 2, axis=1)[mask], 10 ** ((7 - 30) / 10), rtol=1e-13)


def test_srs_periodicity_and_multisymbol_repetition():
    cfg = RunConfig(
        num_slots=4,
        num_rb=4,
        tx=Transmitter(
            waveform=Waveform(
                srs=SRS(period_slots=2, offset_slots=1, symbols=2, start_position=1, repetition=2)
            )
        ),
    )
    x, mask = make_grid(cfg, cfg.tx, np.random.default_rng(1))
    assert np.flatnonzero(mask.any(axis=1)).tolist() == [26, 27, 54, 55]
    np.testing.assert_array_equal(x[26], x[27])


def test_srs_four_ports_with_split_comb_retains_re_power():
    cfg = RunConfig(
        num_slots=1,
        num_rb=4,
        tx=Transmitter(array=Array(cols=4), waveform=Waveform(srs=SRS(ports=4, cyclic_shift=4))),
    )
    x, mask = make_grid(cfg, cfg.tx, np.random.default_rng(1))
    np.testing.assert_allclose(np.sum(abs(x) ** 2, axis=1)[mask], 0.001)


def test_csi_rs_sequence_repeats_at_frame_boundary():
    cfg = RunConfig(num_slots=21, num_rb=4, tx=Transmitter(waveform=Waveform(kind="csi_rs")))
    x, _ = make_grid(cfg, cfg.tx, np.random.default_rng(1))
    np.testing.assert_array_equal(x[4], x[20 * 14 + 4])


def test_custom_x_amplitude_preserved_and_invalid_data_rejected():
    cfg = RunConfig(num_slots=1, num_rb=4, tx=Transmitter(waveform=Waveform(kind="custom"), power_dbm=55))
    x = np.full((14, 1, 48), 3 + 7j)
    actual, _ = make_grid(cfg, cfg.tx, np.random.default_rng(1), x)
    np.testing.assert_array_equal(actual, x)
    for bad in [x.real, x[:2], x * np.nan]:
        with pytest.raises(ValueError):
            make_grid(cfg, cfg.tx, np.random.default_rng(1), bad)


def test_dft_beams_are_orthonormal():
    w = np.stack([dft_weights(8, i) for i in range(8)])
    np.testing.assert_allclose(w @ w.conj().T, np.eye(8), atol=2e-15)


def test_unknown_config_and_out_of_bounds_resources_fail():
    with pytest.raises(ValueError):
        RunConfig.model_validate({"unpublished_setting": True})
    with pytest.raises(ValueError):
        RunConfig(tx={"position_m": [float("nan"), 0, 0]})
    cfg = RunConfig(num_rb=4, tx=Transmitter(waveform=Waveform(srs=SRS(c_srs=63))))
    with pytest.raises(ValueError, match="exceeds"):
        make_grid(cfg, cfg.tx, np.random.default_rng(0))
