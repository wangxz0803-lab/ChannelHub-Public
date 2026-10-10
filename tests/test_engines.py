import os

import numpy as np
import pytest

from channelhub.channels import C
from channelhub.config import RunConfig, Transmitter, Waveform, Array, Receiver, RayTracing
from channelhub.simulation import simulate, save_result


@pytest.mark.statistical
@pytest.mark.parametrize("profile", [f"{family}-{p}" for family in ["CDL", "TDL"] for p in "ABCDE"])
def test_statistical_profiles(profile):
    pytest.importorskip("sionna.phy")
    cfg = RunConfig(profile=profile, num_rb=4, num_slots=1)
    data, meta = simulate(cfg)
    assert data["desired_h"].shape == (14, 2, 1, 48)
    assert all(np.isfinite(v).all() for v in data.values())
    assert np.any(abs(data["desired_h"]) > 0)
    np.testing.assert_allclose(data["y"], data["signal"] + data["interference"] + data["noise"])
    assert meta["axes"]["h"] == ["time", "rx_port", "tx_port", "subcarrier"]


@pytest.mark.statistical
def test_reproducibility_power_and_interference_isolation(tmp_path):
    pytest.importorskip("sionna.phy")
    cfg = RunConfig(
        profile="TDL-A",
        num_rb=4,
        num_slots=1,
        noise_enabled=False,
        tx=Transmitter(waveform=Waveform(kind="qpsk")),
    )
    a, m = simulate(cfg)
    b, _ = simulate(cfg)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    doubled = cfg.model_copy(deep=True)
    doubled.tx.power_dbm = 10 * np.log10(4)
    d, _ = simulate(doubled)
    np.testing.assert_allclose(d["signal"], 2 * a["signal"], rtol=1e-12)
    interfered = cfg.model_copy(deep=True)
    interfered.interferers = [Transmitter(position_m=(40, 30, 10), waveform=Waveform(kind="qpsk"))]
    c, _ = simulate(interfered)
    np.testing.assert_array_equal(a["desired_h"], c["desired_h"])
    np.testing.assert_array_equal(a["signal"], c["signal"])
    assert np.mean(abs(c["interference"]) ** 2) > 0
    np.testing.assert_allclose(c["y"] - a["y"], c["interference"], atol=1e-15)
    save_result(tmp_path, a, m)
    with np.load(tmp_path / "observations.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["y"], a["y"])
    with pytest.raises(FileExistsError):
        save_result(tmp_path, a, m)


@pytest.mark.statistical
def test_noise_psd_scaling():
    pytest.importorskip("sionna.phy")
    cfg = RunConfig(profile="TDL-A", num_slots=2, num_rb=24)
    data, meta = simulate(cfg)
    expected = 10 ** ((-174 + 5 - 30) / 10) * 30000
    assert meta["noise_variance_w_per_re"] == pytest.approx(expected)
    assert np.mean(abs(data["noise"]) ** 2) == pytest.approx(expected, rel=0.05)
    assert abs(np.mean(data["noise"])) < np.sqrt(expected) * 0.04


@pytest.mark.statistical
def test_dual_polarized_cdl_with_motion():
    pytest.importorskip("sionna.phy")
    cfg = RunConfig(
        num_rb=4,
        num_slots=2,
        tx=Transmitter(array=Array(cols=2, polarization="VH")),
        rx=Receiver(array=Array(polarization="VH"), velocity_m_s=(5, 0, 0)),
    )
    data, _ = simulate(cfg)
    assert data["desired_h"].shape == (28, 2, 4, 48)
    assert not np.array_equal(data["desired_h"][0], data["desired_h"][-1])


@pytest.mark.rt
@pytest.mark.skipif(os.environ.get("CHANNELHUB_TEST_RT") != "1", reason="opt-in native RT test")
def test_rt_friis_absolute_delay_and_reflection():
    cfg = RunConfig(
        engine="sionna_rt",
        num_rb=4,
        num_slots=1,
        rx=Receiver(array=Array()),
        rt=RayTracing(max_depth=0, samples_per_source=1000),
    )
    data, _ = simulate(cfg)
    distance = np.linalg.norm(np.asarray(cfg.tx.position_m) - cfg.rx.position_m)
    expected = C / (cfg.carrier_ghz * 1e9) / (4 * np.pi * distance)
    np.testing.assert_allclose(abs(data["desired_h"]), expected, rtol=0.03)
    np.testing.assert_allclose(data["desired_delays_s"], distance / C, rtol=2e-5)
    reflected = cfg.model_copy(deep=True)
    reflected.rt.scene = "reflector"
    reflected.rt.max_depth = 1
    reflection, _ = simulate(reflected)
    assert reflection["desired_cir"].shape[-1] >= 2
    expected_delays = np.sort([distance / C, np.linalg.norm([50, 0, 11.5]) / C])
    np.testing.assert_allclose(np.sort(reflection["desired_delays_s"][0, 0]), expected_delays, rtol=3e-5)


@pytest.mark.rt
@pytest.mark.skipif(os.environ.get("CHANNELHUB_TEST_RT") != "1", reason="opt-in native RT test")
def test_rt_mimo_interferers_keep_each_links_distinct_path_delays():
    array = Array(cols=2)
    cfg = RunConfig(
        engine="sionna_rt",
        num_rb=4,
        num_slots=1,
        tx=Transmitter(array=array),
        rx=Receiver(array=Array(cols=4)),
        interferers=[
            Transmitter(position_m=(70, 10, 12), array=array),
            Transmitter(position_m=(-20, 15, 8), array=array),
        ],
        rt=RayTracing(scene="reflector", max_depth=1, samples_per_source=10000),
    )
    data, _ = simulate(cfg)
    for i, tx in enumerate([cfg.tx, *cfg.interferers]):
        prefix = "desired" if i == 0 else f"interferer_{i - 1}"
        assert data[prefix + "_h"].shape == (14, 4, 2, 48)
        position = np.asarray(tx.position_m)
        image_position = position * [1, 1, -1]
        expected = np.sort(
            [
                np.linalg.norm(position - cfg.rx.position_m) / C,
                np.linalg.norm(image_position - cfg.rx.position_m) / C,
            ]
        )
        actual = np.sort(data[prefix + "_delays_s"], axis=-1)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, (4, 2, 2)), rtol=3e-5)
