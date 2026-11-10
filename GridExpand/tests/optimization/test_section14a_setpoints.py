import numpy as np
import pandas as pd
import pytest

from gridexpand.optimization.urbs.features.section14a import P14A_KW, _smallest_ratio
from gridexpand.optimization.urbs.runfunctions import run_lvds_opt

from lp_fixture import write_input

pytest.importorskip("highspy")


def _fixture(tmp_path):
    """LP fixture with a larger space-heat demand at building 1, so its heat pump is a controllable device."""
    path = write_input(tmp_path / "1_fixture.h5")
    with pd.HDFStore(path, "a") as store:
        demand = store["urbs_in/demand"]
        demand[(1, "space_heat")] *= 8
        store["urbs_in/demand"] = demand
        process = store["urbs_in/process"]
        process.loc[process.Process.eq("heatpump_air") & process.Site.eq(1), "cap-up"] = 20.0
        store["urbs_in/process"] = process
    return path


def _run(path, tmp_path, **extra):
    settings = {"input_file": path.name, "tsam": False, "noTypicalPeriods": 1, "hoursPerPeriod": 168,
                "n_cpu": 1, "scenario_key": "fx", **extra}
    return run_lvds_opt(str(path), str(tmp_path), settings, log_dir=str(tmp_path / "logs"),
                        solver="appsi_highs", concurrency=1)


def _draws(path, process):
    e_in = pd.read_hdf(path, "urbs_out/MILP/e_pro_in").rename("val").reset_index()
    return e_in[e_in.pro.eq(process) & e_in.com.eq("electricity")].set_index(["t", "sit"]).val


def test_smallest_ratio():
    # two devices: 11 kW drawing 11 kW and 11 kW drawing 6 kW; both can go down to 4.2 kW
    span = 11.0 - P14A_KW
    segments = [(0.0, span), ((11.0 - 6.0) / span, span)]
    ratio, cleared = _smallest_ratio(3.0, segments)
    assert cleared and ratio == pytest.approx(3.0 / span)
    ratio, cleared = _smallest_ratio(100.0, segments)
    assert not cleared and ratio == 1.0


def test_setpoints_limit_the_heat_pump_in_congested_hours(tmp_path):
    path = _fixture(tmp_path)
    first = _run(path, tmp_path)
    second = _run(path, tmp_path, p14a_ref=str(first), p14a_rating_kva=5.0)
    assert second.name.endswith("_p14a.h5") and first.exists()

    before, after = _draws(first, "heatpump_air"), _draws(second, "heatpump_air")
    cap_first = pd.read_hdf(first, "urbs_out/MILP/cap_pro")
    cap_second = pd.read_hdf(second, "urbs_out/MILP/cap_pro")
    key = (2026, 1, "heatpump_air")
    # the controlled heat pump keeps its first-run capacity and is held at 4.2 kW in the congested hour
    assert cap_second[key] == pytest.approx(cap_first[key])
    assert before[(1, 1)] > P14A_KW
    assert after[(1, 1)] == pytest.approx(P14A_KW)
    # the heat it cannot deliver comes from the last-resort source
    backup = pd.read_hdf(second, "urbs_out/MILP/e_pro_in").reset_index()
    assert backup.loc[backup.pro.eq("heat_backup"), "e_pro_in"].sum() > 0


def test_no_congestion_needs_no_last_resort_source(tmp_path):
    path = _fixture(tmp_path)
    first = _run(path, tmp_path)
    second = _run(path, tmp_path, p14a_ref=str(first), p14a_rating_kva=1000.0)
    e_in = pd.read_hdf(second, "urbs_out/MILP/e_pro_in").reset_index()
    assert e_in.loc[e_in.pro.eq("heat_backup"), "e_pro_in"].sum() == pytest.approx(0.0)
    # the controlled heat pump never draws more than in the first run plus its share of the headroom
    assert np.all(_draws(second, "heatpump_air").to_numpy() <= 20.0 + 1e-9)
