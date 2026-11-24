import numpy as np
import pandas as pd
import pytest

from gridexpand.powerflow import section14a
from gridexpand.powerflow.demands import _vehicles
from gridexpand.powerflow.run_pwrflw import build_assumptions, parse_args

PF = 1.0  # unit power factors keep the arithmetic of the tests readable


def _heat_pump(heat, cop=3.0, cap_hp=4.0, cap_rod=8.0):
    heat = np.asarray(heat, dtype=float)
    return dict(site=1, heat=heat, cop=np.full(len(heat), cop), cap_hp=cap_hp, cap_rod=cap_rod)


def _vehicle(fraction, session, energy, cap=11.0):
    fraction = np.asarray(fraction, dtype=float)
    session = np.asarray(session, dtype=int)
    # uncontrolled: full power from arrival until the session energy is delivered
    uncontrolled = np.zeros(len(fraction))
    left = dict(energy)
    for t, s in enumerate(session):
        if s >= 0:
            uncontrolled[t] = min(cap * fraction[t], left[s])
            left[s] -= uncontrolled[t]
    return dict(key=(1, "charging_station0"), cap=cap, fraction=fraction, session=session,
                energy=dict(energy), uncontrolled=uncontrolled)


def test_heat_conversion_round_trip():
    heat = np.array([0.0, 6.0, 12.0, 30.0])
    cop = np.array([3.0, 3.0, 3.0, 3.0])
    el = section14a.heat_to_el(heat, cop, cap_hp=4.0)
    assert np.allclose(el, [0.0, 2.0, 4.0, 4.0 + 18.0])
    assert np.allclose(section14a.el_to_heat(el, cop, cap_hp=4.0), heat)


def test_no_congestion_leaves_profiles_unchanged():
    hp = _heat_pump([3.0, 6.0, 9.0, 3.0])
    ev = _vehicle([1, 1, 0, 0], [0, 0, -1, -1], {0: 15.0})
    hps, evs, diag = section14a.control(np.ones(4), np.zeros(4), [hp], [ev], 1000.0, PF, PF)
    assert diag["congested_hours"] == 0
    assert np.allclose(hps[0]["heat_served"], hp["heat"])
    assert np.allclose(evs[0]["charge"], evs[0]["uncontrolled"])
    assert diag["unserved_heat_kwh"] == diag["unserved_ev_kwh"] == 0.0


def test_congested_hour_is_cleared_and_energy_returned_later():
    # hour 0: base 10 + EV 11 + HP 4 = 25 kW against a 20 kVA rating
    hp = _heat_pump([12.0, 0.0, 0.0, 0.0])
    ev = _vehicle([1, 1, 1, 0], [0, 0, 0, -1], {0: 11.0})
    base = np.array([10.0, 2.0, 2.0, 2.0])
    hps, evs, diag = section14a.control(base, np.zeros(4), [hp], [ev], 20.0, PF, PF)

    hp_el = hps[0]["el"]
    load = base + evs[0]["charge"] + hp_el
    assert diag["congested_hours"] == 1 and diag["hours_not_cleared"] == 0
    assert load[0] <= 20.0 + 1e-9
    # the heat pump draws 4 kW, below its 4.2 kW minimum, and is left alone; the charge point takes the cut
    assert hp_el[0] == pytest.approx(4.0)
    assert evs[0]["charge"][0] == pytest.approx(6.0)
    assert evs[0]["charge"][0] >= section14a.P14A_KW
    # deferred energy is returned within the session and the heat backlog is cleared
    assert evs[0]["charge"].sum() == pytest.approx(11.0)
    assert hps[0]["heat_served"].sum() + hps[0]["unserved_kwh"] == pytest.approx(12.0)
    assert diag["unserved_ev_kwh"] == pytest.approx(0.0)
    assert np.all(load <= 20.0 + 1e-9)


def test_ev_energy_not_returned_before_departure_is_unserved():
    # the car leaves right after the congested hour, so the deferred energy cannot come back
    ev = _vehicle([1, 0], [0, -1], {0: 11.0})
    _, evs, diag = section14a.control(np.array([15.0, 0.0]), np.zeros(2), [], [ev], 20.0, PF, PF)
    assert evs[0]["charge"][0] == pytest.approx(5.0)
    assert diag["unserved_ev_kwh"] == pytest.approx(6.0)


def test_small_devices_are_not_controlled():
    hp = _heat_pump([9.0, 0.0], cap_hp=3.0, cap_rod=1.0)   # 4 kW connection power
    hps, _, diag = section14a.control(np.array([30.0, 0.0]), np.zeros(2), [hp], [], 20.0, PF, PF)
    assert diag["controlled_heat_pumps"] == 0
    assert np.allclose(hps[0]["heat_served"], hp["heat"])


def test_vehicles_from_session_tables():
    sessions = pd.DataFrame({"session_id": ["a", "b"], "site": [5, 5], "process": ["cs0", "cs0"],
                             "charger_kw": [11.0, 11.0], "energy_kwh": [15.0, 5.0]})
    hours = pd.DataFrame({"session_id": ["a", "a", "b"], "t": [1, 2, 4], "order": [0, 1, 0],
                          "available_fraction": [1.0, 1.0, 0.5]})
    (v,) = _vehicles(sessions, hours, horizon_hours=4)
    assert v["key"] == (5, "cs0")
    assert np.allclose(v["fraction"], [1.0, 1.0, 0.0, 0.5])
    assert list(v["session"]) == [0, 0, -1, 1]
    assert np.allclose(v["uncontrolled"], [11.0, 4.0, 0.0, 5.0])


def test_cli_requires_inflex():
    with pytest.raises(SystemExit):
        parse_args(["x.h5", "--p14a"])
    args = parse_args(["x.h5", "--post-demand-mode", "inflex", "--p14a"])
    assert args.p14a


def test_assumptions_record_section_14a():
    class Reader:
        path = "unused.h5"

        def temporal_method(self):
            return None

        def solver_audit(self):
            return None

    args = parse_args(["x.h5", "--post-demand-mode", "inflex", "--p14a"])
    assert build_assumptions(args, Reader())["section_14a"] is True


def _inflex_inputs():
    index = pd.RangeIndex(1, 5, name="t")
    raw = pd.DataFrame({(7, "electricity"): [10.0, 2.0, 2.0, 2.0], (7, "space_heat"): [30.0, 3.0, 0.0, 3.0]},
                       index=index)
    eff = pd.DataFrame({(7, "heatpump_air"): [3.0, 3.0, 3.0, 3.0]}, index=index)
    process = pd.DataFrame({"Site": [7, 7], "Process": ["heatpump_air", "heatpump_booster"],
                            "inst-cap": [6.0, 12.0]})
    sessions = pd.DataFrame({"session_id": ["a"], "site": [7], "process": ["charging_station0"],
                             "charger_kw": [11.0], "energy_kwh": [11.0]})
    hours = pd.DataFrame({"session_id": ["a"] * 3, "t": [1, 2, 3], "order": [0, 1, 2],
                          "available_fraction": [1.0, 1.0, 1.0]})
    inputs = {"process": process, "ev_sessions": sessions, "ev_session_hours": hours}
    return inputs, raw, eff


def test_apply_without_congestion_matches_inflex_reconstruction():
    from gridexpand.powerflow import demands

    inputs, raw, eff = _inflex_inputs()
    pre = raw[[(7, "electricity")]]
    pv = demands._empty_electricity_frame(raw.index)
    heat, hp_only, ev, diag = demands._apply_section_14a(inputs, raw, eff, pre, pv, 4, rating_kva=1e6)
    ref_total, ref_hp, _ = demands._inflex_heat_electricity(raw, eff, inputs["process"])
    ref_ev = demands._mobility_electricity(inputs["ev_sessions"], inputs["ev_session_hours"], 4, raw.index)
    assert diag["congested_hours"] == 0
    assert np.allclose(heat.to_numpy(), ref_total.to_numpy())
    assert np.allclose(hp_only.to_numpy(), ref_hp.to_numpy())
    assert np.allclose(ev.to_numpy(), ref_ev.to_numpy())
    assert list(heat.columns) == list(ref_total.columns) and list(ev.columns) == list(ref_ev.columns)


def test_apply_with_congestion_caps_the_peak():
    from gridexpand.powerflow import demands
    from gridexpand.powerflow.config import config

    inputs, raw, eff = _inflex_inputs()
    pre = raw[[(7, "electricity")]]
    pv = demands._empty_electricity_frame(raw.index)
    rating = 25.0
    heat, _, ev, diag = demands._apply_section_14a(inputs, raw, eff, pre, pv, 4, rating_kva=rating)
    apparent = (pre.sum(axis=1) + ev.sum(axis=1)) / config.PF_ELC + heat.sum(axis=1) / config.PF_HP
    assert diag["congested_hours"] == 1 and diag["hours_not_cleared"] == 0
    assert apparent.iloc[0] <= rating + 1e-6
    assert ev.to_numpy().sum() == pytest.approx(11.0)
