"""Grid exclusion and the per-grid expansion cost comparison of the analysis notebooks (no database)."""

from __future__ import annotations

import pandas as pd

from gridexpand.analysis.expansion import notebook_workflow as nw


def test_group_columns():
    assert (nw.group_network("Real ÜZW"), nw.group_provider("Real ÜZW")) == ("Real", "ÜZW")
    assert (nw.group_network("Synthetic SWF"), nw.group_provider("Synthetic")) == ("Synthetic", None)


def test_real_and_synthetic_grids_are_excluded_by_their_keys():
    real = pd.DataFrame({"lv_id": ["038", "59", "7"], "grid": ["a", "b", "c"]})
    assert nw._without_excluded(real, "Real SWF", {"38", "59"})["grid"].tolist() == ["c"]
    synthetic = pd.DataFrame({"grid": ["09474126-91301_1_3", "09474126-91301_3_-1"],
                              "plz": [91301, 91301], "kcid": [1, 3], "bcid": [3, -1]})
    assert nw._without_excluded(synthetic, "Synthetic SWF", {"91301_3_-1"})["bcid"].tolist() == [3]
    # the label alone (cable decomposition) gives the same key
    labels_only = synthetic[["grid"]]
    assert nw.synthetic_grid_keys(labels_only).tolist() == ["91301_1_3", "91301_3_-1"]


def test_excluded_grids_by_group():
    nonconverged = pd.DataFrame({
        "data_source": ["Real SWF", "Real SWF", "Synthetic SWF"],
        "grid_key": ["113", "54", "91301_1_3"],
        "excluded": [True, False, True],
    })
    assert nw.excluded_grids_by_group(nonconverged) == {"Real SWF": ("113",), "Synthetic SWF": ("91301_1_3",)}


def _grid_costs() -> pd.DataFrame:
    rows = []
    for stage in ("status-quo", "HEMS"):
        for lv_id, status in (("1", "complete"), ("2", "complete"), ("3", "complete" if stage == "status-quo" else "incomplete")):
            rows.append({"data_source": "Real SWF", "network": "Real", "provider": "SWF", "stage_label": stage,
                         "grid_label": f"SWF LV_{lv_id}", "lv_id": lv_id, "plz": 91301, "kcid": None, "bcid": None,
                         "cost_status": status, "cable_cost_eur": 100.0 if status == "complete" else None,
                         "transformer_exchange_cost_eur": 10.0, "load_transfer_cost_eur": 0.0,
                         "new_station_cost_eur": 0.0, "voltage_cost_eur": 0.0,
                         "transformer_cost_eur": 10.0, "total_cost_eur": 110.0,
                         "reinforcement_150_count": 1})
        for bcid in (1, 2):
            rows.append({"data_source": "Synthetic SWF", "network": "Synthetic", "provider": "SWF",
                         "stage_label": stage, "grid_label": f"91301-1-{bcid}", "lv_id": None, "plz": 91301,
                         "kcid": 1, "bcid": bcid, "cost_status": "complete", "cable_cost_eur": 50.0,
                         "transformer_exchange_cost_eur": 5.0, "load_transfer_cost_eur": 0.0,
                         "new_station_cost_eur": 0.0, "voltage_cost_eur": 0.0, "transformer_cost_eur": 5.0,
                         "total_cost_eur": 55.0, "reinforcement_150_count": 2})
    return pd.DataFrame(rows)


def test_cost_comparison_uses_one_grid_set_for_all_stages():
    result = nw.expansion_cost_comparison(
        _grid_costs(), stage_labels=("status-quo", "HEMS"), excluded_grids={"Real SWF": ("1",)})
    # LV 1 configured, LV 3 incomplete in HEMS: both leave every stage; LV 2 stays.
    assert result["excluded"]["grid"].tolist() == ["SWF LV_1", "SWF LV_3"]
    assert "cost incomplete in HEMS" in result["excluded"]["reason"].iloc[1]
    grids = result["grids"].set_index(["data_source", "stage"])["grids"]
    assert grids[("Real", "status-quo")] == grids[("Real", "HEMS")] == 1
    assert grids[("Synthetic", "HEMS")] == 2
    costs = result["costs"].set_index(["data_source", "stage", "component"])["cost_eur"]
    assert costs[("Real", "HEMS", "Cables")] == 100.0 and costs[("Synthetic", "HEMS", "Cables")] == 100.0
    assert costs[("Real", "HEMS", "Transformer exchange")] == 10.0
    by_group = nw.expansion_cost_comparison(_grid_costs(), stage_labels=("HEMS",), by="data_source")
    assert set(by_group["costs"]["data_source"]) == {"Real SWF", "Synthetic SWF"}


def test_publication_gate_accepts_the_provider_labels_of_an_aligned_run():
    specs = {"Real SWF": {"HEMS": {"run_name": "run_swf_real_swf_post-hems-heuristic", "stage": "post"}}}
    status = pd.DataFrame({
        "complete": [True], "summary_grids": [3], "failed_timesteps": [0], "timestep_signatures": ["8760"],
        "scenario_labels": ["run_swf_post-hems-heuristic"], "profile_contracts": ["c"],
    })
    gate = nw._publication_gate(scenario_prefix="run", powerflow_status=status, expansion_status=pd.DataFrame(),
                                specs_by_source=specs, expected_grid_counts=None)
    assert gate.set_index("check").loc["Scenario labels of this run", "passed"]
    other = status.assign(scenario_labels=["other_run_swf_post-hems-heuristic"])
    gate = nw._publication_gate(scenario_prefix="run", powerflow_status=other, expansion_status=pd.DataFrame(),
                                specs_by_source=specs, expected_grid_counts=None)
    assert not gate.set_index("check").loc["Scenario labels of this run", "passed"]
