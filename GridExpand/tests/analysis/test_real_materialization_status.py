"""Cost status of a real grid with non-converged power-flow timesteps."""

from gridexpand.analysis.expansion.real_materialization import MAX_FAILED_SHARE, _grid_status


def _run(failed, timesteps=8760, lv_id="LV_153"):
    return {"lv_id": lv_id, "n_failed_timesteps": failed, "n_timesteps": timesteps, "real_powerflow_run_id": 1,
            "real_grid_case_id": 2, "scenario_id": 3, "plz": 91301}


def test_few_failed_timesteps_give_a_lower_bound_cost():
    status = _grid_status(_run(67), excluded=set())
    assert status["cost_status"] == "complete" and "lower bound" in status["status_reason"]
    assert _grid_status(_run(0), excluded=set())["status_reason"] is None


def test_failed_timesteps_from_the_threshold_on_leave_the_cost_unknown():
    limit = int(-(-MAX_FAILED_SHARE * 8760 // 1))  # 88 of 8,760 hours
    assert _grid_status(_run(limit - 1), excluded=set())["cost_status"] == "complete"
    assert _grid_status(_run(limit), excluded=set())["cost_status"] == "incomplete"
    assert _grid_status(_run(5, lv_id="LV_034"), excluded={"34"})["cost_status"] == "excluded"
