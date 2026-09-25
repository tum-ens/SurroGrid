"""Step 2 stage and output tables, output writing and metadata flushing."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import gridexpand.allocation.main as main
from gridexpand.allocation.classes.save_grid import SaveFile

# Stage sequences and HDF key sets of the former hand-written branches.
FORMER_STAGES = {
    "status_quo": [
        "generate_electricity", "align_electricity_output_time", "select_timeframe_after_electricity",
        "apply_timeframe_slice", "create_demand",
    ],
    "heat_library": [
        "retrieve_weather", "select_timeframe_from_weather", "generate_electricity",
        "select_timeframe_after_electricity", "generate_heat", "apply_timeframe_slice",
        "create_demand", "create_tve",
    ],
}
FULL_HEAD = [
    "retrieve_weather", "select_timeframe_from_weather", "generate_electricity", "generate_solar",
    "generate_battery", "select_timeframe_after_electricity",
]
FULL_TAIL = [
    "apply_timeframe_slice", "create_weather_urbs", "create_supim", "create_demand", "create_tve",
    "create_bsp", "create_processes", "create_commodities", "create_process_commodity", "create_storages",
]
FORMER_KEYS = {
    "status_quo": {"raw_data/buildings", "raw_data/building_components", "raw_data/demand_component_audit",
                   "raw_data/weather", "urbs_in/demand"},
    "heat_library": {"raw_data/buildings", "raw_data/building_components", "raw_data/demand_component_audit",
                     "raw_data/weather", "raw_data/heat_asset_plan", "raw_data/heat_asset_audit",
                     "urbs_in/demand", "urbs_in/eff_factor"},
    "full": {"raw_data/building_components", "raw_data/weather", "raw_data/buildings",
             "raw_data/electrification_assignment", "raw_data/electrification_assignment_summary",
             "raw_data/pv_roof_sections", "raw_data/asset_plan", "raw_data/pv_selected_sections",
             "raw_data/pv_asset_audit", "raw_data/battery_asset_plan", "raw_data/battery_asset_audit",
             "raw_data/heat_asset_plan", "raw_data/heat_asset_audit", "raw_data/demand_component_audit",
             "urbs_in/demand", "urbs_in/supim", "urbs_in/eff_factor", "urbs_in/buy_sell_price",
             "urbs_in/weather", "urbs_in/process", "urbs_in/commodity", "urbs_in/process_commodity",
             "urbs_in/storage"},
}


class _RecordingGrid:
    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    def __getattr__(self, name):
        if name.startswith(("generate_", "create_", "select_", "apply_", "align_", "retrieve_")):
            return lambda: self.calls.append(name)
        raise AttributeError(name)


def _stages(profile):
    grid = _RecordingGrid(main.profile_flags(profile))
    main.run_stages(grid, main.STAGES[main.profile_kind(grid.settings)])
    return grid.calls


@pytest.mark.parametrize("profile", ["status_quo", "heat_library"])
def test_single_branch_profiles_keep_their_order(profile):
    assert _stages(profile) == FORMER_STAGES[profile]


@pytest.mark.parametrize(
    "profile, middle",
    [
        ("all", ["generate_heat", "generate_mobility"]),
        ("electricity_heat_mobility", ["generate_heat", "generate_mobility"]),
        ("electricity_heat", ["generate_heat"]),
        ("electricity_mobility", ["align_electricity_output_time", "generate_mobility"]),
    ],
)
def test_full_profiles_keep_their_order(profile, middle):
    assert _stages(profile) == FULL_HEAD + middle + FULL_TAIL


@pytest.mark.parametrize("kind", ["status_quo", "heat_library", "full"])
def test_output_key_sets_are_unchanged(kind):
    keys = [key for key, _, _ in main.OUTPUT_KEYS[kind]]
    assert len(keys) == len(set(keys))
    assert set(keys) == FORMER_KEYS[kind]


class _RecordingSaveFile:
    def __init__(self):
        self.events = []
        self.output_path = "out.h5"

    def copy_save_file(self):
        self.events.append("copy")

    def save_timeframe_metadata(self):
        self.events.append("metadata")

    def flush_metadata(self):
        self.events.append("flush")

    def output_store(self):
        events = self.events

        class _Store:
            def __enter__(self):
                events.append("open")

            def __exit__(self, *exc):
                events.append("close")

        return _Store()

    def save_df(self, frame, key):
        self.events.append(key)

    def save_allocated_vehicles(self, buildings, battery_dict):
        self.events.append("vehicles")


def test_write_outputs_order_and_empty_rules():
    empty, full = pd.DataFrame(), pd.DataFrame({"a": [1]})
    grid = SimpleNamespace(
        SF=_RecordingSaveFile(), battery_dict={}, record_asset_plan_summary=lambda: grid.SF.events.append("summary")
    )
    for _, attribute, _ in main.OUTPUT_KEYS["full"]:
        setattr(grid, attribute, full)
    grid.df_electrification_assignment = empty
    grid.df_electrification_summary = empty
    main.write_outputs(grid, "full")
    events = grid.SF.events
    assert events[:5] == ["summary", "copy", "metadata", "flush", "open"]
    assert "raw_data/electrification_assignment" not in events
    assert events.index("vehicles") == events.index("urbs_in/demand") - 1
    assert events[-1] == "close"

    grid.SF = _RecordingSaveFile()
    grid.df_weather_raw = empty
    main.write_outputs(grid, "status_quo")
    assert "raw_data/weather" not in grid.SF.events and "summary" not in grid.SF.events


def test_parse_args_rejects_inconsistent_cases():
    scenario = ["--scenario-config", "config/scenarios/schweinfurt_2045.yaml"]
    with pytest.raises(SystemExit):
        main.parse_args(["1", "--model-case", "pre", "--profiles", "all", *scenario])
    with pytest.raises(SystemExit):
        main.parse_args(["1", "--profiles", "status_quo", *scenario])
    with pytest.raises(SystemExit):
        main.parse_args(["1", "--timeframe-mode", "min_temperature_week", *scenario])
    with pytest.raises(SystemExit):  # no silent default scenario
        main.parse_args(["1", "--model-case", "pre", "--profiles", "status_quo"])
    args = main.parse_args(["1", "--model-case", "pre", "--profiles", "status_quo", *scenario])
    assert args.storage == "h5" and args.mobility_source == "emobpy" and args.timeseries_storage == "db"


class _FakeDb:
    def __init__(self):
        self.calls = []

    def update_demand_allocation_run_assumptions(self, run_id, assumptions):
        self.calls.append(("run", run_id, dict(assumptions)))

    def ensure_scenario(self, *, scenario_key, assumptions):
        self.calls.append(("scenario", scenario_key, None if assumptions is None else dict(assumptions)))


def _db_save_file():
    save_file = object.__new__(SaveFile)
    save_file.storage = "db"
    save_file.db = _FakeDb()
    save_file.demand_allocation_run_id = 7
    save_file.scenario_key = "key"
    save_file.timeframe_metadata = {}
    save_file._metadata_dirty = False
    save_file._store = None
    return save_file


def test_metadata_is_written_once_with_the_last_update():
    save_file = _db_save_file()
    save_file.update_timeframe_metadata({"timeframe_mode": "full_year", "x": 1})
    save_file.update_timeframe_metadata({"timeframe_mode": "full_year", "x": 2})
    assert save_file.db.calls == []
    save_file.flush_metadata()
    save_file.flush_metadata()
    assert save_file.db.calls == [
        ("run", 7, {"timeframe_mode": "full_year", "x": 2}),
        ("scenario", "key", {"timeframe_mode": "full_year", "x": 2}),
    ]


def test_timeslice_metadata_without_start_is_not_written_to_the_run():
    save_file = _db_save_file()
    save_file.update_timeframe_metadata({"timeframe_mode": "max_base_electricity_demand_week"})
    save_file.flush_metadata()
    assert save_file.db.calls == [("scenario", "key", None)]


def test_output_store_writes_all_keys(tmp_path):
    save_file = SaveFile(
        "grid.h5",
        storage="h5",
        allocation_settings={"output_directory": tmp_path, "scenario_key": "scenario", "grid_filename": "grid.h5"},
    )
    frames = {"urbs_in/demand": pd.DataFrame({"a": [1.0, 2.0]}), "raw_data/buildings": pd.DataFrame({"b": ["x"]})}
    with save_file.output_store():
        for key, frame in frames.items():
            save_file.save_df(frame, key)
    save_file.save_df(pd.DataFrame({"c": [3]}), "urbs_in/storage")
    with pd.HDFStore(save_file.output_path, "r") as store:
        assert set(store.keys()) == {"/urbs_in/demand", "/raw_data/buildings", "/urbs_in/storage"}
        pd.testing.assert_frame_equal(store["urbs_in/demand"], frames["urbs_in/demand"])
