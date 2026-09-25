"""``SurroGridDatabase``: one object for the pipeline steps and notebooks.

A thin facade over :mod:`gridexpand.db.grids`, :mod:`~gridexpand.db.runs`,
:mod:`~gridexpand.db.writers`, :mod:`~gridexpand.db.schema` and
:mod:`~gridexpand.db.maintenance`. Instances share the process-wide engine,
so creating one per call is cheap. ``pylovo_version_id`` (default:
``PYLOVO_VERSION_ID``) restricts grid resolution to one pylovo version.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from gridexpand.db import grids, runs, schema, writers
from gridexpand.db.engine import get_engine
from gridexpand.db.grids import MEAN_HOUSEHOLD_SIZE, get_pylovo_version_id, normalize_ags
from gridexpand.db.runs import (
    DEFAULT_SCENARIO_ASSUMPTIONS,
    DEFAULT_SCENARIO_DESCRIPTION,
    DEFAULT_SCENARIO_KEY,
    DEFAULT_SCENARIO_LABEL,
)
from gridexpand.db.writers import BOUNDARY_SUMMARY_COLUMN_NAMES, TIME_INDEX_START, RunTimestamps

__all__ = [
    "BOUNDARY_SUMMARY_COLUMN_NAMES",
    "DEFAULT_SCENARIO_ASSUMPTIONS",
    "DEFAULT_SCENARIO_DESCRIPTION",
    "DEFAULT_SCENARIO_KEY",
    "DEFAULT_SCENARIO_LABEL",
    "MEAN_HOUSEHOLD_SIZE",
    "TIME_INDEX_START",
    "SurroGridDatabase",
    "get_pylovo_version_id",
    "normalize_ags",
]


class SurroGridDatabase:
    """PostgreSQL/PostGIS read/write helper for SurroGrid pipeline data.

    Args:
        engine: engine to use (default: the cached engine of the ``.env`` URL).
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = get_engine() if engine is None else engine
        self.pylovo_version_id = get_pylovo_version_id()
        self._timestamps = RunTimestamps(self.engine)

    # Schema -------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Initialise a fresh database; raise if an existing one needs migrating."""
        schema.ensure_schema(self.engine)

    def refresh_qgis_views(self) -> None:
        schema.refresh_qgis_views(self.engine)

    def refresh_expansion_materialized_views(self) -> None:
        """Alias of :meth:`refresh_qgis_views`."""
        schema.refresh_qgis_views(self.engine)

    # Grids --------------------------------------------------------------------

    def list_grid_candidates(
        self, ags: str | int, *, min_buildings: int = 5, demand_scope: str = "all"
    ) -> list[dict[str, Any]]:
        """Candidate grids of an AGS (see :func:`gridexpand.db.grids.list_grid_candidates`)."""
        return grids.list_grid_candidates(
            self.engine, ags, min_buildings=min_buildings, demand_scope=demand_scope,
            pylovo_version_id=self.pylovo_version_id,
        )

    def resolve_grid_identifier(
        self,
        input_id: str | int,
        *,
        plz: int | None = None,
        kcid: int | None = None,
        bcid: int | None = None,
        candidate_index: int = 0,
        min_buildings: int = 5,
        demand_scope: str = "all",
    ) -> dict[str, Any]:
        """Resolve an AGS or bridge filename to one grid reference."""
        return grids.resolve_grid_identifier(
            self.engine, input_id, plz=plz, kcid=kcid, bcid=bcid, candidate_index=candidate_index,
            min_buildings=min_buildings, demand_scope=demand_scope, pylovo_version_id=self.pylovo_version_id,
        )

    def parse_grid_filename(self, filename: str) -> dict[str, Any]:
        return grids.parse_grid_filename(filename)

    def _format_grid_ref(
        self,
        *,
        ags: int,
        row: dict[str, Any],
        candidate_index: int | None = None,
        bridge_filename: str | None = None,
    ) -> dict[str, Any]:
        return grids.format_grid_ref(ags=ags, row=row, candidate_index=candidate_index, bridge_filename=bridge_filename)

    def get_or_create_grid_case(self, grid_ref: dict[str, Any]) -> int:
        return grids.get_or_create_grid_case(self.engine, grid_ref)

    def get_or_create_grid_cases(self, grid_refs: list[dict[str, Any]]) -> list[int]:
        return grids.get_or_create_grid_cases(self.engine, grid_refs)

    def read_step2_input_data(self, grid_ref: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, None]:
        """Buildings and region of a grid (no weather in DB mode)."""
        self.get_or_create_grid_case(grid_ref)
        return self.read_buildings(grid_ref), self.read_region(grid_ref), None

    def read_region(self, grid_ref: dict[str, Any]) -> pd.DataFrame:
        return grids.read_region(self.engine, grid_ref)

    def read_buildings(self, grid_ref: dict[str, Any]) -> pd.DataFrame:
        return grids.read_buildings(self.engine, grid_ref)

    def read_building_components(
        self, grid_ref: dict[str, Any], df_buildings: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        return grids.read_building_components(self.engine, grid_ref, df_buildings)

    def read_pandapower_grid(self, grid_ref: dict[str, Any]):
        return grids.read_pandapower_grid(self.engine, grid_ref)

    # Scenarios and runs -----------------------------------------------------------

    def ensure_scenario(
        self,
        *,
        scenario_key: str = DEFAULT_SCENARIO_KEY,
        scenario_label: str = DEFAULT_SCENARIO_LABEL,
        description: str = DEFAULT_SCENARIO_DESCRIPTION,
        assumptions: dict[str, Any] | None = None,
    ) -> int:
        # Runs without own assumptions take their time stamps from the scenario.
        self._timestamps = RunTimestamps(self.engine)
        return runs.ensure_scenario(
            self.engine, scenario_key=scenario_key, scenario_label=scenario_label,
            description=description, assumptions=assumptions,
        )

    def ensure_pipeline_run(self, *, grid_case_id: int, scenario_id: int, run_name: str) -> int:
        return runs.ensure_pipeline_run(self.engine, grid_case_id=grid_case_id, scenario_id=scenario_id, run_name=run_name)

    def default_pipeline_run_name(self, scenario_key: str) -> str:
        return runs.default_pipeline_run_name(scenario_key)

    def default_demand_allocation_run_name(self, scenario_key: str, profiles: str, mobility_source: str) -> str:
        return runs.default_demand_allocation_run_name(scenario_key, profiles, mobility_source)

    def default_powerflow_run_name(self, scenario_key: str, pre_only: bool) -> str:
        return runs.default_powerflow_run_name(scenario_key, pre_only)

    def create_demand_allocation_run(self, grid_ref: dict[str, Any], **kwargs: Any) -> int:
        """Create (or reset) a Step 2 run; see :func:`gridexpand.db.runs.create_demand_allocation_run`."""
        run_id = runs.create_demand_allocation_run(self.engine, grid_ref, **kwargs)
        self._timestamps = RunTimestamps(self.engine)
        return run_id

    def update_demand_allocation_run_assumptions(self, run_id: int, assumptions: dict[str, Any]) -> None:
        runs.update_demand_allocation_run_assumptions(self.engine, run_id, assumptions)
        self._timestamps.forget("demand_allocation_run", run_id)

    def create_powerflow_run(self, grid_ref: dict[str, Any], **kwargs: Any) -> int:
        """Create (or reset) a Step 4 run; see :func:`gridexpand.db.runs.create_powerflow_run`."""
        run_id = runs.create_powerflow_run(self.engine, grid_ref, **kwargs)
        self._timestamps = RunTimestamps(self.engine)
        return run_id

    def promote_powerflow_run(self, staging_run_id: int, run_name: str) -> None:
        """Swap a completed staging run in; see :func:`gridexpand.db.runs.promote_powerflow_run`."""
        runs.promote_powerflow_run(self.engine, staging_run_id, run_name)

    def discard_powerflow_run(self, run_id: int) -> None:
        """Delete a (staging) Step 4 run; see :func:`gridexpand.db.runs.discard_powerflow_run`."""
        runs.discard_powerflow_run(self.engine, run_id)

    def get_or_create_real_grid_case(self, grid_ref: dict[str, Any]) -> int:
        return runs.get_or_create_real_grid_case(self.engine, grid_ref)

    def create_real_powerflow_run(self, grid_ref: dict[str, Any], **kwargs: Any) -> int:
        """Create (or reset) a real-grid run; see :func:`gridexpand.db.runs.create_real_powerflow_run`."""
        run_id = runs.create_real_powerflow_run(self.engine, grid_ref, **kwargs)
        self._timestamps = RunTimestamps(self.engine)
        return run_id

    def find_powerflow_run(self, **filters: Any) -> dict[str, Any] | None:
        """Latest power-flow run matching the filters (see :func:`gridexpand.db.runs.find_powerflow_run`)."""
        return runs.find_powerflow_run(self.engine, **filters)

    def list_powerflow_runs(self, **filters: Any) -> pd.DataFrame:
        """Runs with raw results (see :func:`gridexpand.db.runs.list_powerflow_runs`)."""
        return runs.list_powerflow_runs(self.engine, **filters)

    # Maintenance ----------------------------------------------------------------

    def count_scenario_data(self, scenario_key: str) -> dict[str, int]:
        from gridexpand.db import maintenance

        return maintenance.count_scenario_data(self.engine, scenario_key)

    def delete_scenario_data(
        self,
        scenario_key: str,
        *,
        keep_demands: bool = False,
        dry_run: bool = True,
        refresh_expansion_views: bool = True,
    ) -> dict[str, int]:
        """Delete (or with ``dry_run`` only count) the rows of one scenario key."""
        from gridexpand.db import maintenance

        return maintenance.delete_scenario_data(
            self.engine, scenario_key, keep_demands=keep_demands, dry_run=dry_run,
            refresh_views=refresh_expansion_views,
        )

    # Step 2 writers ---------------------------------------------------------------

    def write_allocated_demand(self, run_id: int, df: pd.DataFrame) -> None:
        writers.write_allocated_timeseries(
            self.engine, self._timestamps, run_id, df, label_column="commodity", table="allocated_demand"
        )

    def write_allocated_eff_factor(self, run_id: int, df: pd.DataFrame) -> None:
        writers.write_allocated_timeseries(
            self.engine, self._timestamps, run_id, df, label_column="component", table="allocated_eff_factor"
        )

    def write_electrification_assignment(self, run_id: int, df: pd.DataFrame) -> None:
        writers.write_electrification_assignment(self.engine, run_id, df)

    def write_demand_component_audit(self, run_id: int, df: pd.DataFrame) -> None:
        writers.write_demand_component_audit(self.engine, run_id, df)

    def write_allocated_vehicles(self, run_id: int, df_buildings: pd.DataFrame, battery_dict: dict | None = None) -> None:
        writers.write_allocated_vehicles(self.engine, run_id, df_buildings, battery_dict)

    # Step 4 writers ---------------------------------------------------------------

    def write_powerflow_summary(self, run_id: int, stage: str, summary: dict[str, Any]) -> None:
        writers.write_summary(self.engine, self._timestamps, run_id, stage, summary)

    def write_real_powerflow_summary(self, run_id: int, stage: str, summary: dict[str, Any]) -> None:
        writers.write_summary(self.engine, self._timestamps, run_id, stage, summary, real=True)

    def write_powerflow_demand(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        writers.write_raw(self.engine, self._timestamps, "powerflow_demand", run_id, stage, df)

    def write_powerflow_import(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        writers.write_raw(self.engine, self._timestamps, "powerflow_import", run_id, stage, df)

    def write_powerflow_bus_voltage(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        writers.write_raw(self.engine, self._timestamps, "powerflow_bus_voltage", run_id, stage, df)

    def write_powerflow_line_result(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        writers.write_raw(self.engine, self._timestamps, "powerflow_line_result", run_id, stage, df)

    def write_powerflow_reactive(self, run_id: int, df: pd.DataFrame) -> None:
        writers.write_raw(self.engine, self._timestamps, "powerflow_reactive_component", run_id, None, df)
