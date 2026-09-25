"""Database loaders and summary tables for synthetic/real power-flow comparisons."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sqlalchemy import text

from gridexpand.analysis.ids import canonical_real_grid_id, optional_ags, real_grid_label, synthetic_grid_label
from gridexpand.analysis.powerflow.scope import scope_filter, scope_params
from gridexpand.db.database import SurroGridDatabase


def _add_headline_asset_percentiles(
    summary: pd.DataFrame,
    db: SurroGridDatabase,
    *,
    cable_table: str,
    voltage_table: str,
    run_id_column: str,
) -> pd.DataFrame:
    if summary.empty:
        return summary
    run_ids = summary["powerflow_run_id"].dropna().astype(int).unique().tolist()
    if not run_ids:
        return summary

    cable_query = text(
        f"""
        SELECT {run_id_column} AS powerflow_run_id,
               stage,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p50_asset_percent,
               percentile_cont(0.90) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p90_asset_percent,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p95_asset_percent_derived,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p99_asset_percent,
               MAX(cable_loading_max_time_percent) AS cable_loading_max_asset_percent
        FROM {cable_table}
        WHERE {run_id_column} = ANY(:run_ids)
          AND cable_loading_max_time_percent IS NOT NULL
        GROUP BY {run_id_column}, stage
        """
    )
    voltage_query = text(
        f"""
        SELECT {run_id_column} AS powerflow_run_id,
               stage,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p50_asset_time_pu,
               percentile_cont(0.10) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p10_asset_time_pu,
               percentile_cont(0.05) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p05_asset_time_pu,
               percentile_cont(0.01) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p01_asset_time_pu,
               MIN(voltage_min_time_pu) AS voltage_min_asset_time_pu
        FROM {voltage_table}
        WHERE {run_id_column} = ANY(:run_ids)
        GROUP BY {run_id_column}, stage
        """
    )
    with db.engine.connect() as conn:
        cable = pd.read_sql_query(cable_query, conn, params={"run_ids": run_ids})
        voltage = pd.read_sql_query(voltage_query, conn, params={"run_ids": run_ids})

    out = summary.copy()
    if not cable.empty:
        out = out.merge(cable, on=["powerflow_run_id", "stage"], how="left")
        if "cable_loading_p95_asset_percent_derived" in out.columns:
            out["cable_loading_p95_asset_percent"] = out["cable_loading_p95_asset_percent"].fillna(
                out["cable_loading_p95_asset_percent_derived"]
            )
            out.drop(columns=["cable_loading_p95_asset_percent_derived"], inplace=True)
    if not voltage.empty:
        out = out.merge(voltage, on=["powerflow_run_id", "stage"], how="left")

    for column in (
        "cable_loading_p50_asset_percent",
        "cable_loading_p90_asset_percent",
        "cable_loading_p95_asset_percent",
        "cable_loading_p99_asset_percent",
        "cable_loading_max_asset_percent",
        "voltage_p50_asset_time_pu",
        "voltage_p10_asset_time_pu",
        "voltage_p05_asset_time_pu",
        "voltage_p01_asset_time_pu",
        "voltage_min_asset_time_pu",
    ):
        if column not in out.columns:
            out[column] = pd.NA
    return out

def _add_synthetic_household_scope(summary: pd.DataFrame, db: SurroGridDatabase) -> pd.DataFrame:
    if summary.empty:
        return summary
    run_ids = summary["powerflow_run_id"].dropna().astype(int).unique().tolist()
    if not run_ids:
        return summary

    query = text(
        """
        WITH selected_runs AS (
            SELECT pr.powerflow_run_id, pr.grid_case_id
            FROM surrogrid.powerflow_run pr
            WHERE pr.powerflow_run_id = ANY(:run_ids)
        )
        SELECT sr.powerflow_run_id,
               COUNT(*) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category = 'Residential'
               ) AS selected_household_load_rows,
               COUNT(DISTINCT gbc.bus) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category = 'Residential'
                     AND gbc.bus IS NOT NULL
               ) AS selected_household_load_buses,
               COALESCE(SUM(gbc.households) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category = 'Residential'
               ), 0) AS selected_household_equivalents,
               COUNT(*) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category IN ('Commercial', 'Public')
               ) AS non_household_load_rows,
               COUNT(DISTINCT gbc.bus) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category IN ('Commercial', 'Public')
                     AND gbc.bus IS NOT NULL
               ) AS non_household_load_buses
        FROM selected_runs sr
        LEFT JOIN surrogrid.grid_building_component gbc USING (grid_case_id)
        GROUP BY sr.powerflow_run_id
        """
    )
    with db.engine.connect() as conn:
        household_scope = pd.read_sql_query(query, conn, params={"run_ids": run_ids})

    out = summary.merge(household_scope, on="powerflow_run_id", how="left")
    for column in (
        "selected_household_load_rows",
        "selected_household_load_buses",
        "selected_household_equivalents",
        "non_household_load_rows",
        "non_household_load_buses",
    ):
        if column not in out.columns:
            out[column] = pd.NA
    return out


_TRAFO_PERCENTILES = {
    "p50": "trafo_loading_p50_time_percent",
    "p90": "trafo_loading_p90_time_percent",
    "p95": "trafo_loading_p95_time_percent",
    "p99": "trafo_loading_p99_time_percent",
    "max": "trafo_loading_max_time_percent",
}
_CABLE_PERCENTILES = {
    "p50": "cable_loading_p50_time_percent",
    "p90": "cable_loading_p90_time_percent",
    "p95": "cable_loading_p95_time_percent",
    "p99": "cable_loading_p99_time_percent",
    "max": "cable_loading_max_time_percent",
}
_VOLTAGE_PERCENTILES = {
    "p50": "voltage_p50_time_pu",
    "p10": "voltage_p10_time_pu",
    "p05": "voltage_p05_time_pu",
    "p01": "voltage_p01_time_pu",
    "min": "voltage_min_time_pu",
}
_PROFILE_COLUMNS = ["metric", "asset_type", "asset_id", "asset_label", "percentile", "percentile_order", "value"]


def percentile_profile_frame(
    grid_summary: pd.DataFrame,
    cable_rows: pd.DataFrame,
    voltage_rows: pd.DataFrame,
    meta_cols: list[str],
) -> pd.DataFrame:
    """Long table of per-asset time percentiles (transformer, cables, voltage) with grid metadata.

    ``cable_rows``/``voltage_rows`` hold one row per asset with ``powerflow_run_id``,
    ``stage``, ``asset_id`` and the percentile columns; ``grid_summary`` the headline rows.
    """
    meta = grid_summary[meta_cols].copy()
    frames = []
    for order, (percentile, column) in enumerate(_TRAFO_PERCENTILES.items()):
        rows = grid_summary[meta_cols + [column]].rename(columns={column: "value"})
        rows["metric"] = "Transformer"
        rows["asset_type"] = "transformer"
        rows["asset_id"] = 0
        rows["asset_label"] = rows["grid"] + " transformer"
        rows["percentile"] = percentile
        rows["percentile_order"] = order
        frames.append(rows)
    for asset_rows, percentiles, metric, asset_type in (
        (cable_rows, _CABLE_PERCENTILES, "Cables", "cable"),
        (voltage_rows, _VOLTAGE_PERCENTILES, "Voltage", "bus"),
    ):
        if asset_rows.empty:
            continue
        asset_rows = asset_rows.merge(meta, on=["powerflow_run_id", "stage"], how="left")
        for order, (percentile, column) in enumerate(percentiles.items()):
            rows = asset_rows[meta_cols + ["asset_id", column]].rename(columns={column: "value"})
            rows["metric"] = metric
            rows["asset_type"] = asset_type
            rows["asset_label"] = f"{asset_type} " + rows["asset_id"].astype(str)
            rows["percentile"] = percentile
            rows["percentile_order"] = order
            frames.append(rows)
    out = pd.concat(frames, ignore_index=True)
    out["value"] = out["value"].astype(float)
    return out[meta_cols + _PROFILE_COLUMNS].dropna(subset=["value"]).reset_index(drop=True)


def _percentile_profile_long(
    grid_summary: pd.DataFrame,
    *,
    stage: str,
    meta_cols: list[str],
    cable_table: str,
    voltage_table: str,
    run_id_column: str,
) -> pd.DataFrame:
    run_ids = grid_summary["powerflow_run_id"].astype(int).tolist()
    if not run_ids:
        return pd.DataFrame()
    cable_query = text(
        f"""
        SELECT {run_id_column} AS powerflow_run_id, stage, cable AS asset_id,
               {", ".join(_CABLE_PERCENTILES.values())}
        FROM {cable_table}
        WHERE {run_id_column} = ANY(:run_ids)
          AND stage = :stage
        """
    )
    voltage_query = text(
        f"""
        SELECT {run_id_column} AS powerflow_run_id, stage, bus AS asset_id,
               {", ".join(_VOLTAGE_PERCENTILES.values())}
        FROM {voltage_table}
        WHERE {run_id_column} = ANY(:run_ids)
          AND stage = :stage
        """
    )
    db = SurroGridDatabase()
    with db.engine.connect() as conn:
        cable_rows = pd.read_sql_query(cable_query, conn, params={"run_ids": run_ids, "stage": stage})
        voltage_rows = pd.read_sql_query(voltage_query, conn, params={"run_ids": run_ids, "stage": stage})
    return percentile_profile_frame(grid_summary, cable_rows, voltage_rows, meta_cols)



def powerflow_headline_summary_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_pre_powerflow",
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read compact DB-backed headline power-flow metrics for comparison plots."""
    db = SurroGridDatabase()
    scope = scope_params(
        db, input_id=input_id, run_name=run_name, scenario_id=scenario_id, ags=ags, plz=plz, kcid=kcid,
        bcid=bcid, candidate_index=candidate_index, min_buildings=min_buildings,
    )
    query = text(
        f"""
        SELECT pr.powerflow_run_id,
               pr.run_name,
               pr.scenario_id,
               sc.scenario_key,
               gc.ags,
               gc.plz,
               gc.kcid,
               gc.bcid,
               gc.pylovo_grid_result_id,
               pfs.stage,
               pfs.n_timesteps,
               pfs.n_converged_timesteps,
               pfs.n_failed_timesteps,
               pfs.n_voltage_buses,
               pfs.n_cables,
               pfs.transformer_s_rated_mva,
               pfs.trafo_loading_p50_time_percent,
               pfs.trafo_loading_p90_time_percent,
               pfs.trafo_loading_p95_time_percent,
               pfs.trafo_loading_p99_time_percent,
               pfs.trafo_loading_max_time_percent,
               pfs.trafo_loading_hours_above_100,
               pfs.cable_loading_p95_asset_percent,
               pfs.cable_hours_above_100_p95_asset,
               pfs.voltage_p05_load_bus_hour_pu,
               pfs.voltage_hours_below_0_90_p95_asset
        FROM surrogrid.powerflow_summary pfs
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pfs.stage = :stage
          AND {scope_filter("pr.powerflow_run_id")}
        ORDER BY gc.ags, gc.plz, gc.kcid, gc.bcid, pr.powerflow_run_id, pfs.stage
        """
    )
    with db.engine.connect() as conn:
        summary = pd.read_sql_query(
            query,
            conn,
            params={"run_name": run_name, "stage": stage, **scope},
        )

    if summary.empty:
        raise ValueError(f"No compact DB power-flow summary found for run name {run_name!r}.")

    summary = _add_headline_asset_percentiles(
        summary,
        db,
        cable_table="surrogrid.powerflow_cable_summary",
        voltage_table="surrogrid.powerflow_bus_voltage_summary",
        run_id_column="powerflow_run_id",
    )
    summary = _add_synthetic_household_scope(summary, db)
    summary["grid"] = summary.apply(synthetic_grid_label, axis=1)
    return summary[
        [
            "grid",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "stage",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "pylovo_grid_result_id",
            "selected_household_load_rows",
            "selected_household_load_buses",
            "selected_household_equivalents",
            "non_household_load_rows",
            "non_household_load_buses",
            "n_timesteps",
            "n_converged_timesteps",
            "n_failed_timesteps",
            "n_voltage_buses",
            "n_cables",
            "transformer_s_rated_mva",
            "trafo_loading_p50_time_percent",
            "trafo_loading_p90_time_percent",
            "trafo_loading_p95_time_percent",
            "trafo_loading_p99_time_percent",
            "trafo_loading_max_time_percent",
            "trafo_loading_hours_above_100",
            "cable_loading_p50_asset_percent",
            "cable_loading_p90_asset_percent",
            "cable_loading_p95_asset_percent",
            "cable_loading_p99_asset_percent",
            "cable_loading_max_asset_percent",
            "cable_hours_above_100_p95_asset",
            "voltage_p50_asset_time_pu",
            "voltage_p10_asset_time_pu",
            "voltage_p05_asset_time_pu",
            "voltage_p01_asset_time_pu",
            "voltage_min_asset_time_pu",
            "voltage_p05_load_bus_hour_pu",
            "voltage_hours_below_0_90_p95_asset",
        ]
    ].reset_index(drop=True)

def powerflow_percentile_profile_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_pre_powerflow",
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read per-asset time-percentiles in long form for duration-profile plots."""
    grid_summary = powerflow_headline_summary_db(
        input_id=input_id,
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        ags=ags,
        plz=plz,
        kcid=kcid,
        bcid=bcid,
        candidate_index=candidate_index,
        min_buildings=min_buildings,
    )
    meta_cols = [
        "grid",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
        "pylovo_grid_result_id",
        "selected_household_load_rows",
        "selected_household_load_buses",
        "n_timesteps",
        "n_converged_timesteps",
        "n_failed_timesteps",
    ]
    return _percentile_profile_long(
        grid_summary,
        stage=stage,
        meta_cols=meta_cols,
        cable_table="surrogrid.powerflow_cable_summary",
        voltage_table="surrogrid.powerflow_bus_voltage_summary",
        run_id_column="powerflow_run_id",
    )

def latest_synthetic_powerflow_summary_run_name(
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    db: SurroGridDatabase | None = None,
) -> str:
    """Return the newest synthetic run name with compact power-flow summaries."""
    db = db or SurroGridDatabase()
    query = text(
        """
        SELECT
            pr.run_name,
            COUNT(DISTINCT pr.powerflow_run_id) AS summary_grids,
            MAX(pfs.created_at) AS latest_summary_at
        FROM surrogrid.powerflow_summary pfs
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pfs.stage = :stage
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
        GROUP BY pr.run_name
        ORDER BY latest_summary_at DESC, summary_grids DESC, pr.run_name DESC
        LIMIT 1
        """
    )
    with db.engine.connect() as conn:
        row = conn.execute(
            query,
            {
                "stage": stage,
                "scenario_id": scenario_id,
                "ags": optional_ags(ags),
                "filter_plz": plz,
            },
        ).mappings().first()
    if row is None:
        raise ValueError(
            "No compact synthetic power-flow summary run found for the selected filters. "
            "Run the pipeline with --powerflow-output summary or --powerflow-output both first."
        )
    return str(row["run_name"])

def load_synthetic_powerflow_cutoff_profile(
    run_name: str | None = None,
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Load synthetic asset-percentile profiles for retained-asset cutoff plots."""
    if run_name is None:
        run_name = latest_synthetic_powerflow_summary_run_name(
            stage=stage,
            scenario_id=scenario_id,
            ags=ags,
            plz=plz,
        )
    profile = powerflow_percentile_profile_db(
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        ags=ags,
        plz=plz,
        kcid=kcid,
        bcid=bcid,
        min_buildings=min_buildings,
    )
    profile = profile.copy()
    profile["comparison_group"] = "Synthetic"
    return profile

REAL_COMPARISON_GROUPS = {"swf": "Real SWF", "uzw": "Real ÜZW"}



def real_powerflow_headline_summary_db(
    run_name: str,
    stage: str = "pre",
    scenario_id: int | None = None,
    plz: int | None = None,
    lv_id: str | int | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Read compact real-grid (SWF or ÜZW) DB-backed headline power-flow metrics."""
    db = SurroGridDatabase()
    lv_id_text = None if lv_id is None else canonical_real_grid_id(lv_id)
    query = text(
        """
        SELECT rpr.real_powerflow_run_id AS powerflow_run_id,
               rpr.run_name,
               rpr.scenario_id,
               sc.scenario_key,
               rgc.source,
               rgc.plz,
               rgc.lv_id,
               rgc.variant,
               rgc.category,
               rgc.load_status,
               rgc.source_file,
               NULLIF(rpr.assumptions ->> 'household_load_rows_before_supply_filter', '')::INTEGER AS household_load_rows_before_supply_filter,
               NULLIF(rpr.assumptions ->> 'household_load_buses_before_supply_filter', '')::INTEGER AS household_load_buses_before_supply_filter,
               NULLIF(rpr.assumptions ->> 'dropped_unsupplied_household_load_rows', '')::INTEGER AS dropped_unsupplied_household_load_rows,
               NULLIF(rpr.assumptions ->> 'dropped_unsupplied_household_load_buses', '')::INTEGER AS dropped_unsupplied_household_load_buses,
               NULLIF(rpr.assumptions ->> 'selected_household_load_rows', '')::INTEGER AS selected_household_load_rows,
               NULLIF(rpr.assumptions ->> 'selected_household_load_buses', '')::INTEGER AS selected_household_load_buses,
               NULLIF(rpr.assumptions ->> 'backbone_voltage_buses', '')::INTEGER AS backbone_voltage_buses,
               NULLIF(rpr.assumptions ->> 'backbone_cables', '')::INTEGER AS backbone_cables,
               rps.stage,
               rps.n_timesteps,
               rps.n_converged_timesteps,
               rps.n_failed_timesteps,
               rps.n_voltage_buses,
               rps.n_cables,
               rps.transformer_s_rated_mva,
               rps.trafo_loading_p50_time_percent,
               rps.trafo_loading_p90_time_percent,
               rps.trafo_loading_p95_time_percent,
               rps.trafo_loading_p99_time_percent,
               rps.trafo_loading_max_time_percent,
               rps.trafo_loading_hours_above_100,
               rps.cable_loading_p95_asset_percent,
               rps.cable_hours_above_100_p95_asset,
               rps.voltage_p05_load_bus_hour_pu,
               rps.voltage_hours_below_0_90_p95_asset
        FROM surrogrid.real_powerflow_summary rps
        JOIN surrogrid.real_powerflow_run rpr USING (real_powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        WHERE rpr.run_name = :run_name
          AND rps.stage = :stage
          AND (:scenario_id IS NULL OR rpr.scenario_id = :scenario_id)
          AND (:filter_plz IS NULL OR rgc.plz = :filter_plz)
          AND (:lv_id IS NULL OR rgc.lv_id = CAST(:lv_id AS TEXT))
          AND (CAST(:source AS TEXT) IS NULL OR rgc.source = CAST(:source AS TEXT))
        ORDER BY rgc.source, LENGTH(rgc.lv_id), rgc.lv_id, rpr.real_powerflow_run_id, rps.stage
        """
    )
    with db.engine.connect() as conn:
        summary = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stage": stage,
                "scenario_id": scenario_id,
                "filter_plz": plz,
                "lv_id": lv_id_text,
                "source": source,
            },
        )

    if summary.empty:
        raise ValueError(f"No compact real-grid DB power-flow summary found for run name {run_name!r}.")

    summary = _add_headline_asset_percentiles(
        summary,
        db,
        cable_table="surrogrid.real_powerflow_cable_summary",
        voltage_table="surrogrid.real_powerflow_bus_voltage_summary",
        run_id_column="real_powerflow_run_id",
    )
    summary["grid"] = [real_grid_label(source, lv_id) for source, lv_id in zip(summary["source"], summary["lv_id"])]
    summary["powerflow_source"] = "real_" + summary["source"].astype(str)
    summary["comparison_group"] = summary["source"].map(REAL_COMPARISON_GROUPS)
    summary["ags"] = pd.NA
    summary["kcid"] = pd.NA
    summary["bcid"] = pd.NA
    summary["pylovo_grid_result_id"] = pd.NA
    return summary[
        [
            "grid",
            "powerflow_source",
            "comparison_group",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "stage",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "pylovo_grid_result_id",
            "lv_id",
            "source_file",
            "household_load_rows_before_supply_filter",
            "household_load_buses_before_supply_filter",
            "dropped_unsupplied_household_load_rows",
            "dropped_unsupplied_household_load_buses",
            "selected_household_load_rows",
            "selected_household_load_buses",
            "backbone_voltage_buses",
            "backbone_cables",
            "n_timesteps",
            "n_converged_timesteps",
            "n_failed_timesteps",
            "n_voltage_buses",
            "n_cables",
            "transformer_s_rated_mva",
            "trafo_loading_p50_time_percent",
            "trafo_loading_p90_time_percent",
            "trafo_loading_p95_time_percent",
            "trafo_loading_p99_time_percent",
            "trafo_loading_max_time_percent",
            "trafo_loading_hours_above_100",
            "cable_loading_p50_asset_percent",
            "cable_loading_p90_asset_percent",
            "cable_loading_p95_asset_percent",
            "cable_loading_p99_asset_percent",
            "cable_loading_max_asset_percent",
            "cable_hours_above_100_p95_asset",
            "voltage_p50_asset_time_pu",
            "voltage_p10_asset_time_pu",
            "voltage_p05_asset_time_pu",
            "voltage_p01_asset_time_pu",
            "voltage_min_asset_time_pu",
            "voltage_p05_load_bus_hour_pu",
            "voltage_hours_below_0_90_p95_asset",
        ]
    ].reset_index(drop=True)

def real_powerflow_percentile_profile_db(
    run_name: str,
    stage: str = "pre",
    scenario_id: int | None = None,
    plz: int | None = None,
    lv_id: str | int | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Read real-grid (SWF or ÜZW) per-asset time-percentiles in long form."""
    grid_summary = real_powerflow_headline_summary_db(
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
        lv_id=lv_id,
        source=source,
    )
    meta_cols = [
        "grid",
        "powerflow_source",
        "comparison_group",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
        "pylovo_grid_result_id",
        "lv_id",
        "source_file",
        "household_load_rows_before_supply_filter",
        "household_load_buses_before_supply_filter",
        "dropped_unsupplied_household_load_rows",
        "dropped_unsupplied_household_load_buses",
        "selected_household_load_rows",
        "selected_household_load_buses",
        "backbone_voltage_buses",
        "backbone_cables",
        "n_timesteps",
        "n_converged_timesteps",
        "n_failed_timesteps",
    ]
    return _percentile_profile_long(
        grid_summary,
        stage=stage,
        meta_cols=meta_cols,
        cable_table="surrogrid.real_powerflow_cable_summary",
        voltage_table="surrogrid.real_powerflow_bus_voltage_summary",
        run_id_column="real_powerflow_run_id",
    )


def powerflow_distribution_similarity_summary(
    profile: pd.DataFrame,
    group_col: str = "comparison_group",
    synthetic_group: str = "Synthetic",
    real_group: str = "Real SWF",
) -> pd.DataFrame:
    """Compare critical synthetic and real power-flow result distributions.

    The table uses the same critical result semantics as the asset-cutoff plots:
    transformer/cable annual maximum loading and annual minimum voltage. Signed
    differences are calculated as synthetic minus real.
    """
    required = {group_col, "metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(f"Missing column(s) for distribution similarity summary: {missing}.")

    critical_percentiles = {
        "Transformer": "max",
        "Cables": "max",
        "Voltage": "min",
    }
    rows = []
    for metric, percentile in critical_percentiles.items():
        metric_rows = profile[
            profile["metric"].eq(metric)
            & profile["percentile"].eq(percentile)
        ]
        synthetic_values = metric_rows.loc[
            metric_rows[group_col].eq(synthetic_group), "value"
        ].astype(float).dropna()
        real_values = metric_rows.loc[
            metric_rows[group_col].eq(real_group), "value"
        ].astype(float).dropna()
        if synthetic_values.empty or real_values.empty:
            rows.append(
                {
                    "metric": metric,
                    "median_diff": np.nan,
                    "std": np.nan,
                    "wasserstein": np.nan,
                }
            )
            continue
        rows.append(
            {
                "metric": metric,
                "median_diff": synthetic_values.median() - real_values.median(),
                "std": synthetic_values.std(ddof=1) - real_values.std(ddof=1),
                "wasserstein": wasserstein_distance(synthetic_values, real_values),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "median_diff", "std", "wasserstein"])

