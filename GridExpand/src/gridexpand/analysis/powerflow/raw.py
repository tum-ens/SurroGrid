"""DB loaders of the raw (and compact fallback) power-flow tables for the notebooks.

Both loaders read the raw hypertables of a run when they exist and fall back to
the compact summary tables of the Step 4 evaluation scope otherwise (runs with
``--outputs summary``): raw rows cover all buses, the summaries the evaluation scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from gridexpand.analysis.ids import synthetic_grid_label
from gridexpand.analysis.powerflow.scope import scope_filter, scope_params
from gridexpand.db.database import SurroGridDatabase


def transformer_import_distribution_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_full_powerflow",
    stage: str = "post",
    reactive_magnitude: bool = True,
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read transformer import time series for one grid or a population scope.

    Pass ``input_id`` for one concrete grid. Leave ``input_id`` as ``None`` to
    aggregate all matching DB runs, optionally narrowed by ``scenario_id``, ``ags``, ``plz``,
    ``kcid``, or ``bcid``.
    """
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
               pi.stage,
               pi.ts,
               pi.t_index,
               pi.p_mw,
               pi.q_mvar
        FROM surrogrid.powerflow_import pi
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pi.stage = :stage
          AND {scope_filter("pi.powerflow_run_id")}
        ORDER BY pr.powerflow_run_id, pi.t_index
        """
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={"run_name": run_name, "stage": stage, **scope},
        )

    if df.empty:
        compact_query = text(
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
                   ptd.stage,
                   ptd.diagnostic,
                   ptd.point_index,
                   ptd.x_value,
                   ptd.t_index,
                   ptd.ts,
                   ptd.p_mw,
                   ptd.q_mvar,
                   ptd.q_abs_mvar,
                   ptd.s_mva,
                   ptd.mean_s_mva,
                   ptd.max_s_mva
            FROM surrogrid.powerflow_transformer_diagnostic ptd
            JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
            JOIN surrogrid.scenario sc USING (scenario_id)
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            WHERE pr.run_name = :run_name
              AND ptd.stage = :stage
              AND {scope_filter("ptd.powerflow_run_id")}
            ORDER BY pr.powerflow_run_id, ptd.diagnostic, ptd.point_index
            """
        )
        with db.engine.connect() as conn:
            df = pd.read_sql_query(
                compact_query,
                conn,
                params={"run_name": run_name, "stage": stage, **scope},
            )
        if df.empty:
            raise ValueError(f"No DB transformer import results found for run name {run_name!r}.")
        df["grid"] = df.apply(synthetic_grid_label, axis=1)
        df["q_import_mvar"] = df["q_abs_mvar"] if reactive_magnitude else df["q_mvar"]
        df["s_import_mva"] = df["s_mva"]
        mean_s = df["mean_s_mva"].replace(0.0, np.nan)
        max_s_by_grid = df.groupby("powerflow_run_id")["max_s_mva"].first().replace(0.0, np.nan)
        ldc_scale = float(max_s_by_grid.mean())
        if not np.isfinite(ldc_scale) or ldc_scale == 0.0:
            ldc_scale = np.nan
        df["p_ts_norm"] = df["p_mw"] / mean_s
        df["q_ts_norm"] = df["q_import_mvar"] / mean_s
        df["s_ts_norm"] = df["s_import_mva"] / mean_s
        df["p_ldc_norm"] = df["p_mw"] / ldc_scale
        df["q_ldc_norm"] = df["q_import_mvar"] / ldc_scale
        df["s_ldc_norm"] = df["s_import_mva"] / ldc_scale
        df.attrs["ldc_scale_mva"] = ldc_scale
        return df.reset_index(drop=True)

    df["grid"] = df.apply(synthetic_grid_label, axis=1)
    df["q_import_mvar"] = df["q_mvar"].abs() if reactive_magnitude else df["q_mvar"]
    df["s_import_mva"] = np.hypot(df["p_mw"].astype(float), df["q_mvar"].astype(float))

    mean_s = df.groupby("powerflow_run_id")["s_import_mva"].transform("mean").replace(0.0, np.nan)
    max_s_by_grid = df.groupby("powerflow_run_id")["s_import_mva"].max().replace(0.0, np.nan)
    ldc_scale = float(max_s_by_grid.mean())
    if not np.isfinite(ldc_scale) or ldc_scale == 0.0:
        ldc_scale = np.nan
    df["p_ts_norm"] = df["p_mw"] / mean_s
    df["q_ts_norm"] = df["q_import_mvar"] / mean_s
    df["s_ts_norm"] = df["s_import_mva"] / mean_s
    df["p_ldc_norm"] = df["p_mw"] / ldc_scale
    df["q_ldc_norm"] = df["q_import_mvar"] / ldc_scale
    df["s_ldc_norm"] = df["s_import_mva"] / ldc_scale
    df.attrs["ldc_scale_mva"] = ldc_scale
    return df.reset_index(drop=True)


def voltage_deviation_summary_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_full_powerflow",
    stages: tuple[str, ...] = ("post",),
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Summarize DB voltage extrema for one grid or a population scope.

    Pass ``input_id`` for one concrete grid. Leave ``input_id`` as ``None`` to
    include all matching results, optionally narrowed by ``scenario_id``, ``ags``, ``plz``,
    ``kcid``, or ``bcid``.
    """
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
               pbv.stage,
               MIN(pbv.vm_pu) AS min_vm_pu,
               MAX(pbv.vm_pu) AS max_vm_pu,
               COUNT(DISTINCT pbv.t_index) AS n_timesteps,
               COUNT(DISTINCT pbv.bus) AS n_buses
        FROM surrogrid.powerflow_bus_voltage pbv
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pbv.stage = ANY(:stages)
          AND {scope_filter("pbv.powerflow_run_id")}
        GROUP BY pr.powerflow_run_id, pr.run_name, pr.scenario_id, sc.scenario_key, gc.ags, gc.plz, gc.kcid, gc.bcid,
                 gc.pylovo_grid_result_id, pbv.stage
        ORDER BY pr.powerflow_run_id, pbv.stage
        """
    )
    with db.engine.connect() as conn:
        summary = pd.read_sql_query(
            query,
            conn,
            params={"run_name": run_name, "stages": list(stages), **scope},
        )

    if summary.empty:
        compact_query = text(
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
                   pbvs.stage,
                   MIN(pbvs.voltage_min_time_pu) AS min_vm_pu,
                   MAX(pbvs.voltage_max_time_pu) AS max_vm_pu,
                   MAX(pfs.n_timesteps) AS n_timesteps,
                   COUNT(DISTINCT pbvs.bus) AS n_buses
            FROM surrogrid.powerflow_bus_voltage_summary pbvs
            JOIN surrogrid.powerflow_summary pfs
              ON pfs.powerflow_run_id = pbvs.powerflow_run_id
             AND pfs.stage = pbvs.stage
            JOIN surrogrid.powerflow_run pr
              ON pr.powerflow_run_id = pbvs.powerflow_run_id
            JOIN surrogrid.scenario sc
              ON sc.scenario_id = pr.scenario_id
            JOIN surrogrid.grid_case gc
              ON gc.grid_case_id = pr.grid_case_id
            WHERE pr.run_name = :run_name
              AND pbvs.stage = ANY(:stages)
              AND {scope_filter("pbvs.powerflow_run_id")}
            GROUP BY pr.powerflow_run_id, pr.run_name, pr.scenario_id, sc.scenario_key, gc.ags, gc.plz, gc.kcid, gc.bcid,
                     gc.pylovo_grid_result_id, pbvs.stage
            ORDER BY pr.powerflow_run_id, pbvs.stage
            """
        )
        with db.engine.connect() as conn:
            summary = pd.read_sql_query(
                compact_query,
                conn,
                params={"run_name": run_name, "stages": list(stages), **scope},
            )

    if summary.empty:
        raise ValueError(f"No DB voltage results found for run name {run_name!r}.")

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
            "n_timesteps",
            "n_buses",
            "min_vm_pu",
            "max_vm_pu",
        ]
    ].reset_index(drop=True)
