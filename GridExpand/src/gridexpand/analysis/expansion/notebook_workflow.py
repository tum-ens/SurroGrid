"""Reusable preparation helpers for the expansion analysis notebook."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from gridexpand.analysis.expansion.cases import LABEL_CASES, analysis_suffix, case_stage
from gridexpand.analysis.ids import ags_string, canonical_real_grid_id
from gridexpand.db.database import SurroGridDatabase
from gridexpand.paths import PROJECT_DIR
from gridexpand.analysis.expansion.overview import load_expansion_overview
from gridexpand.analysis.powerflow.comparison_data import (
    load_synthetic_powerflow_cutoff_profile,
    real_powerflow_headline_summary_db,
    real_powerflow_percentile_profile_db,
)
from gridexpand.analysis.powerflow.raw import transformer_import_distribution_db, voltage_deviation_summary_db


PROVIDER_LABELS = {"swf": "SWF", "uzw": "ÜZW"}
REAL_GROUP_SOURCES = {"Real SWF": "swf", "Real ÜZW": "uzw"}
NETWORK_ORDER = ("Synthetic", "Real")


def provider_group_label(provider: str, network: str) -> str:
    """Return the four-group label, e.g. ``Real ÜZW`` or ``Synthetic SWF``."""
    kind = "Real" if network == "real" else "Synthetic"
    return f"{kind} {PROVIDER_LABELS[provider]}"


def _is_real_group(group: str) -> bool:
    return str(group).startswith("Real")


def group_network(group: str) -> str:
    """Pooled network of a group, ``Real`` or ``Synthetic`` (no provider name)."""
    return "Real" if _is_real_group(group) else "Synthetic"


def group_provider(group: str) -> str | None:
    """Provider label of a group (``SWF``/``ÜZW``); None for the SWF-only label ``Synthetic``."""
    parts = str(group).split(" ", 1)
    return parts[1] if len(parts) == 2 else None


def _with_group_columns(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """``data_source`` (the group), ``network`` and ``provider`` columns of one group's rows."""
    return frame.assign(data_source=group, network=group_network(group), provider=group_provider(group))


def _default_specs_by_source(
    synthetic_specs: Mapping[str, Mapping[str, object]] | None,
    real_specs: Mapping[str, Mapping[str, object]] | None,
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]] | None,
) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    if specs_by_source is not None:
        return specs_by_source
    return {"Synthetic": synthetic_specs or {}, "Real SWF": real_specs or {}}


def synthetic_grid_keys(frame: pd.DataFrame) -> pd.Series:
    """``PLZ_kcid_bcid`` of synthetic rows: from those columns, else from the ``AGS-PLZ_kcid_bcid`` label."""
    if {"plz", "kcid", "bcid"}.issubset(frame.columns):
        parts = [frame[col].astype("Int64").astype("string") for col in ("plz", "kcid", "bcid")]
        return parts[0] + "_" + parts[1] + "_" + parts[2]
    return frame["grid"].astype("string").str.split("-", n=1).str[-1]


def _excluded_ids(excluded_grids: Mapping[str, tuple], group: str) -> set[str]:
    """Excluded grid keys of one group: canonical LV/area ids (real) or ``PLZ_kcid_bcid`` (synthetic)."""
    values = excluded_grids.get(group, ())
    if _is_real_group(group):
        return {canonical_real_grid_id(value) for value in values}
    return {str(value) for value in values}


def _without_excluded(frame: pd.DataFrame, group: str, excluded: set[str]) -> pd.DataFrame:
    if not excluded or frame.empty:
        return frame
    if _is_real_group(group):
        keys = frame["lv_id"].map(canonical_real_grid_id)
    else:
        keys = synthetic_grid_keys(frame)
    return frame[~keys.isin(excluded)].copy()


# Name used by the analysis notebooks.
normalize_ags_string = ags_string


def display_label_from_ags(ags: str | int) -> str:
    """Read a human-readable region label from ``opendata.scope``."""
    normalized_ags = normalize_ags_string(ags)
    db = SurroGridDatabase()
    query = text(
        """
        SELECT gen, bez
        FROM opendata.scope
        WHERE ags = :ags
        ORDER BY wsk DESC NULLS LAST, beginn DESC NULLS LAST
        LIMIT 1
        """
    )
    with db.engine.connect() as conn:
        row = conn.execute(query, {"ags": normalized_ags}).mappings().first()
    if row is None:
        return normalized_ags
    gen = str(row["gen"]).strip()
    bez = str(row["bez"] or "").strip()
    return f"{gen} ({bez})" if bez else gen


def load_expansion_stage_context(
    analysis_keys: Mapping[str, str],
    *,
    default_analysis_label: str,
    allow_empty: bool = False,
) -> dict[str, object]:
    """Load expansion overview tables and availability metadata for all stages."""
    expansion_tables_by_stage = {
        label: load_expansion_overview(analysis_key=key)
        for label, key in analysis_keys.items()
    }

    analysis_meta_by_stage = {}
    analysis_status_rows = []
    for label, tables in expansion_tables_by_stage.items():
        analysis_run = tables["analysis_run"]
        is_available = not analysis_run.empty
        status_row = {
            "stage_label": label,
            "analysis_key": tables["analysis_key"],
            "available": is_available,
            "data_source": None,
            "run_name": None,
            "stage": None,
            "grids_with_expansion_summary": 0,
            "grids_total": 0,
            "grids_complete": 0,
            "grids_incomplete": 0,
            "grids_excluded": 0,
            "total_cost_eur": pd.NA,
        }
        if is_available:
            meta = analysis_run.iloc[0]
            analysis_meta_by_stage[label] = meta
            cost_summary = (
                tables["cost_summary"].iloc[0]
                if not tables["cost_summary"].empty
                else None
            )
            status_row.update(
                {
                    "data_source": meta.get("data_source", "Synthetic"),
                    "run_name": meta["run_name"],
                    "stage": meta["stage"],
                    "grids_with_expansion_summary": int(
                        cost_summary["grids_with_line_rows"]
                    )
                    if cost_summary is not None
                    else 0,
                    "grids_total": int(cost_summary["grids_total"])
                    if cost_summary is not None
                    else 0,
                    "grids_complete": int(cost_summary["grids_complete"])
                    if cost_summary is not None
                    else 0,
                    "grids_incomplete": int(cost_summary["grids_incomplete"])
                    if cost_summary is not None
                    else 0,
                    "grids_excluded": int(cost_summary["grids_excluded"])
                    if cost_summary is not None
                    else 0,
                    "total_cost_eur": float(cost_summary["total_cost_eur"])
                    if cost_summary is not None
                    else 0.0,
                }
            )
        analysis_status_rows.append(status_row)

    available_analysis_keys = {
        label: analysis_keys[label]
        for label in analysis_keys
        if label in analysis_meta_by_stage
    }
    missing_analysis_keys = {
        label: analysis_keys[label]
        for label in analysis_keys
        if label not in analysis_meta_by_stage
    }
    if not available_analysis_keys:
        if not allow_empty:
            raise ValueError(
                "None of the configured analysis keys exist in surrogrid.expansion_analysis_run."
            )
        return {
            "expansion_tables_by_stage": expansion_tables_by_stage,
            "analysis_meta_by_stage": analysis_meta_by_stage,
            "analysis_status": pd.DataFrame(analysis_status_rows),
            "available_analysis_keys": {},
            "missing_analysis_keys": dict(analysis_keys),
            "default_analysis_label": default_analysis_label,
            "expansion_tables": None,
            "analysis_key": None,
        }

    resolved_default_label = default_analysis_label
    if resolved_default_label not in available_analysis_keys:
        resolved_default_label = next(iter(available_analysis_keys))
    expansion_tables = expansion_tables_by_stage[resolved_default_label]

    return {
        "expansion_tables_by_stage": expansion_tables_by_stage,
        "analysis_meta_by_stage": analysis_meta_by_stage,
        "analysis_status": pd.DataFrame(analysis_status_rows),
        "available_analysis_keys": available_analysis_keys,
        "missing_analysis_keys": missing_analysis_keys,
        "default_analysis_label": resolved_default_label,
        "expansion_tables": expansion_tables,
        "analysis_key": expansion_tables["analysis_key"],
    }


STAGED_STATION_COMPONENTS = (
    ("Transformer exchange", "transformer_exchange_cost_eur"),
    ("Load transfer", "load_transfer_cost_eur"),
    ("New substations", "new_station_cost_eur"),
    ("Voltage measures", "voltage_cost_eur"),
)


COST_COMPONENTS = (("Cables", "cable_cost_eur"), *STAGED_STATION_COMPONENTS)
REINFORCEMENT_COLUMNS = (
    "reinforcement_150_count",
    "reinforcement_185_count",
    "reinforcement_240_count",
    "reinforcement_added_capacity_ka",
)


def expansion_grid_costs(
    expansion_tables_by_source: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    stage_labels: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Per-grid expansion costs (``grid_cost_summary`` rows) of every group and stage, with group columns."""
    frames = []
    for source, tables_by_stage in expansion_tables_by_source.items():
        for label in stage_labels:
            tables = tables_by_stage.get(label)
            if tables is None or tables["grid_cost_summary"].empty:
                continue
            frame = _with_group_columns(tables["grid_cost_summary"], source)
            frames.append(frame.assign(stage_label=label))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _cost_grid_keys(frame: pd.DataFrame) -> pd.Series:
    """One key per grid of ``expansion_grid_costs`` rows: group plus LV/area id or ``PLZ_kcid_bcid``."""
    keys = pd.Series(index=frame.index, dtype="string")
    real = frame["network"].eq("Real")
    keys[real] = frame.loc[real, "lv_id"].map(canonical_real_grid_id)
    keys[~real] = synthetic_grid_keys(frame.loc[~real])
    return frame["data_source"].astype("string") + "|" + keys


def expansion_cost_comparison(
    grid_costs: pd.DataFrame,
    *,
    stage_labels: tuple[str, ...] | list[str],
    excluded_grids: Mapping[str, tuple] | None = None,
    by: str = "network",
) -> dict[str, pd.DataFrame]:
    """Expansion cost per component, stage and ``by`` on one grid set for all stages.

    Left out of every stage: the ``excluded_grids`` and each real grid whose cost is
    not ``complete`` in one of the stages (its non-converged hours leave the P100 cost
    unknown), so the stages compare the same grids. ``by`` is ``network`` (pooled
    Real/Synthetic) or ``data_source`` (the provider groups).

    Returns:
        ``costs`` (``stage``, ``data_source`` = the ``by`` value, ``component``, ``cost_eur``,
        the input of ``plot_expansion_cost_comparison_bar``), ``grids`` (grids and total per
        stage), ``reinforcements`` (standard cables added) and ``excluded`` (grid, reason).
    """
    empty = {"costs": pd.DataFrame(), "grids": pd.DataFrame(), "reinforcements": pd.DataFrame(),
             "excluded": pd.DataFrame(columns=["data_source", "grid", "reason"])}
    if grid_costs.empty:
        return empty
    frame = grid_costs[grid_costs["stage_label"].isin(stage_labels)].copy()
    frame["_key"] = _cost_grid_keys(frame)
    excluded_rows = []
    configured = pd.Series(False, index=frame.index)
    for source in frame["data_source"].unique():
        ids = _excluded_ids(excluded_grids or {}, source)
        rows = frame["data_source"].eq(source)
        configured[rows] = ~frame.index[rows].isin(_without_excluded(frame[rows], source, ids).index)
    incomplete_keys = set(frame.loc[frame["cost_status"].fillna("complete").ne("complete"), "_key"])
    for key, rows in frame[configured].groupby("_key"):
        excluded_rows.append({"data_source": rows["data_source"].iloc[0], "grid": rows["grid_label"].iloc[0],
                              "reason": "excluded: non-converged power flow"})
    for key, rows in frame[frame["_key"].isin(incomplete_keys) & ~configured].groupby("_key"):
        stages = rows.loc[rows["cost_status"].fillna("complete").ne("complete"), "stage_label"]
        excluded_rows.append({"data_source": rows["data_source"].iloc[0], "grid": rows["grid_label"].iloc[0],
                              "reason": "cost incomplete in " + ", ".join(dict.fromkeys(map(str, stages)))})
    kept = frame[~configured & ~frame["_key"].isin(incomplete_keys)].copy()

    staged = kept[[column for _, column in STAGED_STATION_COMPONENTS]].notna().any(axis=1)
    components = [(name, column) for name, column in COST_COMPONENTS]
    long_rows = []
    for (group, stage), rows in kept.groupby([by, "stage_label"], sort=False):
        for name, column in components:
            if name == "Cables" or staged[rows.index].all():
                long_rows.append({"stage": stage, "data_source": group, "component": name,
                                  "cost_eur": float(rows[column].fillna(0.0).sum())})
        if not staged[rows.index].all():  # analyses written before migration 0007: one station bar
            long_rows.append({"stage": stage, "data_source": group, "component": "Transformers",
                              "cost_eur": float(rows["transformer_cost_eur"].fillna(0.0).sum())})
    grids = (
        kept.groupby([by, "stage_label"], sort=False)
        .agg(grids=("_key", "nunique"), total_cost_eur=("total_cost_eur", "sum"))
        .reset_index()
        .rename(columns={by: "data_source", "stage_label": "stage"})
    )
    reinforcement_columns = [column for column in REINFORCEMENT_COLUMNS if column in kept.columns]
    reinforcements = (
        kept.groupby([by, "stage_label"], sort=False)[reinforcement_columns].sum(min_count=1)
        .reset_index()
        .rename(columns={by: "data_source", "stage_label": "stage"})
    )
    excluded = pd.DataFrame(excluded_rows, columns=["data_source", "grid", "reason"])
    return {"costs": pd.DataFrame(long_rows), "grids": grids, "reinforcements": reinforcements,
            "excluded": excluded.sort_values(["data_source", "grid"]).reset_index(drop=True)}


def expansion_cost_reduction_summary(
    expansion_cost_comparison: pd.DataFrame,
    *,
    post_inflex_label: str,
    post_flex_label: str,
) -> pd.DataFrame:
    """Calculate flex savings by component and data source."""
    if expansion_cost_comparison.empty:
        return pd.DataFrame()

    cost_data = expansion_cost_comparison.copy()
    if "data_source" not in cost_data.columns:
        cost_data["data_source"] = "Synthetic"
    required_columns = {post_inflex_label, post_flex_label}
    rows = []
    for source, source_data in cost_data.groupby("data_source", sort=False):
        cost_wide = source_data.pivot_table(
            index="component",
            columns="stage",
            values="cost_eur",
            aggfunc="sum",
            fill_value=0.0,
        )
        if required_columns.difference(cost_wide.columns):
            continue
        cost_wide.loc["Total"] = cost_wide.sum(axis=0)
        for component, values in cost_wide.iterrows():
            inflex_cost = float(values[post_inflex_label])
            flex_cost = float(values[post_flex_label])
            rows.append(
                {
                    "data_source": source,
                    "component": component,
                    "inflex_cost_million_eur": inflex_cost / 1_000_000.0,
                    "flex_cost_million_eur": flex_cost / 1_000_000.0,
                    "saving_million_eur": (inflex_cost - flex_cost) / 1_000_000.0,
                    "reduction_percent": (
                        (inflex_cost - flex_cost) / inflex_cost * 100.0
                        if inflex_cost
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows).round(
        {
            "inflex_cost_million_eur": 2,
            "flex_cost_million_eur": 2,
            "saving_million_eur": 2,
            "reduction_percent": 1,
        }
    )


_SYNTHETIC_FAILED_QUERY = text(
    """
    SELECT CONCAT(LPAD(gc.ags::TEXT, 8, '0'), '-', gc.plz, '_', gc.kcid, '_', gc.bcid) AS grid,
           CONCAT(gc.plz, '_', gc.kcid, '_', gc.bcid) AS grid_key,
           pfs.n_timesteps, COALESCE(pfs.n_failed_timesteps, 0) AS n_failed_timesteps
    FROM surrogrid.powerflow_run pr
    JOIN surrogrid.grid_case gc USING (grid_case_id)
    JOIN surrogrid.powerflow_summary pfs USING (powerflow_run_id)
    WHERE pr.run_name = :run_name AND pfs.stage = :stage
    """
)
_REAL_FAILED_QUERY = text(
    """
    SELECT CASE WHEN rgc.source = 'uzw' THEN CONCAT('ÜZW area-', LPAD(rgc.lv_id, 4, '0'))
                ELSE CONCAT(UPPER(rgc.source), ' LV_', LPAD(rgc.lv_id, 3, '0'))
           END AS grid,
           rgc.lv_id AS grid_key,
           rps.n_timesteps, COALESCE(rps.n_failed_timesteps, 0) AS n_failed_timesteps
    FROM surrogrid.real_powerflow_run rpr
    JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
    JOIN surrogrid.real_powerflow_summary rps USING (real_powerflow_run_id)
    WHERE rpr.run_name = :run_name AND rps.stage = :stage
      AND (CAST(:source AS TEXT) IS NULL OR rgc.source = CAST(:source AS TEXT))
    """
)


def nonconverged_grids(
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    min_failed_share: float = 0.01,
) -> pd.DataFrame:
    """Grids with non-converged power-flow hours in any case, and whether they are excluded.

    Non-converged hours are missing from the summaries, so a grid's statistics lack
    exactly its worst hours. A grid is ``excluded`` when its largest share of
    non-converged hours over the cases reaches ``min_failed_share``; it is then left
    out of every case of its own group only (real and synthetic are not matched).
    ``grid_key`` is the id of ``excluded_grids``: LV/area id (real), ``PLZ_kcid_bcid``
    (synthetic).
    """
    frames = []
    with SurroGridDatabase().engine.connect() as conn:
        for source, specs in specs_by_source.items():
            for label, spec in specs.items():
                params = {"run_name": str(spec["run_name"]), "stage": str(spec["stage"])}
                if _is_real_group(source):
                    query = _REAL_FAILED_QUERY
                    params["source"] = REAL_GROUP_SOURCES.get(source)
                else:
                    query = _SYNTHETIC_FAILED_QUERY
                frame = pd.read_sql_query(query, conn, params=params)
                if not frame.empty:
                    frames.append(frame.assign(data_source=source, comparison_stage=label))
    columns = ["data_source", "network", "provider", "grid", "grid_key", "cases", "max_failed_timesteps",
               "n_timesteps", "max_failed_share", "excluded"]
    if not frames:
        return pd.DataFrame(columns=columns)
    summaries = pd.concat(frames, ignore_index=True)
    summaries["grid_key"] = summaries["grid_key"].astype(str)
    summaries["failed_share"] = summaries["n_failed_timesteps"] / summaries["n_timesteps"]
    failed = summaries[summaries["n_failed_timesteps"] > 0]
    result = (
        failed.groupby(["data_source", "grid", "grid_key"], as_index=False)
        .agg(
            cases=("comparison_stage", lambda values: ", ".join(dict.fromkeys(map(str, values)))),
            max_failed_timesteps=("n_failed_timesteps", "max"),
            n_timesteps=("n_timesteps", "max"),
            max_failed_share=("failed_share", "max"),
        )
    )
    result["network"] = result["data_source"].map(group_network)
    result["provider"] = result["data_source"].map(group_provider)
    result["excluded"] = result["max_failed_share"] >= float(min_failed_share)
    return result[columns].sort_values(["excluded", "max_failed_share"], ascending=False).reset_index(drop=True)


def excluded_grids_by_group(nonconverged: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """``excluded_grids`` of the loaders: the ``grid_key`` of every excluded grid, per group."""
    excluded = nonconverged[nonconverged["excluded"]]
    return {
        str(source): tuple(sorted(rows["grid_key"].astype(str).unique()))
        for source, rows in excluded.groupby("data_source", sort=False)
    }


def load_powerflow_cutoff_comparison(
    *,
    synthetic_specs: Mapping[str, Mapping[str, object]] | None = None,
    real_specs: Mapping[str, Mapping[str, object]] | None = None,
    stage_order: list[str],
    ags: str | int | None = None,
    scenario_id: int | None = None,
    plz: int | None = None,
    real_plz: int | None = None,
    excluded_grids: Mapping[str, tuple] | None = None,
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> dict[str, object]:
    """Load compact synthetic and real power-flow summaries for one comparison plot.

    ``specs_by_source`` replaces ``synthetic_specs``/``real_specs`` for more than
    two groups, e.g. the four groups of ``prepare_expansion_analysis(providers=...)``.
    ``excluded_grids`` maps a group to the grids left out of every case (real: LV/area
    ids, synthetic: ``PLZ_kcid_bcid``), e.g. ``excluded_grids_by_group(nonconverged_grids(...))``.
    The profile carries ``data_source`` (group), ``network`` (Real/Synthetic) and ``provider``.
    """
    specs_by_source = _default_specs_by_source(synthetic_specs, real_specs, specs_by_source)
    excluded_grids = excluded_grids or {}
    powerflow_profiles = []
    skipped = {source: {} for source in specs_by_source}
    excluded_rows = []

    for source, specs in specs_by_source.items():
        excluded = _excluded_ids(excluded_grids, source)
        for label, spec in specs.items():
            try:
                if _is_real_group(source):
                    profile = real_powerflow_percentile_profile_db(
                        run_name=str(spec["run_name"]),
                        stage=str(spec["stage"]),
                        plz=real_plz if real_plz is not None else plz,
                        source=REAL_GROUP_SOURCES.get(source),
                    )
                else:
                    profile = load_synthetic_powerflow_cutoff_profile(
                        run_name=str(spec["run_name"]),
                        stage=str(spec["stage"]),
                        scenario_id=scenario_id,
                        ags=ags,
                        plz=plz,
                    )
            except ValueError as exc:
                skipped[source][label] = str(exc)
                continue
            kept = _without_excluded(profile, source, excluded)
            dropped = profile.loc[~profile.index.isin(kept.index), "grid"]
            if not dropped.empty:
                excluded_rows.append(
                    {
                        "data_source": source,
                        "comparison_stage": label,
                        "excluded_grids": dropped.nunique(),
                        "grids": ", ".join(sorted(dropped.astype(str).unique())),
                    }
                )
            kept = _with_group_columns(kept, source)
            kept["comparison_stage"] = label
            powerflow_profiles.append(kept)

    if not powerflow_profiles:
        raise ValueError(
            "No configured synthetic or real compact power-flow summary runs were found."
        )

    powerflow_profile = pd.concat(powerflow_profiles, ignore_index=True, sort=False)
    powerflow_profile["comparison_stage"] = pd.Categorical(
        powerflow_profile["comparison_stage"],
        categories=stage_order,
        ordered=True,
    )
    powerflow_profile["network"] = pd.Categorical(
        powerflow_profile["network"], categories=NETWORK_ORDER, ordered=True
    )
    powerflow_profile = powerflow_profile.sort_values(
        ["comparison_stage", "data_source", "metric", "grid"]
    ).reset_index(drop=True)

    asset_summary = (
        powerflow_profile.groupby(
            ["data_source", "comparison_stage", "metric", "asset_type"],
            observed=True,
            as_index=False,
        )
        .agg(assets=("asset_id", "count"), grids=("grid", "nunique"))
        .sort_values(["comparison_stage", "data_source", "metric", "asset_type"])
    )
    coverage_summary = (
        powerflow_profile.drop_duplicates(["data_source", "comparison_stage", "grid"])
        .groupby(["data_source", "comparison_stage"], observed=True, as_index=False)
        .agg(grids=("grid", "nunique"))
        .sort_values(["comparison_stage", "data_source"])
    )

    return {
        "profile": powerflow_profile,
        "asset_summary": asset_summary,
        "coverage_summary": coverage_summary,
        "excluded_grids": pd.DataFrame(excluded_rows),
        "skipped": skipped,
    }


def load_voltage_summaries_for_powerflow_comparison(
    *,
    synthetic_specs: Mapping[str, Mapping[str, object]] | None = None,
    real_specs: Mapping[str, Mapping[str, object]] | None = None,
    ags: str | int | None = None,
    scenario_id: int | None = None,
    plz: int | None = None,
    real_plz: int | None = None,
    excluded_grids: Mapping[str, tuple] | None = None,
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load source-separated voltage summaries from one scenario's run specs (``excluded_grids`` left out)."""
    specs_by_source = _default_specs_by_source(synthetic_specs, real_specs, specs_by_source)
    excluded_grids = excluded_grids or {}
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for source, specs in specs_by_source.items():
        summaries: dict[str, pd.DataFrame] = {}
        excluded = _excluded_ids(excluded_grids, source)
        for label, spec in specs.items():
            if not _is_real_group(source):
                try:
                    summary = voltage_deviation_summary_db(
                        run_name=str(spec["run_name"]),
                        stages=(str(spec["stage"]),),
                        scenario_id=scenario_id,
                        ags=ags,
                        plz=plz,
                    )
                except ValueError:
                    continue
                summaries[label] = _without_excluded(summary, source, excluded)
                continue
            try:
                summary = real_powerflow_headline_summary_db(
                    run_name=str(spec["run_name"]),
                    stage=str(spec["stage"]),
                    plz=real_plz,
                    source=REAL_GROUP_SOURCES.get(source),
                )
            except ValueError:
                continue
            summary = _without_excluded(summary, source, excluded)
            if summary.empty or "voltage_min_asset_time_pu" not in summary.columns:
                continue
            summaries[label] = pd.DataFrame(
                {
                    "grid": summary["grid"],
                    "stage": summary["stage"],
                    "n_timesteps": summary.get("n_timesteps"),
                    "n_buses": summary.get("n_voltage_buses"),
                    "min_vm_pu": summary["voltage_min_asset_time_pu"],
                    "max_vm_pu": pd.NA,
                }
            )
        if summaries:
            result[source] = summaries
    return result


def load_transformer_import_distributions_for_specs(
    synthetic_specs: Mapping[str, Mapping[str, object]],
    *,
    ags: str | int | None = None,
    scenario_id: int | None = None,
    plz: int | None = None,
) -> pd.DataFrame:
    """Load synthetic transformer diagnostics from scenario-derived run specs."""
    distributions = []
    for label, spec in synthetic_specs.items():
        try:
            distribution = transformer_import_distribution_db(
                run_name=str(spec["run_name"]),
                stage=str(spec["stage"]),
                scenario_id=scenario_id,
                ags=ags,
                plz=plz,
            )
        except ValueError:
            continue
        distribution["comparison_stage"] = label
        distributions.append(distribution)
    if not distributions:
        return pd.DataFrame()
    return pd.concat(distributions, ignore_index=True)


DEFAULT_STAGE_LABELS = {
    "pre": "status-quo",
    "post_inflex": "INFLEX",
    "post_flex": "HEMS",
}

ALL_MODEL_CASE_STAGE_LABELS = {
    **DEFAULT_STAGE_LABELS,
    "post_flex": "HEMS heuristic",
    "post_optimized": "HEMS optimized",
}


def _case_specs(run_prefix: str, labels: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """``{label: {run_name, stage}}`` of the configured model cases (``cases.LABEL_CASES``)."""
    return {
        labels[key]: {"run_name": f"{run_prefix}_{case}", "stage": case_stage(case)}
        for key, case in LABEL_CASES.items()
        if key in labels
    }


def scenario_powerflow_specs(
    scenario_prefix: str,
    stage_labels: Mapping[str, str] | None = None,
    *,
    providers: tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    """Derive all compact-summary run names from one scenario prefix.

    Without ``providers`` this returns the SWF-only groups ``Synthetic`` and
    ``Real SWF``. With providers it returns ``Real <P>``/``Synthetic <P>`` per
    provider for aligned run names ``{run_id}_{provider}_{network}_{case}``.
    """
    labels = dict(stage_labels or DEFAULT_STAGE_LABELS)
    if providers is None:
        return {
            "Synthetic": _case_specs(f"{scenario_prefix}_synthetic", labels),
            "Real SWF": _case_specs(f"{scenario_prefix}_real_swf", labels),
        }
    specs = {}
    for provider in providers:
        specs[provider_group_label(provider, "real")] = _case_specs(
            f"{scenario_prefix}_{provider}_real_{provider}", labels
        )
        specs[provider_group_label(provider, "synthetic")] = _case_specs(
            f"{scenario_prefix}_{provider}_synthetic", labels
        )
    return specs


def scenario_analysis_keys(
    scenario_prefix: str,
    stage_labels: Mapping[str, str] | None = None,
    *,
    data_source: str = "Synthetic",
    provider: str | None = None,
) -> dict[str, str]:
    """Derive stable expansion-analysis keys for one network source.

    With ``provider`` the aligned keys ``{run_id}_{provider}_{real|synthetic}_*``
    written by ``expansion.aligned_expansion`` are returned.
    """
    labels = dict(stage_labels or DEFAULT_STAGE_LABELS)
    if provider is not None:
        network = "real" if _is_real_group(data_source) else "synthetic"
        key_prefix = f"{scenario_prefix}_{provider}_{network}"
    else:
        key_prefix = scenario_prefix + ("" if data_source == "Synthetic" else "_real")
    return {
        labels[key]: f"{key_prefix}_{analysis_suffix(case)}"
        for key, case in LABEL_CASES.items()
        if key in labels
    }


def _powerflow_run_readiness(
    *,
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]],
    expected_grid_counts: Mapping[str, int] | None,
    ags: str | int | None,
    real_plz: int | None,
) -> pd.DataFrame:
    """Audit launched runs, summaries, failures, and temporal contracts."""
    db = SurroGridDatabase()
    synthetic_query = text(
        """
        SELECT COUNT(DISTINCT pr.powerflow_run_id) AS launched_grids,
               COUNT(DISTINCT pfs.powerflow_run_id) AS summary_grids,
               COUNT(DISTINCT pfs.powerflow_run_id)
                   FILTER (WHERE COALESCE(pfs.n_failed_timesteps, 0) > 0) AS grids_with_failed_timesteps,
               COALESCE(SUM(pfs.n_failed_timesteps), 0) AS failed_timesteps,
               STRING_AGG(DISTINCT pfs.n_timesteps::TEXT, ', ' ORDER BY pfs.n_timesteps::TEXT)
                   AS timestep_signatures,
               STRING_AGG(DISTINCT pr.assumptions ->> 'scenario_label', ', ')
                   FILTER (WHERE pr.assumptions ? 'scenario_label') AS scenario_labels,
               STRING_AGG(DISTINCT pr.assumptions ->> 'profile_contract', ', ')
                   FILTER (WHERE pr.assumptions ? 'profile_contract') AS profile_contracts
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        LEFT JOIN surrogrid.powerflow_summary pfs
          ON pfs.powerflow_run_id = pr.powerflow_run_id
         AND pfs.stage = :stage
        WHERE pr.run_name = :run_name
          AND (CAST(:ags AS BIGINT) IS NULL OR gc.ags = CAST(:ags AS BIGINT))
        """
    )
    real_query = text(
        """
        SELECT COUNT(DISTINCT rpr.real_powerflow_run_id) AS launched_grids,
               COUNT(DISTINCT rps.real_powerflow_run_id) AS summary_grids,
               COUNT(DISTINCT rps.real_powerflow_run_id)
                   FILTER (WHERE COALESCE(rps.n_failed_timesteps, 0) > 0) AS grids_with_failed_timesteps,
               COALESCE(SUM(rps.n_failed_timesteps), 0) AS failed_timesteps,
               STRING_AGG(DISTINCT rps.n_timesteps::TEXT, ', ' ORDER BY rps.n_timesteps::TEXT)
                   AS timestep_signatures,
               STRING_AGG(DISTINCT rpr.assumptions ->> 'scenario_label', ', ')
                   FILTER (WHERE rpr.assumptions ? 'scenario_label') AS scenario_labels,
               STRING_AGG(DISTINCT rpr.assumptions ->> 'profile_contract', ', ')
                   FILTER (WHERE rpr.assumptions ? 'profile_contract') AS profile_contracts
        FROM surrogrid.real_powerflow_run rpr
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        LEFT JOIN surrogrid.real_powerflow_summary rps
          ON rps.real_powerflow_run_id = rpr.real_powerflow_run_id
         AND rps.stage = :stage
        WHERE rpr.run_name = :run_name
          AND (:plz IS NULL OR rgc.plz = :plz)
          AND (CAST(:source AS TEXT) IS NULL OR rgc.source = CAST(:source AS TEXT))
        """
    )

    rows = []
    with db.engine.connect() as conn:
        for source, specs in specs_by_source.items():
            for stage_label, spec in specs.items():
                params = {
                    "run_name": str(spec["run_name"]),
                    "stage": str(spec["stage"]),
                }
                if not _is_real_group(source):
                    params["ags"] = None if ags is None else int(normalize_ags_string(ags))
                    result = conn.execute(synthetic_query, params).mappings().one()
                else:
                    params["plz"] = real_plz
                    params["source"] = REAL_GROUP_SOURCES.get(source)
                    result = conn.execute(real_query, params).mappings().one()
                expected = (
                    None
                    if expected_grid_counts is None
                    else expected_grid_counts.get(source)
                )
                launched = int(result["launched_grids"] or 0)
                summaries = int(result["summary_grids"] or 0)
                failed_timesteps = int(result["failed_timesteps"] or 0)
                complete = (
                    summaries > 0
                    and launched == summaries
                    and failed_timesteps == 0
                    and (expected is None or summaries == int(expected))
                )
                rows.append(
                    {
                        "data_source": source,
                        "stage": stage_label,
                        "run_name": str(spec["run_name"]),
                        "expected_grids": expected,
                        "launched_grids": launched,
                        "summary_grids": summaries,
                        "pending_or_missing_summaries": max(launched - summaries, 0),
                        "grids_with_failed_timesteps": int(
                            result["grids_with_failed_timesteps"] or 0
                        ),
                        "failed_timesteps": failed_timesteps,
                        "timestep_signatures": result["timestep_signatures"],
                        "scenario_labels": result["scenario_labels"],
                        "profile_contracts": result["profile_contracts"],
                        "complete": complete,
                    }
                )
    return pd.DataFrame(rows)


def _publication_gate(
    *,
    scenario_prefix: str,
    powerflow_status: pd.DataFrame,
    expansion_status: pd.DataFrame,
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]],
    expected_grid_counts: Mapping[str, int] | None,
) -> pd.DataFrame:
    checks = []
    expected_stage_count = sum(len(specs) for specs in specs_by_source.values())
    all_runs_complete = bool(
        len(powerflow_status) == expected_stage_count
        and powerflow_status["complete"].fillna(False).all()
    )
    checks.append(
        {
            "check": f"All {expected_stage_count} compact power-flow runs complete",
            "passed": all_runs_complete,
            "detail": (
                f"{int(powerflow_status['summary_grids'].sum())} grid-stage summaries; "
                f"{int(powerflow_status['failed_timesteps'].sum())} failed timesteps"
            ),
        }
    )

    signatures = {
        str(value)
        for value in powerflow_status["timestep_signatures"].dropna()
        if str(value).strip()
    }
    checks.append(
        {
            "check": "Shared temporal horizon",
            "passed": (
                len(signatures) == 1 and len(powerflow_status) == expected_stage_count
            ),
            "detail": ", ".join(sorted(signatures)) or "No compact summaries",
        }
    )
    scenario_labels = {
        str(value)
        for value in powerflow_status["scenario_labels"].dropna()
        if str(value).strip()
    }
    # Aligned runs label every provider and case: ``{run_id}_{provider}_{case}``.
    checks.append(
        {
            "check": "Scenario labels of this run",
            "passed": bool(scenario_labels)
            and all(label == scenario_prefix or label.startswith(f"{scenario_prefix}_") for label in scenario_labels),
            "detail": ", ".join(sorted(scenario_labels)) or "No scenario labels",
        }
    )
    profile_contracts = {
        str(value)
        for value in powerflow_status["profile_contracts"].dropna()
        if str(value).strip()
    }
    checks.append(
        {
            "check": "Single paired profile contract",
            "passed": len(profile_contracts) == 1,
            "detail": ", ".join(sorted(profile_contracts)) or "No profile contract",
        }
    )

    rows = expansion_status.copy()
    expected_runs = {
        (source, stage): str(spec["run_name"])
        for source, specs in specs_by_source.items()
        for stage, spec in specs.items()
    }
    if not rows.empty:
        rows["run_matches"] = rows.apply(
            lambda row: (
                bool(row["available"])
                and str(row["run_name"])
                == expected_runs.get((str(row["data_source"]), str(row["stage_label"])))
            ),
            axis=1,
        )
        rows["grid_count_matches"] = rows.apply(
            lambda row: (
                int(row["grids_total"]) > 0
                if expected_grid_counts is None
                else int(row["grids_total"])
                == int(
                    expected_grid_counts.get(
                        str(row["data_source"]), row["grids_total"]
                    )
                )
            ),
            axis=1,
        )
        expansion_complete = bool(
            len(rows) == expected_stage_count
            and rows["run_matches"].all()
            and rows["grid_count_matches"].all()
        )
    else:
        expansion_complete = False
    checks.append(
        {
            "check": f"{expected_stage_count} matching synthetic/real expansion materializations",
            "passed": expansion_complete,
            "detail": f"{int(rows['available'].fillna(False).sum()) if not rows.empty else 0}/"
            f"{expected_stage_count} available",
        }
    )
    incomplete = int(rows["grids_incomplete"].sum()) if not rows.empty else 0
    excluded = int(rows["grids_excluded"].sum()) if not rows.empty else 0
    checks.append(
        {
            "check": "All materialized grid costs complete",
            "passed": expansion_complete and incomplete == 0,
            "detail": f"{incomplete} incomplete and {excluded} explicitly excluded grid-stage rows",
        }
    )
    result = pd.DataFrame(checks)
    result.attrs["publication_ready"] = bool(result["passed"].all())
    return result


def prepare_expansion_analysis(
    *,
    scenario_prefix: str,
    ags: str | int | None = None,
    expected_grid_counts: Mapping[str, int] | None = None,
    stage_labels: Mapping[str, str] | None = None,
    include_optimized: bool = False,
    default_stage: str = "post_flex",
    real_plz: int | None = None,
    require_temporal_method: str | None = None,
    enforce_provenance: bool = True,
    providers: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Prepare one coherent synthetic/real scenario for the analysis notebook.

    Provenance is enforced here, on the path the notebook actually calls, rather
    than left to a helper a reader must remember to invoke. Pass
    ``enforce_provenance=False`` only to inspect a knowingly mixed selection.

    Without ``providers`` the SWF-only groups ``Synthetic``/``Real SWF`` of
    ``ags`` are prepared. With ``providers=("swf", "uzw")`` an aligned run is
    prepared as four groups (``Real SWF``, ``Synthetic SWF``, ``Real ÜZW``,
    ``Synthetic ÜZW``); provider scope then comes from the run names, and
    ``ags``/``real_plz`` are optional extra filters. Pass
    ``specs_by_source=context["specs_by_source"]`` to the loaders.
    """
    labels = dict(
        stage_labels
        or (ALL_MODEL_CASE_STAGE_LABELS if include_optimized else DEFAULT_STAGE_LABELS)
    )
    specs_by_source = scenario_powerflow_specs(
        scenario_prefix, labels, providers=providers
    )
    default_label = labels[default_stage]
    if providers is None:
        analysis_keys_by_source = {
            source: scenario_analysis_keys(scenario_prefix, labels, data_source=source)
            for source in specs_by_source
        }
    else:
        analysis_keys_by_source = {
            provider_group_label(provider, network): scenario_analysis_keys(
                scenario_prefix,
                labels,
                data_source=provider_group_label(provider, network),
                provider=provider,
            )
            for provider in providers
            for network in ("real", "synthetic")
        }
    expansion_context_by_source = {
        source: load_expansion_stage_context(
            keys,
            default_analysis_label=default_label,
            allow_empty=True,
        )
        for source, keys in analysis_keys_by_source.items()
    }
    # The group labels the status rows: synthetic analyses store data_source "Synthetic" for every provider.
    expansion_status = pd.concat(
        [
            context["analysis_status"].assign(data_source=source)
            for source, context in expansion_context_by_source.items()
        ],
        ignore_index=True,
    )
    powerflow_status = _powerflow_run_readiness(
        specs_by_source=specs_by_source,
        expected_grid_counts=expected_grid_counts,
        ags=ags,
        real_plz=real_plz,
    )
    publication_gate = _publication_gate(
        scenario_prefix=scenario_prefix,
        powerflow_status=powerflow_status,
        expansion_status=expansion_status,
        specs_by_source=specs_by_source,
        expected_grid_counts=expected_grid_counts,
    )
    provenance = None
    if enforce_provenance:
        provenance = assert_consistent_temporal_method(
            expansion_status, expected=require_temporal_method
        )

    synthetic_source = next(
        source for source in specs_by_source if not _is_real_group(source)
    )
    synthetic_context = expansion_context_by_source[synthetic_source]
    display_label = (
        display_label_from_ags(ags)
        if providers is None
        else " + ".join(PROVIDER_LABELS[provider] for provider in providers)
    )
    return {
        "temporal_provenance": provenance,
        "scenario_prefix": scenario_prefix,
        "display_label": display_label,
        "stage_labels": labels,
        "providers": providers,
        "analysis_keys": analysis_keys_by_source[synthetic_source],
        "analysis_keys_by_source": analysis_keys_by_source,
        "specs_by_source": specs_by_source,
        "synthetic_specs": specs_by_source.get("Synthetic"),
        "real_specs": specs_by_source.get("Real SWF"),
        "real_plz": real_plz,
        "powerflow_status": powerflow_status,
        "publication_gate": publication_gate,
        "publication_ready": bool(publication_gate.attrs["publication_ready"]),
        "expansion_context_by_source": expansion_context_by_source,
        "expansion_tables_by_source": {
            source: context["expansion_tables_by_stage"]
            for source, context in expansion_context_by_source.items()
        },
        "analysis_meta_by_source": {
            source: context["analysis_meta_by_stage"]
            for source, context in expansion_context_by_source.items()
        },
        "analysis_status": expansion_status,
        **{
            key: value
            for key, value in synthetic_context.items()
            if key != "analysis_status"
        },
    }


def load_cable_loading_decomposition(
    *,
    synthetic_specs: Mapping[str, Mapping[str, object]] | None = None,
    real_specs: Mapping[str, Mapping[str, object]] | None = None,
    ags: str | int | None = None,
    real_plz: int | None = None,
    excluded_grids: Mapping[str, tuple] | None = None,
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> pd.DataFrame:
    """Load one annual-maximum row per analyzed cable for every network group (``excluded_grids`` left out)."""
    specs_by_source = _default_specs_by_source(synthetic_specs, real_specs, specs_by_source)
    excluded_grids = excluded_grids or {}
    db = SurroGridDatabase()
    synthetic_query = text(
        """
        SELECT CONCAT(LPAD(gc.ags::TEXT, 8, '0'), '-', gc.plz, '_', gc.kcid, '_', gc.bcid) AS grid,
               pcs.cable AS asset_id,
               pcs.cable_installed_capacity_ka,
               pcs.cable_loading_max_time_percent
        FROM surrogrid.powerflow_cable_summary pcs
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pcs.stage = :stage
          AND (CAST(:ags AS BIGINT) IS NULL OR gc.ags = CAST(:ags AS BIGINT))
        """
    )
    real_query = text(
        """
        SELECT CASE WHEN rgc.source = 'uzw' THEN CONCAT('ÜZW area-', LPAD(rgc.lv_id, 4, '0'))
                    ELSE CONCAT(UPPER(rgc.source), ' LV_', LPAD(rgc.lv_id, 3, '0'))
               END AS grid,
               rgc.lv_id,
               rpcs.cable AS asset_id,
               rpcs.cable_installed_capacity_ka,
               rpcs.cable_loading_max_time_percent
        FROM surrogrid.real_powerflow_cable_summary rpcs
        JOIN surrogrid.real_powerflow_run rpr USING (real_powerflow_run_id)
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        WHERE rpr.run_name = :run_name
          AND rpcs.stage = :stage
          AND (:plz IS NULL OR rgc.plz = :plz)
          AND (CAST(:source AS TEXT) IS NULL OR rgc.source = CAST(:source AS TEXT))
        """
    )

    frames = []
    with db.engine.connect() as conn:
        for source, specs in specs_by_source.items():
            excluded = _excluded_ids(excluded_grids, source)
            for stage_label, spec in specs.items():
                params = {
                    "run_name": str(spec["run_name"]),
                    "stage": str(spec["stage"]),
                }
                if not _is_real_group(source):
                    params["ags"] = None if ags is None else int(normalize_ags_string(ags))
                    frame = pd.read_sql_query(synthetic_query, conn, params=params)
                else:
                    params["plz"] = real_plz
                    params["source"] = REAL_GROUP_SOURCES.get(source)
                    frame = pd.read_sql_query(real_query, conn, params=params)
                frame = _without_excluded(frame, source, excluded)
                if frame.empty:
                    continue
                frame = _with_group_columns(frame, source)
                frame["comparison_stage"] = stage_label
                frame["installed_capacity_a"] = (
                    pd.to_numeric(frame["cable_installed_capacity_ka"], errors="coerce")
                    * 1000.0
                )
                frame["max_loading_percent"] = pd.to_numeric(
                    frame["cable_loading_max_time_percent"], errors="coerce"
                )
                frame["max_current_a"] = (
                    frame["installed_capacity_a"] * frame["max_loading_percent"] / 100.0
                )
                frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "grid",
                "asset_id",
                "data_source",
                "network",
                "provider",
                "comparison_stage",
                "installed_capacity_a",
                "max_current_a",
                "max_loading_percent",
            ]
        )
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["installed_capacity_a", "max_current_a", "max_loading_percent"]
    )


def export_scenario_analysis_manifest(
    context: Mapping[str, object],
    *,
    output_dir: str | Path,
    excluded_grids: Mapping[str, tuple] | None = None,
) -> dict[str, Path]:
    """Export scenario identity, readiness tables and the excluded grids alongside notebook figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None

    powerflow_status = context["powerflow_status"]
    publication_gate = context["publication_gate"]
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision,
        "scenario_prefix": context["scenario_prefix"],
        "publication_ready": bool(context["publication_ready"]),
        "analysis_keys": context["analysis_keys"],
        "analysis_keys_by_source": context.get("analysis_keys_by_source"),
        "synthetic_specs": context["synthetic_specs"],
        "real_specs": context["real_specs"],
        "specs_by_source": context.get("specs_by_source"),
        "excluded_grids": {
            source: sorted(_excluded_ids(excluded_grids or {}, source)) for source in (excluded_grids or {})
        },
        "publication_checks": publication_gate.to_dict(orient="records"),
    }
    manifest_path = output_dir / "scenario_analysis_manifest.json"
    status_path = output_dir / "powerflow_readiness.csv"
    gate_path = output_dir / "publication_gate.csv"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    powerflow_status.to_csv(status_path, index=False)
    publication_gate.to_csv(gate_path, index=False)
    return {
        "manifest": manifest_path,
        "powerflow_readiness": status_path,
        "publication_gate": gate_path,
    }


def temporal_method_by_run_name(run_names) -> pd.DataFrame:
    """Read the recorded temporal provenance of each power-flow run.

    Identity, not file name, decides whether a result belongs in a reference
    comparison. Step 3 writes ``urbs_out/temporal_method`` for every run and both
    Step-4 entry points copy it into the run assumptions.
    """
    names = sorted({str(name) for name in run_names if name})
    if not names:
        return pd.DataFrame(
            columns=[
                "run_name",
                "temporal_method",
                "operating_hours",
                "storage_boundary_policy",
                "ev_boundary_policy",
            ]
        )
    query = text(
        """
        SELECT run_name,
               assumptions ->> 'temporal_method' AS temporal_method,
               assumptions ->> 'operating_hours' AS operating_hours,
               assumptions ->> 'storage_boundary_policy' AS storage_boundary_policy,
               assumptions ->> 'ev_boundary_policy' AS ev_boundary_policy
        FROM surrogrid.powerflow_run
        WHERE run_name = ANY(:names)
        UNION
        SELECT run_name,
               assumptions ->> 'temporal_method',
               assumptions ->> 'operating_hours',
               assumptions ->> 'storage_boundary_policy',
               assumptions ->> 'ev_boundary_policy'
        FROM surrogrid.real_powerflow_run
        WHERE run_name = ANY(:names)
        """
    )
    db = SurroGridDatabase()
    with db.engine.connect() as conn:
        rows = conn.execute(query, {"names": names}).mappings().all()
    return pd.DataFrame([dict(row) for row in rows]).drop_duplicates()


def assert_consistent_temporal_method(
    analysis_status: pd.DataFrame,
    *,
    expected: str | None = None,
) -> pd.DataFrame:
    """Refuse to compare stages that were not produced the same way.

    Raises if the selected stages mix temporal methods, storage-boundary
    policies or EV service models, or if any stage lacks provenance. Pass
    ``expected`` to additionally pin the comparison to one method.
    """
    available = analysis_status[analysis_status["available"].astype(bool)]
    provenance = temporal_method_by_run_name(available["run_name"])
    merged = available.merge(provenance, on="run_name", how="left")

    # A missing field is never evidence of agreement.
    for column in (
        "temporal_method",
        "operating_hours",
        "storage_boundary_policy",
        "ev_boundary_policy",
    ):
        absent = merged.loc[merged[column].isna(), "stage_label"].tolist()
        if absent:
            raise ValueError(
                f"These stages record no {column} and cannot be used in a "
                f"reference comparison: {absent}. They predate the temporal "
                "provenance record and must be rerun."
            )
        distinct = sorted(merged[column].astype(str).unique())
        if len(distinct) > 1:
            raise ValueError(
                f"Selected stages mix {column} values {distinct}; a paired "
                "comparison requires identical physical inputs and boundary "
                "treatment. Controller dispatch may differ; these fields may not."
            )
    if expected is not None:
        found = sorted(merged["temporal_method"].astype(str).unique())
        if found != [str(expected)]:
            raise ValueError(
                f"Expected temporal_method={expected!r}, found {found}."
            )
    return merged
