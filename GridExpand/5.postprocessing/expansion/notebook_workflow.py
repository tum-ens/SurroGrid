"""Reusable preparation helpers for the expansion analysis notebook."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
import subprocess
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase  # noqa: E402
from expansion import grid_expansion  # noqa: E402
from powerflow.comparison_data import (  # noqa: E402
    load_synthetic_powerflow_cutoff_profile,
    real_powerflow_headline_summary_db,
    real_powerflow_percentile_profile_db,
)
from plotting.powerflow_transformer import transformer_import_distribution_db  # noqa: E402
from plotting.powerflow_voltage import voltage_deviation_summary_db  # noqa: E402


def normalize_ags_string(value: str | int) -> str:
    """Return AGS as an eight-character string with leading zero if needed."""
    return str(int(str(value).strip().lstrip("0") or "0")).zfill(8)


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


def output_slug_from_ags(ags: str | int) -> str:
    """Return the directory-safe AGS slug used for plot exports."""
    return normalize_ags_string(ags)


def analysis_prefix_from_ags(ags: str | int, base_suffix: str) -> str:
    """Find the newest materialized expansion-analysis prefix for a region."""
    normalized_ags = normalize_ags_string(ags)
    db = SurroGridDatabase()
    query = text(
        """
        SELECT analysis_key, created_at
        FROM surrogrid.expansion_analysis_run
        WHERE ags = :ags
          AND (
              analysis_key = :pre_key
              OR analysis_key = :post_key
              OR analysis_key = :post_no_flex_key
              OR analysis_key LIKE :prefixed_pre_key
              OR analysis_key LIKE :prefixed_post_key
              OR analysis_key LIKE :prefixed_post_no_flex_key
          )
        ORDER BY created_at DESC
        """
    )
    params = {
        "ags": int(normalized_ags),
        "pre_key": f"{base_suffix}_pre",
        "post_key": f"{base_suffix}_post",
        "post_no_flex_key": f"{base_suffix}_post_no_flex",
        "prefixed_pre_key": f"%_{base_suffix}_pre",
        "prefixed_post_key": f"%_{base_suffix}_post",
        "prefixed_post_no_flex_key": f"%_{base_suffix}_post_no_flex",
    }
    with db.engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    prefixes: list[str] = []
    for row in rows:
        analysis_key = str(row["analysis_key"])
        for stage_suffix in ("_post_no_flex", "_post", "_pre"):
            ending = f"{base_suffix}{stage_suffix}"
            if analysis_key.endswith(ending):
                prefix = analysis_key[: -len(stage_suffix)]
                if prefix not in prefixes:
                    prefixes.append(prefix)
                break

    if not prefixes:
        raise ValueError(
            f"No materialized expansion analysis found for AGS {normalized_ags} "
            f"and base suffix {base_suffix!r}."
        )
    return prefixes[0]


def load_expansion_stage_context(
    analysis_keys: Mapping[str, str],
    *,
    default_analysis_label: str,
    allow_empty: bool = False,
) -> dict[str, object]:
    """Load expansion overview tables and availability metadata for all stages."""
    expansion_tables_by_stage = {
        label: grid_expansion.load_expansion_overview(analysis_key=key)
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
            cost_summary = tables["cost_summary"].iloc[0] if not tables["cost_summary"].empty else None
            status_row.update(
                {
                    "data_source": meta.get("data_source", "Synthetic"),
                    "run_name": meta["run_name"],
                    "stage": meta["stage"],
                    "grids_with_expansion_summary": int(cost_summary["grids_with_line_rows"])
                    if cost_summary is not None
                    else 0,
                    "grids_total": int(cost_summary["grids_total"]) if cost_summary is not None else 0,
                    "grids_complete": int(cost_summary["grids_complete"]) if cost_summary is not None else 0,
                    "grids_incomplete": int(cost_summary["grids_incomplete"]) if cost_summary is not None else 0,
                    "grids_excluded": int(cost_summary["grids_excluded"]) if cost_summary is not None else 0,
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
            raise ValueError("None of the configured analysis keys exist in surrogrid.expansion_analysis_run.")
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


def expansion_cost_comparison_from_tables(
    expansion_tables_by_stage: Mapping[str, dict[str, pd.DataFrame]],
    analysis_meta_by_stage: Mapping[str, pd.Series],
    *,
    post_no_flex_label: str,
    post_flex_label: str,
    data_source: str = "Synthetic",
) -> pd.DataFrame:
    """Build the cable/transformer cost table used by the comparison bar chart."""
    cost_rows = []
    for label in (post_no_flex_label, post_flex_label):
        tables = expansion_tables_by_stage.get(label)
        if tables is None or label not in analysis_meta_by_stage or tables["cost_summary"].empty:
            continue
        cost_summary_row = tables["cost_summary"].iloc[0]
        cost_rows.extend(
            [
                {
                    "stage": label,
                    "data_source": data_source,
                    "component": "Cables",
                    "cost_eur": float(cost_summary_row["cable_cost_eur"]),
                },
                {
                    "stage": label,
                    "data_source": data_source,
                    "component": "Transformers",
                    "cost_eur": float(cost_summary_row["transformer_cost_eur"]),
                },
            ]
        )
    return pd.DataFrame(cost_rows)



def reinforcement_catalog_summary(
    expansion_tables_by_source: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
) -> pd.DataFrame:
    """Return selected standard reinforcement-cable counts by source and stage."""
    rows: list[dict[str, object]] = []
    for source, tables_by_stage in expansion_tables_by_source.items():
        for stage_label, tables in tables_by_stage.items():
            summary = tables.get("cost_summary", pd.DataFrame())
            if summary.empty:
                continue
            row = summary.iloc[0]
            rows.append(
                {
                    "data_source": source,
                    "stage_label": stage_label,
                    "NAYY_4_150": int(row.get("reinforcement_150_count", 0) or 0),
                    "NAYY_4_185": int(row.get("reinforcement_185_count", 0) or 0),
                    "NAYY_4_240": int(row.get("reinforcement_240_count", 0) or 0),
                    "added_capacity_ka": round(
                        float(row.get("reinforcement_added_capacity_ka", 0.0) or 0.0),
                        3,
                    ),
                }
            )
    return pd.DataFrame(rows)



def expansion_cost_reduction_summary(
    expansion_cost_comparison: pd.DataFrame,
    *,
    post_no_flex_label: str,
    post_flex_label: str,
) -> pd.DataFrame:
    """Calculate flex savings by component and data source."""
    if expansion_cost_comparison.empty:
        return pd.DataFrame()

    cost_data = expansion_cost_comparison.copy()
    if "data_source" not in cost_data.columns:
        cost_data["data_source"] = "Synthetic"
    required_columns = {post_no_flex_label, post_flex_label}
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
            no_flex_cost = float(values[post_no_flex_label])
            flex_cost = float(values[post_flex_label])
            rows.append(
                {
                    "data_source": source,
                    "component": component,
                    "no_flex_cost_million_eur": no_flex_cost / 1_000_000.0,
                    "flex_cost_million_eur": flex_cost / 1_000_000.0,
                    "saving_million_eur": (no_flex_cost - flex_cost) / 1_000_000.0,
                    "reduction_percent": (
                        (no_flex_cost - flex_cost) / no_flex_cost * 100.0
                        if no_flex_cost
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows).round(
        {
            "no_flex_cost_million_eur": 2,
            "flex_cost_million_eur": 2,
            "saving_million_eur": 2,
            "reduction_percent": 1,
        }
    )


def expansion_cost_coverage_summary(analysis_status: pd.DataFrame) -> pd.DataFrame:
    """Report comparable totals and per-complete-grid costs for each source/stage."""
    if analysis_status.empty:
        return pd.DataFrame()
    columns = [
        "data_source",
        "stage_label",
        "grids_total",
        "grids_complete",
        "grids_incomplete",
        "grids_excluded",
        "total_cost_eur",
    ]
    result = analysis_status.loc[analysis_status["available"], columns].copy()
    result["cost_per_complete_grid_eur"] = result["total_cost_eur"] / result["grids_complete"].replace(0, pd.NA)
    return result.round({"total_cost_eur": 0, "cost_per_complete_grid_eur": 0})


def load_powerflow_cutoff_comparison(
    *,
    synthetic_specs: Mapping[str, Mapping[str, object]],
    real_specs: Mapping[str, Mapping[str, object]],
    stage_order: list[str],
    ags: str | int,
    scenario_id: int | None = None,
    plz: int | None = None,
    real_plz: int | None = None,
    excluded_real_lv_ids: tuple[int, ...] = (),
) -> dict[str, object]:
    """Load compact synthetic and real power-flow summaries for one comparison plot."""
    powerflow_profiles = []
    skipped = {"Synthetic": {}, "Real SWF": {}}
    excluded_real_grids = []
    excluded_lv_ids = {str(int(lv_id)) for lv_id in excluded_real_lv_ids}

    for label, spec in synthetic_specs.items():
        try:
            profile = load_synthetic_powerflow_cutoff_profile(
                run_name=str(spec["run_name"]),
                stage=str(spec["stage"]),
                scenario_id=scenario_id,
                ags=ags,
                plz=plz,
            )
        except ValueError as exc:
            skipped["Synthetic"][label] = str(exc)
            continue
        profile["comparison_stage"] = label
        profile["data_source"] = "Synthetic"
        powerflow_profiles.append(profile)

    if real_plz is None and normalize_ags_string(ags) == "09474126":
        real_plz = 91301
    for label, spec in real_specs.items():
        try:
            profile = real_powerflow_percentile_profile_db(
                run_name=str(spec["run_name"]),
                stage=str(spec["stage"]),
                plz=real_plz if real_plz is not None else plz,
            )
        except ValueError as exc:
            skipped["Real SWF"][label] = str(exc)
            continue
        if excluded_lv_ids and "lv_id" in profile.columns:
            excluded = profile[profile["lv_id"].astype(str).isin(excluded_lv_ids)]
            if not excluded.empty:
                excluded_real_grids.append(
                    {
                        "comparison_stage": label,
                        "excluded_lv_ids": ", ".join(
                            f"LV_{int(lv_id):03d}"
                            for lv_id in sorted(excluded["lv_id"].astype(int).unique())
                        ),
                        "excluded_grids": excluded["grid"].nunique(),
                    }
                )
            profile = profile[~profile["lv_id"].astype(str).isin(excluded_lv_ids)].copy()
        profile["comparison_stage"] = label
        profile["data_source"] = "Real SWF"
        powerflow_profiles.append(profile)

    if not powerflow_profiles:
        raise ValueError("No configured synthetic or real compact power-flow summary runs were found.")

    powerflow_profile = pd.concat(powerflow_profiles, ignore_index=True, sort=False)
    powerflow_profile["comparison_stage"] = pd.Categorical(
        powerflow_profile["comparison_stage"],
        categories=stage_order,
        ordered=True,
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
        "excluded_real_grids": pd.DataFrame(excluded_real_grids),
        "skipped": skipped,
    }


def meta_filter(meta: pd.Series) -> dict[str, object]:
    """Convert one expansion-analysis metadata row to DB loader filters."""
    return {
        "run_name": meta["run_name"],
        "stage": meta["stage"],
        "scenario_id": None if pd.isna(meta["scenario_id"]) else int(meta["scenario_id"]),
        "ags": None if pd.isna(meta["ags"]) else int(meta["ags"]),
        "plz": None if pd.isna(meta["plz"]) else int(meta["plz"]),
    }


def load_voltage_summaries_for_analysis(
    analysis_meta_by_stage: Mapping[str, pd.Series],
) -> dict[str, pd.DataFrame]:
    """Load grid-level voltage-extreme summaries for each available synthetic stage."""
    voltage_summaries = {}
    for label, meta in analysis_meta_by_stage.items():
        filters = meta_filter(meta)
        voltage_summaries[label] = voltage_deviation_summary_db(
            run_name=filters["run_name"],
            stages=(filters["stage"],),
            scenario_id=filters["scenario_id"],
            ags=filters["ags"],
            plz=filters["plz"],
        )
    if not voltage_summaries:
        raise ValueError("No available expansion analyses for voltage diagnostics.")
    return voltage_summaries



def load_voltage_summaries_for_powerflow_comparison(
    *,
    synthetic_specs: Mapping[str, Mapping[str, object]],
    real_specs: Mapping[str, Mapping[str, object]],
    ags: str | int,
    scenario_id: int | None = None,
    plz: int | None = None,
    real_plz: int | None = None,
    excluded_real_lv_ids: tuple[int, ...] = (),
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load source-separated voltage summaries from one scenario's run specs."""
    synthetic: dict[str, pd.DataFrame] = {}
    for label, spec in synthetic_specs.items():
        try:
            synthetic[label] = voltage_deviation_summary_db(
                run_name=str(spec["run_name"]),
                stages=(str(spec["stage"]),),
                scenario_id=scenario_id,
                ags=ags,
                plz=plz,
            )
        except ValueError:
            continue

    if real_plz is None and normalize_ags_string(ags) == "09474126":
        real_plz = 91301
    real: dict[str, pd.DataFrame] = {}
    excluded_lv_ids = {str(int(lv_id)) for lv_id in excluded_real_lv_ids}
    for label, spec in real_specs.items():
        try:
            summary = real_powerflow_headline_summary_db(
                run_name=str(spec["run_name"]),
                stage=str(spec["stage"]),
                plz=real_plz,
            )
        except ValueError:
            continue
        if excluded_lv_ids and "lv_id" in summary.columns:
            summary = summary[~summary["lv_id"].astype(str).isin(excluded_lv_ids)].copy()
        if summary.empty or "voltage_min_asset_time_pu" not in summary.columns:
            continue
        real[label] = pd.DataFrame(
            {
                "grid": summary["grid"],
                "stage": summary["stage"],
                "n_timesteps": summary.get("n_timesteps"),
                "n_buses": summary.get("n_voltage_buses"),
                "min_vm_pu": summary["voltage_min_asset_time_pu"],
                "max_vm_pu": pd.NA,
            }
        )
    result: dict[str, dict[str, pd.DataFrame]] = {}
    if synthetic:
        result["Synthetic"] = synthetic
    if real:
        result["Real SWF"] = real
    return result


def load_transformer_import_distributions_for_specs(
    synthetic_specs: Mapping[str, Mapping[str, object]],
    *,
    ags: str | int,
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
    "post_no_flex": "no-flex",
    "post_flex": "HEMS",
}


def scenario_powerflow_specs(
    scenario_prefix: str,
    stage_labels: Mapping[str, str] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    """Derive all compact-summary run names from one scenario prefix."""
    labels = dict(stage_labels or DEFAULT_STAGE_LABELS)
    return {
        "Synthetic": {
            labels["pre"]: {"run_name": f"{scenario_prefix}_synthetic_pre", "stage": "pre"},
            labels["post_no_flex"]: {
                "run_name": f"{scenario_prefix}_synthetic_post-inflex-heuristic",
                "stage": "post",
            },
            labels["post_flex"]: {"run_name": f"{scenario_prefix}_synthetic_post-hems-heuristic", "stage": "post"},
        },
        "Real SWF": {
            labels["pre"]: {"run_name": f"{scenario_prefix}_real_swf_pre", "stage": "pre"},
            labels["post_no_flex"]: {
                "run_name": f"{scenario_prefix}_real_swf_post-inflex-heuristic",
                "stage": "post",
            },
            labels["post_flex"]: {"run_name": f"{scenario_prefix}_real_swf_post-hems-heuristic", "stage": "post"},
        },
    }


def scenario_analysis_keys(
    scenario_prefix: str,
    stage_labels: Mapping[str, str] | None = None,
    *,
    data_source: str = "Synthetic",
) -> dict[str, str]:
    """Derive stable expansion-analysis keys for one network source."""
    labels = dict(stage_labels or DEFAULT_STAGE_LABELS)
    source_suffix = "" if data_source == "Synthetic" else "_real"
    return {
        labels["pre"]: f"{scenario_prefix}{source_suffix}_pre",
        labels["post_no_flex"]: f"{scenario_prefix}{source_suffix}_post_no_flex",
        labels["post_flex"]: f"{scenario_prefix}{source_suffix}_post",
    }


def _powerflow_run_readiness(
    *,
    specs_by_source: Mapping[str, Mapping[str, Mapping[str, object]]],
    expected_grid_counts: Mapping[str, int] | None,
    ags: str | int,
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
          AND gc.ags = :ags
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
        """
    )

    rows = []
    with db.engine.connect() as conn:
        for source, specs in specs_by_source.items():
            for stage_label, spec in specs.items():
                params = {"run_name": str(spec["run_name"]), "stage": str(spec["stage"])}
                if source == "Synthetic":
                    params["ags"] = int(normalize_ags_string(ags))
                    result = conn.execute(synthetic_query, params).mappings().one()
                else:
                    params["plz"] = real_plz
                    result = conn.execute(real_query, params).mappings().one()
                expected = None if expected_grid_counts is None else expected_grid_counts.get(source)
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
                        "grids_with_failed_timesteps": int(result["grids_with_failed_timesteps"] or 0),
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
    all_runs_complete = bool(
        len(powerflow_status) == 6 and powerflow_status["complete"].fillna(False).all()
    )
    checks.append(
        {
            "check": "All six compact power-flow runs complete",
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
            "passed": len(signatures) == 1 and len(powerflow_status) == 6,
            "detail": ", ".join(sorted(signatures)) or "No compact summaries",
        }
    )
    scenario_labels = {
        str(value)
        for value in powerflow_status["scenario_labels"].dropna()
        if str(value).strip()
    }
    checks.append(
        {
            "check": "Single scenario label",
            "passed": scenario_labels == {scenario_prefix},
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
            lambda row: bool(row["available"])
            and str(row["run_name"])
            == expected_runs.get((str(row["data_source"]), str(row["stage_label"]))),
            axis=1,
        )
        rows["grid_count_matches"] = rows.apply(
            lambda row: (
                int(row["grids_total"]) > 0
                if expected_grid_counts is None
                else int(row["grids_total"])
                == int(expected_grid_counts.get(str(row["data_source"]), row["grids_total"]))
            ),
            axis=1,
        )
        expansion_complete = bool(
            len(rows) == 6 and rows["run_matches"].all() and rows["grid_count_matches"].all()
        )
    else:
        expansion_complete = False
    checks.append(
        {
            "check": "Six matching synthetic/real expansion materializations",
            "passed": expansion_complete,
            "detail": f"{int(rows['available'].fillna(False).sum()) if not rows.empty else 0}/6 available",
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
    ags: str | int,
    expected_grid_counts: Mapping[str, int] | None = None,
    stage_labels: Mapping[str, str] | None = None,
    default_stage: str = "post_flex",
    real_plz: int | None = None,
) -> dict[str, object]:
    """Prepare one coherent synthetic/real scenario for the analysis notebook."""
    labels = dict(stage_labels or DEFAULT_STAGE_LABELS)
    if real_plz is None and normalize_ags_string(ags) == "09474126":
        real_plz = 91301
    specs_by_source = scenario_powerflow_specs(scenario_prefix, labels)
    default_label = labels[default_stage]
    analysis_keys_by_source = {
        source: scenario_analysis_keys(scenario_prefix, labels, data_source=source)
        for source in ("Synthetic", "Real SWF")
    }
    expansion_context_by_source = {
        source: load_expansion_stage_context(
            keys,
            default_analysis_label=default_label,
            allow_empty=True,
        )
        for source, keys in analysis_keys_by_source.items()
    }
    expansion_status = pd.concat(
        [context["analysis_status"] for context in expansion_context_by_source.values()],
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
    synthetic_context = expansion_context_by_source["Synthetic"]
    return {
        "scenario_prefix": scenario_prefix,
        "display_label": display_label_from_ags(ags),
        "stage_labels": labels,
        "analysis_keys": analysis_keys_by_source["Synthetic"],
        "analysis_keys_by_source": analysis_keys_by_source,
        "synthetic_specs": specs_by_source["Synthetic"],
        "real_specs": specs_by_source["Real SWF"],
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
        **{key: value for key, value in synthetic_context.items() if key != "analysis_status"},
    }


def load_cable_loading_decomposition(
    *,
    synthetic_specs: Mapping[str, Mapping[str, object]],
    real_specs: Mapping[str, Mapping[str, object]],
    ags: str | int,
    real_plz: int | None = None,
    excluded_real_lv_ids: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Load one annual-maximum row per analyzed cable for both networks."""
    if real_plz is None and normalize_ags_string(ags) == "09474126":
        real_plz = 91301
    db = SurroGridDatabase()
    synthetic_query = text(
        """
        SELECT CONCAT(gc.plz, '-', gc.kcid, '-', gc.bcid) AS grid,
               pcs.cable AS asset_id,
               pcs.cable_installed_capacity_ka,
               pcs.cable_loading_max_time_percent
        FROM surrogrid.powerflow_cable_summary pcs
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pcs.stage = :stage
          AND gc.ags = :ags
        """
    )
    real_query = text(
        """
        SELECT CONCAT('LV_', LPAD(rgc.lv_id::TEXT, 3, '0')) AS grid,
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
        """
    )

    frames = []
    excluded = {int(value) for value in excluded_real_lv_ids}
    with db.engine.connect() as conn:
        for source, specs in (("Synthetic", synthetic_specs), ("Real SWF", real_specs)):
            for stage_label, spec in specs.items():
                params = {"run_name": str(spec["run_name"]), "stage": str(spec["stage"])}
                if source == "Synthetic":
                    params["ags"] = int(normalize_ags_string(ags))
                    frame = pd.read_sql_query(synthetic_query, conn, params=params)
                else:
                    params["plz"] = real_plz
                    frame = pd.read_sql_query(real_query, conn, params=params)
                    if excluded and not frame.empty:
                        frame = frame[~frame["lv_id"].astype(int).isin(excluded)].copy()
                if frame.empty:
                    continue
                frame["data_source"] = source
                frame["comparison_stage"] = stage_label
                frame["installed_capacity_a"] = (
                    pd.to_numeric(frame["cable_installed_capacity_ka"], errors="coerce") * 1000.0
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
    excluded_real_lv_ids: tuple[int, ...] = (),
) -> dict[str, Path]:
    """Export scenario identity and readiness tables alongside notebook figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=GRIDEXPAND_DIR.parent,
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
        "excluded_real_lv_ids": [int(value) for value in excluded_real_lv_ids],
        "publication_checks": publication_gate.to_dict(orient="records"),
    }
    manifest_path = output_dir / "scenario_analysis_manifest.json"
    status_path = output_dir / "powerflow_readiness.csv"
    gate_path = output_dir / "publication_gate.csv"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    powerflow_status.to_csv(status_path, index=False)
    publication_gate.to_csv(gate_path, index=False)
    return {
        "manifest": manifest_path,
        "powerflow_readiness": status_path,
        "publication_gate": gate_path,
    }
