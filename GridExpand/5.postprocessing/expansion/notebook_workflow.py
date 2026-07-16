"""Reusable preparation helpers for the expansion analysis notebook."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase
from expansion import grid_expansion
from powerflow.comparison_data import (
    load_synthetic_powerflow_cutoff_profile,
    real_powerflow_headline_summary_db,
    real_powerflow_percentile_profile_db,
)
from plotting.powerflow_transformer import transformer_import_distribution_db
from plotting.powerflow_voltage import voltage_deviation_summary_db


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
            "run_name": None,
            "stage": None,
            "grids_with_expansion_summary": 0,
            "total_cost_eur": pd.NA,
        }
        if is_available:
            meta = analysis_run.iloc[0]
            analysis_meta_by_stage[label] = meta
            status_row.update(
                {
                    "run_name": meta["run_name"],
                    "stage": meta["stage"],
                    "grids_with_expansion_summary": tables["grid_cost_summary"]["grid_case_id"].nunique()
                    if not tables["grid_cost_summary"].empty
                    else 0,
                    "total_cost_eur": float(tables["cost_summary"].iloc[0]["total_cost_eur"])
                    if not tables["cost_summary"].empty
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
        raise ValueError("None of the configured analysis keys exist in surrogrid.expansion_analysis_run.")

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


def expansion_cost_reduction_summary(
    expansion_cost_comparison: pd.DataFrame,
    *,
    post_no_flex_label: str,
    post_flex_label: str,
) -> pd.DataFrame:
    """Calculate cost savings of flex compared with no-flex by component and total."""
    if expansion_cost_comparison.empty:
        return pd.DataFrame()

    cost_data = expansion_cost_comparison.copy()
    if "data_source" in cost_data.columns:
        cost_data = cost_data[cost_data["data_source"].astype(str) == "Synthetic"].copy()
    cost_wide = cost_data.pivot_table(
        index="component",
        columns="stage",
        values="cost_eur",
        aggfunc="sum",
        fill_value=0.0,
    )
    cost_wide.loc["Total"] = cost_wide.sum(axis=0)
    required_columns = {post_no_flex_label, post_flex_label}
    if required_columns.difference(cost_wide.columns):
        return pd.DataFrame()

    cost_reduction_summary = pd.DataFrame(
        {
            "component": cost_wide.index,
            "no_flex_cost_million_eur": cost_wide[post_no_flex_label].to_numpy() / 1_000_000.0,
            "flex_cost_million_eur": cost_wide[post_flex_label].to_numpy() / 1_000_000.0,
            "saving_million_eur": (
                cost_wide[post_no_flex_label] - cost_wide[post_flex_label]
            ).to_numpy()
            / 1_000_000.0,
            "reduction_percent": (
                (cost_wide[post_no_flex_label] - cost_wide[post_flex_label])
                / cost_wide[post_no_flex_label].replace(0, pd.NA)
                * 100.0
            ).to_numpy(),
        }
    )
    return cost_reduction_summary.round(
        {
            "no_flex_cost_million_eur": 2,
            "flex_cost_million_eur": 2,
            "saving_million_eur": 2,
            "reduction_percent": 1,
        }
    )


def load_powerflow_cutoff_comparison(
    *,
    synthetic_specs: Mapping[str, Mapping[str, object]],
    real_specs: Mapping[str, Mapping[str, object]],
    stage_order: list[str],
    ags: str | int,
    scenario_id: int | None = None,
    plz: int | None = None,
    real_plz: int | None = None,
) -> dict[str, object]:
    """Load compact synthetic and real power-flow summaries for one comparison plot."""
    powerflow_profiles = []
    skipped = {"Synthetic": {}, "Real SWF": {}}

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
    synthetic_analysis_meta_by_stage: Mapping[str, pd.Series],
    real_specs: Mapping[str, Mapping[str, object]],
    real_plz: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load source-separated voltage summaries for synthetic and real comparisons.

    Real SWF compact summaries currently store low-voltage extrema but not
    high-voltage extrema. The returned real frames therefore contain
    ``max_vm_pu`` as NA so the voltage histogram only draws the lower tail for
    real data.
    """
    synthetic = load_voltage_summaries_for_analysis(synthetic_analysis_meta_by_stage)
    real: dict[str, pd.DataFrame] = {}
    for label, spec in real_specs.items():
        try:
            summary = real_powerflow_headline_summary_db(
                run_name=str(spec["run_name"]),
                stage=str(spec["stage"]),
                plz=real_plz,
            )
        except ValueError:
            continue
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
    result = {"Synthetic": synthetic}
    if real:
        result["Real SWF"] = real
    return result

def load_transformer_import_distributions_for_analysis(
    analysis_meta_by_stage: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Load transformer import distributions for each available synthetic stage."""
    transformer_distributions = []
    for label, meta in analysis_meta_by_stage.items():
        filters = meta_filter(meta)
        distribution = transformer_import_distribution_db(
            run_name=filters["run_name"],
            stage=filters["stage"],
            scenario_id=filters["scenario_id"],
            ags=filters["ags"],
            plz=filters["plz"],
        )
        distribution["comparison_stage"] = label
        transformer_distributions.append(distribution)
    if not transformer_distributions:
        return pd.DataFrame()
    return pd.concat(transformer_distributions, ignore_index=True)
