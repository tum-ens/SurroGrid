"""Shared setup of the joint SWF + ÜZW analysis notebooks (``analysis_powerflow``, ``analysis_expansion``).

One run and two exclusion rules for both notebooks, applied to every case:
- a grid whose power flow did not converge in at least ``MIN_FAILED_SHARE`` of the
  hours of any case is left out of its own group (grid level: its real or synthetic
  counterparts stay). Non-converged hours are missing from the summaries, so such a
  grid would be compared without its worst hours;
- a synthetic grid with fewer than ``MIN_SYNTHETIC_BUILDINGS`` buildings (a fragment
  of pylovo's clustering) is left out, as the reference has no grids that small. The
  rule reads the building count of each synthetic grid of the run, so it follows
  whatever grids a pylovo version generates.
The paper figures pool the providers as ``Real`` and ``Synthetic`` so that no DSO is named.
"""

from __future__ import annotations

import pandas as pd
import yaml

from gridexpand.analysis.expansion.notebook_workflow import (
    PROVIDER_LABELS,
    excluded_grids_by_group,
    nonconverged_grids,
    prepare_expansion_analysis,
)
from gridexpand.paths import ANALYSIS_OUTPUT_DIR, RUN_CONFIG_DIR, SCENARIO_CALIBRATION_OUTPUT_DIR

RUN_ID = "joint_2045_internal_2k_a3"
PROVIDERS = ("swf", "uzw")
EXPECTED_GRID_COUNTS = {"Real SWF": 44, "Synthetic SWF": 45, "Real ÜZW": 213, "Synthetic ÜZW": 203}
MIN_FAILED_SHARE = 0.01  # 1 % of the hours of one case (88 h of 8,760)
MIN_SYNTHETIC_BUILDINGS = 5  # smaller synthetic grids (clustering fragments) are left out, as in the reference

STAGE_LABELS = {"pre": "status-quo", "post_inflex": "INFLEX", "post_flex": "HEMS"}
CASE_COLORS = {"status-quo": "#2E7D32", "INFLEX": "#D62728", "HEMS": "#1F77B4"}
CASE_DISPLAY = {"status-quo": "Status quo", "INFLEX": "INFLEX", "HEMS": "HEMS"}
NETWORK_COLORS = {"Synthetic": "#335C81", "Real": "#D95D39"}

# Top row of both paper figures (status quo, and status quo + INFLEX + HEMS): one wide, fixed scale, so that
# narrow limits do not magnify the synthetic-vs-real gaps. Loading starts at 0; the upper bounds leave headroom
# above the largest curve of either figure (P99 cutoff, 2026-09-30: transformer 130 %, cables 42 %). Voltage runs
# from the 0.90 p.u. lower limit to nominal (curves 0.92-0.96 p.u.).
CURVE_Y_LIMITS = {"Transformer": (0.0, 150.0), "Cables": (0.0, 60.0), "Voltage": (0.90, 1.00)}

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "plots" / RUN_ID


def small_synthetic_grids(run_id: str = RUN_ID, min_buildings: int = MIN_SYNTHETIC_BUILDINGS) -> pd.DataFrame:
    """Synthetic grids of the run with fewer than ``min_buildings`` buildings.

    Reads ``paired_registered_synthetic_grids.csv`` of each provider's paired dataset (named in the run YAML) and
    returns ``data_source``, the ``PLZ_kcid_bcid`` ``grid_key`` of the loaders and ``n_buildings``.
    """
    run = yaml.safe_load((RUN_CONFIG_DIR / f"{run_id}.yaml").read_text(encoding="utf-8"))
    frames = []
    for provider in PROVIDERS:
        dataset_id = run["resources"]["providers"][provider]["paired_dataset_id"]
        grids = pd.read_csv(SCENARIO_CALIBRATION_OUTPUT_DIR / dataset_id / "paired_registered_synthetic_grids.csv")
        small = grids[grids["n_buildings"] < min_buildings]
        keys = small["plz"].astype(str) + "_" + small["kcid"].astype(str) + "_" + small["bcid"].astype(str)
        frames.append(pd.DataFrame({
            "data_source": f"Synthetic {PROVIDER_LABELS[provider]}",
            "grid_key": keys.to_numpy(),
            "n_buildings": small["n_buildings"].to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def prepare_joint_analysis(
    run_id: str = RUN_ID,
    min_failed_share: float = MIN_FAILED_SHARE,
    min_buildings: int = MIN_SYNTHETIC_BUILDINGS,
) -> dict:
    """``prepare_expansion_analysis`` of the joint run plus ``nonconverged``, ``small_synthetic_grids`` and
    ``excluded_grids`` (both rules of the module docstring)."""
    context = prepare_expansion_analysis(
        scenario_prefix=run_id,
        providers=PROVIDERS,
        stage_labels=STAGE_LABELS,
        expected_grid_counts=EXPECTED_GRID_COUNTS,
    )
    nonconverged = nonconverged_grids(context["specs_by_source"], min_failed_share=min_failed_share)
    small = small_synthetic_grids(run_id, min_buildings)
    excluded = excluded_grids_by_group(nonconverged)
    for source, rows in small.groupby("data_source"):
        excluded[source] = tuple(sorted(set(excluded.get(source, ())) | set(rows["grid_key"])))
    return {**context, "nonconverged": nonconverged, "small_synthetic_grids": small, "excluded_grids": excluded}


def stage_specs(context: dict, *labels: str) -> dict:
    """``specs_by_source`` of the context restricted to the given case labels."""
    return {
        source: {label: specs[label] for label in labels if label in specs}
        for source, specs in context["specs_by_source"].items()
    }
