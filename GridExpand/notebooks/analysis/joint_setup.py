"""Shared setup of the joint SWF + ÜZW analysis notebooks (``analysis_powerflow``, ``analysis_expansion``).

One run and one exclusion rule for both notebooks. A grid whose power flow did not
converge in at least ``MIN_FAILED_SHARE`` of the hours of any case is left out of
every case of its own group (grid level: its real or synthetic counterparts stay).
Non-converged hours are missing from the summaries, so such a grid would be
compared without its worst hours. The paper figures pool the providers as ``Real``
and ``Synthetic`` so that no DSO is named.
"""

from __future__ import annotations

from gridexpand.analysis.expansion.notebook_workflow import (
    excluded_grids_by_group,
    nonconverged_grids,
    prepare_expansion_analysis,
)
from gridexpand.paths import ANALYSIS_OUTPUT_DIR

RUN_ID = "joint_2045_v1_full_year"
PROVIDERS = ("swf", "uzw")
EXPECTED_GRID_COUNTS = {"Real SWF": 49, "Synthetic SWF": 52, "Real ÜZW": 213, "Synthetic ÜZW": 230}
MIN_FAILED_SHARE = 0.01  # 1 % of the hours of one case (88 h of 8,760)

STAGE_LABELS = {"pre": "status-quo", "post_inflex": "INFLEX", "post_flex": "HEMS"}
CASE_COLORS = {"status-quo": "#2E7D32", "INFLEX": "#D62728", "HEMS": "#1F77B4"}
CASE_DISPLAY = {"status-quo": "Status quo", "INFLEX": "No HEMS", "HEMS": "HEMS"}
NETWORK_COLORS = {"Synthetic": "#335C81", "Real": "#D95D39"}

# Top row of both paper figures (status quo, and status quo + INFLEX + HEMS): one wide, fixed scale, so that
# narrow limits do not magnify the synthetic-vs-real gaps. Loading starts at 0; the upper bounds leave headroom
# above the largest curve of either figure (P99 cutoff, 2026-09-30: transformer 130 %, cables 42 %). Voltage runs
# from the 0.90 p.u. lower limit to nominal (curves 0.92-0.96 p.u.).
CURVE_Y_LIMITS = {"Transformer": (0.0, 150.0), "Cables": (0.0, 60.0), "Voltage": (0.90, 1.00)}

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "plots" / RUN_ID


def prepare_joint_analysis(run_id: str = RUN_ID, min_failed_share: float = MIN_FAILED_SHARE) -> dict:
    """``prepare_expansion_analysis`` of the joint run plus ``nonconverged`` and ``excluded_grids``."""
    context = prepare_expansion_analysis(
        scenario_prefix=run_id,
        providers=PROVIDERS,
        stage_labels=STAGE_LABELS,
        expected_grid_counts=EXPECTED_GRID_COUNTS,
    )
    nonconverged = nonconverged_grids(context["specs_by_source"], min_failed_share=min_failed_share)
    return {**context, "nonconverged": nonconverged, "excluded_grids": excluded_grids_by_group(nonconverged)}


def stage_specs(context: dict, *labels: str) -> dict:
    """``specs_by_source`` of the context restricted to the given case labels."""
    return {
        source: {label: specs[label] for label in labels if label in specs}
        for source, specs in context["specs_by_source"].items()
    }
