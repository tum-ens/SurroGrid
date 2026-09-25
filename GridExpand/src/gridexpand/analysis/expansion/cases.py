"""Model case -> power-flow stage and expansion analysis key suffix.

One table for the expansion entry points (``aligned_expansion``) and the notebook
helpers (``notebook_workflow``); ``gridexpand run`` (``scenario.commands``) uses
the same suffixes.
"""

from __future__ import annotations

# model case -> (power-flow stage, analysis key suffix)
CASE_STAGES: dict[str, tuple[str, str]] = {
    "pre": ("pre", "pre"),
    "post-inflex-heuristic": ("post", "post_inflex"),
    "post-hems-heuristic": ("post", "post"),
    "post-hems-optimized": ("post", "post_hems_optimized"),
}

# notebook stage-label key -> model case
LABEL_CASES: dict[str, str] = {
    "pre": "pre",
    "post_inflex": "post-inflex-heuristic",
    "post_flex": "post-hems-heuristic",
    "post_optimized": "post-hems-optimized",
}


def case_stage(model_case: str) -> str:
    """Power-flow stage (``pre``/``post``) whose summaries a model case is costed from."""
    return CASE_STAGES[model_case][0]


def analysis_suffix(model_case: str) -> str:
    """Suffix of the analysis key of a model case (``pre``, ``post_inflex``, ``post``, ``post_hems_optimized``)."""
    return CASE_STAGES[model_case][1]
