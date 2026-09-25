"""The four model cases and how they are grouped into shared solves.

A model case fixes how assets are sized (``asset_plan``), how the post
power flow reconstructs demand (``powerflow_mode``) and how its results are
named. This module is the only place that encodes those rules; the synthetic
runner, the paired runner, the real-grid adapters, the scenario config and
``gridexpand run`` read them from here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

ASSET_PLANS = ("none", "heuristic", "optimization")


@dataclass(frozen=True)
class ModelCase:
    """One model case.

    Attributes:
        name: Public case name (CLI choices, run names, YAML ``model_cases``).
        asset_plan: ``none`` (status quo), ``heuristic`` (rule-sized assets) or
            ``optimization`` (Step 3 sizes the assets).
        powerflow_mode: Step 4 ``--post-demand-mode`` of the post stage
            (``flexible`` or ``inflex``); None for ``pre`` (pre stage only).
        analysis_suffix: Suffix of paired expansion analysis keys.
        profiles: Step 2 ``--profiles`` value (``status_quo`` or ``all``).
        label: Human-readable label of real-grid power-flow runs.
    """

    name: str
    asset_plan: str
    powerflow_mode: str | None
    analysis_suffix: str
    profiles: str
    label: str

    @property
    def stage(self) -> str:
        """Power-flow stage that carries this case's result (``pre`` or ``post``)."""
        return "pre" if self.powerflow_mode is None else "post"

    @property
    def real_powerflow_mode(self) -> str:
        """``--post-demand-mode`` of the real-grid power flow (``pre-only`` for pre)."""
        return self.powerflow_mode or "pre-only"


# Order matters: POST_MODEL_CASES are argparse choices in this order.
MODEL_CASES: dict[str, ModelCase] = {
    case.name: case
    for case in (
        ModelCase("pre", "none", None, "pre", "status_quo", "pre electricity-only"),
        ModelCase(
            "post-inflex-heuristic", "heuristic", "inflex", "post_inflex", "all",
            "heuristic-assets INFLEX",
        ),
        ModelCase(
            "post-hems-optimized", "optimization", "flexible", "post_hems_optimized", "all",
            "optimized HEMS",
        ),
        ModelCase(
            "post-hems-heuristic", "heuristic", "flexible", "post", "all",
            "heuristic-assets HEMS",
        ),
    )
}

POST_MODEL_CASES = tuple(name for name in MODEL_CASES if name != "pre")
# Heuristic cases share one asset plan; this fixed order is their result order.
HEURISTIC_CASES = ("post-inflex-heuristic", "post-hems-heuristic")
# review-optpf B3: the synthetic Step 2 writes no urbs_in/ev_sessions, which the
# INFLEX power flow needs; the methodological fix is an open question.
SYNTHETIC_UNSUPPORTED_CASES = {
    "post-inflex-heuristic": (
        "The synthetic INFLEX power flow cannot run: the synthetic Step 2 writes no "
        "urbs_in/ev_sessions (INFLEX needs the EV sessions of the paired pipeline)."
    ),
}
# The case whose Step 2 materialization and Step 3 solve serve an asset plan.
MATERIALIZATION_CASE = {
    "none": "pre",
    "heuristic": "post-hems-heuristic",
    "optimization": "post-hems-optimized",
}


def get_model_case(name: str) -> ModelCase:
    """Return the model case ``name``.

    Raises:
        ValueError: unknown name.
    """
    try:
        return MODEL_CASES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model case {name!r}.") from exc


def compatible_result_cases(materialization_case: str) -> tuple[str, ...]:
    """Post cases that can be emitted from the asset plan of ``materialization_case``."""
    plan = get_model_case(materialization_case).asset_plan
    return tuple(name for name in POST_MODEL_CASES if MODEL_CASES[name].asset_plan == plan)


@dataclass(frozen=True)
class ExecutionGroup:
    """Model cases that share one asset plan, i.e. one Step 2 + Step 3 solve.

    Attributes:
        name: Run sub-directory of the group (paired runs keep the historical
            ``heuristic-assets`` / ``post-hems-optimized`` names).
        materialization_case: Case that is materialized and solved.
        result_cases: Cases whose power flows come from that solve, in order.
        emits_pre: True for the first group only: it also emits the shared pre
            stage (paired runners get ``--skip-pre`` otherwise).
    """

    name: str
    materialization_case: str
    result_cases: tuple[str, ...]
    emits_pre: bool


def validate_cases(cases: Iterable[str], *, allowed: Iterable[str] = MODEL_CASES) -> tuple[str, ...]:
    """Return ``cases`` as a tuple after checking names, emptiness and duplicates.

    Raises:
        ValueError: empty, unknown, not allowed or duplicated cases.
    """
    cases = tuple(str(case) for case in cases)
    if not cases:
        raise ValueError("model_cases cannot be empty.")
    unknown = sorted(set(cases).difference(MODEL_CASES))
    if unknown:
        raise ValueError(f"Unknown model cases: {unknown}")
    not_allowed = sorted(set(cases).difference(allowed))
    if not_allowed:
        raise ValueError(f"Model cases not allowed here: {not_allowed}")
    if len(cases) != len(set(cases)):
        raise ValueError("model_cases contains duplicates.")
    return cases


def execution_groups(cases: Iterable[str]) -> list[ExecutionGroup]:
    """Group requested cases into shared solves.

    ``pre`` (synthetic runs only) is its own group; the heuristic cases share
    one ``post-hems-heuristic`` materialization; the optimized case gets its
    own. Only the first group emits the pre stage.
    """
    cases = validate_cases(cases)
    groups: list[ExecutionGroup] = []
    if "pre" in cases:
        groups.append(ExecutionGroup("pre", "pre", ("pre",), emits_pre=True))
    heuristic = tuple(case for case in HEURISTIC_CASES if case in cases)
    if heuristic:
        groups.append(
            ExecutionGroup("heuristic-assets", "post-hems-heuristic", heuristic, emits_pre=not groups)
        )
    if "post-hems-optimized" in cases:
        groups.append(
            ExecutionGroup(
                "post-hems-optimized", "post-hems-optimized", ("post-hems-optimized",),
                emits_pre=not groups,
            )
        )
    return groups
