"""Input selection and Step 2 identity checks of Step 3 (pure, no solver)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gridexpand.common.electrification import (
    assignment_manifest_hash,
    validate_electrification_assignment_config,
)
from gridexpand.common.io import resolve_input_file  # noqa: F401  (re-export)
from gridexpand.common.timeframe import scenario_key_for_timeframe
from gridexpand.scenario.config_loader import scenario_identity_key


@dataclass(frozen=True)
class Step2Identity:
    """Identity of a Step 2 input as recorded in its ``metadata/timeframe``."""

    model_case: str | None
    scenario_key: str
    assignment_hash: str | None


def validate_step2_input(
    metadata: dict[str, Any],
    assignment: pd.DataFrame | None,
    scenario,
    scenario_hash: str,
) -> Step2Identity:
    """Check that a Step 2 input belongs to the active scenario configuration.

    Args:
        metadata: ``metadata/timeframe`` of the input.
        assignment: ``raw_data/electrification_assignment`` (None if absent).
        scenario: the loaded ScenarioConfig.
        scenario_hash: hash of the scenario YAML.

    Raises:
        ValueError: scenario hash, electrification assignment or scenario key do
            not match.
    """
    if metadata.get("scenario_hash") not in {None, scenario_hash}:
        raise ValueError(
            "Step-3 scenario YAML does not match the Step-2 input scenario_hash."
        )
    model_case = metadata.get("model_case")
    assignment_hash = metadata.get("electrification_assignment_hash")
    if model_case == "pre":
        # Status-quo inputs intentionally contain only base electricity and
        # therefore have no post-electrification assignment manifest.
        assignment_hash = None
    else:
        if not assignment_hash:
            raise ValueError(
                "Step-3 post-electrification input is missing "
                "electrification_assignment_hash."
            )
        if assignment is None:
            raise ValueError(
                "Step-3 post-electrification input is missing "
                "raw_data/electrification_assignment."
            )
        validate_electrification_assignment_config(
            assignment,
            scenario.electrification,
            profile_seed=metadata.get("profile_seed"),
            exact_share=False,
        )
        local_assignment_hash = assignment_manifest_hash(assignment, exact_share=False)
        expected_local_hash = metadata.get(
            "electrification_assignment_local_hash", assignment_hash
        )
        if local_assignment_hash != expected_local_hash:
            raise ValueError(
                "Step-2 assignment metadata does not match the stored "
                "electrification assignment."
            )
    scenario_key = metadata.get("scenario_key")
    if not scenario_key:
        raise ValueError(
            "Step-2 input is missing scenario_key metadata; "
            "rerun Step 2 with the active scenario configuration."
        )
    expected_scenario_key = scenario_key_for_timeframe(
        str(metadata.get("timeframe_mode", "full_year")),
        base_key=scenario_identity_key(scenario.scenario_id, scenario_hash),
    )
    if str(scenario_key) != expected_scenario_key:
        raise ValueError(
            "Step-2 scenario_key is not canonical for the active scenario "
            f"and timeframe: expected={expected_scenario_key}, got={scenario_key}"
        )
    return Step2Identity(model_case, str(scenario_key), assignment_hash)
