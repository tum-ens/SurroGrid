"""Input selection and Step 2 identity checks of Step 3 (pure, no solver)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gridexpand.common.electrification import (
    assignment_manifest_hash,
    validate_electrification_assignment_config,
)
from gridexpand.common.timeframe import scenario_key_for_timeframe
from gridexpand.scenario.config_loader import scenario_identity_key


def resolve_input_file(directory: Path | str, file_id: str | Path) -> Path:
    """Return the input HDF5 file named by ``file_id``.

    ``file_id`` is a path to an existing file, an exact file name in
    ``directory`` or the unique id prefix before the first underscore of one
    ``.h5`` file there (e.g. ``9184137-03``).

    Raises:
        FileNotFoundError: nothing matches.
        ValueError: the prefix matches several files.
    """
    directory = Path(directory)
    candidate = Path(file_id)
    if candidate.is_file() and (candidate.is_absolute() or len(candidate.parts) > 1):
        return candidate
    name = str(file_id)
    if name.endswith(".h5"):
        path = directory / name
        if path.is_file():
            return path
        raise FileNotFoundError(f"No input file {name} in {directory}.")
    matches = sorted(
        path for path in directory.glob("*.h5") if path.name.split("_", 1)[0] == name
    )
    if not matches:
        raise FileNotFoundError(f"No input file matches {name} in {directory}.")
    if len(matches) > 1:
        raise ValueError(
            f"Input id {name} is ambiguous in {directory}: "
            f"{[path.name for path in matches]}; pass the file name."
        )
    return matches[0]


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
