"""Per-building-type adoption shares and the battery share of the PV buildings."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.common.electrification import (
    STRATUM_COLUMN,
    assignment_manifest_hash,
    assignment_summary,
    battery_buildings,
    build_electrification_assignment,
    validate_electrification_assignment,
    validate_electrification_assignment_config,
)
from gridexpand.scenario.scenario_config import TechnologyAdoptionConfig

TYPES = ["SFH"] * 20 + ["MFH"] * 10 + [None] * 4


def _inventory():
    frame = pd.DataFrame({"building_objectid": [f"B{i:02d}" for i in range(len(TYPES))], "building_type": TYPES})
    for technology in ("heat", "mobility", "pv_battery"):
        frame[f"{technology}_eligible"] = True
        frame[f"{technology}_exclusion_reason"] = None
    return frame


def _adoption(**mobility):
    shared = {"adoption_mode": "deterministic_share", "building_share": 0.5}
    return {"heat": shared, "pv_battery": shared, "mobility": {**shared, **mobility}}


def test_without_per_type_shares_the_manifest_keeps_its_columns():
    assignment = build_electrification_assignment(
        _inventory(), _adoption(), selection_scope_id="s", profile_seed=3
    )
    assert STRATUM_COLUMN not in assignment.columns
    assert assignment.loc[assignment["technology"].eq("mobility"), "selected"].sum() == 17


def test_per_type_shares_select_within_each_type():
    adoption = _adoption(building_share_by_type={"SFH": 0.85, "TH": 0.85, "MFH": 0.5})
    assignment = build_electrification_assignment(
        _inventory(), adoption, selection_scope_id="s", profile_seed=3
    )
    mobility = assignment.loc[assignment["technology"].eq("mobility")]
    selected = mobility.groupby(STRATUM_COLUMN)["selected"].sum().to_dict()
    assert selected == {"SFH": 17, "MFH": 5, "other": 2}
    shares = mobility.groupby(STRATUM_COLUMN)["configured_share"].unique().map(list).to_dict()
    assert shares == {"SFH": [0.85], "MFH": [0.5], "other": [0.5]}
    # Other technologies keep one stratum.
    assert set(assignment.loc[assignment["technology"].eq("heat"), STRATUM_COLUMN]) == {"other"}
    validate_electrification_assignment(assignment)
    validate_electrification_assignment_config(assignment, adoption, profile_seed=3)
    assert assignment_summary(assignment).set_index("technology").loc["mobility", "configured_share"] != 0.5
    # The stratum is part of the hash and survives a CSV round trip.
    assert assignment_manifest_hash(assignment) == assignment_manifest_hash(assignment.copy())


def test_a_wrong_count_in_one_stratum_is_rejected():
    adoption = _adoption(building_share_by_type={"SFH": 0.85, "MFH": 0.5})
    assignment = build_electrification_assignment(
        _inventory(), adoption, selection_scope_id="s", profile_seed=3
    )
    rows = assignment.index[
        assignment["technology"].eq("mobility") & assignment[STRATUM_COLUMN].eq("MFH") & ~assignment["selected"]
    ]
    tampered = assignment.copy()
    tampered.loc[rows[0], "selected"] = True
    with pytest.raises(ValueError, match="selected 6 buildings; expected 5"):
        validate_electrification_assignment(tampered)


def test_config_validation_rejects_a_changed_type_share():
    built = _adoption(building_share_by_type={"SFH": 0.85, "MFH": 0.5})
    assignment = build_electrification_assignment(_inventory(), built, selection_scope_id="s", profile_seed=3)
    with pytest.raises(ValueError, match="does not"):
        validate_electrification_assignment_config(
            assignment, _adoption(building_share_by_type={"SFH": 0.9, "MFH": 0.5})
        )
    with pytest.raises(ValueError, match="strata the scenario does not define"):
        validate_electrification_assignment_config(assignment, _adoption())


def test_per_type_shares_need_the_building_type():
    with pytest.raises(ValueError, match="building_type column"):
        build_electrification_assignment(
            _inventory().drop(columns="building_type"),
            _adoption(building_share_by_type={"SFH": 0.85}),
            selection_scope_id="s", profile_seed=3,
        )


def test_battery_buildings_are_a_seeded_share_of_the_pv_buildings():
    assignment = build_electrification_assignment(
        _inventory(), _adoption(), selection_scope_id="s", profile_seed=3
    )
    pv = set(assignment.loc[assignment["technology"].eq("pv_battery") & assignment["selected"], "building_objectid"])
    assert battery_buildings(assignment, 1.0) == pv
    subset = battery_buildings(assignment, 0.8)
    assert subset <= pv and len(subset) == round(0.8 * len(pv))
    assert subset == battery_buildings(assignment, 0.8)
    with pytest.raises(ValueError):
        battery_buildings(assignment, 1.2)


def test_adoption_config_parses_type_shares_and_battery_share():
    config = TechnologyAdoptionConfig.from_dict(
        {"adoption_mode": "deterministic_share", "building_share": 0.5,
         "building_share_by_type": {"SFH": 0.85, "MFH": 0.5}},
        "electrification.mobility",
    )
    assert config.share_for_type("SFH") == 0.85 and config.share_for_type(None) == 0.5
    pv = TechnologyAdoptionConfig.from_dict(
        {"adoption_mode": "deterministic_share", "building_share": 0.7, "battery_share_of_selected": 0.8},
        "electrification.pv_battery",
    )
    assert pv.battery_share_of_selected == 0.8
    with pytest.raises(ValueError, match="Unknown"):
        TechnologyAdoptionConfig.from_dict(
            {"adoption_mode": "deterministic_share", "building_share": 0.5, "battery_share_of_selected": 0.8},
            "electrification.mobility",
        )
    with pytest.raises(ValueError, match="building_share_by_type"):
        TechnologyAdoptionConfig.from_dict(
            {"adoption_mode": "deterministic_share", "building_share": 0.5, "building_share_by_type": {"Villa": 1}},
            "electrification.mobility",
        )
