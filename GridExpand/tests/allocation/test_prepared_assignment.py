"""Loading a prepared regional electrification assignment (alloc E3)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from gridexpand.allocation.electrification import file_sha256, load_prepared_assignment
from gridexpand.common.electrification import (
    assignment_manifest_hash,
    assignment_summary,
    build_electrification_assignment,
)

ADOPTION = {
    technology: {"adoption_mode": "deterministic_share", "building_share": 0.5}
    for technology in ("heat", "mobility", "pv_battery")
}


def _write(tmp_path, *, with_file_hash=True):
    ids = [f"B{i}" for i in range(6)]
    inventory = pd.DataFrame({"building_objectid": ids})
    for technology in ("heat", "mobility", "pv_battery"):
        inventory[f"{technology}_eligible"] = [True, True, False, True, True, True]
        inventory[f"{technology}_exclusion_reason"] = [None, None, "no_residential_component", None, None, None]
    assignment = build_electrification_assignment(
        inventory, ADOPTION, selection_scope_id="region", profile_seed=7
    )
    path = tmp_path / "electrification_assignment.csv"
    assignment.to_csv(path, index=False)
    metadata = {
        "scenario_hash": "abc",
        "profile_seed": 7,
        "assignment_hash": assignment_manifest_hash(pd.read_csv(path)),
        "assignment_summary": assignment_summary(assignment).to_dict("records"),
    }
    if with_file_hash:
        metadata["assignment_file_sha256"] = file_sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path, metadata


@pytest.mark.parametrize("with_file_hash", [True, False])
def test_loads_the_rows_of_one_grid(tmp_path, with_file_hash):
    path, metadata = _write(tmp_path, with_file_hash=with_file_hash)
    rows, source_hash, summary = load_prepared_assignment(
        path, pd.Series(["B1", "B4"]), scenario_hash="abc", profile_seed=7
    )
    assert source_hash == metadata["assignment_hash"]
    assert summary == metadata["assignment_summary"]
    assert sorted(set(rows["building_objectid"])) == ["B1", "B4"]
    assert len(rows) == 6 and list(rows.index) == list(range(6))
    expected = pd.read_csv(path)
    expected["building_objectid"] = expected["building_objectid"].astype(str)
    expected = expected.loc[expected["building_objectid"].isin({"B1", "B4"})].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows, expected, check_exact=True)


@pytest.mark.parametrize("with_file_hash", [True, False])
def test_rejects_modified_content(tmp_path, with_file_hash):
    path, _ = _write(tmp_path, with_file_hash=with_file_hash)
    frame = pd.read_csv(path)
    frame.loc[frame["exclusion_reason"].notna(), "exclusion_reason"] = "no_household"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="sidecar"):
        load_prepared_assignment(path, ["B1"], scenario_hash="abc", profile_seed=7)


def test_rejects_other_scenario_seed_and_missing_buildings(tmp_path):
    path, _ = _write(tmp_path)
    with pytest.raises(ValueError, match="scenario_hash"):
        load_prepared_assignment(path, ["B1"], scenario_hash="other", profile_seed=7)
    with pytest.raises(ValueError, match="profile_seed"):
        load_prepared_assignment(path, ["B1"], scenario_hash="abc", profile_seed=8)
    with pytest.raises(ValueError, match="one row per"):
        load_prepared_assignment(path, ["B1", "B99"], scenario_hash="abc", profile_seed=7)


def test_requires_the_sidecar(tmp_path):
    path, _ = _write(tmp_path)
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="missing sidecar"):
        load_prepared_assignment(path, ["B1"], scenario_hash="abc", profile_seed=7)
