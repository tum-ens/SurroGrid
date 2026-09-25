"""common.io: input-file resolution and HDF5 key lookup (moved from Steps 3/4)."""

from __future__ import annotations

import h5py
import pytest

from gridexpand.common import io
from gridexpand.optimization import identity
from gridexpand.powerflow import io as powerflow_io


def test_resolve_input_file(tmp_path):
    for name in ("9184137-00_a_pre.h5", "9184137-00_b_pre.h5", "9184137-01_b_pre.h5"):
        (tmp_path / name).write_bytes(b"")
    assert io.resolve_input_file(tmp_path, "9184137-01").name == "9184137-01_b_pre.h5"
    assert io.resolve_input_file(tmp_path, "9184137-00_a_pre.h5").name == "9184137-00_a_pre.h5"
    assert io.resolve_input_file(tmp_path / "x", tmp_path / "9184137-01_b_pre.h5").parent == tmp_path
    with pytest.raises(ValueError):
        io.resolve_input_file(tmp_path, "9184137-00")
    with pytest.raises(FileNotFoundError):
        io.resolve_input_file(tmp_path, "missing.h5")


def test_key_exists(tmp_path):
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("urbs_in/demand")
    assert io.key_exists(path, "/urbs_in/demand") and io.key_exists(path, "urbs_in")
    assert not io.key_exists(path, "raw_data/net")


def test_old_import_locations_still_work():
    assert identity.resolve_input_file is io.resolve_input_file
    assert powerflow_io.key_exists is io.key_exists
