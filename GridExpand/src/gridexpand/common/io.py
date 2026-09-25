"""Small file helpers shared by the pipeline steps (input selection, HDF5 keys)."""

from __future__ import annotations

from pathlib import Path

import h5py


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


def key_exists(path: Path | str, key: str) -> bool:
    """True if ``key`` (with or without leading slash) exists in the HDF5 file."""
    with h5py.File(path, "r") as hdf_file:
        return str(key).strip("/") in hdf_file
