"""Deterministic random realizations shared by general scenario pipelines."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import random
from typing import Iterator

import numpy as np
import pandas as pd


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a deterministic seed independent of row and execution ordering."""
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def physical_building_id(row: pd.Series) -> str:
    """Return the topology-independent identifier used for stochastic sampling."""
    for column in ("building_objectid", "objectid", "building_match_id"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    raise ValueError(
        "Reproducible profile generation requires building_objectid, objectid, "
        "or building_match_id for every building."
    )


@contextmanager
def legacy_random_state(seed: int) -> Iterator[None]:
    """Temporarily seed and subsequently restore Python and NumPy global RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def realization_id(base_seed: int, building_ids: list[str]) -> str:
    """Identify one physical stochastic realization independently of model case."""
    payload = "|".join([str(int(base_seed)), *sorted(map(str, building_ids))])
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash a profile frame including its ordered labels and numeric values."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(list(frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()
