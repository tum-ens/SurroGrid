"""Identifiers and labels shared by the analysis loaders, plots and expansion tools."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from gridexpand.db.grids import normalize_ags

__all__ = [
    "ags_string",
    "canonical_real_grid_id",
    "normalize_ags",
    "optional_ags",
    "real_grid_label",
    "synthetic_grid_label",
]

_REAL_ID_PREFIXES = ("LV_", "area-", "area_")


def ags_string(value: str | int) -> str:
    """AGS as the eight-character string with leading zero (``9184137`` -> ``09184137``)."""
    return str(normalize_ags(value)).zfill(8)


def optional_ags(value: str | int | None) -> int | None:
    """``normalize_ags`` that passes None through."""
    return None if value is None else normalize_ags(value)


def canonical_real_grid_id(value: Any) -> str:
    """A real grid id (SWF LV or ÜZW area) as text without prefix or zero padding.

    ``"LV_007"``, ``"007"``, ``7`` and ``7.0`` give ``"7"``; ``"area-12"`` gives
    ``"12"``; other text ids are returned stripped.
    """
    text_value = str(value).strip()
    for prefix in _REAL_ID_PREFIXES:
        text_value = text_value.removeprefix(prefix)
    if text_value.isdigit():
        return str(int(text_value))
    try:
        number = float(text_value)
    except ValueError:
        return text_value
    return str(int(number)) if math.isfinite(number) and number.is_integer() else text_value


def synthetic_grid_label(row: Mapping[str, Any]) -> str:
    """``<AGS>-<plz>_<kcid>_<bcid>`` of a synthetic grid (row with ags, plz, kcid, bcid)."""
    return f"{ags_string(row['ags'])}-{int(row['plz'])}_{int(row['kcid'])}_{int(row['bcid'])}"


def real_grid_label(source: str, lv_id: Any) -> str:
    """``ÜZW area-0012`` or ``SWF LV_007`` of a real grid."""
    number = int(canonical_real_grid_id(lv_id))
    if source == "uzw":
        return f"ÜZW area-{number:04d}"
    return f"{str(source).upper()} LV_{number:03d}"
