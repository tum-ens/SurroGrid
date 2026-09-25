"""Occupancy sampling of residential components (no database, no data files)."""

from __future__ import annotations

import random

import gridexpand.allocation.functions.electricity as electricity

PROB = {1: 0.4, 2: 0.3, 3: 0.15, 4: 0.1, 5.394: 0.05}


def test_occupancy_sums_to_building_occupants():
    sizes = electricity._get_occupancy_distribution(PROB, 4, 9, rng=random.Random(7))
    assert len(sizes) == 4
    assert abs(sum(sizes) - 9) < 1.0


def test_occupancy_bounds_use_min_and_max_household_size():
    assert electricity._get_occupancy_distribution(PROB, 2, 20) == [5.394, 5.394]
    assert electricity._get_occupancy_distribution(PROB, 3, 1) == [1, 1, 1]


def test_occupancy_missing_inputs_give_no_households():
    assert electricity._get_occupancy_distribution(PROB, float("nan"), 3) == []


def test_occupancy_fallback_uses_closest_allowed_size(monkeypatch):
    """B5: the fallback swapped its arguments and raised TypeError."""

    def no_continuation(*args, **kwargs):
        raise ValueError("No valid continuation found; consider increasing tol.")

    monkeypatch.setattr(electricity, "_sample_sequence_with_tolerance", no_continuation)
    assert electricity._get_occupancy_distribution(PROB, 3, 7) == [2, 2, 2]


def test_occupancy_fallback_does_not_swallow_other_errors(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(electricity, "_sample_sequence_with_tolerance", interrupted)
    try:
        electricity._get_occupancy_distribution(PROB, 3, 7)
    except KeyboardInterrupt:
        return
    raise AssertionError("KeyboardInterrupt must propagate")
