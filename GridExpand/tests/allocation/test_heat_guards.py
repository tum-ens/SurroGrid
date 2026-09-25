"""Guards that keep a selected heat building from silently getting 0 kW (alloc B8)."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.allocation.functions.heat import require_heat_profiles


def _profiles(buses, commodity):
    frame = pd.DataFrame({bus: [1.0, 2.0] for bus in buses})
    frame.columns = pd.MultiIndex.from_product([frame.columns, [commodity]])
    return frame


def test_require_heat_profiles_accepts_complete_coverage():
    require_heat_profiles(
        pd.Series([3, 5, 5]), _profiles([3, 5], "space_heat"), _profiles([3, 5], "water_heat")
    )


def test_require_heat_profiles_names_missing_buses():
    with pytest.raises(ValueError, match=r"\[5\]"):
        require_heat_profiles(
            pd.Series([3, 5]), _profiles([3], "space_heat"), _profiles([3, 5], "water_heat")
        )


def test_districtgenerator_no_longer_skips_invalid_buildings():
    from gridexpand.allocation.external.districtgenerator.classes import Datahandler

    duplicated = pd.DataFrame({"id": [0, 0]}, index=[0, 0])
    handler = Datahandler(duplicated)
    with pytest.raises(ValueError, match="not unique"):
        handler.initializeBuildings()

    handler = Datahandler(pd.DataFrame({"id": ["x"]}, index=["x"]))
    with pytest.raises(ValueError, match="not a number"):
        handler.initializeBuildings()
