"""Scenario identity and run settings of one urbs input."""

import pandas as pd


def read_scenario_name(global_settings, data=None):
    """Return the canonical scenario identity prepared by Step 2."""
    scenario_key = global_settings.get("scenario_key")
    if not scenario_key:
        raise ValueError(
            "URBS input is missing scenario_key; use the Step 2 scenario metadata."
        )
    return str(scenario_key)


def insert_scenario(data, global_settings):
    """Store the run settings as ``data['global_prop']`` (one row per setting)."""
    support_timeframes = data["demand"].index.get_level_values("support_timeframe").unique()
    if len(support_timeframes) != 1:
        raise ValueError(
            "Expected exactly one support timeframe in demand input, "
            f"found {list(support_timeframes)}."
        )
    support_timeframe = support_timeframes[0]
    index = pd.MultiIndex.from_tuples([(support_timeframe, prop) for prop in global_settings.keys()], names=["support_timeframe", "property"])
    # Create DataFrame
    df = pd.DataFrame({"value": list(global_settings.values())}, index=index)
    data["global_prop"]=df

    return data
