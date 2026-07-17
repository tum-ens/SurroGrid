"""Shared URBS input-table constructors for calibrated scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_or_create_weather(weather_source_hdf: Path | None, hours: int) -> pd.DataFrame:
    if weather_source_hdf is None:
        weather = pd.DataFrame(
            {
                ("ambient", "Tamb"): [0.0] * hours,
                ("ambient", "Irradiation"): [0.0] * hours,
            }
        )
        weather.columns = pd.MultiIndex.from_tuples(weather.columns)
    else:
        weather = pd.read_hdf(weather_source_hdf, key="urbs_in/weather")
        if len(weather) < hours:
            raise ValueError(
                f"Weather source {weather_source_hdf} has {len(weather)} "
                f"rows, expected at least {hours}."
            )
        weather = weather.iloc[:hours].reset_index(drop=True)
    weather.index.name = "t"
    return weather


def empty_timeseries(hours: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.RangeIndex(hours, name="t"))
    frame.columns = pd.MultiIndex(
        levels=[[], []],
        codes=[[], []],
        names=["Site", "Commodity"],
    )
    return frame


def buy_sell_price(hours: int, electricity_module) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "electricity_import": [electricity_module.config.BSP_IMPORT] * hours,
            "electricity_feed_in": [electricity_module.config.BSP_FEED_IN] * hours,
        },
        index=pd.RangeIndex(hours, name="t"),
    )


def urbs_static_tables(
    active_buses: list[int], electricity_module
) -> dict[str, pd.DataFrame]:
    return {
        "process": electricity_module.create_pro_elec(active_buses),
        "commodity": electricity_module.create_com_elec(active_buses),
        "process_commodity": electricity_module.create_pro_com_elec(),
        "storage": electricity_module.create_sto_elec(active_buses),
    }
