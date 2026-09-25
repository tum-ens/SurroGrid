"""Shared URBS input-table constructors for calibrated scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import gridexpand.allocation.functions.electricity as electricity


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


def buy_sell_price(
    hours: int,
    *,
    import_price_eur_per_kwh: float,
    pv_feed_in_tariff_eur_per_kwh: float,
) -> pd.DataFrame:
    import_price = float(import_price_eur_per_kwh)
    feed_in_tariff = float(pv_feed_in_tariff_eur_per_kwh)
    return pd.DataFrame(
        {
            "electricity_import": [import_price] * hours,
            "electricity_feed_in": [feed_in_tariff] * hours,
        },
        index=pd.RangeIndex(hours, name="t"),
    )


def urbs_static_tables(
    active_buses: list[int],
    technologies,
    *,
    include_generic_battery: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return the grid-connection process, commodity and (generic) storage tables.

    Args:
        active_buses: Buses with demand.
        technologies: ``ScenarioConfig.technologies``.
        include_generic_battery: Add a ``battery_private`` row per bus.
    """
    battery = technologies.storages["stationary_battery"]
    storage = (
        electricity.create_sto_elec(active_buses, battery)
        if include_generic_battery
        else electricity.create_sto_elec([], battery)
    )
    return {
        "process": electricity.create_pro_elec(
            active_buses, technologies.processes["grid_connection"]
        ),
        "commodity": electricity.create_com_elec(active_buses),
        "process_commodity": electricity.create_pro_com_elec(),
        "storage": storage,
    }
