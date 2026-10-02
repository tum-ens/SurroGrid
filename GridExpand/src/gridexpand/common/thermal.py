"""Chronological implicit-Euler 1R1C physics shared by both HEMS backends.

Power is kW, capacitance kWh/K, conductance kW/K and duration hours.
The thermostat reference is independent of installed equipment and HEMS dispatch.
"""

from __future__ import annotations

import numpy as np

THERMAL_PARAMETER_KEY = "building_thermal_parameters"
THERMAL_TIMESERIES_KEY = "building_thermal_timeseries"
FIELDS = (
    "outside_temperature_c",
    "internal_gains_kw",
    "solar_gains_kw",
    "minimum_temperature_c",
    "upper_temperature_c",
)


def thermostat_reference(
    outside,
    gains,
    *,
    conductance_kw_per_k,
    capacitance_kwh_per_k,
    minimum_temperature_c=20.0,
    delta_t_hours=1.0,
    tolerance_k=1e-9,
):
    """Return heat, end-of-interval temperatures and the periodic initial state.

    Passive overheating is permitted; no active cooling or heat rejection exists.
    Repeat the supplied chronological cycle until its boundary closes.
    """
    outside, gains = np.asarray(outside, float), np.asarray(gains, float)
    minimum = np.broadcast_to(np.asarray(minimum_temperature_c, float), outside.shape)
    h, c, dt = conductance_kw_per_k, capacitance_kwh_per_k, delta_t_hours
    if outside.ndim != 1 or outside.size == 0 or gains.shape != outside.shape:
        raise ValueError("Thermostat requires equally sized, nonempty hourly arrays.")
    if (
        not np.isfinite(np.r_[outside, gains, minimum, h, c, dt]).all()
        or min(h, c, dt) <= 0
    ):
        raise ValueError("Thermostat requires finite inputs and positive H, C and dt.")
    capacity = c / dt
    decay = capacity / (capacity + h)
    forcing = (h * outside + gains) / (capacity + h)

    def cycle(initial):
        temperature, heat = np.empty(outside.size), np.empty(outside.size)
        previous = initial
        for i in range(outside.size):
            free = decay * previous + forcing[i]
            temperature[i] = max(minimum[i], free)
            heat[i] = max(0.0, (minimum[i] - free) * (capacity + h))
            previous = temperature[i]
        return heat, temperature

    initial = float(minimum[0])
    for _ in range(10000):
        heat, temperature = cycle(initial)
        new_initial = float(temperature[-1])
        if abs(new_initial - initial) <= tolerance_k:
            # Exact matching boundaries are stored; verify the recomputed cycle.
            initial = new_initial
            heat, temperature = cycle(initial)
            return heat, temperature, initial
        initial = new_initial
    raise ValueError("Thermostat periodic boundary did not converge.")


def validate_thermal_input(parameters, timeseries, *, steps=None):
    """Reject incomplete, invalid or mismapped physical-building thermal inputs."""
    if parameters.empty:
        if not timeseries.empty:
            raise ValueError("Thermal time series exist without building parameters.")
        return
    required = {
        "building_objectid",
        "Site",
        "heat_commodity",
        "conductance_kw_per_k",
        "capacitance_kwh_per_k",
        "initial_temperature_c",
        "terminal_temperature_c",
        "room_heat_upper_kw",
    }
    if required.difference(parameters.columns):
        raise ValueError(
            f"Thermal parameters lack {sorted(required.difference(parameters.columns))}."
        )
    if parameters.building_objectid.duplicated().any():
        raise ValueError("Thermal parameters require one state per physical building.")
    if parameters[["Site", "heat_commodity"]].duplicated().any():
        raise ValueError("Thermal buildings require distinct room heat balances.")
    for row in parameters.to_dict("records"):
        bid = str(row["building_objectid"])
        for field in FIELDS:
            if (bid, field) not in timeseries.columns:
                raise ValueError(f"Missing thermal time series {bid}/{field}.")
        values = timeseries.loc[:, [(bid, field) for field in FIELDS]]
        if steps is not None:
            values = values.loc[steps]
        if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Incomplete or non-finite thermal time series for {bid}.")
        if (
            values[(bid, "upper_temperature_c")]
            < values[(bid, "minimum_temperature_c")]
        ).any():
            raise ValueError(f"Inverted thermal comfort bounds for {bid}.")
        if not np.isfinite(
            [
                row[k]
                for k in (
                    "conductance_kw_per_k",
                    "capacitance_kwh_per_k",
                    "initial_temperature_c",
                    "terminal_temperature_c",
                )
            ]
        ).all():
            raise ValueError(f"Non-finite thermal parameters for {bid}.")
        if not np.isfinite(row["room_heat_upper_kw"]) or row["room_heat_upper_kw"] < 0:
            raise ValueError(f"Invalid finite room heat bound for {bid}.")
        if min(row["conductance_kw_per_k"], row["capacitance_kwh_per_k"]) <= 0:
            raise ValueError(f"Non-positive thermal H/C for {bid}.")


def dispatch_heat_services(heat_kw, cops, hp_capacity_kw_el):
    """Greedy instantaneous HP dispatch, highest COP first; rod supplies residual.

    No forecast or thermal shifting. Arrays are hours x services and a shared
    electrical HP capacity is consumed once across all simultaneous services.
    """
    heat, cops = np.asarray(heat_kw, float), np.asarray(cops, float)
    if (
        heat.shape != cops.shape
        or heat.ndim != 2
        or not np.isfinite(np.r_[heat.ravel(), cops.ravel(), hp_capacity_kw_el]).all()
        or (heat < 0).any()
        or (cops <= 0).any()
        or hp_capacity_kw_el < 0
    ):
        raise ValueError(
            "Heat dispatch requires nonnegative heat/capacity and positive finite service COPs."
        )
    order = np.argsort(-cops, axis=1, kind="stable")
    hp = np.zeros_like(heat)
    remaining = np.full(len(heat), float(hp_capacity_kw_el))
    hours = np.arange(len(heat))
    for rank in range(heat.shape[1]):
        service = order[:, rank]
        electricity = np.minimum(remaining, heat[hours, service] / cops[hours, service])
        hp[hours, service] = electricity
        remaining -= electricity
    residual = np.maximum(0, heat - hp * cops)
    return hp, residual
