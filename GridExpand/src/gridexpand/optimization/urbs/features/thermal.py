"""Physical building temperatures and delivered room heat (implicit Euler 1R1C)."""

import pyomo.environ as pyomo
from gridexpand.common.thermal import validate_thermal_input


def add_thermal(m, data):
    if m.mode["tsam"] or m.mode["tdy"]:
        raise ValueError(
            "Internal thermal inertia requires chronological inputs; TSAM is unsupported."
        )
    params = data["building_thermal_parameters"].reset_index()
    series = data["building_thermal_timeseries"]
    validate_thermal_input(
        params, series, steps=[(stf, t) for stf in m.stf for t in m.tm]
    )
    rows = {
        (r["support_timeframe"], r["Site"], str(r["building_objectid"])): r
        for r in params.to_dict("records")
    }
    m.building = pyomo.Set(initialize=[key[2] for key in rows])
    m.thermal_tuples = pyomo.Set(
        within=m.stf * m.sit * m.building, initialize=list(rows)
    )
    m._thermal_commodities = {
        (stf, sit, row["heat_commodity"]): bid for (stf, sit, bid), row in rows.items()
    }

    def value(stf, t, bid, field):
        return float(series.loc[(stf, t), (bid, field)])

    def bounds(m, t, stf, sit, bid):
        row = rows[stf, sit, bid]
        if t == m.t.first():
            temp = row["initial_temperature_c"]
            return temp, temp
        return value(stf, t, bid, "minimum_temperature_c"), value(
            stf, t, bid, "upper_temperature_c"
        )

    m.building_temperature = pyomo.Var(
        m.t,
        m.thermal_tuples,
        bounds=bounds,
        doc="End of interval room temperature [degC]",
    )
    m.building_heat = pyomo.Var(
        m.tm,
        m.thermal_tuples,
        within=pyomo.NonNegativeReals,
        doc="Delivered room heat [kW]",
    )

    def balance(m, t, stf, sit, bid):
        row = rows[stf, sit, bid]
        return row["capacitance_kwh_per_k"] / m.dt * (
            m.building_temperature[t, stf, sit, bid]
            - m.building_temperature[t - 1, stf, sit, bid]
        ) == m.building_heat[t, stf, sit, bid] + value(
            stf, t, bid, "internal_gains_kw"
        ) + value(stf, t, bid, "solar_gains_kw") - row["conductance_kw_per_k"] * (
            m.building_temperature[t, stf, sit, bid]
            - value(stf, t, bid, "outside_temperature_c")
        )

    m.building_thermal_balance = pyomo.Constraint(m.tm, m.thermal_tuples, rule=balance)
    m.building_gains = pyomo.Expression(
        m.tm,
        m.thermal_tuples,
        rule=lambda m, t, stf, sit, bid: (
            value(stf, t, bid, "internal_gains_kw")
            + value(stf, t, bid, "solar_gains_kw")
        ),
        doc="Internal plus solar gains [kW]",
    )
    m.building_loss = pyomo.Expression(
        m.tm,
        m.thermal_tuples,
        rule=lambda m, t, stf, sit, bid: (
            rows[stf, sit, bid]["conductance_kw_per_k"]
            * (
                m.building_temperature[t, stf, sit, bid]
                - value(stf, t, bid, "outside_temperature_c")
            )
        ),
        doc="Transmission plus ventilation loss [kW]",
    )
    m.building_energy_residual = pyomo.Expression(
        m.tm,
        m.thermal_tuples,
        rule=lambda m, t, stf, sit, bid: (
            rows[stf, sit, bid]["capacitance_kwh_per_k"]
            / m.dt
            * (
                m.building_temperature[t, stf, sit, bid]
                - m.building_temperature[t - 1, stf, sit, bid]
            )
            - m.building_heat[t, stf, sit, bid]
            - m.building_gains[t, stf, sit, bid]
            + m.building_loss[t, stf, sit, bid]
        ),
        doc="1R1C energy balance residual [kW]",
    )
    m.building_terminal_temperature = pyomo.Constraint(
        m.thermal_tuples,
        rule=lambda m, stf, sit, bid: (
            m.building_temperature[m.t.last(), stf, sit, bid]
            == rows[stf, sit, bid]["terminal_temperature_c"]
        ),
    )
    # Prevent the tank charging COP route from bypassing the physical buffer.
    tank_rows = [
        (stf, sit, bid)
        for stf, sit, bid in rows
        if (stf, sit, "heat_storage_" + bid, "tank_heat_" + bid)
        in getattr(m, "sto_tuples", ())
    ]
    m.building_tank_tuples = pyomo.Set(within=m.thermal_tuples, initialize=tank_rows)
    m.building_tank_charge = pyomo.Constraint(
        m.tm,
        m.building_tank_tuples,
        rule=lambda m, t, stf, sit, bid: (
            m.e_pro_out[t, stf, sit, "HP_buffer_" + bid, "tank_heat_" + bid]
            == m.e_sto_in[t, stf, sit, "heat_storage_" + bid, "tank_heat_" + bid]
        ),
    )
    return m
