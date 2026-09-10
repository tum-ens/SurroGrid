"""Dedicated EV charging-session constraints.

Implements the shared contract of ``GridExpand/common/ev_sessions.py`` inside
URBS. One session is one home stay of one vehicle; its required grid energy must
be delivered inside that stay, at or below the charger rating scaled by the
connected fraction of each hour:

    tau_pro[t, sit, charger]  <=  dt * cap_pro[sit, charger] * frac[s,t]
    sum_{t in H(s)} tau_pro[t, sit, charger]  ==  energy_kwh[s]
    tau_pro[t, sit, charger]  ==  0                      for t outside any session

``tau_pro`` is throughput *energy* per timestep in URBS, and the charger's only
process-commodity row is ``('electricity', 'In', 1)``, so the charging energy
enters the site electricity balance exactly once through ``e_pro_in``. There is
no mobility commodity, no mobility demand column and no virtual mobility
storage: a session's obligation can never be served during a different session.
"""

import pyomo.core as pyomo


def add_ev_sessions(m, data):
    """Add the dedicated session variables' constraints to model ``m``."""
    sessions = data["ev_sessions"]
    hours = data["ev_session_hours"]

    session_records = sessions.to_dict("records")
    if not session_records:
        return m

    stf = m.stf_list[0]

    # session key -> (site, process); admissible (t, fraction) pairs per session
    session_process = {}
    session_energy = {}
    for row in session_records:
        key = str(row["session_id"])
        session_process[key] = (row["site"], str(row["process"]))
        session_energy[key] = float(row["energy_kwh"])

    session_hours = {key: [] for key in session_process}
    charger_hours = {}
    for row in hours.to_dict("records"):
        key = str(row["session_id"])
        if key not in session_hours:
            raise ValueError(
                f"EV session hour references unknown session {key!r}."
            )
        timestep = int(row["t"])
        fraction = float(row["available_fraction"])
        session_hours[key].append((timestep, fraction))
        charger_key = session_process[key] + (timestep,)
        if charger_key in charger_hours:
            raise ValueError(
                "Two EV sessions of one vehicle claim model hour "
                f"{timestep} at {session_process[key]}, which would duplicate "
                "charger capacity."
            )
        charger_hours[charger_key] = fraction

    empty = [key for key, entries in session_hours.items() if not entries]
    if empty:
        raise ValueError(f"EV sessions without admissible hours: {sorted(empty)[:5]}")

    chargers = sorted({value for value in session_process.values()})
    missing = [
        charger for charger in chargers if (stf,) + charger not in m.pro_tuples
    ]
    if missing:
        raise ValueError(
            f"EV sessions reference processes that are not in the model: {missing[:5]}"
        )

    m.ev_session_tuples = pyomo.Set(
        initialize=sorted(session_process),
        ordered=True,
        doc="EV charging sessions served on this model instance",
    )
    m.ev_session_energy = session_energy
    m.ev_session_process = session_process
    m.ev_session_hours = session_hours
    m.ev_charger_availability = charger_hours
    m.ev_charger_tuples = pyomo.Set(
        within=m.pro_tuples,
        initialize=[(stf,) + charger for charger in chargers],
        ordered=True,
        doc="Processes governed by the dedicated EV session contract",
    )

    m.res_ev_charge_by_availability = pyomo.Constraint(
        m.tm, m.ev_charger_tuples,
        rule=res_ev_charge_by_availability_rule,
        doc="EV charging <= charger rating * connected fraction of the hour",
    )
    m.def_ev_session_energy = pyomo.Constraint(
        m.ev_session_tuples,
        rule=def_ev_session_energy_rule,
        doc="EV charging summed over a session == its required grid energy",
    )
    return m


def res_ev_charge_by_availability_rule(m, tm, stf, sit, pro):
    """Hard instantaneous limit, including a hard zero while disconnected."""
    fraction = m.ev_charger_availability.get((sit, pro, tm), 0.0)
    if fraction <= 0.0:
        return m.tau_pro[tm, stf, sit, pro] == 0
    return (
        m.tau_pro[tm, stf, sit, pro]
        <= m.dt * m.cap_pro[stf, sit, pro] * fraction
    )


def def_ev_session_energy_rule(m, session):
    stf = m.stf_list[0]
    sit, pro = m.ev_session_process[session]
    return (
        sum(
            m.tau_pro[timestep, stf, sit, pro]
            for timestep, _fraction in m.ev_session_hours[session]
        )
        == m.ev_session_energy[session]
    )
