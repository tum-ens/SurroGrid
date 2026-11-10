"""Section 14a EnWG control on the INFLEX post demand (Lee, 2026, Section 3.6).

Rule-based grid-oriented control following BNetzA BK6-22-300, applied to the
inflexible heat-pump and EV charging profiles before the power flow:

- Congested hours are the hours in which the uncontrolled net load at the
  transformer (household demand + heat pumps + EV charging - PV, converted to
  apparent power with the power factors of ``config``) exceeds the rating.
- Controllable devices are heat-pump systems (heat pump + auxiliary heater of a
  bus, Ziffer 2.4.1.b and 2.4.2) and charge points with more than 4.2 kW
  (Ziffer 2.4.1). Stationary batteries are not controlled: under INFLEX they
  charge only from PV surplus.
- In a congested hour every controllable device gets the setpoint
  ``cap - ratio * (cap - p_min)`` with one common ratio, the smallest that
  clears the overload (Ziffer 4.2, 4.3, 4.4.a). ``p_min`` is 4.2 kW, or
  0.4 x connection power for heat-pump systems above 11 kW (Ziffer 4.5.1).
- Deferred energy is returned outside congested hours, within the headroom left
  below the rating and shared in proportion to the backlog (Ziffer 4.3, staged
  release). Deferred heat is carried as a heat backlog; deferred EV energy stays
  in its session and can only be returned before departure. What is left at the
  end of a session or of the horizon is reported as unserved energy.
"""

from __future__ import annotations

import numpy as np

P14A_KW = 4.2            # minimum power per controllable device (Ziffer 4.5.1)
LARGE_HP_KW = 11.0       # heat-pump systems above this get 0.4 x connection power
LARGE_HP_SHARE = 0.4
TOL = 1e-9


def heat_to_el(heat, cop, cap_hp):
    """Electricity drawn to deliver ``heat``: heat pump up to its capacity, auxiliary heater (COP 1) above it."""
    hp_heat = np.minimum(heat, cap_hp * cop)
    return hp_heat / cop + np.maximum(heat - hp_heat, 0.0)


def el_to_heat(el, cop, cap_hp):
    """Heat delivered when drawing ``el``."""
    hp_el = np.minimum(el, cap_hp)
    return hp_el * cop + np.maximum(el - hp_el, 0.0)


def _smallest_ratio(over, segments):
    """Smallest common ratio that removes ``over`` kVA.

    Each device contributes from the ratio at which its setpoint falls to its
    current draw (breakpoint) with a constant slope (curtailable range in kVA).
    Returns ``(ratio, cleared)``.
    """
    if not segments:
        return 0.0, False
    segments = sorted(segments)
    slope = offset = 0.0
    for k, (breakpoint, step) in enumerate(segments):
        nxt = segments[k + 1][0] if k + 1 < len(segments) else 1.0
        slope += step
        offset += step * breakpoint
        if slope * nxt - offset >= over - TOL:
            return min(1.0, max(breakpoint, (over + offset) / slope)), True
    return 1.0, False


def control(base_kw, pv_kw, heat_pumps, vehicles, rating_kva, pf_elc, pf_hp, staged=True):
    """Apply Section 14a control to the uncontrolled INFLEX profiles.

    Args:
        base_kw: household electricity of all buses, kW per hour.
        pv_kw: PV generation of all buses, kW per hour.
        heat_pumps: list of dicts with ``heat`` (kW per hour), ``cop`` (per hour,
            > 0), ``cap_hp`` and ``cap_rod`` (installed electrical kW).
        vehicles: list of dicts with ``cap`` (charger kW), ``fraction`` (plugged-in
            share per hour), ``session`` (session index per hour, -1 if away),
            ``energy`` (kWh per session index) and ``uncontrolled`` (kW per hour,
            the INFLEX charging profile).
        rating_kva: transformer rating.

    Returns:
        ``(heat_pumps, vehicles, diagnostics)``; each heat pump gets ``heat_served``,
        ``el`` and ``unserved_kwh``, each vehicle ``charge`` and ``unserved_kwh``.
    """
    base_kw = np.asarray(base_kw, dtype=float)
    pv_kw = np.asarray(pv_kw, dtype=float)
    T = len(base_kw)
    limit = float(rating_kva)

    hp_fixed = np.zeros(T)      # heat pumps that are not controllable devices
    hp_total = np.zeros(T)
    controlled_hp = []
    for d in heat_pumps:
        d["heat"] = np.asarray(d["heat"], dtype=float)
        d["cop"] = np.asarray(d["cop"], dtype=float)
        d["el"] = heat_to_el(d["heat"], d["cop"], d["cap_hp"])
        d["heat_served"] = d["heat"].copy()
        d["unserved_kwh"] = 0.0
        hp_total += d["el"]
        p_conn = float(d["cap_hp"]) + float(d["cap_rod"])
        if p_conn <= P14A_KW:
            hp_fixed += d["el"]
            continue
        d["p_conn"] = p_conn
        d["p_min"] = LARGE_HP_SHARE * p_conn if p_conn > LARGE_HP_KW else P14A_KW
        d["carry"] = 0.0
        d["hmax"] = float(d["heat"].max())
        controlled_hp.append(d)

    ev_total = np.zeros(T)
    for v in vehicles:
        v["fraction"] = np.asarray(v["fraction"], dtype=float)
        v["session"] = np.asarray(v["session"], dtype=int)
        v["uncontrolled"] = np.asarray(v["uncontrolled"], dtype=float)
        v["remaining"] = dict(v["energy"])
        v["charge"] = np.zeros(T)
        ev_total += v["uncontrolled"]

    total = np.maximum((base_kw + ev_total - pv_kw) / pf_elc + hp_total / pf_hp, 0.0)
    congested = total > limit + TOL

    ratio = np.zeros(T)
    n_insufficient = n_blocked = 0
    for t in range(T):
        # what each vehicle would draw this hour without control, given what it still needs
        for v in vehicles:
            s = v["session"][t]
            v["want"] = min(v["cap"] * v["fraction"][t], v["remaining"][s]) if s >= 0 else 0.0
            v["draw"] = min(v["want"], v["uncontrolled"][t])

        if congested[t]:
            segments = []
            for v in vehicles:
                span = v["cap"] - P14A_KW
                if span > TOL and v["draw"] > P14A_KW:
                    segments.append(((v["cap"] - v["draw"]) / span, span / pf_elc))
            for d in controlled_hp:
                span = d["p_conn"] - d["p_min"]
                if span > TOL and d["el"][t] > d["p_min"]:
                    segments.append(((d["p_conn"] - d["el"][t]) / span, span / pf_hp))
            ratio[t], cleared = _smallest_ratio(total[t] - limit, segments)
            n_insufficient += int(not cleared)
            for v in vehicles:
                setpoint = v["cap"] - ratio[t] * (v["cap"] - P14A_KW)
                v["charge"][t] = min(v["draw"], max(P14A_KW, setpoint))
            for d in controlled_hp:
                setpoint = max(d["p_min"], d["p_conn"] - ratio[t] * (d["p_conn"] - d["p_min"]))
                # no recovery while the control event runs (Ziffer 4.3 sentence 3)
                d["heat_served"][t] = min(d["heat"][t], el_to_heat(setpoint, d["cop"][t], d["cap_hp"]))
        else:
            for v in vehicles:
                v["charge"][t] = v["draw"] if staged else v["want"]
            if staged:
                # headroom below the rating once every device has its uncontrolled draw
                # (as in the thesis, PV is not netted off here, which keeps the release conservative)
                guaranteed = ((base_kw[t] + sum(v["charge"][t] for v in vehicles)) / pf_elc
                              + hp_total[t] / pf_hp)
                head = limit - guaranteed
                need_v = [max(0.0, v["want"] - v["charge"][t]) / pf_elc for v in vehicles]
                need_h = [(heat_to_el(d["heat"][t] + d["carry"], d["cop"][t], d["cap_hp"]) - d["el"][t]) / pf_hp
                          for d in controlled_hp]
                need = sum(need_v) + sum(need_h)
                if need > TOL:
                    if head <= TOL:
                        n_blocked += 1
                    else:
                        share = min(1.0, head / need)
                        for v, n in zip(vehicles, need_v):
                            v["charge"][t] += share * n * pf_elc
                        for d, n in zip(controlled_hp, need_h):
                            grant = min(share * n * pf_hp, max(0.0, d["p_conn"] - d["el"][t]))
                            reachable = el_to_heat(d["el"][t] + grant, d["cop"][t], d["cap_hp"]) - d["heat"][t]
                            room = max(0.0, d["hmax"] - d["heat"][t])
                            d["heat_served"][t] += min(d["carry"], max(0.0, reachable), room)
            else:
                for d in controlled_hp:
                    d["heat_served"][t] += min(d["carry"], d["p_min"] * d["cop"][t])

        # advance the state
        for v in vehicles:
            s = v["session"][t]
            if s >= 0:
                v["remaining"][s] = max(0.0, v["remaining"][s] - v["charge"][t])
        for d in controlled_hp:
            d["carry"] = max(0.0, d["heat"][t] + d["carry"] - d["heat_served"][t])

    for v in vehicles:
        v["unserved_kwh"] = float(sum(v["remaining"].values()))
    for d in controlled_hp:
        d["unserved_kwh"] = float(d["carry"])
    for d in heat_pumps:
        d["el"] = heat_to_el(d["heat_served"], d["cop"], d["cap_hp"])

    diagnostics = {
        "rating_kva": limit,
        "peak_uncontrolled_kva": float(total.max()) if T else 0.0,
        "congested_hours": int(congested.sum()),
        "hours_not_cleared": n_insufficient,
        "hours_backlog_without_headroom": n_blocked,
        "mean_ratio_congested": float(ratio[congested].mean()) if congested.any() else 0.0,
        "controlled_heat_pumps": len(controlled_hp),
        "controlled_charge_points": sum(1 for v in vehicles if v["cap"] > P14A_KW),
        "unserved_heat_kwh": float(sum(d["unserved_kwh"] for d in heat_pumps)),
        "unserved_ev_kwh": float(sum(v["unserved_kwh"] for v in vehicles)),
    }
    return heat_pumps, vehicles, diagnostics
