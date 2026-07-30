"""Materialize the shared expansion-cost heuristic for real SWF grids."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pandapower as pp
from sqlalchemy import text


CORRIDOR_LENGTH_RELATIVE_TOLERANCE = 0.05


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bus_coordinates(net: pp.pandapowerNet, bus: int) -> tuple[float, float] | None:
    if bus not in net.bus.index or "geo" not in net.bus.columns:
        return None
    value = net.bus.at[bus, "geo"]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    coordinates = value.get("coordinates")
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
        return None
    x = _finite(coordinates[0])
    y = _finite(coordinates[1])
    if x is None or y is None:
        return None
    return x, y


def _line_wkt(net: pp.pandapowerNet, from_bus: int, to_bus: int) -> str | None:
    start = _bus_coordinates(net, from_bus)
    end = _bus_coordinates(net, to_bus)
    if start is None or end is None or start == end:
        return None
    return f"LINESTRING({start[0]} {start[1]}, {end[0]} {end[1]})"


def _point_wkt(net: pp.pandapowerNet) -> str | None:
    candidate_buses: list[int] = []
    if not net.ext_grid.empty and "bus" in net.ext_grid.columns:
        candidate_buses.extend(net.ext_grid["bus"].dropna().astype(int).tolist())
    if not net.trafo.empty and "lv_bus" in net.trafo.columns:
        candidate_buses.extend(net.trafo["lv_bus"].dropna().astype(int).tolist())
    for bus in candidate_buses:
        coordinates = _bus_coordinates(net, bus)
        if coordinates is not None:
            return f"POINT({coordinates[0]} {coordinates[1]})"
    return None


def _settlement_type(db, plz: int | None) -> int | None:
    if plz is None:
        return None
    query = text(
        """
        SELECT settlement_type
        FROM pylovo.postcode_result
        WHERE postcode_result_plz = :plz
          AND version_id = :version_id
          AND settlement_type IS NOT NULL
        LIMIT 1
        """
    )
    with db.engine.connect() as conn:
        value = conn.execute(
            query,
            {"plz": int(plz), "version_id": db.pylovo_version_id},
        ).scalar_one_or_none()
    return None if value is None else int(value)


def _line_cost(
    *,
    required_added_capacity_ka: float,
    settlement_type: int | None,
    length_km: float,
    assumption: dict[str, Any],
    duct_share_override: float | None,
) -> dict[str, Any]:
    """Select the least-cost adequate combination from the shared cable catalogue."""
    if settlement_type == 1:
        reopen_cost = float(assumption["line_reopen_rural_eur_per_km"])
        settlement_label = "rural"
    elif settlement_type == 3:
        reopen_cost = float(assumption["line_reopen_urban_eur_per_km"])
        settlement_label = "urban"
    else:
        reopen_cost = float(assumption["line_reopen_suburban_eur_per_km"])
        settlement_label = "semiurban"

    duct_share = (
        float(assumption["line_existing_duct_share"])
        if duct_share_override is None
        else float(duct_share_override)
    )
    duct_share = min(max(duct_share, 0.0), 1.0)
    trenching_share = 1.0 - duct_share
    catalog = (
        (
            "150",
            float(assumption["line_reinforcement_150_max_i_ka"]),
            float(assumption["line_parallel_150_eur_per_km"]),
        ),
        (
            "185",
            float(assumption["line_reinforcement_185_max_i_ka"]),
            float(assumption["line_parallel_185_eur_per_km"]),
        ),
        (
            "240",
            float(assumption["line_reinforcement_240_max_i_ka"]),
            float(assumption["line_parallel_240_eur_per_km"]),
        ),
    )
    required = max(float(required_added_capacity_ka), 0.0)
    if required <= 1e-12:
        return {
            "line_existing_duct_share": duct_share,
            "line_trenching_share": trenching_share,
            "cost_eur_per_km": 0.0,
            "estimated_cost_eur": 0.0,
            "cost_basis": "none_existing_capacity_sufficient",
            "duct_cost_eur_per_km": 0.0,
            "reopen_cost_eur_per_km": reopen_cost,
            "reinforcement_150_count": 0,
            "reinforcement_185_count": 0,
            "reinforcement_240_count": 0,
            "reinforcement_added_capacity_ka": 0.0,
            "reinforcement_catalog": "NAYY_4_150|NAYY_4_185|NAYY_4_240",
        }

    limit = int(math.ceil(required / min(item[1] for item in catalog)))
    best: tuple[tuple[float, int, float, int, int], dict[str, Any]] | None = None
    for n150 in range(limit + 1):
        for n185 in range(limit + 1):
            for n240 in range(limit + 1):
                count = n150 + n185 + n240
                if count == 0:
                    continue
                counts = (n150, n185, n240)
                added_capacity = sum(
                    count * item[1] for count, item in zip(counts, catalog)
                )
                if added_capacity + 1e-12 < required:
                    continue
                total_duct_cost = sum(
                    count * item[2] for count, item in zip(counts, catalog)
                )
                primary_duct_cost = max(
                    item[2] for count, item in zip(counts, catalog) if count > 0
                )
                cost_per_km = total_duct_cost + trenching_share * (
                    reopen_cost - primary_duct_cost
                )
                choice = {
                    "line_existing_duct_share": duct_share,
                    "line_trenching_share": trenching_share,
                    "cost_eur_per_km": cost_per_km,
                    "estimated_cost_eur": length_km * cost_per_km,
                    "cost_basis": (
                        f"catalog_{settlement_label}_duct{round(duct_share * 100)}_"
                        f"trench{round(trenching_share * 100)}_"
                        f"150x{n150}_185x{n185}_240x{n240}"
                    ),
                    "duct_cost_eur_per_km": total_duct_cost,
                    "reopen_cost_eur_per_km": reopen_cost,
                    "reinforcement_150_count": n150,
                    "reinforcement_185_count": n185,
                    "reinforcement_240_count": n240,
                    "reinforcement_added_capacity_ka": added_capacity,
                    "reinforcement_catalog": "NAYY_4_150|NAYY_4_185|NAYY_4_240",
                }
                key = (
                    cost_per_km,
                    count,
                    added_capacity - required,
                    -n240,
                    -n185,
                )
                if best is None or key < best[0]:
                    best = (key, choice)

    if best is None:
        raise RuntimeError(f"No reinforcement combination covers {required:.6f} kA.")
    return best[1]


def _transformer_cost(
    required_kva: float, rated_kva: float, assumption: dict[str, Any]
) -> tuple[float, str]:
    if required_kva <= rated_kva:
        return 0.0, "none_existing_capacity_sufficient"
    for capacity, column in (
        (400.0, "transformer_replace_400_eur"),
        (630.0, "transformer_replace_630_eur"),
        (800.0, "transformer_replace_800_eur"),
        (1000.0, "transformer_replace_1000_eur"),
    ):
        if required_kva <= capacity:
            return float(
                assumption[column]
            ), f"all_in_replacement_to_{int(capacity)}kva"
    return (
        float(assumption["transformer_station_rebuild_boundary_eur"]),
        "station_rebuild_boundary_case_gt_1000kva",
    )


def _same_recorded_corridor(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two parallel rows plausibly occupy the same cable route."""
    if left["bus_pair"] != right["bus_pair"]:
        return False
    shorter = min(left["length_km"], right["length_km"])
    longer = max(left["length_km"], right["length_km"])
    if shorter <= 0.0 or longer <= 0.0:
        return False
    return (longer - shorter) / longer <= CORRIDOR_LENGTH_RELATIVE_TOLERANCE


def _corridor_groups(cables: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group same-endpoint rows only when all recorded lengths agree."""
    by_bus_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cable in cables:
        by_bus_pair.setdefault(cable["bus_pair"], []).append(cable)

    groups: list[list[dict[str, Any]]] = []
    for candidates in by_bus_pair.values():
        route_groups: list[list[dict[str, Any]]] = []
        for candidate in sorted(
            candidates, key=lambda row: (row["length_km"], row["cable"])
        ):
            matching = next(
                (
                    group
                    for group in route_groups
                    if all(
                        _same_recorded_corridor(candidate, member) for member in group
                    )
                ),
                None,
            )
            if matching is None:
                route_groups.append([candidate])
            else:
                matching.append(candidate)
        groups.extend(route_groups)
    return groups


def _selected_runs(db, args) -> pd.DataFrame:
    query = text(
        """
        SELECT
            rpr.real_powerflow_run_id,
            rpr.real_grid_case_id,
            rpr.scenario_id,
            rgc.plz,
            rgc.lv_id,
            rgc.source_file,
            rps.n_timesteps,
            COALESCE(rps.n_failed_timesteps, 0) AS n_failed_timesteps,
            rps.transformer_s_rated_mva,
            rps.trafo_loading_max_time_percent
        FROM surrogrid.real_powerflow_run rpr
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        JOIN surrogrid.real_powerflow_summary rps USING (real_powerflow_run_id)
        WHERE rpr.run_name = :run_name
          AND rps.stage = :stage
          AND (:scenario_id IS NULL OR rpr.scenario_id = :scenario_id)
          AND (:plz IS NULL OR rgc.plz = :plz)
        ORDER BY rgc.lv_id
        """
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": args.run_name,
                "stage": args.stage,
                "scenario_id": args.scenario_id,
                "plz": args.plz,
            },
        )


def _assumption(db, key: str) -> dict[str, Any]:
    with db.engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT * FROM surrogrid.expansion_cost_assumption WHERE assumption_key = :key"
                ),
                {"key": key},
            )
            .mappings()
            .one()
        )
    return dict(row)


def _critical_indices(db, run_id: int, stage: str) -> dict[tuple[str, int], int]:
    query = text(
        """
        SELECT DISTINCT ON (metric, asset_id)
            metric, asset_id, t_index
        FROM surrogrid.real_powerflow_tail_value
        WHERE real_powerflow_run_id = :run_id
          AND stage = :stage
          AND tail = 'upper'
        ORDER BY metric, asset_id, value DESC
        """
    )
    with db.engine.connect() as conn:
        rows = conn.execute(query, {"run_id": run_id, "stage": stage}).mappings()
        return {
            (str(row["metric"]), int(row["asset_id"])): int(row["t_index"])
            for row in rows
        }


def _cable_summaries(db, run_id: int, stage: str) -> pd.DataFrame:
    query = text(
        """
        SELECT *
        FROM surrogrid.real_powerflow_cable_summary
        WHERE real_powerflow_run_id = :run_id
          AND stage = :stage
        ORDER BY cable
        """
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"run_id": run_id, "stage": stage})


def materialize_real_results(
    db, *, expansion_analysis_run_id: int, args
) -> dict[str, int]:
    runs = _selected_runs(db, args)
    if runs.empty:
        raise RuntimeError(
            "No real SWF power-flow summaries match the requested expansion scope."
        )
    assumption = _assumption(db, args.assumption_key)
    excluded = {int(value) for value in (args.exclude_real_lv_id or [])}
    settlement_by_plz = {
        int(plz): _settlement_type(db, int(plz))
        for plz in runs["plz"].dropna().astype(int).unique()
    }
    status_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    transformer_rows: list[dict[str, Any]] = []

    for run in runs.to_dict("records"):
        run_id = int(run["real_powerflow_run_id"])
        lv_id = int(str(run["lv_id"]).removeprefix("LV_"))
        failed = int(run["n_failed_timesteps"] or 0)
        if lv_id in excluded:
            cost_status = "excluded"
            reason = "Explicitly excluded from the comparison scope."
        elif failed > 0:
            cost_status = "incomplete"
            reason = f"{failed} power-flow timesteps did not converge; P100 expansion cost is unknown."
        else:
            cost_status = "complete"
            reason = None
        status_rows.append(
            {
                "expansion_analysis_run_id": expansion_analysis_run_id,
                "real_powerflow_run_id": run_id,
                "real_grid_case_id": int(run["real_grid_case_id"]),
                "scenario_id": int(run["scenario_id"]),
                "plz": None if pd.isna(run["plz"]) else int(run["plz"]),
                "lv_id": str(lv_id),
                "n_timesteps": int(run["n_timesteps"]),
                "n_failed_timesteps": failed,
                "cost_status": cost_status,
                "status_reason": reason,
            }
        )
        if cost_status != "complete":
            continue

        source_file = Path(str(run["source_file"]))
        if not source_file.exists():
            raise FileNotFoundError(
                f"Real SWF grid source does not exist: {source_file}"
            )
        net = pp.from_excel(source_file)
        critical = _critical_indices(db, run_id, args.stage)
        settlement_type = (
            settlement_by_plz.get(int(run["plz"])) if not pd.isna(run["plz"]) else None
        )
        prepared_cables = []
        for cable in _cable_summaries(db, run_id, args.stage).to_dict("records"):
            cable_id = int(cable["cable"])
            if cable_id not in net.line.index:
                raise KeyError(
                    f"Cable {cable_id} from real summary is absent in {source_file}."
                )
            line = net.line.loc[cable_id]
            existing_parallel = max(
                int(round(_finite(cable.get("cable_parallel"), 1.0) or 1.0)), 1
            )
            max_i_ka = _finite(cable.get("cable_max_i_ka"))
            installed_capacity_ka = _finite(cable.get("cable_installed_capacity_ka"))
            loading_percent = _finite(cable.get("cable_loading_max_time_percent"))
            if max_i_ka is None and installed_capacity_ka is not None:
                max_i_ka = installed_capacity_ka / existing_parallel
            if installed_capacity_ka is None and max_i_ka is not None:
                installed_capacity_ka = max_i_ka * existing_parallel
            if (
                max_i_ka is None
                or max_i_ka <= 0
                or installed_capacity_ka is None
                or loading_percent is None
            ):
                raise ValueError(
                    f"Real cable {lv_id}:{cable_id} lacks a finite installed "
                    "capacity or P100 loading."
                )
            from_bus = int(line["from_bus"])
            to_bus = int(line["to_bus"])
            prepared_cables.append(
                {
                    "cable": cable_id,
                    "cable_name": str(line.get("name") or ""),
                    "std_type": str(line.get("std_type") or ""),
                    "from_bus": from_bus,
                    "to_bus": to_bus,
                    "bus_pair": tuple(sorted((from_bus, to_bus))),
                    "length_km": max(_finite(line.get("length_km"), 0.0) or 0.0, 0.0),
                    "existing_parallel": existing_parallel,
                    "installed_capacity_ka": installed_capacity_ka,
                    "max_i_from_ka": (loading_percent / 100.0 * installed_capacity_ka),
                    "critical_t_index": critical.get(("Cables", cable_id)),
                }
            )

        for corridor in _corridor_groups(prepared_cables):
            representative = min(corridor, key=lambda row: row["cable"])
            cable_ids = sorted(row["cable"] for row in corridor)
            installed_capacity_ka = sum(
                row["installed_capacity_ka"] for row in corridor
            )
            max_i_from_ka = sum(row["max_i_from_ka"] for row in corridor)
            existing_parallel = sum(row["existing_parallel"] for row in corridor)
            max_i_ka = installed_capacity_ka / existing_parallel
            loading_percent = max_i_from_ka / installed_capacity_ka * 100.0
            required_added_capacity_ka = max(max_i_from_ka - installed_capacity_ka, 0.0)
            length_km = max(row["length_km"] for row in corridor)
            costs = _line_cost(
                required_added_capacity_ka=required_added_capacity_ka,
                settlement_type=settlement_type,
                length_km=length_km,
                assumption=assumption,
                duct_share_override=args.line_existing_duct_share,
            )
            additional_parallel = (
                costs["reinforcement_150_count"]
                + costs["reinforcement_185_count"]
                + costs["reinforcement_240_count"]
            )
            critical_indices = {
                row["critical_t_index"]
                for row in corridor
                if row["critical_t_index"] is not None
            }
            line_rows.append(
                {
                    **status_rows[-1],
                    "cable": representative["cable"],
                    "cable_name": " | ".join(row["cable_name"] for row in corridor),
                    "std_type": " | ".join(
                        sorted({row["std_type"] for row in corridor})
                    ),
                    "corridor_cable_ids": "|".join(map(str, cable_ids)),
                    "corridor_line_count": len(corridor),
                    "corridor_grouping_method": (
                        "same_bus_pair_length_within_5pct"
                        if len(corridor) > 1
                        else "single_line"
                    ),
                    "from_bus": representative["from_bus"],
                    "to_bus": representative["to_bus"],
                    "length_km": length_km,
                    "settlement_type": settlement_type,
                    "existing_parallel": existing_parallel,
                    "max_i_from_ka": max_i_from_ka,
                    "max_i_ka": max_i_ka,
                    "installed_capacity_ka": installed_capacity_ka,
                    "loading_percent": loading_percent,
                    "required_parallel": existing_parallel + additional_parallel,
                    "additional_parallel": additional_parallel,
                    "requires_expansion": additional_parallel > 0,
                    "overloaded_at_100_percent": loading_percent > 100.0,
                    "critical_t_index": (
                        next(iter(critical_indices))
                        if len(critical_indices) == 1
                        else None
                    ),
                    "geom_wkt": _line_wkt(
                        net, representative["from_bus"], representative["to_bus"]
                    ),
                    **costs,
                }
            )

        rated_kva = (_finite(run["transformer_s_rated_mva"]) or 0.0) * 1000.0
        loading_percent = _finite(run["trafo_loading_max_time_percent"])
        if rated_kva <= 0 or loading_percent is None:
            raise ValueError(
                f"Real grid LV {lv_id} lacks a finite transformer rating or P100 loading."
            )
        max_s_mva = loading_percent / 100.0 * rated_kva / 1000.0
        step = float(assumption["transformer_capacity_step_kva"])
        required_kva = max(
            rated_kva, math.ceil(max_s_mva * 1000.0 / step - 1e-12) * step
        )
        transformer_cost, transformer_basis = _transformer_cost(
            required_kva, rated_kva, assumption
        )
        equipment_name = None
        if not net.trafo.empty:
            equipment_name = str(
                net.trafo.iloc[0].get("name") or net.trafo.iloc[0].get("std_type") or ""
            )
        transformer_rows.append(
            {
                **status_rows[-1],
                "transformer_rated_power_kva": rated_kva,
                "transformer_equipment_name": equipment_name,
                "max_s_mva": max_s_mva,
                "loading_percent": loading_percent,
                "required_transformer_kva": required_kva,
                "additional_transformer_kva": max(required_kva - rated_kva, 0.0),
                "requires_expansion": required_kva > rated_kva,
                "overloaded_at_100_percent": loading_percent > 100.0,
                "estimated_cost_eur": transformer_cost,
                "transformer_cost_basis": transformer_basis,
                "critical_t_index": critical.get(("Transformer", 0)),
                "geom_wkt": _point_wkt(net),
            }
        )

    status_sql = text(
        """
        INSERT INTO surrogrid.expansion_real_grid_status (
            expansion_analysis_run_id, real_powerflow_run_id, real_grid_case_id,
            scenario_id, plz, lv_id, n_timesteps, n_failed_timesteps,
            cost_status, status_reason
        ) VALUES (
            :expansion_analysis_run_id, :real_powerflow_run_id, :real_grid_case_id,
            :scenario_id, :plz, :lv_id, :n_timesteps, :n_failed_timesteps,
            :cost_status, :status_reason
        )
        """
    )
    line_sql = text(
        """
        INSERT INTO surrogrid.expansion_real_line_result (
            expansion_analysis_run_id, real_powerflow_run_id, real_grid_case_id,
            scenario_id, plz, lv_id, cable, cable_name, std_type, corridor_cable_ids,
            corridor_line_count, corridor_grouping_method, from_bus, to_bus,
            length_km, settlement_type, line_existing_duct_share, line_trenching_share,
            existing_parallel, max_i_from_ka, max_i_ka, installed_capacity_ka,
            loading_percent, required_parallel, additional_parallel,
            reinforcement_150_count, reinforcement_185_count,
            reinforcement_240_count, reinforcement_added_capacity_ka,
            reinforcement_catalog, requires_expansion,
            overloaded_at_100_percent, estimated_cost_eur,
            cost_eur_per_km, cost_basis, duct_cost_eur_per_km,
            reopen_cost_eur_per_km, critical_t_index, geom
        ) VALUES (
            :expansion_analysis_run_id, :real_powerflow_run_id, :real_grid_case_id,
            :scenario_id, :plz, :lv_id, :cable, :cable_name, :std_type, :corridor_cable_ids,
            :corridor_line_count, :corridor_grouping_method, :from_bus, :to_bus,
            :length_km, :settlement_type, :line_existing_duct_share, :line_trenching_share,
            :existing_parallel, :max_i_from_ka, :max_i_ka, :installed_capacity_ka,
            :loading_percent, :required_parallel, :additional_parallel,
            :reinforcement_150_count, :reinforcement_185_count,
            :reinforcement_240_count, :reinforcement_added_capacity_ka,
            :reinforcement_catalog, :requires_expansion,
            :overloaded_at_100_percent, :estimated_cost_eur,
            :cost_eur_per_km, :cost_basis, :duct_cost_eur_per_km,
            :reopen_cost_eur_per_km, :critical_t_index,
            CASE WHEN :geom_wkt IS NULL THEN NULL ELSE ST_GeomFromText(:geom_wkt, 25832) END
        )
        """
    )
    transformer_sql = text(
        """
        INSERT INTO surrogrid.expansion_real_transformer_result (
            expansion_analysis_run_id, real_powerflow_run_id, real_grid_case_id,
            scenario_id, plz, lv_id, transformer_rated_power_kva,
            transformer_equipment_name, max_s_mva, loading_percent,
            required_transformer_kva, additional_transformer_kva,
            requires_expansion, overloaded_at_100_percent, estimated_cost_eur,
            transformer_cost_basis, critical_t_index, geom
        ) VALUES (
            :expansion_analysis_run_id, :real_powerflow_run_id, :real_grid_case_id,
            :scenario_id, :plz, :lv_id, :transformer_rated_power_kva,
            :transformer_equipment_name, :max_s_mva, :loading_percent,
            :required_transformer_kva, :additional_transformer_kva,
            :requires_expansion, :overloaded_at_100_percent, :estimated_cost_eur,
            :transformer_cost_basis, :critical_t_index,
            CASE WHEN :geom_wkt IS NULL THEN NULL ELSE ST_GeomFromText(:geom_wkt, 25832) END
        )
        """
    )
    with db.engine.begin() as conn:
        conn.execute(status_sql, status_rows)
        if line_rows:
            conn.execute(line_sql, line_rows)
        if transformer_rows:
            conn.execute(transformer_sql, transformer_rows)
    return {
        "grid_status_rows": len(status_rows),
        "line_rows": len(line_rows),
        "transformer_rows": len(transformer_rows),
    }
