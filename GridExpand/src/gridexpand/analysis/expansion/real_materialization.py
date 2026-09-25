"""Expansion-cost rows for the real SWF and ÜZW grids (the Python path of ``heuristics``).

Real grids have no pylovo display lines, so parallel rows between the same bus pair
with lengths within 5 % form one cable corridor (capacity and peak current summed,
longest length). Grids with failed power-flow timesteps are ``incomplete`` (P100 cost
unknown); ``--exclude-real-lv-id`` grids are ``excluded``; both stay in the status table.

``prepare_real_results`` reads everything (summaries, grid files) without writing;
``insert_real_results`` writes the rows inside the caller's transaction.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from gridexpand.analysis.ids import canonical_real_grid_id

from .heuristics import (
    required_transformer_kva,
    select_cable_reinforcement,
    transformer_upgrade_cost,
)

if TYPE_CHECKING:
    import pandapower as pp

CORRIDOR_LENGTH_RELATIVE_TOLERANCE = 0.05
REAL_GRID_SOURCES = {"real_swf": "swf", "real_uzw": "uzw"}


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


@lru_cache(maxsize=256)
def _cached_real_net(path: str, mtime_ns: int) -> pp.pandapowerNet:
    import pandapower as pp

    source_file = Path(path)
    if source_file.suffix == ".json":
        return pp.from_json(source_file)
    return pp.from_excel(source_file)


def load_real_net(source_file: Path) -> pp.pandapowerNet:
    """The stored real grid (SWF Excel workbook or ÜZW pandapower JSON), cached per file version.

    The returned network is shared between callers and must not be modified.
    """
    return _cached_real_net(str(source_file), os.stat(source_file).st_mtime_ns)


def _settlement_types(db, plzs: list[int], version_id: str | None) -> dict[int, int | None]:
    if version_id is None or not plzs:
        return {plz: None for plz in plzs}
    query = text(
        """
        SELECT DISTINCT ON (postcode_result_plz) postcode_result_plz, settlement_type
        FROM pylovo.postcode_result
        WHERE postcode_result_plz = ANY(:plzs)
          AND version_id = :version_id
          AND settlement_type IS NOT NULL
        ORDER BY postcode_result_plz
        """
    )
    with db.engine.connect() as conn:
        found = {
            int(row["postcode_result_plz"]): int(row["settlement_type"])
            for row in conn.execute(query, {"plzs": plzs, "version_id": str(version_id)}).mappings()
        }
    return {plz: found.get(plz) for plz in plzs}


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
        for candidate in sorted(candidates, key=lambda row: (row["length_km"], row["cable"])):
            matching = next(
                (
                    group
                    for group in route_groups
                    if all(_same_recorded_corridor(candidate, member) for member in group)
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
            rgc.source,
            rgc.plz,
            rgc.lv_id,
            rgc.source_file,
            rpr.assumptions ->> 'pylovo_version_id' AS pylovo_version_id,
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
          AND rgc.source = :source
          AND (CAST(:plz AS INTEGER[]) IS NULL OR rgc.plz = ANY(CAST(:plz AS INTEGER[])))
        ORDER BY LENGTH(rgc.lv_id), rgc.lv_id
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
                "source": REAL_GRID_SOURCES[args.data_source],
                "plz": [int(value) for value in args.plz] if args.plz else None,
            },
        )


def _assumption(db, key: str) -> dict[str, Any]:
    with db.engine.connect() as conn:
        row = (
            conn.execute(
                text("SELECT * FROM surrogrid.expansion_cost_assumption WHERE assumption_key = :key"),
                {"key": key},
            )
            .mappings()
            .one()
        )
    return dict(row)


def _critical_indices(db, run_ids: list[int], stage: str) -> dict[int, dict[tuple[str, int], int]]:
    """Time index of each asset's largest upper-tail value, per run (ties: earliest t_index).

    Step 4 labels the tails ``p99_upper`` / ``p01_lower`` (``powerflow.engine``).
    """
    by_run: dict[int, dict[tuple[str, int], int]] = {run_id: {} for run_id in run_ids}
    if not run_ids:
        return by_run
    query = text(
        """
        SELECT DISTINCT ON (real_powerflow_run_id, metric, asset_id)
            real_powerflow_run_id, metric, asset_id, t_index
        FROM surrogrid.real_powerflow_tail_value
        WHERE real_powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
          AND tail LIKE '%upper'
        ORDER BY real_powerflow_run_id, metric, asset_id, value DESC, t_index
        """
    )
    with db.engine.connect() as conn:
        for row in conn.execute(query, {"run_ids": run_ids, "stage": stage}).mappings():
            by_run[int(row["real_powerflow_run_id"])][(str(row["metric"]), int(row["asset_id"]))] = int(
                row["t_index"]
            )
    return by_run


def _cable_summaries(db, run_ids: list[int], stage: str) -> dict[int, list[dict[str, Any]]]:
    """Cable summary rows per run, ordered by cable."""
    by_run: dict[int, list[dict[str, Any]]] = {run_id: [] for run_id in run_ids}
    if not run_ids:
        return by_run
    query = text(
        """
        SELECT *
        FROM surrogrid.real_powerflow_cable_summary
        WHERE real_powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
        ORDER BY real_powerflow_run_id, cable
        """
    )
    with db.engine.connect() as conn:
        for row in conn.execute(query, {"run_ids": run_ids, "stage": stage}).mappings():
            by_run[int(row["real_powerflow_run_id"])].append(dict(row))
    return by_run


def _grid_status(run: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    """Status row of one real grid: ``excluded``, ``incomplete`` (failed timesteps) or ``complete``."""
    lv_id = canonical_real_grid_id(run["lv_id"])
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
    return {
        "real_powerflow_run_id": int(run["real_powerflow_run_id"]),
        "real_grid_case_id": int(run["real_grid_case_id"]),
        "scenario_id": int(run["scenario_id"]),
        "plz": None if pd.isna(run["plz"]) else int(run["plz"]),
        "lv_id": lv_id,
        "n_timesteps": int(run["n_timesteps"]),
        "n_failed_timesteps": failed,
        "cost_status": cost_status,
        "status_reason": reason,
    }


def _prepare_cables(
    net: pp.pandapowerNet,
    summaries: list[dict[str, Any]],
    critical: dict[tuple[str, int], int],
    *,
    lv_id: str,
    source_file: Path,
) -> list[dict[str, Any]]:
    """One record per summarized cable: endpoints, length, installed capacity and peak current."""
    prepared = []
    for cable in summaries:
        cable_id = int(cable["cable"])
        if cable_id not in net.line.index:
            raise KeyError(f"Cable {cable_id} from real summary is absent in {source_file}.")
        line = net.line.loc[cable_id]
        existing_parallel = max(int(round(_finite(cable.get("cable_parallel"), 1.0) or 1.0)), 1)
        max_i_ka = _finite(cable.get("cable_max_i_ka"))
        installed_capacity_ka = _finite(cable.get("cable_installed_capacity_ka"))
        loading_percent = _finite(cable.get("cable_loading_max_time_percent"))
        if max_i_ka is None and installed_capacity_ka is not None:
            max_i_ka = installed_capacity_ka / existing_parallel
        if installed_capacity_ka is None and max_i_ka is not None:
            installed_capacity_ka = max_i_ka * existing_parallel
        if max_i_ka is None or max_i_ka <= 0 or installed_capacity_ka is None or loading_percent is None:
            raise ValueError(
                f"Real cable {lv_id}:{cable_id} lacks a finite installed capacity or P100 loading."
            )
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        prepared.append(
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
    return prepared


def _corridor_row(
    corridor: list[dict[str, Any]],
    status: dict[str, Any],
    net: pp.pandapowerNet,
    *,
    settlement_type: int | None,
    assumption: dict[str, Any],
    duct_share_override: float | None,
) -> dict[str, Any]:
    """Line result row of one cable corridor."""
    representative = min(corridor, key=lambda row: row["cable"])
    cable_ids = sorted(row["cable"] for row in corridor)
    installed_capacity_ka = sum(row["installed_capacity_ka"] for row in corridor)
    max_i_from_ka = sum(row["max_i_from_ka"] for row in corridor)
    existing_parallel = sum(row["existing_parallel"] for row in corridor)
    max_i_ka = installed_capacity_ka / existing_parallel
    loading_percent = max_i_from_ka / installed_capacity_ka * 100.0
    required_added_capacity_ka = max(max_i_from_ka - installed_capacity_ka, 0.0)
    length_km = max(row["length_km"] for row in corridor)
    costs = select_cable_reinforcement(
        required_added_capacity_ka=required_added_capacity_ka,
        settlement_type=settlement_type,
        length_km=length_km,
        assumption=assumption,
        duct_share_override=duct_share_override,
    )
    additional_parallel = (
        costs["reinforcement_150_count"] + costs["reinforcement_185_count"] + costs["reinforcement_240_count"]
    )
    critical_indices = {row["critical_t_index"] for row in corridor if row["critical_t_index"] is not None}
    return {
        **status,
        "cable": representative["cable"],
        "cable_name": " | ".join(row["cable_name"] for row in corridor),
        "std_type": " | ".join(sorted({row["std_type"] for row in corridor})),
        "corridor_cable_ids": "|".join(map(str, cable_ids)),
        "corridor_line_count": len(corridor),
        "corridor_grouping_method": (
            "same_bus_pair_length_within_5pct" if len(corridor) > 1 else "single_line"
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
        "critical_t_index": next(iter(critical_indices)) if len(critical_indices) == 1 else None,
        "geom_wkt": _line_wkt(net, representative["from_bus"], representative["to_bus"]),
        **costs,
    }


def _transformer_row(
    run: dict[str, Any],
    status: dict[str, Any],
    net: pp.pandapowerNet,
    *,
    assumption: dict[str, Any],
    critical: dict[tuple[str, int], int],
) -> dict[str, Any] | None:
    """Transformer result row of one grid, or None for the ÜZW area without a rating."""
    lv_id = status["lv_id"]
    rated_kva = (_finite(run["transformer_s_rated_mva"]) or 0.0) * 1000.0
    loading_percent = _finite(run["trafo_loading_max_time_percent"])
    if rated_kva <= 0 or loading_percent is None:
        if run["source"] == "uzw" and rated_kva <= 0:
            # One ÜZW station (area 113) is delivered without a rating: its
            # cables are costed, its transformer is not.
            print(
                f"Warning: real uzw grid {lv_id} has no transformer rating; "
                "transformer loading and cost are omitted."
            )
            return None
        raise ValueError(
            f"Real {run['source']} grid {lv_id} lacks a finite transformer rating or P100 loading."
        )
    max_s_mva = loading_percent / 100.0 * rated_kva / 1000.0
    required_kva = max(
        rated_kva, required_transformer_kva(max_s_mva, assumption["transformer_capacity_step_kva"])
    )
    transformer_cost, transformer_basis = transformer_upgrade_cost(required_kva, rated_kva, assumption)
    equipment_name = None
    if not net.trafo.empty:
        equipment_name = str(net.trafo.iloc[0].get("name") or net.trafo.iloc[0].get("std_type") or "")
    return {
        **status,
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


@dataclass
class RealResults:
    """Rows of one real-grid analysis (without ``expansion_analysis_run_id``)."""

    status_rows: list[dict[str, Any]] = field(default_factory=list)
    line_rows: list[dict[str, Any]] = field(default_factory=list)
    transformer_rows: list[dict[str, Any]] = field(default_factory=list)
    scenario_ids: set[int] = field(default_factory=set)


def _pylovo_version(args, runs: pd.DataFrame) -> str | None:
    recorded_versions = set(runs["pylovo_version_id"].dropna().astype(str))
    if args.pylovo_version_id is not None:
        return str(args.pylovo_version_id)
    if len(recorded_versions) == 1:
        return next(iter(recorded_versions))
    print(
        "Warning: no unique pylovo version for the real run "
        f"(recorded: {sorted(recorded_versions)}); settlement type is unknown "
        "and the semiurban reopening cost is used. Pass --pylovo-version-id."
    )
    return None


def prepare_real_results(db, args) -> RealResults:
    """Compute all status, corridor and transformer rows of one real-grid analysis (reads only).

    Raises:
        RuntimeError: no matching real power-flow summary.
        FileNotFoundError, KeyError, ValueError: a grid file or a rating is missing.
    """
    runs = _selected_runs(db, args)
    if runs.empty:
        raise RuntimeError(
            f"No {args.data_source} power-flow summaries match the requested expansion scope."
        )
    assumption = _assumption(db, args.assumption_key)
    excluded = {canonical_real_grid_id(value) for value in (args.exclude_real_lv_id or [])}
    settlement_by_plz = _settlement_types(
        db, [int(plz) for plz in runs["plz"].dropna().astype(int).unique()], _pylovo_version(args, runs)
    )
    records = runs.to_dict("records")
    statuses = [_grid_status(run, excluded) for run in records]
    complete_ids = [s["real_powerflow_run_id"] for s in statuses if s["cost_status"] == "complete"]
    critical_by_run = _critical_indices(db, complete_ids, args.stage)
    cables_by_run = _cable_summaries(db, complete_ids, args.stage)

    results = RealResults(scenario_ids={s["scenario_id"] for s in statuses})
    for run, status in zip(records, statuses):
        results.status_rows.append(status)
        if status["cost_status"] != "complete":
            continue
        source_file = Path(str(run["source_file"]))
        if not source_file.exists():
            raise FileNotFoundError(f"Real grid source does not exist: {source_file}")
        net = load_real_net(source_file)
        run_id = status["real_powerflow_run_id"]
        critical = critical_by_run[run_id]
        settlement_type = settlement_by_plz.get(int(run["plz"])) if not pd.isna(run["plz"]) else None
        cables = _prepare_cables(
            net, cables_by_run[run_id], critical, lv_id=status["lv_id"], source_file=source_file
        )
        for corridor in _corridor_groups(cables):
            results.line_rows.append(
                _corridor_row(
                    corridor,
                    status,
                    net,
                    settlement_type=settlement_type,
                    assumption=assumption,
                    duct_share_override=args.line_existing_duct_share,
                )
            )
        transformer = _transformer_row(run, status, net, assumption=assumption, critical=critical)
        if transformer is not None:
            results.transformer_rows.append(transformer)
    return results


STATUS_SQL = text(
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
LINE_SQL = text(
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
TRANSFORMER_SQL = text(
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


def insert_real_results(conn: Connection, expansion_analysis_run_id: int, results: RealResults) -> dict[str, int]:
    """Write the rows of ``results`` under one analysis id (caller's transaction)."""
    for statement, rows in (
        (STATUS_SQL, results.status_rows),
        (LINE_SQL, results.line_rows),
        (TRANSFORMER_SQL, results.transformer_rows),
    ):
        if rows:
            conn.execute(
                statement,
                [{**row, "expansion_analysis_run_id": expansion_analysis_run_id} for row in rows],
            )
    return {
        "grid_status_rows": len(results.status_rows),
        "line_rows": len(results.line_rows),
        "transformer_rows": len(results.transformer_rows),
    }
