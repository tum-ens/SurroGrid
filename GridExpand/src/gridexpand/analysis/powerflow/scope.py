"""Grid and run scope of the DB-backed power-flow loaders.

A loader reads either one concrete grid (``input_id``: AGS, ``AGS-<index>`` or a
bridge filename, resolved like the pipeline does) or a population narrowed by
scenario, AGS, PLZ, KCID and BCID.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from gridexpand.analysis.ids import optional_ags
from gridexpand.db.database import SurroGridDatabase


def resolve_db_grid(
    db: SurroGridDatabase,
    input_id: str,
    plz: int | None,
    kcid: int | None,
    bcid: int | None,
    candidate_index: int,
    min_buildings: int,
) -> dict:
    """Grid reference of one concrete grid (see ``SurroGridDatabase.resolve_grid_identifier``)."""
    return db.resolve_grid_identifier(
        input_id,
        plz=plz,
        kcid=kcid,
        bcid=bcid,
        candidate_index=candidate_index,
        min_buildings=min_buildings,
    )


def resolve_powerflow_run(
    db: SurroGridDatabase,
    grid_ref: dict,
    run_name: str,
    scenario_id: int | None = None,
) -> dict:
    """The newest power-flow run ``run_name`` of one grid (``db.find_powerflow_run``).

    Raises:
        ValueError: the grid has no grid case or no such run.
    """
    with db.engine.connect() as conn:
        grid_case_id = conn.execute(
            text(
                """
                SELECT grid_case_id
                FROM surrogrid.grid_case
                WHERE ags = :ags AND plz = :plz AND kcid = :kcid AND bcid = :bcid
                  AND pylovo_grid_result_id = :grid_result_id
                """
            ),
            {key: grid_ref[key] for key in ("ags", "plz", "kcid", "bcid", "grid_result_id")},
        ).scalar_one_or_none()
    run = (
        None
        if grid_case_id is None
        else db.find_powerflow_run(run_name=run_name, scenario_id=scenario_id, grid_case_id=int(grid_case_id))
    )
    if run is None:
        raise ValueError(
            f"No DB power-flow run named {run_name!r} found for "
            f"scenario_id={scenario_id!r}, PLZ={grid_ref['plz']}, "
            f"KCID={grid_ref['kcid']}, BCID={grid_ref['bcid']}."
        )
    return run


def scope_filter(run_column: str) -> str:
    """SQL condition of the loader scope (aliases ``pr`` = powerflow_run, ``gc`` = grid_case)."""
    return f"""(:run_id IS NULL OR {run_column} = :run_id)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
          AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
          AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)"""


def scope_params(
    db: SurroGridDatabase,
    *,
    input_id: str | None,
    run_name: str,
    scenario_id: int | None,
    ags: str | int | None,
    plz: int | None,
    kcid: int | None,
    bcid: int | None,
    candidate_index: int,
    min_buildings: int,
) -> dict[str, Any]:
    """Parameters of ``scope_filter``: one resolved run for ``input_id``, else the population filters."""
    run_id = None
    if input_id is not None:
        grid_ref = resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
        run_id = int(resolve_powerflow_run(db, grid_ref, run_name, scenario_id)["powerflow_run_id"])
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "ags": optional_ags(ags),
        "filter_plz": plz if input_id is None else None,
        "filter_kcid": kcid if input_id is None else None,
        "filter_bcid": bcid if input_id is None else None,
    }
