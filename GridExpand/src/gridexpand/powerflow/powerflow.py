"""Time-series power flow entry points (raw tables, summaries) of the real-grid runners.

``pf`` returns the raw tables and ``pf_summary`` the compact summary of one demand
frame. The implementation lives in ``engine`` (solves, matrices, summary) and
``network`` (grid preparation, scopes).
"""

from __future__ import annotations

from gridexpand.powerflow.engine import raw_tables, run_timeseries, summarize
from gridexpand.powerflow.network import (  # noqa: F401  (used as pwrflw.* by the real-grid runners)
    comparison_backbone_scope,
    comparison_evaluation_scope,
)


def pf(grid, df, parallel, n_cpu):
    """Raw tables ``(demand_import, vm, line_loads)`` of ``df`` (bfsw, raises on failure).

    ``n_cpu`` time chunks are solved in parallel processes when ``parallel``.
    """
    matrices = run_timeseries(grid, df, n_workers=int(n_cpu) if parallel else 1)
    return raw_tables(matrices)


def pf_summary(
    grid,
    df,
    transformer_s_rated_mva,
    cable_max_i_ka,
    voltage_buses,
    algorithm="bfsw",
    cable_ids=None,
    on_nonconvergence="raise",
    protect_grid_state=False,
    n_workers=1,
):
    """Run power flow and return compact violation-hour and percentile metrics.

    ``on_nonconvergence="nan"`` keeps the annual summary running and records
    failed timesteps as missing values. The default stays strict and raises.
    """
    matrices = run_timeseries(
        grid,
        df,
        algorithm=algorithm,
        on_nonconvergence=on_nonconvergence,
        n_workers=n_workers,
        protect_grid_state=protect_grid_state,
    )
    return summarize(
        grid,
        matrices,
        transformer_s_rated_mva=transformer_s_rated_mva,
        cable_max_i_ka=cable_max_i_ka,
        voltage_buses=voltage_buses,
        cable_ids=cable_ids,
    )
