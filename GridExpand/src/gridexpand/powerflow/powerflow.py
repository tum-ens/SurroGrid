"""Time-series power flow entry points (raw tables, summaries) for Step 4.

``pf`` returns the raw tables and ``pf_summary`` the compact summary of one demand
frame; ``pf_outputs`` returns both from a single pass. The implementation lives in
``engine`` (solves, matrices, summary) and ``network`` (grid preparation, scopes);
the names below are kept for the Step 5 tools that import this module.
"""

from __future__ import annotations

from gridexpand.powerflow.engine import (  # noqa: F401  (re-exported)
    PowerflowMatrices,
    _safe_nanmax,
    _safe_nanpercentile,
    _tail_values_frame,
    _transformer_import_diagnostic_frame,
    annual_boundary_diagnostic,
    raw_tables,
    run_timeseries,
    summarize,
    summarize_powerflow_matrices,
)
from gridexpand.powerflow.network import (  # noqa: F401  (re-exported)
    active_line_index as _active_line_index,
    comparison_backbone_scope,
    comparison_evaluation_scope,
    grid_adjacency as _grid_adjacency,
    parent_tree_from_root as _parent_tree_from_root,
    prepare_synthetic_grid as prepare_grid,
    root_bus as _root_bus,
    set_scenario_load_buses,
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
