# UV Environment Setup (GridExpand)

This project can now be run with uv-managed environments per pipeline step.

## Prerequisites

- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Install required Python versions once:

```bash
uv python install 3.12
uv python install 3.10
```

## Step 1: Grid Sampling

```bash
cd GridExpand/1.grid_sampling
uv sync
uv run python gridreadout/export_single_grid.py --plz 80803 --list-candidates
```

## Step 2: Demand Allocation

```bash
cd GridExpand/2.demand_allocation
uv sync
cd gridalloc
uv run --project .. python main.py <inputfile_id> --n_cpu <N>
```

## Step 3: URBS Optimization

```bash
cd GridExpand/3.urbs
uv sync
uv run python run_urbs_cluster.py <inputfile_id> --n_cpu <N>
```

Notes:
- Step 3 still requires a working Gurobi installation/license.
- uv handles Python packages only; solver binaries/licenses are external.

## Step 4: Power Flow

```bash
cd GridExpand/4.powerflow
uv sync
uv run python run_pwrflw.py <inputfile_id> --n_cpu <N>
```

## Legacy Conda files

The original environment files remain available:
- `environment.yml`
- `environment_HPC.yml` (where present)

They are kept for backward compatibility but uv is now the preferred path.
