# Forchheim Expansion Pipeline Commands

These commands document the residential HH-only Forchheim runs used for the
pre, post-flex, and post-no-flex expansion comparison.

Run from the repository root:

```bash
cd /home/breveron/git/github/SurroGrid
```

## Pre + Post-Flex

Runs residential HH-only Step 2, TSAM Step 3 with URBS optimization, Step 4
summary power flow, and materializes both `pre` and `post`.

```bash
uv run --project GridExpand/2.demand_allocation python GridExpand/runme/synthetic_ags_pipeline_runner.py \
  --repo-root /home/breveron/git/github/SurroGrid \
  --ags 9474126 \
  --profiles all \
  --demand-scope residential \
  --powerflow-output summary \
  --tsam \
  --workers 6 \
  --step2-cpus 1 \
  --step4-cpus 1 \
  --run-dir GridExpand/run_logs/forchheim_post_flex_hh_tsam_$(date -u +%Y%m%dT%H%M%SZ)
```

To run post-flex and post-no-flex in one pass, add:

```bash
--include-no-flex-powerflow
```

Creates:

```text
baseline_static_hh_only_post_electrification_summary_powerflow
full_year_post_electrification_hh_only_tsam_pre
full_year_post_electrification_hh_only_tsam_post
```

## Post-No-Flex

Runs residential HH-only Step 2 and Step 3 optimization to obtain post-flex
heat-pump and auxiliary-heater capacities, then runs fixed/no-flex post power
flow and materializes `pre` and `post_no_flex`.

```bash
uv run --project GridExpand/2.demand_allocation python GridExpand/runme/synthetic_ags_pipeline_runner.py \
  --repo-root /home/breveron/git/github/SurroGrid \
  --ags 9474126 \
  --profiles all \
  --demand-scope residential \
  --powerflow-output summary \
  --no-flex-only \
  --tsam \
  --workers 6 \
  --step2-cpus 1 \
  --step4-cpus 1 \
  --run-dir GridExpand/run_logs/forchheim_post_no_flex_hh_tsam_$(date -u +%Y%m%dT%H%M%SZ)
```

Creates:

```text
baseline_static_hh_only_post_electrification_summary_no_flex_powerflow
full_year_post_electrification_hh_only_tsam_pre
full_year_post_electrification_hh_only_tsam_post_no_flex
```

## Rerun Failed Candidates

Use this only with an existing `--run-dir` from a failed or incomplete run.
`--resume --rerun-failed` should not be used for a clean first run.

```bash
uv run --project GridExpand/2.demand_allocation python GridExpand/runme/synthetic_ags_pipeline_runner.py \
  --repo-root /home/breveron/git/github/SurroGrid \
  --ags 9474126 \
  --profiles all \
  --demand-scope residential \
  --powerflow-output summary \
  --tsam \
  --workers 6 \
  --step2-cpus 1 \
  --step4-cpus 1 \
  --resume \
  --rerun-failed \
  --run-dir GridExpand/run_logs/<EXISTING_RUN_DIR>
```

For no-flex reruns, add:

```bash
--no-flex-only
```
