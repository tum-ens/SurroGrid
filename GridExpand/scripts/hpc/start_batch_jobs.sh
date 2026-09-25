#!/bin/bash
#
# TEMPLATE: submit one Slurm job per input index for one pipeline step.
#
#   scripts/hpc/start_batch_jobs.sh <allocation|optimization|powerflow> <START> <END>
#
# Run from GridExpand/ after `uv sync`. START and END are inclusive, e.g.
# 0 24 submits INDEX=0,1,...,24. Each job runs the matching
# scripts/hpc/<step>/run_cluster_serialstd.sh; adapt its #SBATCH header to your site.
set -euo pipefail
STEP=$1
START=$2
END=$3
SCRIPT="$(dirname "$0")/$STEP/run_cluster_serialstd.sh"
[ -f "$SCRIPT" ] || { echo "unknown step: $STEP" >&2; exit 2; }
mkdir -p work/runs/slurm

for INDEX in $(seq "$START" "$END"); do
  echo "Submitting $STEP job for INDEX=$INDEX"
  sbatch "$SCRIPT" "$INDEX"
done
