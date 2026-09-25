#!/bin/bash
#
# TEMPLATE Slurm job for Step 2 (demand allocation) of one input file:
#
#   sbatch scripts/hpc/allocation/run_cluster_serialstd.sh <inputfile_id>
#
# Submit from GridExpand/ (or use scripts/hpc/start_batch_jobs.sh). The Python
# environment comes from `uv sync`; adapt cluster, partition and resources to
# your site. Directories follow gridexpand.paths (set GRIDEXPAND_WORK_DIR to use
# a scratch file system).

#SBATCH -J gridexpand_allocate
#SBATCH --output=work/runs/slurm/%j_output.log
#SBATCH --error=work/runs/slurm/%j_error.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-06:00:00
#SBATCH --mem-per-cpu=6200M

module list

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

INDEX="$1"
echo "Fileindex: $INDEX"

srun uv run --frozen gridexpand allocate "$INDEX" --n_cpu "$SLURM_CPUS_PER_TASK"
wait

### Delete error log file at end of run if it is empty
ERR_FILE="work/runs/slurm/${SLURM_JOB_ID}_error.log"

if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
  rm "$ERR_FILE"
fi
