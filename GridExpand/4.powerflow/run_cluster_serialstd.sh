#!/bin/bash

#SBATCH -J test_run
#SBATCH --output=logs/normal/%j_output.log
#SBATCH --error=logs/errors/%j_error.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --mem-per-cpu=6200M


module load miniconda3/24.7.1
module list

eval "$(conda shell.bash hook)"
conda activate pwrflw-hpc
conda env list

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

INDEX="$1"
echo "Fileindex: $INDEX"

srun python3 run_pwrflw.py $INDEX --n_cpu $SLURM_CPUS_PER_TASK
wait

### Delete error log file at end of run if it is empty
ERR_FILE="logs/errors/${SLURM_JOB_ID}_error.log"

if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
  rm "$ERR_FILE"
fi