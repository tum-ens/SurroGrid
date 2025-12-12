#!/bin/bash
#SBATCH --job-name=mlp_hpo
#SBATCH --output=logs/out/%x-%j.out
#SBATCH --error=logs/error/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

##SBATCH --job-name=mlp_hpo
##SBATCH --output=logs/out/%x-%j.out
##SBATCH --error=logs/error/%x-%j.err
##SBATCH --time=00:30:00
##SBATCH --partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=14
##SBATCH --gres=gpu:1
##SBATCH --mem=128G

# Positional parameters with defaults:
#   $1 -> agg_hours (default: 1)
#   $2 -> loss_type (default: alpha_peak)
AGG_HOURS=${1:-1}
LOSS_TYPE=${2:-alpha_peak}

echo "Running HPO with: agg_hours=${AGG_HOURS}, loss_type=${LOSS_TYPE}"



srun	--container-mounts=/dss/dsshome1/05/ge96ton2/GridForecast/3_transformer/:/workspace,\
/dss/dsshome1/05/ge96ton2/GridForecast/3_transformer/data:/data \
	    --container-image='/dss/dssfs04/lwp-dss-0002/pn98cu/pn98cu-dss-0000/EliasH/containers/nvidia+pytorch+25.06-py3.sqsh' \
	    python run_transformer_model_HPO.py --agg-hours "${AGG_HOURS}" --loss-type "${LOSS_TYPE}"