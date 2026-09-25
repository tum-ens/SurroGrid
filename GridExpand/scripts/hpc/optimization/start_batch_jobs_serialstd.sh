#!/bin/bash
#
# manager.sh
#
# Define your START and END here (inclusive), then loop over them.
# For example, START=0 and END=24 will submit INDEX=0,1,…,24.
#
START=$1
END=$2

for INDEX in $(seq "$START" "$END"); do
  echo "Submitting job for INDEX=$INDEX"
  sbatch run_cluster_serialstd.sh "$INDEX"
done