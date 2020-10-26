#!/bin/bash

# Use this shell script to parallelize 02_calc_distance.py. Use offset to skip
# a number of indices (presumably because they've already been computed). Then
# deploy N_WORKERS processes to compute N_DIST_PER_WORKER distances each.

OFFSET=${1:-0}
N_WORKERS=${3:-20}
LAST_I=$(($N_WORKERS-1))
N_DIST_PER_WORKER=${2:-5000}

echo $LAST_I

for i in $(seq 0 $LAST_I)
do
    let "START_INDEX = $OFFSET + ($i * $N_DIST_PER_WORKER)"
    let "END_INDEX = $START_INDEX + $N_DIST_PER_WORKER - 1"
    echo "i: ${i}, start index: ${START_INDEX}, end index: ${END_INDEX}."
    python calc_distance.py -si=$START_INDEX -ei=$END_INDEX &
done