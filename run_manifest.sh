#!/bin/bash

gpu=${1:-6}
eval_set=${2:-valid}
manifest_name=${3:-exp_manifest}
result_log_name=${4:-results_log}

# Loop through the experiment manifest and run the experiments.

while read name lr dpt mid_dpt cvf cvs mid_cvs batch_size blocks growthrate epochs usel2 usetb opt scaling data earlystop patience RLR usesw; do
    if [ "$name" != 'name' ]
    then
        python main.py \
        -gpu=$gpu \
        -name=$name \
        -lr=$lr \
        -dpt=$dpt \
        -mid_dpt=$mid_dpt \
        -cvf=$cvf \
        -cvs=$cvs \
        -mid_cvs=$mid_cvs \
        -batch_size=$batch_size \
        -blocks=$blocks \
        -growthrate=$growthrate \
        -epochs=$epochs \
        -use_l2=$usel2 \
        -use_tb=$usetb \
        -opt=$opt \
        -scaling=$scaling \
        -data=$data \
        -earlystop=$earlystop \
        -patience=$patience \
        -RLR=$RLR \
        -use_sw=$usesw \
        -eval_set=$eval_set \
        -resfilename=$result_log_name
    fi
done < ./manifests/$manifest_name.txt

