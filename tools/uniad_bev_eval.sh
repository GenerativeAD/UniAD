#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
SAVE_BEV_FEAT_DIR=${4:-"workspace/bev_similarity"}  # Default save directory
START_FRAME=${5:-3}                                # Start frame index
END_FRAME=${6:-3}                                  # End frame index
SCENE_NAME=${7:-"scene-0014"}                      # Scene name to process (e.g., scene-0014)
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test_bev.py \
    $CFG \
    $CKPT \
    ${@:8} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    --start_frame ${START_FRAME} \
    --end_frame ${END_FRAME} \
    --save_bev_feat_dir ${SAVE_BEV_FEAT_DIR} \
    --scene_name ${SCENE_NAME} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T
