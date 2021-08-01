#!/bin/bash
# This script can be used to plot p(a_t | a_{t-1}) of an agent
# continually trained with some partners after each task
# These plots are based on 1000 games generated through Self-play

# Please create a folder with all the models. e.g. ${BASE_DIR}/Adam-EWC
BASE_DIR=<base_dir>
CONFIG_FILE=iql_2p_210.txt
SAVE_DIR=<save_dir>
tasks=(1 2 3 4 5 6 7 8 9)

for task in ${tasks[@]}
do
  python tools/analyze_policies.py \
      --weight_file ${BASE_DIR}/Adam-EWC/Adam-EWC-t${task}_zero_shot.pthw \
      --config_file ${BASE_DIR}${CONFIG_FILE} \
      --save_fig ${SAVE_DIR}
done