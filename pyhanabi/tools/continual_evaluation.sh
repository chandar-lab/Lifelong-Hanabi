#!/bin/bash
# the last agent in --weight_2 is the pre-trained agent itself that was used for continual training
LOAD_MODEL_DIR=<path-to-pretrained-model-pool-dir>
python continual_evaluation.py \
       --weight_1_dir <path-to-ckpts-saved-during-continual-training> \
       --weight_2 ${LOAD_MODEL_DIR}/iql_2p_310.pthw ${LOAD_MODEL_DIR}/vdn_2p_720.pthw \
                  ${LOAD_MODEL_DIR}/vdn_2p_7140.pthw ${LOAD_MODEL_DIR}/iql_op_2p_710.pthw \
                  ${LOAD_MODEL_DIR}/vdn_op_2p_729.pthw ${LOAD_MODEL_DIR}/iql_2p_210.pthw \
       --num_player 2 \
