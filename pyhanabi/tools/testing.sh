#!/bin/bash
# the last agent in --weight_2 is the pre-trained agent itself that was used for continual training
LOAD_MODEL_DIR=<path-to-pretrained-model-pool-dir>
python testing.py \
       --weight_1_dir <path-to-ckpts-saved-during-continual-training> \
       --weight_2 ${LOAD_MODEL_DIR}/iql_2p_510.pthw ${LOAD_MODEL_DIR}/iql_2p_7.pthw \
                  ${LOAD_MODEL_DIR}/iql_op_2p_612.pthw ${LOAD_MODEL_DIR}/iql_op_2p_6140.pthw \
                  ${LOAD_MODEL_DIR}/vdn_aux_2p_941.pthw ${LOAD_MODEL_DIR}/vdn_aux_2p_970.pthw \
                  ${LOAD_MODEL_DIR}/sad_op_2p_1.pthw ${LOAD_MODEL_DIR}/sad_op_2p_2501.pthw \
                  ${LOAD_MODEL_DIR}/sad_aux_op_2p_1.pthw ${LOAD_MODEL_DIR}/sad_aux_op_2p_25001.pthw \
                  ${LOAD_MODEL_DIR}/sad_aux_2p_1.pthw ${LOAD_MODEL_DIR}/sad_aux_2p_20001.pthw \
                  ${LOAD_MODEL_DIR}/sad_2p_1.pthw ${LOAD_MODEL_DIR}/sad_2p_2006.pthw \
                  ${LOAD_MODEL_DIR}/iql_aux_2p_800.pthw ${LOAD_MODEL_DIR}/iql_aux_2p_811.pthw \
                  ${LOAD_MODEL_DIR}/vdn_2p_726.pthw ${LOAD_MODEL_DIR}/vdn_2p_740.pthw \
                  ${LOAD_MODEL_DIR}/vdn_op_2p_727.pthw ${LOAD_MODEL_DIR}/vdn_op_2p_77111.pthw \
       --num_player 2 \
