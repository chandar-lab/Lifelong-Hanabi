#!/bin/bash
EVAL_METHOD="few_shot"
LOAD_MODEL_DIR="../models/iql_2p"
python contplay_full_eval_ER_noeval.py \
       --save_dir /network/tmp1/badrinaa/hanabi_sad_models/exps/iql_2p_ind_RB_${EVAL_METHOD}_None_noeval_easy \
       --load_model_dir ${LOAD_MODEL_DIR} \
       --method iql \
       --ll_algo None \
       --load_learnable_model ${LOAD_MODEL_DIR}/iql_2p_5.pthw \
       --load_fixed_model ${LOAD_MODEL_DIR}/iql_2p_6.pthw ${LOAD_MODEL_DIR}/iql_2p_11.pthw ${LOAD_MODEL_DIR}/iql_2p_113.pthw ${LOAD_MODEL_DIR}/iql_2p_210.pthw \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --eval_num_thread 10 \
       --eval_num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 1 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --eval_burn_in_frames 1000 \
       --replay_buffer_size 32768 \
       --eval_replay_buffer_size 32768 \
       --epoch_len 200 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --eval_epoch_len 50 \
       --eval_method ${EVAL_METHOD} \
       --eval_freq 25 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --act_device cuda:1,cuda:2 \
       --shuffle_color 0 \
