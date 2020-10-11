#!/bin/bash

EVAL_METHOD="zero_shot"
python contplay_full_eval_AGEM.py \
       --save_dir exps/iql_2p_3_ind_RB_${EVAL_METHOD}_AGEM \
       --method iql \
       --ll_algo AGEM \
       --use_wandb \
       --num_thread 10 \
       --load_learnable_model ../models/iql_2p_3.pthw \
       --load_fixed_model ../models/iql_2p_1.pthw ../models/iql_2p_4.pthw ../models/iql_2p_5.pthw \
       --num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 1 \
       --batchsize 128 \
       --burn_in_frames 5000 \
       --eval_burn_in_frames 1000 \
       --replay_buffer_size 131072 \
       --eval_replay_buffer_size 32768 \
       --epoch_len 200 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --eval_epoch_len 100 \
       --eval_method ${EVAL_METHOD} \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --act_device cuda:0,cuda:1 \
       --shuffle_color 0 \
