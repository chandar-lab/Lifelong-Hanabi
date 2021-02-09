#!/bin/bash
python selfplay.py \
       --save_dir <save_dir> \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --method iql \
       --pred_weight 0 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 25000 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 65536 \
       --epoch_len 1000 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --num_player 2 \
       --num_fflayer 1 \
       --rnn_type lstm \
       --rnn_hid_dim 512 \
       --num_rnn_layer 2 \
       --shuffle_color 0 \
       --multi_step 3 \
       --act_device cuda:1,cuda:2 \
