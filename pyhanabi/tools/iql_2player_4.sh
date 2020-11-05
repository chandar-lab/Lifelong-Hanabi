#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH -o /scratch/akb/iql_seed-%j.out

USER="akb"
python selfplay.py \
       --save_dir /scratch/${USER}/iql_2p_608 \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 608 \
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
       --rnn_hid_dim 256 \
       --num_rnn_layer 2 \
       --shuffle_color 1 \
       --multi_step 3 \
       --act_device cuda:1,cuda:2 \
