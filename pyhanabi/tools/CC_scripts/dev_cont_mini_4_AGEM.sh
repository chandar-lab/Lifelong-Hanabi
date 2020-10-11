#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH -o /scratch/akb/agem_4_fewshot-%j.out

export PYTHONPATH=/home/akb/CMAL_Hanabi:$PYTHONPATH
export OMP_NUM_THREADS=1

EVAL_METHOD="few_shot"
python contplay_full_eval_AGEM.py \
       --save_dir exps/iql_2p_4_ind_RB_${EVAL_METHOD}_AGEM \
       --method iql \
       --ll_algo AGEM \
       --use_wandb \
       --run_wandb_offline \
       --num_thread 10 \
       --load_learnable_model ../models/iql_2p_3.pthw \
       --load_fixed_model ../models/iql_2p_4.pthw ../models/iql_2p_5.pthw ../models/iql_2p_11.pthw ../models/iql_2p_204.pthw \
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
       --eval_epoch_len 50 \
       --eval_method ${EVAL_METHOD} \
       --eval_freq 50 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --act_device cuda:0,cuda:1 \
       --shuffle_color 0 \
