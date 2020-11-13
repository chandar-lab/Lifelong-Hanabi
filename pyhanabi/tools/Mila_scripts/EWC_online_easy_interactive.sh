#!/bin/bash
## specify optim_name to be either Adam or SGD.
## specify --decay_lr for learning rate decay.
## dropout_p should be 0 for no dropout. dropout_p is drop probability.
USER="akb"
EVAL_METHOD="few_shot"
LOAD_MODEL_DIR="../models/iql_2p"
INITIAL_LR=0.02
BATCH_SIZE=128
python cont_EWC.py \
       --save_dir /miniscratch/${USER}/cont_hanabi_models/exps/ind_RB_${EVAL_METHOD}_EWC_online_noeval_easy \
       --load_model_dir ${LOAD_MODEL_DIR} \
       --method iql \
       --ll_algo EWC \
       --load_learnable_model ${LOAD_MODEL_DIR}/iql_2p_5.pthw \
       --load_fixed_model ${LOAD_MODEL_DIR}/iql_2p_6.pthw ${LOAD_MODEL_DIR}/iql_2p_11.pthw ${LOAD_MODEL_DIR}/iql_2p_113.pthw ${LOAD_MODEL_DIR}/iql_2p_210.pthw \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --eval_num_thread 10 \
       --eval_num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 1 \
       --online 1 \
       --ewc_lambda 5000 \
       --ewc_gamma 1 \
       --initial_lr ${INITIAL_LR} \
       --final_lr 6.25e-05 \
       --lr_gamma 0.2 \
       --dropout_p 0 \
       --sgd_momentum 0.8 \
       --batchsize ${BATCH_SIZE} \
       --burn_in_frames 10000 \
       --eval_burn_in_frames 1000 \
       --replay_buffer_size 32768 \
       --eval_replay_buffer_size 10000 \
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
