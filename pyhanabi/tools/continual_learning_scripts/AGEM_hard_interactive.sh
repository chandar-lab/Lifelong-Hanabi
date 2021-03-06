#!/bin/bash
## specify optim_name to be either Adam or SGD.
## specify --decay_lr for learning rate decay.
## dropout_p should be 0 for no dropout. dropout_p is drop probability.
OPTIM_NAME="SGD"
SEED=10
EVAL_METHOD="few_shot"
LOAD_MODEL_DIR=<path-to-pretrained-model-pool-dir>
SAVE_DIR=<path-to-save-dir>
INITIAL_LR=0.02
BATCH_SIZE=32
python continual_training.py \
       --save_dir ${SAVE_DIR} \
       --load_model_dir ${LOAD_MODEL_DIR} \
       --method iql \
       --ll_algo AGEM \
       --load_learnable_model ${LOAD_MODEL_DIR}/iql_2p_210.pthw \
       --load_partner_model ${LOAD_MODEL_DIR}/vdn_op_2p_7771.pthw ${LOAD_MODEL_DIR}/vdn_2p_726.pthw \
                          ${LOAD_MODEL_DIR}/vdn_2p_7140.pthw ${LOAD_MODEL_DIR}/iql_op_2p_600.pthw \
                          ${LOAD_MODEL_DIR}/vdn_2p_720.pthw \
       --num_thread 10 \
       --num_game_per_thread 80 \
       --eval_num_thread 10 \
       --eval_num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed ${SEED} \
       --initial_lr ${INITIAL_LR} \
       --final_lr 6.25e-05 \
       --lr_gamma 0.2 \
       --dropout_p 0 \
       --sgd_momentum 0.8 \
       --optim_name ${OPTIM_NAME} \
       --batchsize ${BATCH_SIZE} \
       --max_train_steps 200000000 \
       --max_eval_steps 500000 \
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
       --rnn_type lstm \
       --num_fflayer 1 \
       --num_rnn_layer 2 \
       --rnn_hid_dim 512 \
       --act_device cuda:1,cuda:2 \
       --shuffle_color 0 \
