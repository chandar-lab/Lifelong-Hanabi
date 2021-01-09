#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=22:00:00
#SBATCH -o /scratch/akb/final_experiments/out_files/ewc/cont_eval/sgd_ewc_online_hard-cont-eval-%j.out

USER="akb"
LOAD_MODEL_DIR="/scratch/akb/final_experiments/final_model_pool_for_csv"
OPTIM_NAME="SGD"
SEED=10
python continual_evaluation.py \
       --weight_1_dir /scratch/${USER}/final_experiments/EWC/batch/${OPTIM_NAME}_EWC_online_hard_${SEED} \
       --weight_2 ${LOAD_MODEL_DIR}/sad_2p_2025.pthw ${LOAD_MODEL_DIR}/sad_op_2p_2506.pthw \
                  ${LOAD_MODEL_DIR}/sad_op_2p_2507.pthw ${LOAD_MODEL_DIR}/vdn_2p_7141.pthw \
                  ${LOAD_MODEL_DIR}/vdn_op_2p_7741.pthw ${LOAD_MODEL_DIR}/vdn_op_2p_7742.pthw \
                  ${LOAD_MODEL_DIR}/iql_2p_4.pthw \
       --num_player 2 \