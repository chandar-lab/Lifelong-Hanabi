#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH -o /scratch/akb/final_experiments/out_files/ewc/cont_eval/sgd_ewc_offline_easy-cont-eval-%j.out

USER="akb"
LOAD_MODEL_DIR="/scratch/akb/final_experiments/final_model_pool_for_csv"
OPTIM_NAME="SGD"
python continual_evaluation.py \
       --weight_1_dir /scratch/${USER}/final_experiments/EWC/batch/${OPTIM_NAME}_EWC_offline_easy \
       --weight_2 ${LOAD_MODEL_DIR}/vdn_2p_720.pthw ${LOAD_MODEL_DIR}/sad_op_2p_1.pthw \
                  ${LOAD_MODEL_DIR}/vdn_2p_726.pthw ${LOAD_MODEL_DIR}/sad_2p_2001.pthw \
                  ${LOAD_MODEL_DIR}/vdn_2p_7140.pthw ${LOAD_MODEL_DIR}/iql_2p_210.pthw \
       --num_player 2 \
