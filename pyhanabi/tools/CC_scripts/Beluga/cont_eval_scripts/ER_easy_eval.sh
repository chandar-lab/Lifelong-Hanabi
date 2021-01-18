#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH -o /scratch/akb/final_experiments/out_files/er/cont_eval/sgd_er_easy-cont-eval-%j.out

USER="akb"
LOAD_MODEL_DIR="/scratch/akb/final_experiments/final_model_pool_for_csv"
OPTIM_NAME="SGD"
SEED=10
python continual_evaluation.py \
       --weight_1_dir /scratch/${USER}/final_experiments/ER/batch/${OPTIM_NAME}_ER_easy_${SEED} \
       --weight_2 ${LOAD_MODEL_DIR}/iql_2p_310.pthw ${LOAD_MODEL_DIR}/vdn_2p_720.pthw \
                  ${LOAD_MODEL_DIR}/vdn_2p_7140.pthw ${LOAD_MODEL_DIR}/iql_op_2p_710.pthw \
                  ${LOAD_MODEL_DIR}/vdn_op_2p_729.pthw ${LOAD_MODEL_DIR}/iql_2p_210.pthw \
       --num_player 2 \
