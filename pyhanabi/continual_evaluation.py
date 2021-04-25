""" Evaluating all the checkpoints saved periodically during train args.eval_freq
Requires only 1 GPU.
Sample usage: 
python continual_evaluation.py 
--weight_1_dir <path-to-saved-ckpts-dir> 
--weight_2 <list of ckpts separated by a space> i.e a.pthw b.pthw ...
--num_player 2
note the last arg of --weight_2 is the self-play agent that is the agent that was being trained in continual fashion...
"""

import argparse
import os
import sys
import glob
import wandb
import json

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import numpy as np
import torch
import utils
from eval import evaluate_legacy_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_1_dir", default=None, type=str, required=True)
    parser.add_argument("--weight_2", default=None, type=str, nargs="+", required=True)
    parser.add_argument("--is_rand", action="store_true", default=True)
    parser.add_argument("--num_player", default=None, type=int, required=True)
    args = parser.parse_args()

    test_dir = f"{args.weight_1_dir}_test_models"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    cont_train_args_txt = glob.glob(f"{args.weight_1_dir}/*.txt")

    # move cont_args.txt to test_dir
    move_cont_args = f"cp {cont_train_args_txt[0]} {test_dir}/"
    os.system(move_cont_args)

    with open(cont_train_args_txt[0], "r") as f:
        agent_args = {**json.load(f)}

    ## move learnable model to test_dir
    if agent_args["load_learnable_model"] != "":
        initial_learnable_model = agent_args["load_learnable_model"]
        move_model_0 = (
            f"cp {initial_learnable_model} {test_dir}/model_epoch0_zero_shot.pthw"
        )
        os.system(move_model_0)

    exp_name = agent_args["save_dir"].split("/")[-1]
    wandb.init(project="Lifelong_Hanabi_project", name=exp_name)
    wandb.config.update(agent_args)

    assert os.path.exists(args.weight_1_dir)
    weight_1 = []
    weight_1 = glob.glob(args.weight_1_dir + "/*.pthw")
    weight_1.sort(key=os.path.getmtime)

    ## check if everything in weights_2 exist
    for ag2 in args.weight_2:
        assert os.path.exists(ag2)

    slice_epoch = int(agent_args["num_epoch"]) * (len(args.weight_2) - 1)
    act_steps = utils.get_act_steps(args.weight_1_dir, slice_epoch)

    cur_task = 0
    prev_max = [0] * len(args.weight_2)
    prev_task_max = [0] * len(args.weight_2)
    prev_max_fs = [0] * len(args.weight_2)
    prev_task_max_fs = [0] * len(args.weight_2)
    avg_fs_score = 0
    avg_fs_future_score = 0
    avg_fs_forgetting = 0
    all_done = 0
    total_prev_act_steps = 0

    for ag1_idx, ag1 in enumerate(weight_1):
        ag1_name = ag1.split("/")[-1].split("_")[-1]
        act_epoch_cnt = int(ag1.split("/")[-1].split("_")[1][5:])
        ### move zs ckpts after every task to test dir
        if act_epoch_cnt % int(agent_args["num_epoch"]) == 0:
            if ag1_name == "shot.pthw":
                move_zs_ckpt = f"cp {ag1} {test_dir}/"
                os.system(move_zs_ckpt)

        ### this is for different zero-shot evaluations...
        total_tasks = len(args.weight_2)
        if ag1_name == "shot.pthw":
            all_done += 1
            avg_score = 0
            avg_future_score = 0
            avg_forgetting = 0

            for fixed_agent_idx in range(len(args.weight_2)):
                weight_files = [ag1, args.weight_2[fixed_agent_idx]]
                mean_score, sem, perfect_rate = evaluate_legacy_model(
                    weight_files, 1000, 1, 0, agent_args, args, num_run=5
                )

                if mean_score > prev_max[fixed_agent_idx]:
                    prev_max[fixed_agent_idx] = mean_score
                wandb.log(
                    {
                        "epoch_zeroshot": act_epoch_cnt,
                        "eval_score_zeroshot_" + str(fixed_agent_idx): mean_score,
                        "perfect_zeroshot_" + str(fixed_agent_idx): perfect_rate,
                        "sem_zeroshot_" + str(fixed_agent_idx): sem,
                        "total_act_steps": (
                            total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                        ),
                    }
                )
                if fixed_agent_idx == cur_task:
                    wandb.log(
                        {
                            "epoch_zs_curtask": act_epoch_cnt,
                            "eval_score_zs_curtask": mean_score,
                            "perfect_zs_curtask": perfect_rate,
                            "sem_zs_curtask": sem,
                            "total_act_steps": (
                                total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                            ),
                        }
                    )

                if fixed_agent_idx <= cur_task:
                    avg_score += mean_score
                if fixed_agent_idx > cur_task:
                    avg_future_score += mean_score
                if cur_task > 0:
                    forgetting = prev_task_max[fixed_agent_idx] - mean_score
                    if fixed_agent_idx < cur_task:
                        avg_forgetting += forgetting
                    wandb.log(
                        {
                            "epoch_zs_forgetting": act_epoch_cnt,
                            "forgetting_zs_" + str(fixed_agent_idx): forgetting,
                            "total_act_steps": (
                                total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                            ),
                        }
                    )

            avg_score = avg_score / (cur_task + 1)
            avg_future_score = avg_future_score / (total_tasks - (cur_task + 1))
            wandb.log(
                {
                    "epoch_zs_avg_score": act_epoch_cnt,
                    "avg_zs_score": avg_score,
                    "avg_future_zs_score": avg_future_score,
                    "total_act_steps": (
                        total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                    ),
                }
            )

            if cur_task > 0:
                avg_forgetting = avg_forgetting / (cur_task)
                wandb.log(
                    {
                        "epoch_zs_avg_forgetting": act_epoch_cnt,
                        "avg_zs_forgetting": avg_forgetting,
                        "total_act_steps": (
                            total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                        ),
                    }
                )

        else:
            ## for different few shot evaluations ...
            for i in range(len(args.weight_2)):
                if ag1_name == str(i) + ".pthw":
                    all_done += 1
                    weight_files = [ag1, args.weight_2[i]]

            cur_ag_id = ag1_name.split(".")[0]

            mean_score, sem, perfect_rate = evaluate_legacy_model(
                weight_files, 1000, 1, 0, agent_args, args, num_run=5
            )
            if mean_score > prev_max_fs[int(cur_ag_id)]:
                prev_max_fs[int(cur_ag_id)] = mean_score

            wandb.log(
                {
                    "epoch_fewshot": act_epoch_cnt,
                    "eval_score_fewshot_" + cur_ag_id: mean_score,
                    "perfect_fewshot_" + cur_ag_id: perfect_rate,
                    "sem_fewshot_" + cur_ag_id: sem,
                    "total_act_steps": (
                        total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                    ),
                }
            )

            if int(cur_ag_id) <= cur_task:
                avg_fs_score += mean_score
            if int(cur_ag_id) > cur_task:
                avg_fs_future_score += mean_score
            if int(cur_ag_id) == cur_task:
                wandb.log(
                    {
                        "epoch_fs_curtask": act_epoch_cnt,
                        "eval_score_fs_curtask": mean_score,
                        "perfect_fs_curtask": perfect_rate,
                        "sem_fs_curtask": sem,
                        "total_act_steps": (
                            total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                        ),
                    }
                )

            if cur_task > 0:
                forgetting_fs = prev_task_max_fs[int(cur_ag_id)] - mean_score
                if int(cur_ag_id) < cur_task:
                    avg_fs_forgetting += forgetting_fs
                wandb.log(
                    {
                        "epoch_fs_forgetting": act_epoch_cnt,
                        "forgetting_fs_" + cur_ag_id: forgetting_fs,
                        "total_act_steps": (
                            total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                        ),
                    }
                )

        if all_done % (total_tasks + 1) == 0:
            avg_fs_score = avg_fs_score / (cur_task + 1)
            avg_fs_future_score = avg_fs_future_score / (total_tasks - (cur_task + 1))
            wandb.log(
                {
                    "epoch_fs_avg_score": act_epoch_cnt,
                    "avg_fs_score": avg_fs_score,
                    "avg_fs_future_score": avg_fs_future_score,
                    "total_act_steps": (
                        total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                    ),
                }
            )

            avg_fs_score = 0
            avg_fs_future_score = 0

            if cur_task > 0:
                avg_fs_forgetting = avg_fs_forgetting / cur_task
                wandb.log(
                    {
                        "epoch_fs_avg_forgetting": act_epoch_cnt,
                        "avg_fs_forgetting": avg_fs_forgetting,
                        "total_act_steps": (
                            total_prev_act_steps + act_steps[act_epoch_cnt - 1]
                        ),
                    }
                )

                avg_fs_forgetting = 0

        if (
            act_epoch_cnt == agent_args["num_epoch"] * (cur_task + 1)
            and all_done % (total_tasks + 1) == 0
        ):
            cur_task += 1
            for fixed_agent_idx in range(len(args.weight_2)):
                prev_task_max[fixed_agent_idx] = prev_max[fixed_agent_idx]
                prev_task_max_fs[fixed_agent_idx] = prev_max_fs[fixed_agent_idx]
            all_done = 0
            total_prev_act_steps += act_steps[act_epoch_cnt - 1]
