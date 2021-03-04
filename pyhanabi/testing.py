""" Evaluating all the checkpoints saved periodically during train args.eval_freq
Requires only 1 GPU.
Sample usage: 
python testing.py 
--weight_1_dir <path-to-final-eval-models-dir>
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

    ## Note: This assumes that we have access to configuration file
    ## during continual training in the models directory specifying architecture details
    ## like type of RNN, num of RNN layers etc.
    ## else to evaluate models otherwise created, please specify these in
    cont_train_args_txt = glob.glob(f"{args.weight_1_dir}/*.txt")
    with open(cont_train_args_txt[0], "r") as f:
        agent_args = {**json.load(f)}
    save_dir = agent_args["save_dir"].split("/")[-1]
    exp_name = f"test_{save_dir}"
    wandb.init(project="Lifelong_Hanabi_project", name=exp_name)
    wandb.config.update(agent_args)

    assert os.path.exists(args.weight_1_dir)
    weight_1 = []
    weight_1 = glob.glob(f"{args.weight_1_dir}/*.pthw")
    weight_1.sort(key=os.path.getmtime)

    ## check if everything in weights_2 exist
    for ag2 in args.weight_2:
        assert os.path.exists(ag2)

    for ag1_idx, ag1 in enumerate(weight_1):
        ag1_name = ag1.split("/")[-1].split("_")[-1]
        act_epoch_cnt = int(ag1.split("/")[-1].split("_")[1][5:])

        ### this is for different zero-shot evaluations...
        if ag1_name == "shot.pthw":
            for fixed_agent_idx in range(len(args.weight_2)):
                weight_files = [ag1, args.weight_2[fixed_agent_idx]]
                mean_score, sem, perfect_rate = evaluate_legacy_model(
                    weight_files,
                    1000,
                    1,
                    0,
                    agent_args,
                    args,
                    num_run=5,
                )
                wandb.log(
                    {
                        "epoch_zeroshot": act_epoch_cnt,
                        "final_eval_score_zeroshot_" + str(fixed_agent_idx): mean_score,
                        "perfect_zeroshot_" + str(fixed_agent_idx): perfect_rate,
                        "sem_zeroshot_" + str(fixed_agent_idx): sem,
                    }
                )
        else:
            ## for different few shot evaluations ...
            for i in range(len(args.weight_2)):
                if ag1_name == f"{i}.pthw":
                    weight_files = [ag1, args.weight_2[i]]

            mean_score, sem, perfect_rate = evaluate_legacy_model(
                weight_files,
                1000,
                1,
                0,
                agent_args,
                args,
                num_run=5,
            )
            wandb.log(
                {
                    "epoch_fewshot": act_epoch_cnt,
                    "final_eval_score_fewshot_" + ag1_name.split(".")[0]: mean_score,
                    "perfect_fewshot_" + ag1_name.split(".")[0]: perfect_rate,
                    "sem_fewshot_" + ag1_name.split(".")[0]: sem,
                }
            )
