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
from eval import evaluate
import r2d2


def evaluate_legacy_model(
    weight_files,
    num_game,
    seed,
    bomb,
    learnable_agent_args,
    cont_train_args,
    num_run=1,
    verbose=True,
):

    agents = []
    num_player = len(weight_files)
    assert num_player > 1, "1 weight file per player"

    env_sad = False
    for i, weight_file in enumerate(weight_files):
        if verbose:
            print(
                "evaluating: %s\n\tfor %dx%d games" % (weight_file, num_run, num_game)
            )
        if "sad" in weight_file:
            sad = True
            env_sad = True
        else:
            sad = False

        device = "cuda:0"

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]

        learnable_pretrain = False
        no_CL = False
        if i == 0:
            if "load_learnable_model" in learnable_agent_args:
                if learnable_agent_args["load_learnable_model"] != "":
                    agent_args_file = (
                        learnable_agent_args["load_learnable_model"][:-4] + "txt"
                    )
                    learnable_pretrain = True
            else:
                agent_args_file = cont_train_args
                no_CL = True
        else:
            agent_args_file = weight_file[:-4] + "txt"

        if learnable_pretrain == True or no_CL == True or i == 1:
            with open(agent_args_file, "r") as f:
                agent_args = {**json.load(f)}
            rnn_type = agent_args["rnn_type"]
            rnn_hid_dim = agent_args["rnn_hid_dim"]
            num_fflayer = agent_args["num_fflayer"]
            num_rnn_layer = agent_args["num_rnn_layer"]
        elif learnable_pretrain == False and no_CL == False:
            rnn_type = learnable_agent_args["rnn_type"]
            rnn_hid_dim = learnable_agent_args["rnn_hid_dim"]
            num_fflayer = learnable_agent_args["num_fflayer"]
            num_rnn_layer = learnable_agent_args["num_rnn_layer"]

        # if rnn_type == "lstm":
        #     import r2d2_lstm as r2d2
        # elif rnn_type == "gru":
        #     import r2d2_gru as r2d2

        agent = r2d2.R2D2Agent(
            False,
            3,
            0.999,
            0.9,
            device,
            input_dim,
            rnn_hid_dim,
            output_dim,
            num_fflayer,
            rnn_type,
            num_rnn_layer,
            5,
            False,
            sad=sad,
        ).to(device)

        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)

    scores = []
    perfect = 0
    for i in range(num_run):
        if args.is_rand:
            flag = np.random.randint(0, num_player)
            if flag == 0:
                new_agents = [agents[0], agents[1]]
            elif flag == 1:
                new_agents = [agents[1], agents[0]]
        else:
            new_agents = [agents[0], agents[1]]

        _, _, score, p = evaluate(
            new_agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,
            env_sad,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: ", perfect_rate)
    return mean, sem, perfect_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_1_dir", default=None, type=str, required=True)
    parser.add_argument("--weight_2", default=None, type=str, nargs="+", required=True)
    parser.add_argument("--is_rand", action="store_true", default=True)
    parser.add_argument("--num_player", default=None, type=int, required=True)
    args = parser.parse_args()

    cont_train_args_txt = glob.glob(args.weight_1_dir + "/*.txt")
    with open(cont_train_args_txt[0], "r") as f:
        learnable_agent_args = {**json.load(f)}

    exp_name = "final_eval_" + learnable_agent_args["save_dir"].split("/")[-1]
    wandb.init(project="Lifelong_Hanabi_project", name=exp_name)
    wandb.config.update(learnable_agent_args)

    assert os.path.exists(args.weight_1_dir)
    weight_1 = []
    weight_1 = glob.glob(args.weight_1_dir + "/*.pthw")
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
                    learnable_agent_args,
                    cont_train_args_txt[0],
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
                if ag1_name == str(i) + ".pthw":
                    weight_files = [ag1, args.weight_2[i]]

            mean_score, sem, perfect_rate = evaluate_legacy_model(
                weight_files,
                1000,
                1,
                0,
                learnable_agent_args,
                cont_train_args_txt[0],
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
