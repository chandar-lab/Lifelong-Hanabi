## evaluating all the checkpoints saved periodically: args.eval_freq
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''
Sample usage: 
'''
#
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
import r2d2_gru_unify as r2d2_gru
import r2d2_lstm_unify as r2d2_lstm
import utils
from eval import evaluate


def evaluate_legacy_model(
    weight_files, num_game, seed, bomb, learnable_agent_args, num_run=1, verbose=True
):
    # model_lockers = []
    # greedy_extra = 0
    agents = []
    num_player = len(weight_files)
    assert num_player > 1, "1 weight file per player"

    for i, weight_file in enumerate(weight_files):
        if verbose:
            print(
                "evaluating: %s\n\tfor %dx%d games" % (weight_file, num_run, num_game)
            )
        if "sad" in weight_file or "aux" in weight_file:
            sad = True
        else:
            sad = False

        device = "cuda:0"

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]

        if i == 0:
            agent_args_file = learnable_agent_args['load_learnable_model'][:-4]+"txt"
        else:
            agent_args_file = weight_file[:-4] + "txt"

        with open(agent_args_file, 'r') as f:
            agent_args = {**json.load(f)}

        if agent_args['rnn_type'] == "lstm":
            agent = r2d2_lstm.R2D2Agent(
                False, 3, 0.999, 0.9, device, input_dim, agent_args['rnn_hid_dim'], output_dim, agent_args['num_fflayer'], agent_args['num_rnn_layer'], 5, False
            ).to(device)
        elif agent_args['rnn_type'] == "gru":
            agent = r2d2_gru.R2D2Agent(
                False, 3, 0.999, 0.9, device, input_dim, agent_args['rnn_hid_dim'], output_dim, agent_args['num_fflayer'], agent_args['num_rnn_layer'], 5, False
            ).to(device)

        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,
            sad,
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
    parser.add_argument("--weight_2", default=None, type=str, nargs='+', required=True)

    parser.add_argument("--num_player", default=None, type=int, required=True)
    args = parser.parse_args()  

    cont_train_args_txt = glob.glob(args.weight_1_dir+"/*.txt")

    with open(cont_train_args_txt[0], 'r') as f:
            learnable_agent_args = {**json.load(f)}


    rb_exp_name = int(learnable_agent_args['replay_buffer_size']) // 1000
    lr_str = learnable_agent_args['load_learnable_model'].split("/")[-1].split(".")[0]
    exp_name = lr_str+"_fixed_"+str(len(learnable_agent_args['load_fixed_models']))+"_ind_RB_"+learnable_agent_args['eval_method']+"_"+ str(rb_exp_name)+"k_"+learnable_agent_args['ll_algo']    
    
    wandb.init(project="ContPlay_Hanabi_complete", name=exp_name)
    wandb.config.update(learnable_agent_args)

    print("weights_1 is ", args.weight_1_dir)
    print("weights_2 is ", args.weight_2)

    assert os.path.exists(args.weight_1_dir)    
    weight_1 = []
    weight_1 = glob.glob(args.weight_1_dir+"/*.pthw")
    weight_1.sort(key=os.path.getmtime)

    print("ckpt files are ", weight_1)

    ## check if everything in weights_2 exist
    for ag2 in args.weight_2:
        assert os.path.exists(ag2)
        # ag2_names.append(ag2.split("/")[2].split(".")[0])

    for ag1_idx, ag1 in enumerate(weight_1):
        ag1_name = ag1.split("/")[-1].split("_")[-1]
        act_epoch_cnt = int(ag1.split("/")[-1].split("_")[1][5:])


    ### this is for different zero-shot evaluations...
        if ag1_name == "shot.pthw":
            for fixed_agent_idx in range(len(args.weight_2)):
                weight_files = [ag1, args.weight_2[fixed_agent_idx]]
                mean_score, sem, perfect_rate = evaluate_legacy_model(weight_files, 1000, 1, 0, learnable_agent_args, num_run=5)
                wandb.log({"epoch_zeroshot_"+str(fixed_agent_idx): act_epoch_cnt, "eval_score_zeroshot_"+str(fixed_agent_idx): mean_score, "perfect_zeroshot_"+str(fixed_agent_idx): perfect_rate, "sem_zeroshot_"+str(fixed_agent_idx):sem})
        else:
            ## for different few shot evaluations ... 
            for i in range(len(args.weight_2)):
                if ag1_name == str(i)+".pthw":
                    weight_files = [ag1, args.weight_2[i]]

            # print("weight ")
            mean_score, sem, perfect_rate = evaluate_legacy_model(weight_files, 1000, 1, 0, learnable_agent_args, num_run=5)
            wandb.log({"epoch_fewshot_"+ag1_name.split(".")[0]: act_epoch_cnt, "eval_score_fewshot_"+ag1_name.split(".")[0]: mean_score, "perfect_fewshot_"+ag1_name.split(".")[0]: perfect_rate, "sem_fewshot_"+ag1_name.split(".")[0]:sem})

        

        
        
