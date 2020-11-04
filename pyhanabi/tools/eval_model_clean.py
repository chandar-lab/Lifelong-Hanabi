'''
Sample usage: 
'''
import argparse
import os
import sys
import json
import glob

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import numpy as np
import pandas as pd
import torch
import r2d2_gru as r2d2_gru
import r2d2_lstm as r2d2_lstm
import utils
from eval import evaluate


def evaluate_legacy_model(
    weight_files, num_game, seed, bomb, args, num_run=1, verbose=True):
    # model_lockers = []
    # greedy_extra = 0
    agents = []
    num_player = len(weight_files)
    assert num_player > 1, "1 weight file per player"
    # print("args weight 1 inside evaluate leg model is ", args.weight_1_dir)

    for weight_file in weight_files:
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
        num_lstm_layer = 2
        output_dim = state_dict["fc_a.weight"].size()[0]

        agent_name = weight_file.split("/")[-1].split(".")[0]

        with open(args.weight_1_dir+"/"+agent_name+".txt", 'r') as f:
            agent_args = {**json.load(f)}

        # print("args after merging dictionaries is ...", args)

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
    parser.add_argument("--num_player", default=None, type=int, required=True)

    args = parser.parse_args()
    print("weights_1 dir is ", args.weight_1_dir)

    assert os.path.exists(args.weight_1_dir) 
    weight_1 = []
    weight_1 = glob.glob(args.weight_1_dir+"/*.pthw")
    # weight_1.sort(key=os.path.getmtime)

    print("ckpt files are ", weight_1)
    
    scores_arr = np.zeros([len(weight_1), len(weight_1)])
    sem_arr = np.zeros([len(weight_1), len(weight_1)])
    ag1_names = []

    ## check if everything in weights_1 exist
    for ag1 in weight_1:
        ag1_names.append(ag1.split("/")[-1].split(".")[0])

    print("ag1 names is ", ag1_names)

    for ag1_idx, ag1 in enumerate(weight_1):
        for ag2_idx, ag2 in enumerate(weight_1):
            ## we are doing cross player, the 2 players use different weights
            print("Current game is ", str(ag1_idx) + " vs " + str(ag2_idx))
            weight_files = [ag1, ag2]
            # # fast evaluation for 10k games
            mean, sem, _ = evaluate_legacy_model(weight_files, 1000, 1, 0, args, num_run=5)
            scores_arr[ag1_idx, ag2_idx] = mean
            sem_arr[ag1_idx, ag2_idx] = sem 


    scores_df = pd.DataFrame(data=scores_arr, index=ag1_names, columns=ag1_names)
    sem_df = pd.DataFrame(data=sem_arr, index=ag1_names, columns=ag1_names)

    scores_df.to_csv('scores_data.csv')
    sem_df.to_csv('sem_data.csv')


