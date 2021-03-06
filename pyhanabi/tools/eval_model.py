"""
Requires only 1 GPU.
Sample usage: 
python tools/eval_model.py --weight_1_dir ../models/iql_2p --num_player 2
It dumps a .csv as output
"""
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
from eval import evaluate_legacy_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_1_dir", default=None, type=str, required=True)
    parser.add_argument("--num_player", default=None, type=int, required=True)
    parser.add_argument("--is_rand", action="store_true", default=True)

    args = parser.parse_args()

    assert os.path.exists(args.weight_1_dir)
    weight_1 = []
    weight_1 = glob.glob(f"{args.weight_1_dir}/*.pthw")

    scores_arr = np.zeros([len(weight_1), len(weight_1)])
    sem_arr = np.zeros([len(weight_1), len(weight_1)])
    ag1_names = []

    for ag1 in weight_1:
        ag1_names.append(ag1.split("/")[-1].split(".")[0])

    for ag1_idx, ag1 in enumerate(weight_1):
        for ag2_idx, ag2 in enumerate(weight_1):
            ## we are doing cross player, the 2 players use different weights
            print("Current game is ", str(ag1_idx) + " vs " + str(ag2_idx))
            weight_files = [ag1, ag2]
            agent_args = {}
            # # fast evaluation for 5k games
            mean, sem, _ = evaluate_legacy_model(
                weight_files,
                1000,
                1,
                0,
                agent_args,
                args,
                num_run=5,
                gen_cross_play=True,
            )
            scores_arr[ag1_idx, ag2_idx] = mean
            sem_arr[ag1_idx, ag2_idx] = sem
            np.save("scores_data", scores_arr)
            np.save("sem_data", sem_arr)

    scores_df = pd.DataFrame(data=scores_arr, index=ag1_names, columns=ag1_names)
    sem_df = pd.DataFrame(data=sem_arr, index=ag1_names, columns=ag1_names)

    scores_df.to_csv("scores_data.csv")
    sem_df.to_csv("sem_data.csv")
