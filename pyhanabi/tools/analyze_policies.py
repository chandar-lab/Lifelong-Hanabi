# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import pprint
from collections import defaultdict
import json
import torch

import matplotlib.pyplot as plt

plt.switch_backend("agg")

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

from eval import *
import utils
import common_utils
import rela
import create


def create_dataset(agent, sad, device):
    # use it in "vdn" mode so that trajecoties from the same game are
    # grouped together
    agent = agent.clone(device, {"vdn": True})
    runner = rela.BatchRunner(agent, device, 100, ["act", "compute_priority"])

    dataset_size = 1000
    replay_buffer = rela.RNNPrioritizedReplay(
        dataset_size,  # args.dataset_size,
        1,  # args.seed,
        0,  # args.priority_exponent, uniform sampling
        1,  # args.priority_weight,
        0,  # args.prefetch,
    )

    num_thread = 100
    num_game_per_thread = 1
    max_len = 80
    actors = []
    for i in range(num_thread):
        # thread_actors = []
        actor = rela.R2D2Actor(
            runner,
            1,  # multi_step,
            num_game_per_thread,
            0.99,  # gamma,
            0.9,  # eta
            max_len,  # max_len,
            2,  # num_player
            replay_buffer,
        )
        actors.append(actor)

    eps = [0]  # for _ in range(num_game_per_thread)]
    num_game = num_thread * num_game_per_thread
    games = create.create_envs(num_game, 1, 2, 5, 0, [0], max_len, sad, False, False)
    context, threads = create.create_threads(
        num_thread, num_game_per_thread, actors, games
    )

    runner.start()
    context.start()
    while replay_buffer.size() < dataset_size:
        print("collecting data from replay buffer:", replay_buffer.size())
        time.sleep(0.2)

    context.pause()

    # remove extra data
    for _ in range(2):
        data, unif = replay_buffer.sample(10, "cpu")
        replay_buffer.update_priority(unif.detach().cpu())
        time.sleep(0.2)

    print("dataset size:", replay_buffer.size())
    print("done about to return")
    return replay_buffer, agent, context


def analyze(dataset):
    p0_p1 = np.zeros((20, 20))
    for i in range(dataset.size()):
        epsd = dataset.get(i)
        action = epsd.action["a"]
        for t in range(int(epsd.seq_len.item()) - 1):
            if t % 2 == 0:
                a0 = int(action[t][0].item())
                a1 = int(action[t + 1][1].item())
            else:
                a0 = int(action[t][1].item())
                a1 = int(action[t + 1][0].item())

            p0_p1[a0][a1] += 1

    denom = p0_p1.sum(1, keepdims=True)
    normed_p0_p1 = p0_p1 / denom
    return normed_p0_p1, p0_p1


idx2action = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
]


plt.rc("image", cmap="viridis")
plt.rc("xtick", labelsize=10)  # fontsize of the tick labels
plt.rc("ytick", labelsize=10)
plt.rc("axes", labelsize=10)
plt.rc("axes", titlesize=10)


def plot(mat, title, num_player, *, fig=None, ax=None, savefig=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(mat)
    ax.set_title(title)
    if num_player == 2:
        ax.set_xticks(range(20))
        ax.set_xticklabels(idx2action)
        ax.set_yticks(range(20))
        ax.set_yticklabels(idx2action)
    elif num_player == 3:
        ax.set_xticks(range(30))
        ax.set_xticklabels(idx2action_p3)
        ax.set_yticks(range(30))
        ax.set_yticklabels(idx2action_p3)
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # config for model from sad paper
    parser.add_argument("--weight_file", default=None, type=str)
    parser.add_argument("--config_file", default=None, type=str)
    # config for model from op paper
    # output
    parser.add_argument("--save_fig", required=True, type=str, help="where to save?")

    args = parser.parse_args()
    if not os.path.exists(args.save_fig):
        os.makedirs(args.save_fig)

    device = "cuda"
    state_dict = torch.load(args.weight_file)
    input_dim = state_dict["net.0.weight"].size()[1]
    output_dim = state_dict["fc_a.weight"].size()[0]
    with open(args.config_file, "r") as f:
        agent_args = {**json.load(f)}

    rnn_type = agent_args["rnn_type"]
    rnn_hid_dim = agent_args["rnn_hid_dim"]
    num_fflayer = agent_args["num_fflayer"]
    num_rnn_layer = agent_args["num_rnn_layer"]

    if rnn_type == "lstm":
        import r2d2_lstm as r2d2
    elif rnn_type == "gru":
        import r2d2_gru as r2d2

    if "sad" in args.weight_file:
        sad = True
        env_sad = True
    else:
        sad = False
        env_sad = False

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
        num_rnn_layer,
        5,
        False,
        sad=sad,
    ).to(device)

    utils.load_weight(agent.online_net, args.weight_file, device)

    dataset, _, _ = create_dataset(agent, sad, device)
    normed_p0_p1, p0_p1 = analyze(dataset)
    plot_name = args.weight_file.split("/")[-1][:-5]
    plot(normed_p0_p1, "action_matrix", 2, savefig=f"{args.save_fig}{plot_name}")
