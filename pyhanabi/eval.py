import os
import time
import json
import numpy as np
import torch
import random
from create import *
import rela
import utils


def evaluate(agents, num_game, seed, bomb, eps, sad, *, hand_size=5, runners=None):
    """
    evaluate agents as long as they have a "act" function
    """
    assert agents is None or runners is None
    if agents is not None:
        runners = [rela.BatchRunner(agent, "cuda:0", 1000, ["act"]) for agent in agents]
    num_player = len(runners)

    context = rela.Context()
    games = create_envs(
        num_game,
        seed,
        num_player,
        hand_size,
        bomb,
        [eps],
        -1,
        sad,
        False,
        False,
    )

    for g in games:
        env = hanalearn.HanabiVecEnv()
        env.append(g)
        actors = []
        for i in range(num_player):
            actors.append(rela.R2D2Actor(runners[i], 1))
        thread = hanalearn.HanabiThreadLoop(actors, env, True)
        context.push_env_thread(thread)

    for runner in runners:
        runner.start()

    context.start()
    while not context.terminated():
        time.sleep(0.5)
    context.terminate()
    while not context.terminated():
        time.sleep(0.5)

    for runner in runners:
        runner.stop()

    scores = [g.last_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect


def evaluate_legacy_model(
    weight_files,
    num_game,
    seed,
    bomb,
    agent_args,
    args,
    num_run=1,
    gen_cross_play=False,
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
        output_dim = state_dict["fc_a.weight"].size()[0]

        if gen_cross_play:
            agent_name = weight_file.split("/")[-1].split(".")[0]

            with open(f"{args.weight_1_dir}/{agent_name}.txt", "r") as f:
                agent_args = {**json.load(f)}
        else:
            learnable_pretrain = True

            if i == 0:
                learnable_agent_name = agent_args["load_learnable_model"]
                if learnable_agent_name != "":
                    agent_args_file = f"{learnable_agent_name[:-4]}txt"
                else:
                    learnable_pretrain = False
            else:
                agent_args_file = f"{weight_file[:-4]}txt"

            if learnable_pretrain == True:
                with open(agent_args_file, "r") as f:
                    agent_args = {**json.load(f)}

        rnn_type = agent_args["rnn_type"]
        rnn_hid_dim = agent_args["rnn_hid_dim"]
        num_fflayer = agent_args["num_fflayer"]
        num_rnn_layer = agent_args["num_rnn_layer"]

        if rnn_type == "lstm":
            import r2d2_lstm as r2d2
        elif rnn_type == "gru":
            import r2d2_gru as r2d2

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

        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)

    scores = []
    perfect = 0
    for i in range(num_run):
        if args.is_rand:
            random.shuffle(agents)

        _, _, score, p = evaluate(
            agents,
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


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    overwrite=None,
    num_run=1,
    verbose=True,
):
    agents = []
    sad = []
    hide_action = []
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = "cuda:0"
    overwrite["boltzmann_act"] = False

    for weight_file in weight_files:
        agent, cfg = utils.load_agent(
            weight_file,
            overwrite,
        )
        agents.append(agent)
        sad.append(cfg["sad"] if "sad" in cfg else cfg["greedy_extra"])
        hide_action.append(bool(cfg["hide_action"]))

    hand_size = cfg.get("hand_size", 5)

    assert all(s == sad[0] for s in sad)
    sad = sad[0]
    if all(h == hide_action[0] for h in hide_action):
        hide_action = hide_action[0]
        process_game = None
    else:
        hide_actions = hide_action
        process_game = lambda g: g.set_hide_actions(hide_actions)
        hide_action = False

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p, _ = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,  # eps
            sad,
            hide_action,
            process_game=process_game,
            hand_size=hand_size,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print(
            "score: %f +/- %f" % (mean, sem), "; perfect: %.2f%%" % (100 * perfect_rate)
        )
    return mean, sem, perfect_rate, scores
