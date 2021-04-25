import set_path

set_path.append_sys_path()

import os
import pprint
import time
import copy

import numpy as np
import torch
import rela
import hanalearn

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")


def create_envs(
    num_env,
    seed,
    num_player,
    hand_size,
    bomb,
    explore_eps,
    max_len,
    sad,
    shuffle_obs,
    shuffle_color,
):
    games = []
    for game_idx in range(num_env):
        params = {
            "players": str(num_player),
            "hand_size": str(hand_size),
            "seed": str(seed + game_idx),
            "bomb": str(bomb),
        }
        game = hanalearn.HanabiEnv(
            params,
            explore_eps,
            max_len,
            sad,
            shuffle_obs,
            shuffle_color,
            False,
        )
        games.append(game)
    return games


def create_threads(
    num_thread,
    num_game_per_thread,
    actors,
    games,
):
    context = rela.Context()
    threads = []
    for thread_idx in range(num_thread):
        env = hanalearn.HanabiVecEnv()
        for game_idx in range(num_game_per_thread):
            env.append(games[thread_idx * num_game_per_thread + game_idx])
        thread = hanalearn.HanabiThreadLoop(actors[thread_idx], env, False)
        threads.append(thread)
        context.push_env_thread(thread)
    print(
        "Finished creating %d threads with %d games and %d actors"
        % (len(threads), len(games), len(actors))
    )
    return context, threads


class ActGroup:
    """
    Creates actors given the agents. Starts to stores transitions in the replay_buffer by calling ActGroup.start()
    Args:
        method(str): iql or vdn
        devices(str): cuda1
        agent(object): an R2d2 agent object
        num_thread(int): default=10
        num_game_per_thread(int): default=80
        multi_step(int): default=3
        gamma(float): discount factor
        eta(float): eta for aggregate priority
        max_len(int): max seq len
        num_player(int):  default=2
        replay_buffer(object): a replay buffer onject
    Returns:
        None
    """

    def __init__(
        self,
        method,
        devices,
        agent,
        num_thread,
        num_game_per_thread,
        multi_step,
        gamma,
        eta,
        max_len,
        num_player,
        replay_buffer,
    ):
        self.devices = devices.split(",")

        self.model_runners = []
        for dev in self.devices:
            runner = rela.BatchRunner(
                agent.clone(dev), dev, 100, ["act", "compute_priority"]
            )
            self.model_runners.append(runner)

        self.num_runners = len(self.model_runners)

        self.actors = []
        self.eval_actors = []
        if method == "vdn":
            for i in range(num_thread):
                actor = rela.R2D2Actor(
                    self.model_runners[i % self.num_runners],
                    multi_step,
                    num_game_per_thread,
                    gamma,
                    eta,
                    max_len,
                    num_player,
                    replay_buffer,
                )
                self.actors.append(actor)
        elif method == "iql":
            for i in range(num_thread):
                thread_actors = []
                for _ in range(num_player):
                    actor = rela.R2D2Actor(
                        self.model_runners[i % self.num_runners],
                        multi_step,
                        num_game_per_thread,
                        gamma,
                        eta,
                        max_len,
                        1,
                        replay_buffer,
                    )
                    thread_actors.append(actor)
                self.actors.append(thread_actors)
        print("ActGroup created")
        self.state_dicts = []

    def start(self):
        for runner in self.model_runners:
            runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)


class ContActGroup:
    """
    Creates actors given the agents. Starts to stores transitions in the replay_buffer by calling ContActGroup.start()
    Args:
        method(str): iql or vdn
        devices(str): cuda1
        agent_list(list): list of a learner and its partners
        num_thread(int): default=10
        num_game_per_thread(int): default=80
        multi_step(int): default=3
        gamma(float): discount factor
        eta(float): eta for aggregate priority
        max_len(int): max seq len
        num_player(int):  default=2
        is_rand(bool): To randomize ordering of the learner and its partners or not
        replay_buffer(object): a replay buffer onject
    Returns:
        None
    """

    def __init__(
        self,
        method,
        devices,
        agent_list,
        num_thread,
        num_game_per_thread,
        multi_step,
        gamma,
        eta,
        max_len,
        num_player,
        is_rand,
        replay_buffer,
    ):
        self.devices = devices.split(",")
        self.flags = []
        self.model_runners = []
        self.is_rand = is_rand

        for dev in self.devices:
            learnable_runner = rela.BatchRunner(
                agent_list[0].clone(dev), dev, 100, ["act", "compute_priority"]
            )
            fixed_runner = rela.BatchRunner(
                agent_list[1].clone(dev), dev, 100, ["act", "compute_priority"]
            )
            if self.is_rand:
                flag = np.random.randint(0, num_player)
                if flag == 0:
                    self.model_runners.append([learnable_runner, fixed_runner])
                elif flag == 1:
                    self.model_runners.append([fixed_runner, learnable_runner])

                self.flags.append(flag)
            else:
                self.model_runners.append([learnable_runner, fixed_runner])

        self.num_runners = len(self.model_runners)

        self.actors = []
        if method == "vdn":
            for i in range(num_thread):
                actor = rela.R2D2Actor(
                    self.model_runners[i % self.num_runners],
                    multi_step,
                    num_game_per_thread,
                    gamma,
                    eta,
                    max_len,
                    num_player,
                    replay_buffer,
                )
                self.actors.append(actor)
        elif method == "iql":
            for i in range(num_thread):
                thread_actors = []
                for n in range(num_player):
                    actor = rela.R2D2Actor(
                        self.model_runners[i % self.num_runners][n],
                        multi_step,
                        num_game_per_thread,
                        gamma,
                        eta,
                        max_len,
                        1,
                        replay_buffer,
                    )
                    thread_actors.append(actor)
                self.actors.append(thread_actors)
        self.state_dicts = []

    def start(self):
        for runner in self.model_runners:
            runner[0].start()
            runner[1].start()

    def update_model(self, agent):
        for idx, runner in enumerate(self.model_runners):
            if self.is_rand:
                if self.flags[idx] == 0:
                    runner[0].update_model(agent)
                elif self.flags[idx] == 1:
                    runner[1].update_model(agent)
            else:
                runner[0].update_model(agent)
