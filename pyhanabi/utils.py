#
import os
import time
from collections import OrderedDict
import json
import torch
import numpy as np
import glob
import rela
from create import *
import common_utils


def parse_first_dict(lines):
    config_lines = []
    open_count = 0
    for i, l in enumerate(lines):
        if l.strip()[0] == "{":
            open_count += 1
        if open_count:
            config_lines.append(l)
        if l.strip()[-1] == "}":
            open_count -= 1
        if open_count == 0 and len(config_lines) != 0:
            break

    config = "".join(config_lines).replace("'", '"')
    config = config.replace("True", "true")
    config = config.replace("False", "false")
    config = json.loads(config)
    return config, lines[i + 1 :]


def get_train_config(weight_file):
    log = os.path.join(os.path.dirname(weight_file), "train.log")
    if not os.path.exists(log):
        return None

    lines = open(log, "r").readlines()
    cfg, rest = parse_first_dict(lines)
    return cfg


def flatten_dict(d, new_dict):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_dict)
        else:
            new_dict[k] = v


def load_agent(weight_file, overwrite):
    """
    overwrite has to contain "device"
    """
    cfg = get_train_config(weight_file)
    assert cfg is not None

    if "core" in cfg:
        new_cfg = {}
        flatten_dict(cfg, new_cfg)
        cfg = new_cfg

    game = create_envs(
        1,
        1,
        cfg["num_player"],
        cfg["train_bomb"],
        [0],  # explore_eps,
        [100],  # boltzmann_t,
        cfg["max_len"],
        cfg["sad"] if "sad" in cfg else cfg["greedy_extra"],
        cfg["shuffle_obs"],
        cfg["shuffle_color"],
        cfg["hide_action"],
        True,
    )[0]

    config = {
        "vdn": overwrite["vdn"] if "vdn" in overwrite else cfg["method"] == "vdn",
        "multi_step": overwrite.get("multi_step", cfg["multi_step"]),
        "gamma": overwrite.get("gamma", cfg["gamma"]),
        "eta": 0.9,
        "device": overwrite["device"],
        "in_dim": game.feature_size(),
        "hid_dim": cfg["hid_dim"] if "hid_dim" in cfg else cfg["rnn_hid_dim"],
        "out_dim": game.num_action(),
        "num_fflayer": overwrite.get("num_fflayer", cfg["num_fflayer"]),
        "num_rnn_layer": overwrite.get("num_rnn_layer", cfg["num_rnn_layer"]),
        "boltzmann_act": overwrite.get("boltzmann_act", cfg["boltzmann_act"]),
        "hand_size": overwrite.get("hand_size", cfg["hand_size"]),
        "uniform_priority": overwrite.get("uniform_priority", False),
    }
    if cfg["rnn_type"] == "lstm":
        import r2d2_lstm as r2d2_lstm

        agent = r2d2_lstm.R2D2Agent(**config).to(config["device"])
    elif cfg["rnn_type"] == "gru":
        import r2d2_gru as r2d2_gru

        agent = r2d2_gru.R2D2Agent(**config).to(config["device"])

    load_weight(agent.online_net, weight_file, config["device"])
    agent.sync_target_with_online()
    return agent, cfg


def log_explore_ratio(games, expected_eps):
    explore = []
    for g in games:
        explore.append(g.get_explore_count())
    explore = np.stack(explore)
    explore = explore.sum(0)  # .reshape((8, 10)).sum(1)

    step_counts = []
    for g in games:
        step_counts.append(g.get_step_count())
    step_counts = np.stack(step_counts)
    step_counts = step_counts.sum(0)  # .reshape((8, 10)).sum(1)

    factor = []
    for i in range(len(explore)):
        if step_counts[i] == 0:
            factor.append(1.0)
        else:
            f = expected_eps / max(1e-5, (explore[i] / step_counts[i]))
            f = max(0.5, min(f, 2))
            factor.append(f)
    print(">>>explore factor:", len(factor))

    explore = explore.reshape((8, 10)).sum(1)
    step_counts = step_counts.reshape((8, 10)).sum(1)

    print("exploration:")
    for i in range(len(explore)):
        ratio = 100 * explore[i] / step_counts[i]
        print(
            "\tbucket [%2d, %2d]: %5d, %5d, %2.2f%%"
            % (i * 10, (i + 1) * 10, explore[i], step_counts[i], ratio)
        )

    for g in games:
        g.reset_count()

    return factor


class Tachometer:
    def __init__(self, iseval=False):
        self.num_act = 0
        self.num_buffer = 0
        self.num_train = 0
        self.t = None
        self.total_time = 0
        self.iseval = iseval

    def start(self):
        self.t = time.time()

    def lap(self, actors, replay_buffer, num_train, factor, prev_steps):
        t = time.time() - self.t
        self.total_time += t
        num_act = get_num_acts(actors, prev_steps)
        act_rate = factor * (num_act - self.num_act) / t
        num_buffer = replay_buffer.num_add()
        buffer_rate = factor * (num_buffer - self.num_buffer) / t
        train_rate = factor * num_train / t
        if self.iseval:
            print(
                "Eval Speed: train: %.1f, act: %.1f, buffer_add: %.1f, buffer_size: %d"
                % (train_rate, act_rate, buffer_rate, replay_buffer.size())
            )
        else:
            print(
                "Speed: train: %.1f, act: %.1f, buffer_add: %.1f, buffer_size: %d"
                % (train_rate, act_rate, buffer_rate, replay_buffer.size())
            )
        self.num_act = num_act
        self.num_buffer = num_buffer
        self.num_train += num_train
        if self.iseval:
            print(
                "Eval Total Time: %s, %ds"
                % (common_utils.sec2str(self.total_time), self.total_time)
            )
        else:
            print(
                "Total Time: %s, %ds"
                % (common_utils.sec2str(self.total_time), self.total_time)
            )
        if self.iseval:
            print(
                "Eval Total Sample: train: %s, act: %s"
                % (
                    common_utils.num2str(self.num_train),
                    common_utils.num2str(self.num_act),
                )
            )
        else:
            print(
                "Total Sample: train: %s, act: %s"
                % (
                    common_utils.num2str(self.num_train),
                    common_utils.num2str(self.num_act),
                )
            )

    def lap2(self, actors, num_buffer, num_train):
        t = time.time() - self.t
        self.total_time += t
        num_act = get_num_acts(actors, 0)
        act_rate = (num_act - self.num_act) / t
        buffer_rate = (num_buffer - self.num_buffer) / t
        train_rate = num_train / t
        print(
            "Speed: train: %.1f, act: %.1f, buffer_add: %.1f"
            % (train_rate, act_rate, buffer_rate)
        )
        self.num_act = num_act
        self.num_buffer = num_buffer
        self.num_train += num_train
        print(
            "Total Time: %s, %ds"
            % (common_utils.sec2str(self.total_time), self.total_time)
        )
        print(
            "Total Sample: train: %s, act: %s"
            % (common_utils.num2str(self.num_train), common_utils.num2str(self.num_act))
        )


def load_weight(model, weight_file, device):
    state_dict = torch.load(weight_file, map_location=device)
    source_state_dict = OrderedDict()
    target_state_dict = model.state_dict()
    for k, v in target_state_dict.items():
        if k not in state_dict:
            print("warning: %s not loaded" % k)
            state_dict[k] = v
    for k in state_dict:
        if k not in target_state_dict:
            print("removing: %s not used" % k)
        else:
            source_state_dict[k] = state_dict[k]

    model.load_state_dict(source_state_dict)
    return


def make_batch_ER(args, episodic_memory, batch, weight, stat, learnable_agent):
    prev_tasks_b = []
    prev_tasks_w = []

    for prev_task_idx in range(len(episodic_memory)):
        samples_per_task = args.batchsize // len(episodic_memory)
        n_residual = args.batchsize - (samples_per_task * len(episodic_memory))
        if prev_task_idx < n_residual:
            samples_per_task += 1
        b, w = episodic_memory[prev_task_idx].sample(args.batchsize, args.train_device)
        prev_tasks_b.append(b)
        prev_tasks_w.append(w)
        batch_obs = {}
        batch_act = {}
        for k in batch.obs.keys():
            if k == "eps":
                batch_obs[k] = torch.cat(
                    [batch.obs[k], b.obs[k][:, :samples_per_task]], dim=1
                )
            else:
                batch_obs[k] = torch.cat(
                    [batch.obs[k], b.obs[k][:, :samples_per_task, :]], dim=1
                )

        batch.obs = batch_obs

        for k in batch.action.keys():
            batch_act[k] = torch.cat(
                [batch.action[k], b.action[k][:, :samples_per_task]], dim=1
            )

        batch.action = batch_act

        batch.reward = torch.cat([batch.reward, b.reward[:, :samples_per_task]], dim=1)
        batch.terminal = torch.cat(
            [batch.terminal, b.terminal[:, :samples_per_task]], dim=1
        )
        batch.bootstrap = torch.cat(
            [batch.bootstrap, b.bootstrap[:, :samples_per_task]], dim=1
        )

        batch.seq_len = torch.cat([batch.seq_len, b.seq_len[:samples_per_task]], dim=0)
        weight = torch.cat([weight, w[:samples_per_task]], dim=0)

    for prev_task_idx in range(len(episodic_memory)):
        _, p = learnable_agent.loss(prev_tasks_b[prev_task_idx], args.pred_weight, stat)
        p = rela.aggregate_priority(
            p.cpu(), prev_tasks_b[prev_task_idx].seq_len.cpu(), args.eta
        )
        episodic_memory[prev_task_idx].update_priority(p)

    return batch, weight, episodic_memory


def make_batch_AGEM(args, episodic_memory, stat, learnable_agent):
    prev_tasks_b = []
    prev_tasks_w = []
    batch_obs = {}
    batch_act = {}

    for prev_task_idx in range(len(episodic_memory)):
        samples_per_task = args.batchsize // len(episodic_memory)
        n_residual = args.batchsize - (samples_per_task * len(episodic_memory))
        if prev_task_idx < n_residual:
            samples_per_task += 1
        b, w = episodic_memory[prev_task_idx].sample(args.batchsize, args.train_device)
        prev_tasks_b.append(b)
        prev_tasks_w.append(w)

        if prev_task_idx == 0:
            for k in b.obs.keys():
                if k == "eps":
                    batch_obs[k] = b.obs[k][:, :samples_per_task]
                else:
                    batch_obs[k] = b.obs[k][:, :samples_per_task, :]
        elif prev_task_idx > 0:
            for k in b.obs.keys():
                if k == "eps":
                    batch_obs[k] = torch.cat(
                        [batch_obs[k], b.obs[k][:, :samples_per_task]], dim=1
                    )
                else:
                    batch_obs[k] = torch.cat(
                        [batch_obs[k], b.obs[k][:, :samples_per_task, :]], dim=1
                    )

        if prev_task_idx == 0:
            for k in b.action.keys():
                batch_act[k] = b.action[k][:, :samples_per_task]
        elif prev_task_idx > 0:
            for k in b.action.keys():
                batch_act[k] = torch.cat(
                    [batch_act[k], b.action[k][:, :samples_per_task]], dim=1
                )

        if prev_task_idx == 0:
            batch_reward = b.reward[:, :samples_per_task]
            batch_terminal = b.terminal[:, :samples_per_task]
            batch_bootstrap = b.bootstrap[:, :samples_per_task]
            batch_seq_len = b.seq_len[:samples_per_task]
            batch_weight = w[:samples_per_task]
        elif prev_task_idx > 0:
            batch_reward = torch.cat(
                [batch_reward, b.reward[:, :samples_per_task]], dim=1
            )
            batch_terminal = torch.cat(
                [batch_terminal, b.terminal[:, :samples_per_task]], dim=1
            )
            batch_bootstrap = torch.cat(
                [batch_bootstrap, b.bootstrap[:, :samples_per_task]], dim=1
            )
            batch_seq_len = torch.cat(
                [batch_seq_len, b.seq_len[:samples_per_task]], dim=0
            )
            batch_weight = torch.cat([batch_weight, w[:samples_per_task]], dim=0)

        if prev_task_idx == len(episodic_memory) - 1:
            b.obs = batch_obs
            b.action = batch_act
            b.reward = batch_reward
            b.terminal = batch_terminal
            b.bootstrap = batch_bootstrap
            b.seq_len = batch_seq_len

            w = batch_weight

    for prev_task_idx in range(len(episodic_memory)):
        _, p = learnable_agent.loss(prev_tasks_b[prev_task_idx], args.pred_weight, stat)
        p = rela.aggregate_priority(
            p.cpu(), prev_tasks_b[prev_task_idx].seq_len.cpu(), args.eta
        )
        episodic_memory[prev_task_idx].update_priority(p)

    if len(episodic_memory) > 0:
        return b, w, episodic_memory
    else:
        return None, None, episodic_memory


def get_grad_list(learnable_agent):
    ## reorganize the gradient of batch as a single vector
    grad = []
    for p in learnable_agent.online_net.parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad.append(p.grad.view(-1))
    grad = torch.cat(grad)
    return grad


def grad_proj(grad_cur, grad_rep, learnable_agent):
    ## adding A-GEM projection inequality.
    angle = (grad_cur * grad_rep).sum()
    if angle < 0:
        # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
        length_rep = (grad_rep * grad_rep).sum()
        grad_proj = grad_cur - (angle / length_rep) * grad_rep
        # -...and replace all the gradients within the model with this projected gradient
        index = 0
        for p in learnable_agent.online_net.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    n_param = p.numel()  # number of parameters in [p]
                    p.grad.copy_(grad_proj[index : index + n_param].view_as(p))
                    index += n_param

    return learnable_agent


def get_game_info(num_player, greedy_extra, feed_temperature, extra_args=None):
    params = {"players": str(num_player)}
    if extra_args is not None:
        params.update(extra_args)
    game = hanalearn.HanabiEnv(
        params,
        [0],
        [],
        -1,
        greedy_extra,
        False,
        False,
        False,
        feed_temperature,
        False,
        False,
    )

    if num_player < 5:
        hand_size = 5
    else:
        hand_size = 4

    info = {
        "input_dim": game.feature_size(),
        "num_action": game.num_action(),
        "hand_size": hand_size,
        "hand_feature_size": game.hand_feature_size(),
    }
    return info


def compute_input_dim(num_player):
    hand = 126 * num_player
    board = 76
    discard = 50
    last_action = 51 + 2 * num_player
    card_knowledge = num_player * 5 * 35
    return hand + board + discard + last_action + card_knowledge


def get_act_steps(save_dir, last_epoch_restore):
    cont_train_log = glob.glob(f"{save_dir}/*.log")
    cont_train_log.sort(key=os.path.getmtime)

    act_steps_lns = []
    act_steps = []

    for i in range(len(cont_train_log)):
        if i > 0:
            with open(cont_train_log[i], "r") as f:
                for ln in f:
                    if ln.startswith("epoch restore is"):
                        epoch_restore = int(ln.split(" ")[-1])
                        act_steps_lns = act_steps_lns[:epoch_restore]

        with open(cont_train_log[i], "r") as f:
            for ln in f:
                if ln.startswith("Total Sample:"):
                    act_steps_lns.append(ln)

    act_steps_lns = act_steps_lns[:last_epoch_restore]

    for asl in act_steps_lns:
        ac_st = asl.split(" ")[-1]
        if ac_st[-2] == "K":
            st = float(ac_st[:-2]) * float(1000)
        elif ac_st[-2] == "M":
            st = float(ac_st[:-2]) * float(1000000)
        act_steps.append(st)

    return act_steps


# returns the number of steps in all actors
def get_num_acts(actors, prev_steps):
    total_acts = prev_steps
    for actor in actors:
        if isinstance(actor, list):
            total_acts += get_num_acts(actor)
        else:
            total_acts += actor.num_act()
    return total_acts


# num_acts is the total number of acts, so total number of acts is num_acts * num_game_per_actor
# num_buffer is the total number of elements inserted into the buffer
# time elapsed is in seconds
def get_frame_stat(num_game_per_thread, time_elapsed, num_acts, num_buffer, frame_stat):
    total_sample = (num_acts - frame_stat["num_acts"]) * num_game_per_thread
    act_rate = total_sample / time_elapsed
    buffer_rate = (num_buffer - frame_stat["num_buffer"]) / time_elapsed
    frame_stat["num_acts"] = num_acts
    frame_stat["num_buffer"] = num_buffer
    return total_sample, act_rate, buffer_rate


def generate_explore_eps(base_eps, alpha, num_env):
    if num_env == 1:
        if base_eps < 1e-6:
            base_eps = 0
        return [base_eps]

    eps_list = []
    for i in range(num_env):
        eps = base_eps ** (1 + i / (num_env - 1) * alpha)
        if eps < 1e-6:
            eps = 0
        eps_list.append(eps)
    return eps_list


def generate_log_uniform(min_val, max_val, n):
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    uni = np.linspace(log_min, log_max, n)
    uni_exp = np.exp(uni)
    return uni_exp.tolist()


@torch.jit.script
def get_v1(v0_joind, card_counts, ref_mask):
    v0_joind = v0_joind.cpu()
    card_counts = card_counts.cpu()

    batch, num_player, dim = v0_joind.size()
    num_player = 3
    v0_joind = v0_joind.view(batch, 1, num_player * 5, 25)

    mask = (v0_joind > 0).float()
    total_viable_cards = mask.sum()
    v1_old = v0_joind
    thres = 0.0001
    max_count = 100
    weight = 0.1
    v1_new = v1_old
    for i in range(max_count):  # can't use a variable count for tracing
        # torch.Size([256, 99, 25]) torch.Size([256, 99, 10, 25])
        # Calculate how many cards of what types are sitting in other hands.
        hand_cards = v1_old.sum(2)
        total_cards = card_counts - hand_cards
        # Exclude the cards I am holding myself.
        excluding_self = total_cards.unsqueeze(2) + v1_old
        # Negative numbers shouldn't happen, but they might (for all I know)
        excluding_self.clamp_(min=0)
        # Calculate unnormalised likelihood of cards: Adjusted count * Mask
        v1_new = excluding_self * mask
        # this is avoiding NaNs for when there are no cards.
        v1_new = v1_old * (1 - weight) + weight * v1_new
        v1_new = v1_new / (v1_new.sum(-1, keepdim=True) + 1e-8)
        # if False: # this is strictly for debugging / diagnostics
        #     # Normalise the diff by total viable cards.
        #     diff = (v1_new - v1_old).abs().sum() / total_viable_cards
        #     xent = get_xent(data, v1_old[:,:,:5,:])
        #     print('diff %8.3g  xent %8.3g' % (diff, xent))
        v1_old = v1_new

    return v1_new


@torch.jit.script
def check_v1(v0, v1, card_counts, mask):
    ref_v1 = get_v1(v0, card_counts, mask)
    batch, num_player, dim = v1.size()
    v1 = v1.view(batch, 1, 3 * 5, 25).cpu()
    print("diff: ", (ref_v1 - v1).max())
    if (ref_v1 - v1).max() > 1e-4:
        print((ref_v1 - v1)[0][0][0])
        assert False


def check_trajectory(batch):
    assert batch.obs["h"][0][0].sum() == 0
    length = batch.obs["h"][0].size(0)
    end = 0
    for i in range(length):
        t = batch.terminal[0][i]

        if end != 0:
            assert t

        if not t:
            continue

        if end == 0:
            end = i
    print("trajectory ends at:", end)
