# AGEM implementation

import time
import os
import sys
import argparse
import pprint
import json
import gc
import numpy as np
import torch
from torch import nn

from create_cont import create_envs, create_threads, ActGroup
from eval import evaluate
import common_utils
import rela
import r2d2_gru as r2d2_gru
import r2d2_lstm as r2d2_lstm
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--load_model_dir", type=str, default="../models/iql_2p")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_obs", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--pred_weight", type=float, default=0)
    parser.add_argument("--num_eps", type=int, default=80)

    parser.add_argument("--load_learnable_model", type=str, default="")
    parser.add_argument("--load_fixed_models", type=str, nargs='+', default="")

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--eta", type=float, default=0.9, help="eta for aggregate priority")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)

    # optimization/training settings
    parser.add_argument("--initial_lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--final_lr", type=float, default=6.25e-5, help="Final learning rate")
    parser.add_argument("--lr_gamma", type=float, default=0.2, help="lr decay")
    parser.add_argument("--decay_lr", action="store_true", default=False)
    parser.add_argument("--dropout_p", type=float, default=0, help="drop probability")
    parser.add_argument("--optim_name", type=str, default="Adam")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--sgd_momentum", type=float, default=0.8, help="SGD momentum")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")

    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--num_fflayer", type=int, default=1)
    parser.add_argument("--num_rnn_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--eval_num_epoch", type=int, default=1)
    parser.add_argument("--eval_epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--eval_num_update_between_sync", type=int, default=5)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--eval_burn_in_frames", type=int, default=1000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument("--eval_replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)
    parser.add_argument("--eval_num_thread", type=int, default=40, help="#eval_thread_loop")
    parser.add_argument("--eval_num_game_per_thread", type=int, default=20)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--eval_actor_sync_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=1)


    # life long learning settings
    parser.add_argument("--ll_algo", type=str, default="AGEM")
    parser.add_argument("--eval_method", type=str, default="zero_shot")
    parser.add_argument("--add_agent_id", action="store_true", default=False)

    ## args dump settings
    parser.add_argument("--args_dump_name", type=str, default="AGEM_commandline_args.txt")

    args = parser.parse_args()
    assert args.method in ["vdn", "iql"]
    assert args.ll_algo in ["ER", "AGEM", "EWC", "None"]
    assert args.eval_method in ["zero_shot", "few_shot"]
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.args_dump_name = "cont_args.txt"

    with open(args.save_dir+"/"+args.args_dump_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_eps
    )
    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.hand_size,
        args.train_bomb,
        explore_eps,
        args.max_len,
        args.sad,
        args.shuffle_obs,
        args.shuffle_color,
    )
## this is the learnable agent.
    if args.load_learnable_model:
        learnable_agent_name = args.load_learnable_model.split("/")[-1].split(".")[0]
        with open(args.load_model_dir+"/"+learnable_agent_name+".txt") as f:
            learnable_agent_args = {**json.load(f)}

        if learnable_agent_args['rnn_type'] == "lstm":
            learnable_agent = r2d2_lstm.R2D2Agent(
                (args.method == "vdn"),
                args.multi_step,
                args.gamma,
                args.eta,
                args.train_device,
                games[0].feature_size(),
                learnable_agent_args['rnn_hid_dim'],
                games[0].num_action(),
                learnable_agent_args['num_fflayer'],
                learnable_agent_args['num_rnn_layer'],
                args.hand_size,
                False,  # uniform priority
            )
        elif learnable_agent_args['rnn_type'] == "gru":
            learnable_agent = r2d2_gru.R2D2Agent(
                (args.method == "vdn"),
                args.multi_step,
                args.gamma,
                args.eta,
                args.train_device,
                games[0].feature_size(),
                learnable_agent_args['rnn_hid_dim'],
                games[0].num_action(),
                learnable_agent_args['num_fflayer'],
                learnable_agent_args['num_rnn_layer'],
                args.hand_size,
                False,  # uniform priority
            )
    else:
        if args.rnn_type == "lstm":
            learnable_agent = r2d2_lstm.R2D2Agent(
                (args.method == "vdn"),
                args.multi_step,
                args.gamma,
                args.eta,
                args.train_device,
                games[0].feature_size(),
                args.rnn_hid_dim,
                games[0].num_action(),
                args.num_fflayer,
                args.num_rnn_layer,
                args.hand_size,
                False,  # uniform priority
            )
        elif args.rnn_type == "gru":
            learnable_agent = r2d2_gru.R2D2Agent(
                (args.method == "vdn"),
                args.multi_step,
                args.gamma,
                args.eta,
                args.train_device,
                games[0].feature_size(),
                args.rnn_hid_dim,
                games[0].num_action(),
                args.num_fflayer,
                args.num_rnn_layer,
                args.hand_size,
                False,  # uniform priority
            ) 

    learnable_agent.sync_target_with_online()

    if args.load_learnable_model:
        print("*****loading pretrained model for learnable agent *****")
        utils.load_weight(learnable_agent.online_net, args.load_learnable_model, args.train_device)
        print("*****done*****")

    learnable_agent = learnable_agent.to(args.train_device)
    print(learnable_agent)

    eval_agent = learnable_agent.clone(args.train_device, {"vdn": False})

    fixed_learnable_agent = learnable_agent.clone(args.train_device, {"vdn": False})

    fixed_agents = []
    episodic_memory = []

    for opp_idx, opp_model in enumerate(args.load_fixed_models):
        opp_model_name = opp_model.split("/")[-1].split(".")[0]

        with open(args.load_model_dir+"/"+opp_model_name+".txt") as f:
            opp_model_args = {**json.load(f)}

        if opp_model_args['rnn_type'] == "lstm":
            fixed_agent = r2d2_lstm.R2D2Agent(
                (args.method == "vdn"),
                args.multi_step,
                args.gamma,
                args.eta,
                args.train_device,
                games[0].feature_size(),
                opp_model_args['rnn_hid_dim'],
                games[0].num_action(),
                opp_model_args['num_fflayer'],
                opp_model_args['num_rnn_layer'],
                args.hand_size,
                False,  # uniform priority
            )
        elif opp_model_args['rnn_type'] == "gru":
            fixed_agent = r2d2_gru.R2D2Agent(
                (args.method == "vdn"),
                args.multi_step,
                args.gamma,
                args.eta,
                args.train_device,
                games[0].feature_size(),
                opp_model_args['rnn_hid_dim'],
                games[0].num_action(),
                opp_model_args['num_fflayer'],
                opp_model_args['num_rnn_layer'],
                args.hand_size,
                False,  # uniform priority
            )
        
        if opp_model:
            print("*****loading pretrained model for fixed agent *****")
            utils.load_weight(fixed_agent.online_net, opp_model, args.train_device)
            print("*****done*****")

        fixed_agent = fixed_agent.to(args.train_device)
        fixed_agents.append(fixed_agent)

    total_epochs = 0

    for task_idx, fixed_agent in enumerate(fixed_agents):
        if args.decay_lr:
            lr = max(args.initial_lr * args.lr_gamma**(task_idx), args.final_lr)
        else:
            lr = args.final_lr

        if args.optim_name == "Adam":
            optim = torch.optim.Adam(learnable_agent.online_net.parameters(), lr=lr, eps=args.eps)
        elif args.optim_name == "SGD":
            optim = torch.optim.SGD(learnable_agent.online_net.parameters(), lr=lr, momentum=args.sgd_momentum)

        replay_buffer = rela.RNNPrioritizedReplay(
            args.replay_buffer_size,
            args.seed,
            args.priority_exponent,
            args.priority_weight,
            args.prefetch,
        )

        games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.hand_size,
        args.train_bomb,
        explore_eps,
        args.max_len,
        args.sad,
        args.shuffle_obs,
        args.shuffle_color,
        )

        act_group = ActGroup(
            args.method,
            args.act_device,
            [learnable_agent, fixed_agent],
            args.num_thread,
            args.num_game_per_thread,
            args.multi_step,
            args.gamma,
            args.eta,
            args.max_len,
            args.num_player,
            replay_buffer,
        )

        assert args.shuffle_obs == False, 'not working with 2nd order aux'
        context, threads = create_threads(
            args.num_thread, args.num_game_per_thread, act_group.actors, games,
        )
        act_group.start()
        context.start()
        while replay_buffer.size() < args.burn_in_frames:
            print("warming up replay buffer:", replay_buffer.size())
            time.sleep(1)

        print("Success, Done")
        print("=======================")

        frame_stat = dict()
        frame_stat["num_acts"] = 0
        frame_stat["num_buffer"] = 0

        stat = common_utils.MultiCounter(args.save_dir)
        tachometer = utils.Tachometer()
        stopwatch = common_utils.Stopwatch()

        for epoch in range(args.num_epoch):
            total_epochs += 1
            print("beginning of epoch: ", total_epochs)
            cnt_angle_less = 0
            print(common_utils.get_mem_usage())
            tachometer.start()
            stat.reset()
            stopwatch.reset()

            for batch_idx in range(args.epoch_len):
                num_update = batch_idx + epoch * args.epoch_len
                if num_update % args.num_update_between_sync == 0:
                    learnable_agent.sync_target_with_online()
                if num_update % args.actor_sync_freq == 0:
                    act_group.update_model(learnable_agent)

                torch.cuda.synchronize()
                stopwatch.time("sync and updating")

                batch, weight = replay_buffer.sample(args.batchsize, args.train_device)

                # Looping over the replay buffers of previous tasks.
                ## TODO: Figure out what happens to batch.h0 -- why is it empty? should it be concat?
                ## TODO: If possible, fix the actual C++ thing where batch size is interfaced so that .sample would return desired number of samples.
                if args.ll_algo == "AGEM":
                    prev_tasks_b = []
                    prev_tasks_w = []
                    batch_obs = {}
                    batch_act = {}

                    for prev_task_idx in range(len(episodic_memory)):
                        samples_per_task = (args.batchsize // len(episodic_memory))
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
                                    batch_obs[k] = torch.cat([batch_obs[k], b.obs[k][:, :samples_per_task]], dim=1)
                                else:
                                    batch_obs[k] = torch.cat([batch_obs[k], b.obs[k][:, :samples_per_task, :]], dim=1)

                        if prev_task_idx == 0:
                            for k in b.action.keys():
                                batch_act[k] = b.action[k][:, :samples_per_task]
                        elif prev_task_idx > 0:
                            for k in b.action.keys():
                                batch_act[k] = torch.cat([batch_act[k], b.action[k][:, :samples_per_task]], dim=1)

                        if prev_task_idx == 0:
                            batch_reward = b.reward[:, :samples_per_task]
                            batch_terminal = b.terminal[:, :samples_per_task]
                            batch_bootstrap = b.bootstrap[:, :samples_per_task]
                            batch_seq_len = b.seq_len[:samples_per_task]
                            batch_weight = w[:samples_per_task]
                        elif prev_task_idx > 0:
                            batch_reward = torch.cat([batch_reward, b.reward[:, :samples_per_task]], dim=1)
                            batch_terminal = torch.cat([batch_terminal, b.terminal[:, :samples_per_task]], dim=1)
                            batch_bootstrap = torch.cat([batch_bootstrap, b.bootstrap[:, :samples_per_task]], dim=1)
                            batch_seq_len = torch.cat([batch_seq_len, b.seq_len[:samples_per_task]], dim=0)
                            batch_weight = torch.cat([batch_weight, w[:samples_per_task]], dim=0)

                        if prev_task_idx == len(episodic_memory)-1:
                            b.obs = batch_obs
                            b.action = batch_act
                            b.reward = batch_reward
                            b.terminal = batch_terminal
                            b.bootstrap = batch_bootstrap
                            b.seq_len = batch_seq_len

                            w = batch_weight   

                    stopwatch.time("sample data")
                    ## TODO: find a better solution instead of this hack of slicing priority
                    for prev_task_idx in range(len(episodic_memory)):
                        _, p = learnable_agent.loss(prev_tasks_b[prev_task_idx], args.pred_weight, stat)
                        p = rela.aggregate_priority(
                        p.cpu(), prev_tasks_b[prev_task_idx].seq_len.cpu(), args.eta
                        )
                        episodic_memory[prev_task_idx].update_priority(p)

                ## previous tasks average loss
                if task_idx > 0:
                    loss_replay, _ = learnable_agent.loss(b, args.pred_weight, stat)
                    loss_replay = (loss_replay * w).mean()
                    loss_replay.backward()

                    ## reorganize the gradient of replayed batch as a single vector
                    grad_rep = []
                    for p in learnable_agent.online_net.parameters():
                        if p.requires_grad:
                            if p.grad is not None:
                                grad_rep.append(p.grad.view(-1))
                    grad_rep = torch.cat(grad_rep)
                    ## reset gradients (with A-GEM the gradients of replayed batch should only be used as inequality constraints)
                    optim.zero_grad()


                ## current task loss
                loss_cur, priority = learnable_agent.loss(batch, args.pred_weight, stat)

                priority = rela.aggregate_priority(
                    priority.cpu(), batch.seq_len.cpu(), args.eta
                )

                loss_cur = (loss_cur * weight).mean()
                loss_cur.backward()

                if task_idx > 0:
                    ## reorganize the gradient of the current batch as a single vector
                    grad_cur = []
                    for i,p in enumerate(list(learnable_agent.online_net.parameters())):
                        if p.requires_grad:
                            if p.grad is not None:
                                grad_cur.append(p.grad.view(-1))
                    grad_cur = torch.cat(grad_cur)

                    ## adding A-GEM projection inequality.
                    angle = (grad_cur*grad_rep).sum()
                    if angle < 0:
                        cnt_angle_less += 1
                    # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                        length_rep = (grad_rep*grad_rep).sum()
                        grad_proj = grad_cur-(angle/length_rep)*grad_rep
                        # -...and replace all the gradients within the model with this projected gradient
                        index = 0
                        for p in learnable_agent.online_net.parameters():
                            if p.requires_grad:
                                if p.grad is not None:
                                    n_param = p.numel()  # number of parameters in [p]
                                    p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                                    index += n_param

                torch.cuda.synchronize()
                stopwatch.time("forward & backward")

                g_norm = torch.nn.utils.clip_grad_norm_(
                    learnable_agent.online_net.parameters(), args.grad_clip
                )
                optim.step()
                optim.zero_grad()

                torch.cuda.synchronize()
                stopwatch.time("update model")

                ## current task priority
                replay_buffer.update_priority(priority[:args.batchsize])
                stopwatch.time("updating priority")

                stat["loss"].feed(loss_cur.detach().item())
                stat["grad_norm"].feed(g_norm.detach().item())

            count_factor = args.num_player if args.method == "vdn" else 1
            print("EPOCH: %d" % total_epochs)
            tachometer.lap(
                act_group.actors, replay_buffer, args.epoch_len * args.batchsize, count_factor
            )
            stopwatch.summary()
            stat.summary(epoch)

            context.pause()
            eval_seed = (9917 + epoch * 999999) % 7777777
            
            if (epoch+1) % args.eval_freq == 0:
                for eval_fixed_ag_idx, eval_fixed_agent in enumerate(fixed_agents + [fixed_learnable_agent]):
                    print("evaluating learnable agent with fixed agent %d "%eval_fixed_ag_idx)

                    if args.eval_method == 'few_shot':
                        print("Few Shot Learning ...")
                        few_shot_learnable_agent = learnable_agent.clone(args.train_device, {"vdn": False})
                        if args.optim_name == "Adam":
                            eval_optim = torch.optim.Adam(few_shot_learnable_agent.online_net.parameters(), lr=args.final_lr,
                                    eps=args.eps)
                        elif args.optim_name == "SGD":
                            eval_optim = torch.optim.SGD(few_shot_learnable_agent.online_net.parameters(), lr=args.final_lr,
                                    momentum=args.sgd_momentum)

                        eval_replay_buffer = rela.RNNPrioritizedReplay(
                            args.eval_replay_buffer_size,
                            eval_seed,
                            args.priority_exponent,
                            args.priority_weight,
                            args.prefetch,
                        )

                        eval_games = create_envs(
                            args.eval_num_thread * args.eval_num_game_per_thread,
                            eval_seed,
                            args.num_player,
                            args.hand_size,
                            args.train_bomb,
                            explore_eps,
                            args.max_len,
                            args.sad,
                            args.shuffle_obs,
                            args.shuffle_color,
                        )

                        eval_act_group = ActGroup(
                            args.method,
                            args.act_device,
                            [few_shot_learnable_agent, eval_fixed_agent],
                            args.eval_num_thread,
                            args.eval_num_game_per_thread,
                            args.multi_step,
                            args.gamma,
                            args.eta,
                            args.max_len,
                            args.num_player,
                            eval_replay_buffer,
                        )
                        eval_context, eval_threads = create_threads(
                            args.eval_num_thread, args.eval_num_game_per_thread, eval_act_group.actors, eval_games,
                        )
                        eval_act_group.start()
                        eval_context.start()
                        while eval_replay_buffer.size() < args.eval_burn_in_frames:
                            print("warming up replay buffer:", eval_replay_buffer.size())
                            time.sleep(1)
                        eval_tachometer = utils.Tachometer()
                        eval_stat = common_utils.MultiCounter(args.save_dir)

                        for eval_epoch in range(args.eval_num_epoch):
                            print("beginning of eval epoch: ", eval_epoch)
                            eval_stat.reset()
                            eval_tachometer.start()
                            for eval_batch_idx in range(args.eval_epoch_len):
                                eval_num_update = eval_batch_idx + eval_epoch * args.eval_epoch_len
                                if eval_num_update % args.eval_num_update_between_sync == 0:
                                    few_shot_learnable_agent.sync_target_with_online()
                                if eval_num_update % args.eval_actor_sync_freq == 0:
                                    eval_act_group.update_model(few_shot_learnable_agent)

                                batch, weight = eval_replay_buffer.sample(args.batchsize, args.train_device)

                                torch.cuda.synchronize()
                                stopwatch.time("sync and updating")

                                loss, priority = few_shot_learnable_agent.loss(batch, args.pred_weight, eval_stat)

                                priority = rela.aggregate_priority(
                                    priority.cpu(), batch.seq_len.cpu(), args.eta
                                )

                                loss = (loss * weight).mean()
                                loss.backward()

                                torch.cuda.synchronize()

                                g_norm = torch.nn.utils.clip_grad_norm_(
                                    few_shot_learnable_agent.online_net.parameters(), args.grad_clip
                                )
                                eval_optim.step()
                                eval_optim.zero_grad()

                                torch.cuda.synchronize()

                                ## current task priority
                                eval_replay_buffer.update_priority(priority[:args.batchsize])

                                eval_stat["loss"].feed(loss.detach().item())
                                eval_stat["grad_norm"].feed(g_norm.detach().item())
                            eval_tachometer.lap(eval_act_group.actors, eval_replay_buffer, args.eval_epoch_len * args.batchsize, count_factor)
                            eval_stat.summary(eval_epoch)
                        eval_context.pause()
                        fs_force_save_name = "model_epoch%d_few_shot_%d" % (total_epochs, eval_fixed_ag_idx)
                        few_shot_model_saved = saver.save(None, few_shot_learnable_agent.online_net.state_dict(), force_save_name=fs_force_save_name)      
                        print("few shot model saved: %s "%(few_shot_model_saved))

                ## zero shot learnable agent. 
                zs_force_save_name = "model_epoch%d_zero_shot" %(total_epochs)
                zero_shot_model_saved = saver.save(None, learnable_agent.online_net.state_dict(), force_save_name=zs_force_save_name)
                print("zero shot model saved: %s "%(zero_shot_model_saved))

            print("number of times angle less than 0 is ", cnt_angle_less)
            gc.collect()
            context.resume()
            print("==========")
    
        episodic_memory.append(replay_buffer)
