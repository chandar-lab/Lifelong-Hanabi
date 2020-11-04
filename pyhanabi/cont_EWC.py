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
import r2d2_gru_unify as r2d2_gru
import r2d2_lstm_unify as r2d2_lstm
import utils
# from EWC import EWC
import EWC as ewc
# os.environ["WANDB_API_KEY"] = "b002db5ed8e9de3af350e301d5c25d0dcd8ea320"
# os.environ['WANDB_MODE'] = 'dryrun'

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
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
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
    parser.add_argument("--ll_algo", type=str, default="ER")
    parser.add_argument("--eval_method", type=str, default="zero_shot")
    parser.add_argument("--add_agent_id", action="store_true", default=False)

    ## EWC settings
    parser.add_argument("--online", type=int, default=0)
    parser.add_argument("--ewc_lambda", type=float, default=5000)
    parser.add_argument("--ewc_gamma", type=float, default=1)


    ## args dump settings
    parser.add_argument("--args_dump_name", type=str, default="ER_commandline_args.txt")

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

    rb_exp_name = int(args.replay_buffer_size) // 1000
    args.args_dump_name = str(rb_exp_name)+"k_"+args.ll_algo+".txt"

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

    learnable_agent.sync_target_with_online()

    if args.load_learnable_model:
        print("*****loading pretrained model for learnable agent *****")
        utils.load_weight(learnable_agent.online_net, args.load_learnable_model, args.train_device)
        print("*****done*****")

    learnable_agent = learnable_agent.to(args.train_device)
    optim = torch.optim.Adam(learnable_agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(learnable_agent)

    if args.ll_algo == "EWC":
        ewc_class = ewc.EWC(args)
    
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

    act_epoch_cnt = 0

    for task_idx, fixed_agent in enumerate(fixed_agents):
        ## TODO: Exp decision : do we want different replay buffer when playing with diff opponents
        ## i.e do we want to replay prev experiences? 
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
            act_epoch_cnt += 1
            print("beginning of epoch: ", act_epoch_cnt)
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

                loss, priority = learnable_agent.loss(batch, args.pred_weight, stat)

                priority = rela.aggregate_priority(
                    priority.cpu(), batch.seq_len.cpu(), args.eta
                )

                loss = (loss * weight).mean() 

                if args.ll_algo == "EWC":
                    if epoch == (args.num_epoch-1) and batch_idx == (args.epoch_len-1):
                        ewc_class.estimate_fisher(learnable_agent, batch, weight, stat, task_idx)

                    ewc_loss = ewc_class.compute_ewc_loss(learnable_agent, task_idx)
                    
                    #print("task idx is ", task_idx)
                    #print("orig loss is ", loss)
                    #print("EWC loss is ", args.ewc_lambda*ewc_loss)
                    #print("\n")
                    loss += args.ewc_lambda * ewc_loss


                optim.zero_grad()
                loss.backward()

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

                stat["loss"].feed(loss.detach().item())
                stat["grad_norm"].feed(g_norm.detach().item())

            count_factor = args.num_player if args.method == "vdn" else 1
            print("EPOCH: %d" % act_epoch_cnt)
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
                        eval_optim = torch.optim.Adam(few_shot_learnable_agent.online_net.parameters(), lr=args.lr,
                                                      eps=args.eps)
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
                        eval_stat = common_utils.MultiCounter(args.save_dir)

                        for eval_epoch in range(args.eval_num_epoch):
                            print("beginning of eval epoch: ", eval_epoch)
                            eval_stat.reset()
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
                            eval_stat.summary(eval_epoch)
                        eval_context.pause()
                        fs_force_save_name = "model_epoch%d_few_shot_%d" % (act_epoch_cnt, eval_fixed_ag_idx)
                        few_shot_model_saved = saver.save(None, few_shot_learnable_agent.online_net.state_dict(), force_save_name=fs_force_save_name)      
                        print("few shot model saved: %s "%(few_shot_model_saved))

                ## zero shot learnable agent. 
                zs_force_save_name = "model_epoch%d_zero_shot" %(act_epoch_cnt)
                zero_shot_model_saved = saver.save(None, learnable_agent.online_net.state_dict(), force_save_name=zs_force_save_name)
                print("zero shot model saved: %s "%(zero_shot_model_saved))

            gc.collect()
            context.resume()
            print("==========")
    
        episodic_memory.append(replay_buffer)
