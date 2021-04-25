import time
import os
import sys
import argparse
import pprint
import json
import gc
import glob
import numpy as np
import torch
from torch import nn

from create import create_envs, create_threads, ContActGroup
import common_utils
import rela
import utils
import ewc as ewc


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--load_model_dir", type=str, default="../models/iql_2p")
    parser.add_argument("--log_file", type=str, default="train.log")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_obs", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--pred_weight", type=float, default=0)
    parser.add_argument("--num_eps", type=int, default=80)

    parser.add_argument("--load_learnable_model", type=str, default="")
    parser.add_argument("--resume_cont_training", action="store_true", default=False)
    parser.add_argument("--load_partner_models", type=str, nargs="+", default="")

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--eta", type=float, default=0.9, help="eta for aggregate priority"
    )
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--is_rand", action="store_true", default=True)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)

    # optimization/training settings
    parser.add_argument(
        "--initial_lr", type=float, default=0.1, help="Initial learning rate"
    )
    parser.add_argument(
        "--final_lr", type=float, default=6.25e-5, help="Final learning rate"
    )
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
    parser.add_argument("--max_train_steps", type=int, default=10000000)
    parser.add_argument("--max_eval_steps", type=int, default=50000)
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
    parser.add_argument("--episodic_replay_buffer_size", type=int, default=8000)
    parser.add_argument("--store_sliced_rb", action="store_true", default=False)
    parser.add_argument(
        "--priority_exponent",
        type=float,
        default=0.6,
        help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight",
        type=float,
        default=0.4,
        help="prioritized replay beta",
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)
    parser.add_argument(
        "--eval_num_thread", type=int, default=40, help="#eval_thread_loop"
    )
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
    parser.add_argument("--eval_method", type=str, default="few_shot")

    ## EWC settings
    parser.add_argument("--online", type=int, default=0)
    parser.add_argument("--ewc_lambda", type=float, default=5000)
    parser.add_argument("--ewc_gamma", type=float, default=1)

    ## args dump settings
    parser.add_argument("--args_dump_name", type=str, default="ER_commandline_args.txt")

    args = parser.parse_args()
    assert args.method in ["vdn", "iql"]
    assert args.ll_algo in ["ER", "AGEM", "EWC", "Naive"]
    assert args.eval_method in ["zero_shot", "few_shot"]
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.args_dump_name = "cont_args.txt"
    with open(f"{args.save_dir}/{args.args_dump_name}", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    logger_path = os.path.join(args.save_dir, args.log_file)
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

    ## this is the learnable agent.
    learnable_sad = False

    if args.load_learnable_model:
        learnable_agent_name = args.load_learnable_model.split("/")[-1].split(".")[0]
        with open(f"{args.load_model_dir}/{learnable_agent_name}.txt") as f:
            learnable_agent_args = {**json.load(f)}

        rnn_type = learnable_agent_args["rnn_type"]
        rnn_hid_dim = learnable_agent_args["rnn_hid_dim"]
        num_fflayer = learnable_agent_args["num_fflayer"]
        num_rnn_layer = learnable_agent_args["num_rnn_layer"]

        if "sad" in args.load_learnable_model:
            learnable_sad = True
    else:
        rnn_type = args.rnn_type
        rnn_hid_dim = args.rnn_hid_dim
        num_fflayer = args.num_fflayer
        num_rnn_layer = args.num_rnn_layer

    if rnn_type == "lstm":
        import r2d2_lstm as r2d2_learnable
    elif rnn_type == "gru":
        import r2d2_gru as r2d2_learnable

    learnable_games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.hand_size,
        args.train_bomb,
        explore_eps,
        args.max_len,
        learnable_sad,
        args.shuffle_obs,
        args.shuffle_color,
    )

    learnable_agent = r2d2_learnable.R2D2Agent(
        (args.method == "vdn"),
        args.multi_step,
        args.gamma,
        args.eta,
        args.train_device,
        learnable_games[0].feature_size(),
        rnn_hid_dim,
        learnable_games[0].num_action(),
        num_fflayer,
        num_rnn_layer,
        args.hand_size,
        False,
        sad=learnable_sad,
    )

    learnable_agent.sync_target_with_online()

    if args.load_learnable_model:
        print("***** loading pretrained model for learnable agent *****")
        utils.load_weight(
            learnable_agent.online_net, args.load_learnable_model, args.train_device
        )
        print("***** done *****")

    if args.resume_cont_training:
        print("***** resuming continual training ***** ")
        learnable_agent_ckpts = glob.glob(f"{args.save_dir}/*_zero_shot.pthw")
        learnable_agent_ckpts.sort(key=os.path.getmtime)
        print("restoring from ... ", learnable_agent_ckpts[-1])
        utils.load_weight(
            learnable_agent.online_net, learnable_agent_ckpts[-1], args.train_device
        )
        epoch_restore = int(
            learnable_agent_ckpts[-1].split("/")[-1].split(".")[0].split("_")[1][5:]
        )
        print("epoch restore is ... ", epoch_restore)

    learnable_agent = learnable_agent.to(args.train_device)
    print(learnable_agent)

    if args.ll_algo == "EWC":
        ewc_class = ewc.EWC(args)

    eval_agent = learnable_agent.clone(args.train_device, {"vdn": False})

    fixed_learnable_agent = learnable_agent.clone(args.train_device, {"vdn": False})

    partner_agents = []
    episodic_memory = []

    for opp_idx, opp_model in enumerate(args.load_partner_models):
        opp_model_name = opp_model.split("/")[-1].split(".")[0]
        with open(f"{args.load_model_dir}/{opp_model_name}.txt") as f:
            opp_model_args = {**json.load(f)}

        opp_sad = False
        if "sad" in opp_model:
            opp_sad = True

        if opp_model_args["rnn_type"] == "lstm":
            import r2d2_lstm as r2d2_partner
        elif opp_model_args["rnn_type"] == "gru":
            import r2d2_gru as r2d2_partner

        partner_games = create_envs(
            args.num_thread * args.num_game_per_thread,
            args.seed,
            args.num_player,
            args.hand_size,
            args.train_bomb,
            explore_eps,
            args.max_len,
            opp_sad,
            args.shuffle_obs,
            args.shuffle_color,
        )

        partner_agent = r2d2_partner.R2D2Agent(
            (args.method == "vdn"),
            args.multi_step,
            args.gamma,
            args.eta,
            args.train_device,
            partner_games[0].feature_size(),
            opp_model_args["rnn_hid_dim"],
            partner_games[0].num_action(),
            opp_model_args["num_fflayer"],
            opp_model_args["num_rnn_layer"],
            args.hand_size,
            False,
            sad=opp_sad,
        )

        if opp_model:
            print(
                "***** loading pretrained model for partner " + str(opp_idx) + " *****"
            )
            utils.load_weight(partner_agent.online_net, opp_model, args.train_device)
            print("***** done *****")

        partner_agent = partner_agent.to(args.train_device)
        partner_agents.append(partner_agent)

    ### Populate episodic memory
    if args.resume_cont_training:
        task_idx_restore = epoch_restore // args.num_epoch
        total_epochs = epoch_restore

        learnable_agent_populate_em = learnable_agent.clone(
            args.train_device, {"vdn": False}
        )
        for emi in range(task_idx_restore):
            model_to_load = (
                f"{args.save_dir}/model_epoch{(emi+1)*args.num_epoch}_zero_shot.pthw"
            )
            utils.load_weight(
                learnable_agent_populate_em.online_net, model_to_load, args.train_device
            )

            em_replay_buffer = rela.RNNPrioritizedReplay(
                args.replay_buffer_size,
                args.seed,
                args.priority_exponent,
                args.priority_weight,
                args.prefetch,
            )

            cont_sad = False
            if "sad" in args.load_partner_models[emi] or learnable_sad == True:
                cont_sad = True

            em_games = create_envs(
                args.num_thread * args.num_game_per_thread,
                args.seed,
                args.num_player,
                args.hand_size,
                args.train_bomb,
                explore_eps,
                args.max_len,
                cont_sad,
                args.shuffle_obs,
                args.shuffle_color,
            )

            print("Creating ContActGroup for restoring buffer with partner " + str(emi))
            em_act_group = ContActGroup(
                args.method,
                args.act_device,
                [learnable_agent_populate_em, partner_agents[emi]],
                args.num_thread,
                args.num_game_per_thread,
                args.multi_step,
                args.gamma,
                args.eta,
                args.max_len,
                args.num_player,
                args.is_rand,
                em_replay_buffer,
            )
            em_context, em_threads = create_threads(
                args.num_thread,
                args.num_game_per_thread,
                em_act_group.actors,
                em_games,
            )
            em_act_group.start()
            em_context.start()

            while em_replay_buffer.size() < args.replay_buffer_size:
                print(
                    "warming up em replay buffer:" + str(emi), em_replay_buffer.size()
                )
                time.sleep(1)
            em_context.pause()
            episodic_memory.append(em_replay_buffer)

            if args.ll_algo == "EWC":
                em_batch, em_weight = em_replay_buffer.sample(
                    args.batchsize, args.train_device
                )

                em_stat = common_utils.MultiCounter(args.save_dir)
                em_stat.reset()

                em_priority = ewc_class.estimate_fisher(
                    learnable_agent_populate_em, em_batch, em_weight, em_stat, emi
                )

                em_priority = rela.aggregate_priority(
                    em_priority.cpu(), em_batch.seq_len.cpu(), args.eta
                )
                em_replay_buffer.update_priority(em_priority[: args.batchsize])

        act_steps = utils.get_act_steps(args.save_dir, epoch_restore)
    else:
        task_idx_restore = 0
        total_epochs = 0

    for task_idx, partner_agent in enumerate(partner_agents):
        if task_idx < task_idx_restore:
            continue

        if args.decay_lr:
            lr = max(args.initial_lr * args.lr_gamma ** (task_idx), args.final_lr)
        else:
            lr = args.final_lr

        if args.optim_name == "Adam":
            optim = torch.optim.Adam(
                learnable_agent.online_net.parameters(), lr=lr, eps=args.eps
            )
        elif args.optim_name == "SGD":
            optim = torch.optim.SGD(
                learnable_agent.online_net.parameters(),
                lr=lr,
                momentum=args.sgd_momentum,
            )

        replay_buffer = rela.RNNPrioritizedReplay(
            args.replay_buffer_size,
            args.seed,
            args.priority_exponent,
            args.priority_weight,
            args.prefetch,
        )
        cont_sad = False
        if "sad" in args.load_partner_models[task_idx] or learnable_sad == True:
            cont_sad = True

        games = create_envs(
            args.num_thread * args.num_game_per_thread,
            args.seed,
            args.num_player,
            args.hand_size,
            args.train_bomb,
            explore_eps,
            args.max_len,
            cont_sad,
            args.shuffle_obs,
            args.shuffle_color,
        )

        print("Creating ContActGroup for training with partner " + str(task_idx))
        act_group = ContActGroup(
            args.method,
            args.act_device,
            [learnable_agent, partner_agent],
            args.num_thread,
            args.num_game_per_thread,
            args.multi_step,
            args.gamma,
            args.eta,
            args.max_len,
            args.num_player,
            args.is_rand,
            replay_buffer,
        )

        assert args.shuffle_obs == False, "not working with 2nd order aux"
        context, threads = create_threads(
            args.num_thread,
            args.num_game_per_thread,
            act_group.actors,
            games,
        )
        act_group.start()
        context.start()

        if args.resume_cont_training and task_idx == task_idx_restore:
            burn_in_frames = args.replay_buffer_size
        else:
            burn_in_frames = args.burn_in_frames

        while replay_buffer.size() < burn_in_frames:
            print("warming up replay buffer:", replay_buffer.size())
            time.sleep(1)

        print("Success, Done")
        print("=======================")

        stat = common_utils.MultiCounter(args.save_dir)
        tachometer = utils.Tachometer()
        stopwatch = common_utils.Stopwatch()

        task_done = False
        if args.resume_cont_training and task_idx == task_idx_restore:
            initial_epoch = epoch_restore - (task_idx_restore * args.num_epoch)
        else:
            initial_epoch = 0

        for epoch in range(initial_epoch, args.num_epoch):
            total_epochs += 1
            print("beginning of epoch: ", total_epochs)
            print(common_utils.get_mem_usage())
            tachometer.start()
            stat.reset()
            stopwatch.reset()

            for batch_idx in range(args.epoch_len):
                learnable_agent_actors = [x[0] for x in act_group.actors]
                if args.resume_cont_training:
                    total_task_steps = utils.get_num_acts(
                        learnable_agent_actors, act_steps[epoch_restore - 1]
                    )
                else:
                    total_task_steps = utils.get_num_acts(learnable_agent_actors, 0)

                if total_task_steps > args.max_train_steps:
                    print(
                        "Training with agent  ",
                        task_idx,
                        " is done after ",
                        total_task_steps,
                    )
                    task_done = True
                    break

                num_update = batch_idx + epoch * args.epoch_len
                if num_update % args.num_update_between_sync == 0:
                    learnable_agent.sync_target_with_online()
                if num_update % args.actor_sync_freq == 0:
                    act_group.update_model(learnable_agent)

                torch.cuda.synchronize()
                stopwatch.time("sync and updating")

                batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
                # Looping over the replay buffers of previous tasks.
                ## The below portion of code creates a batch consisting of equal sized samples (samples_per_task) from previous RBs and
                ## concatenates to the batch of samples from the current task.

                if args.ll_algo == "ER":
                    batch, weight, episodic_memory = utils.make_batch_ER(
                        args, episodic_memory, batch, weight, stat, learnable_agent
                    )
                    stopwatch.time("sample data")

                elif args.ll_algo == "AGEM":
                    b, w, episodic_memory = utils.make_batch_AGEM(
                        args, episodic_memory, stat, learnable_agent
                    )
                    stopwatch.time("sample data")

                    if task_idx > 0:
                        loss_replay, _ = learnable_agent.loss(b, args.pred_weight, stat)
                        loss_replay = (loss_replay * w).mean()
                        loss_replay.backward()

                        grad_rep = utils.get_grad_list(learnable_agent)
                        ## reset gradients (with A-GEM the gradients of replayed batch should only be used as inequality constraints)
                        optim.zero_grad()

                loss, priority = learnable_agent.loss(batch, args.pred_weight, stat)

                priority = rela.aggregate_priority(
                    priority.cpu(), batch.seq_len.cpu(), args.eta
                )

                loss = (loss * weight).mean()
                if args.ll_algo == "EWC":
                    if epoch == (args.num_epoch - 1) and batch_idx == (
                        args.epoch_len - 1
                    ):
                        _ = ewc_class.estimate_fisher(
                            learnable_agent, batch, weight, stat, task_idx
                        )

                    ewc_loss = ewc_class.compute_ewc_loss(learnable_agent, task_idx)
                    loss += args.ewc_lambda * ewc_loss

                optim.zero_grad()
                loss.backward()

                if args.ll_algo == "AGEM" and task_idx > 0:
                    ## reorganize the gradient of the current batch as a single vector
                    grad_cur = utils.get_grad_list(learnable_agent)
                    ## project gradient and return new learnable agent
                    learnable_agent = utils.grad_proj(
                        grad_cur, grad_rep, learnable_agent
                    )

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
                replay_buffer.update_priority(priority[: args.batchsize])
                stopwatch.time("updating priority")

                stat["loss"].feed(loss.detach().item())
                stat["grad_norm"].feed(g_norm.detach().item())

            count_factor = args.num_player if args.method == "vdn" else 1
            print("EPOCH: %d" % total_epochs)
            learnable_agent_actors = [x[0] for x in act_group.actors]
            if args.resume_cont_training:
                tachometer.lap(
                    learnable_agent_actors,
                    replay_buffer,
                    args.epoch_len * args.batchsize,
                    count_factor,
                    act_steps[epoch_restore - 1],
                )
            else:
                tachometer.lap(
                    learnable_agent_actors,
                    replay_buffer,
                    args.epoch_len * args.batchsize,
                    count_factor,
                    0,
                )

            stopwatch.summary()
            stat.summary(epoch)

            eval_seed = (9917 + epoch * 999999) % 7777777

            if (epoch + 1) % args.eval_freq == 0 or epoch == 0 or task_done == True:
                context.pause()
                for eval_partner_ag_idx, eval_partner_agent in enumerate(
                    partner_agents + [fixed_learnable_agent]
                ):
                    print(
                        "evaluating learnable agent with partner %d "
                        % eval_partner_ag_idx
                    )

                    if args.eval_method == "few_shot":
                        print("Few Shot Learning ...")
                        few_shot_learnable_agent = learnable_agent.clone(
                            args.train_device, {"vdn": False}
                        )
                        if args.optim_name == "Adam":
                            eval_optim = torch.optim.Adam(
                                few_shot_learnable_agent.online_net.parameters(),
                                lr=args.final_lr,
                                eps=args.eps,
                            )
                        elif args.optim_name == "SGD":
                            eval_optim = torch.optim.SGD(
                                few_shot_learnable_agent.online_net.parameters(),
                                lr=args.final_lr,
                                momentum=args.sgd_momentum,
                            )
                        eval_replay_buffer = rela.RNNPrioritizedReplay(
                            args.eval_replay_buffer_size,
                            eval_seed,
                            args.priority_exponent,
                            args.priority_weight,
                            args.prefetch,
                        )
                        eval_sad = False
                        if eval_partner_ag_idx != (
                            len(partner_agents + [fixed_learnable_agent]) - 1
                        ):
                            if "sad" in args.load_partner_models[eval_partner_ag_idx]:
                                eval_sad = True
                        elif learnable_sad == True:
                            eval_sad = True

                        eval_games = create_envs(
                            args.eval_num_thread * args.eval_num_game_per_thread,
                            eval_seed,
                            args.num_player,
                            args.hand_size,
                            args.train_bomb,
                            explore_eps,
                            args.max_len,
                            eval_sad,
                            args.shuffle_obs,
                            args.shuffle_color,
                        )

                        print(
                            "Creating ContActGroup for finetuning with partner "
                            + str(eval_partner_ag_idx)
                        )
                        eval_act_group = ContActGroup(
                            args.method,
                            args.act_device,
                            [few_shot_learnable_agent, eval_partner_agent],
                            args.eval_num_thread,
                            args.eval_num_game_per_thread,
                            args.multi_step,
                            args.gamma,
                            args.eta,
                            args.max_len,
                            args.num_player,
                            args.is_rand,
                            eval_replay_buffer,
                        )
                        eval_context, eval_threads = create_threads(
                            args.eval_num_thread,
                            args.eval_num_game_per_thread,
                            eval_act_group.actors,
                            eval_games,
                        )
                        eval_act_group.start()
                        eval_context.start()
                        while eval_replay_buffer.size() < args.eval_burn_in_frames:
                            print(
                                "warming up replay buffer:", eval_replay_buffer.size()
                            )
                            time.sleep(1)
                        eval_tachometer = utils.Tachometer(iseval=True)
                        eval_stat = common_utils.MultiCounter(args.save_dir)
                        eval_done = False
                        for eval_epoch in range(args.eval_num_epoch):
                            print("beginning of eval epoch: ", eval_epoch)
                            eval_stat.reset()
                            eval_tachometer.start()
                            for eval_batch_idx in range(args.eval_epoch_len):
                                eval_learnable_agent_actors = [
                                    x[0] for x in eval_act_group.actors
                                ]
                                total_eval_steps = utils.get_num_acts(
                                    eval_learnable_agent_actors, 0
                                )
                                if total_eval_steps > args.max_eval_steps:
                                    print(
                                        "Finetuning with ",
                                        eval_partner_ag_idx,
                                        " is done after ",
                                        total_eval_steps,
                                    )
                                    eval_done = True
                                    break
                                eval_num_update = (
                                    eval_batch_idx + eval_epoch * args.eval_epoch_len
                                )
                                if (
                                    eval_num_update % args.eval_num_update_between_sync
                                    == 0
                                ):
                                    few_shot_learnable_agent.sync_target_with_online()
                                if eval_num_update % args.eval_actor_sync_freq == 0:
                                    eval_act_group.update_model(
                                        few_shot_learnable_agent
                                    )

                                batch, weight = eval_replay_buffer.sample(
                                    args.batchsize, args.train_device
                                )

                                torch.cuda.synchronize()
                                stopwatch.time("sync and updating")

                                loss, priority = few_shot_learnable_agent.loss(
                                    batch, args.pred_weight, eval_stat
                                )

                                priority = rela.aggregate_priority(
                                    priority.cpu(), batch.seq_len.cpu(), args.eta
                                )

                                loss = (loss * weight).mean()
                                loss.backward()

                                torch.cuda.synchronize()

                                g_norm = torch.nn.utils.clip_grad_norm_(
                                    few_shot_learnable_agent.online_net.parameters(),
                                    args.grad_clip,
                                )
                                eval_optim.step()
                                eval_optim.zero_grad()

                                torch.cuda.synchronize()

                                ## current task priority
                                eval_replay_buffer.update_priority(
                                    priority[: args.batchsize]
                                )

                                eval_stat["loss"].feed(loss.detach().item())
                                eval_stat["grad_norm"].feed(g_norm.detach().item())

                            eval_learnable_agent_actors = [
                                x[0] for x in eval_act_group.actors
                            ]
                            eval_tachometer.lap(
                                eval_learnable_agent_actors,
                                eval_replay_buffer,
                                args.eval_epoch_len * args.batchsize,
                                count_factor,
                                0,
                            )
                            eval_stat.summary(eval_epoch)
                            if eval_done == True:
                                break
                        eval_context.pause()
                        fs_force_save_name = "model_epoch%d_few_shot_%d" % (
                            total_epochs,
                            eval_partner_ag_idx,
                        )
                        few_shot_model_saved = saver.save(
                            None,
                            few_shot_learnable_agent.online_net.state_dict(),
                            force_save_name=fs_force_save_name,
                        )
                        print("few shot model saved: %s " % (few_shot_model_saved))

                ## zero shot learnable agent.
                zs_force_save_name = "model_epoch%d_zero_shot" % (total_epochs)
                zero_shot_model_saved = saver.save(
                    None,
                    learnable_agent.online_net.state_dict(),
                    force_save_name=zs_force_save_name,
                )
                print("zero shot model saved: %s " % (zero_shot_model_saved))

            if task_done == True:
                break
            gc.collect()
            context.resume()
            print("==========")

        if args.store_sliced_rb:
            if args.episodic_replay_buffer_size < replay_buffer.size():
                replay_buffer.slice(
                    replay_buffer.size() - args.episodic_replay_buffer_size
                )
            else:
                print(
                    "Episodic replay buffer size can't be larger than replay buffer size!!!"
                )
        episodic_memory.append(replay_buffer)
