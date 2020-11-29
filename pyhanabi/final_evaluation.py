''' Evaluating all the checkpoints saved periodically during train args.eval_freq
Usually done only in Mila cluster as we need wandb to log.
Requires only 1 GPU.
Please ignore ./models/iql_2p/iql_2p_228.pthw for the final evaluation ...
Sample usage: 
python final_evaluation.py 
--weight_1_dir /miniscratch/akb/cont_hanabi_models/exps/ind_RB_few_shot_ER_noeval_easy 
--weight_2 ../models/iql_2p/iql_2p_6.pthw ../models/iql_2p/iql_2p_11.pthw ../models/iql_2p/iql_2p_113.pthw ../models/iql_2p/iql_2p_210.pthw ../models/iql_2p/iql_2p_5.pthw 
--num_player 2
note the last arg of --weight_2 is the self-play that is the agent that was being trained.
'''

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
import r2d2_gru as r2d2_gru
import r2d2_lstm as r2d2_lstm
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

        learnable_pretrain = True
        if i == 0 and learnable_agent_args['load_learnable_model'] != "":
            agent_args_file = learnable_agent_args['load_learnable_model'][:-4]+"txt"
        elif i == 0:
            learnable_pretrain = False
        else:
            agent_args_file = weight_file[:-4] + "txt"

        if learnable_pretrain == True:
            with open(agent_args_file, 'r') as f:
                agent_args = {**json.load(f)}

        if learnable_pretrain == False:
            rnn_type = learnable_agent_args['rnn_type']
            rnn_hid_dim = learnable_agent_args['rnn_hid_dim']
            num_fflayer = learnable_agent_args['num_fflayer']
            num_rnn_layer = learnable_agent_args['num_rnn_layer']
        elif learnable_pretrain == True:
            rnn_type = agent_args['rnn_type']
            rnn_hid_dim = agent_args['rnn_hid_dim']
            num_fflayer = agent_args['num_fflayer']
            num_rnn_layer = agent_args['num_rnn_layer']

        if rnn_type == "lstm":
            import r2d2_lstm as r2d2
        elif rnn_type == "gru":
            import r2d2_gru as r2d2 

        agent = r2d2.R2D2Agent(
        False, 3, 0.999, 0.9, device, input_dim, rnn_hid_dim, output_dim, num_fflayer, num_rnn_layer, 5, False
        ).to(device)

        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)

    scores = []
    perfect = 0
    for i in range(num_run):
        flag = np.random.randint(0, num_player)

        if flag == 0:
            new_agents = [agents[0], agents[1]]
        elif flag == 1:
            new_agents = [agents[1], agents[0]]

        _, _, score, p = evaluate(
            new_agents,
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

    if learnable_agent_args['load_learnable_model'] != "":
        lr_str = learnable_agent_args['load_learnable_model'].split("/")[-1].split(".")[0]
    else:
        lr_str = "no_pretrain"
    exp_name = "final_evaluation_" + lr_str+"_fixed_"+str(len(learnable_agent_args['load_fixed_models']))+"_"+learnable_agent_args['ll_algo'] 

    wandb.init(project="ContPlay_Hanabi_complete", name=exp_name)
    wandb.config.update(learnable_agent_args)

    assert os.path.exists(args.weight_1_dir)    
    weight_1 = []
    weight_1 = glob.glob(args.weight_1_dir+"/*.pthw")
    weight_1.sort(key=os.path.getmtime)

    print("weight 1 after sorting ", weight_1)
    ## check if everything in weights_2 exist
    for ag2 in args.weight_2:
        assert os.path.exists(ag2)

    for ag1_idx, ag1 in enumerate(weight_1):
        ag1_name = ag1.split("/")[-1].split("_")[-1]
        act_epoch_cnt = int(ag1.split("/")[-1].split("_")[1][5:])
        
        ### this is for different zero-shot evaluations...
        if ag1_name == "shot.pthw":
            for fixed_agent_idx in range(len(args.weight_2)):
                weight_files = [ag1, args.weight_2[fixed_agent_idx]]
                mean_score, sem, perfect_rate = evaluate_legacy_model(weight_files, 1000, 1, 0, learnable_agent_args, num_run=10)
                wandb.log({"epoch_zeroshot": act_epoch_cnt, "final_eval_score_zeroshot_"+str(fixed_agent_idx): mean_score, "perfect_zeroshot_"+str(fixed_agent_idx): perfect_rate, "sem_zeroshot_"+str(fixed_agent_idx):sem})
        else:
            ## for different few shot evaluations ... 
            for i in range(len(args.weight_2)):
                if ag1_name == str(i)+".pthw":
                    weight_files = [ag1, args.weight_2[i]]

            mean_score, sem, perfect_rate = evaluate_legacy_model(weight_files, 1000, 1, 0, learnable_agent_args, num_run=10)
            wandb.log({"epoch_fewshot": act_epoch_cnt, "final_eval_score_fewshot_"+ag1_name.split(".")[0]: mean_score, "perfect_fewshot_"+ag1_name.split(".")[0]: perfect_rate, "sem_fewshot_"+ag1_name.split(".")[0]:sem})

        
        