# Lifelong Hanabi

## Introduction

This repo contains code and models for [Continuous Coordination As a Realistic Scenario for Lifelong Learning](https://arxiv.org/pdf/2103.03216.pdf), a multi-agent lifelong learning testbed that supports both zero-shot and few-shot settings. Our setup is based on [hanabi](https://github.com/deepmind/hanabi-learning-environment) — a partially-observable, fully cooperative multi-agent game. 


<br/>

![LifelongHanabi_setup](https://user-images.githubusercontent.com/43013139/107289273-c4f17680-6a32-11eb-93c2-0a70a9e342f3.png)

<br/>



Lifelong Hanabi consists of 3 phases: 1- [Pre-training](https://github.com/chandar-lab/Lifelong-Hanabi/blob/master/README.md#1--pre-trained-agents), 2- [Continual training](https://github.com/chandar-lab/Lifelong-Hanabi#2--continual-training), 3- [Testing](https://github.com/chandar-lab/Lifelong-Hanabi#3--testing). 

The code is built on top of the [Other-Play & Simplified Action Decoder in Hanabi](https://github.com/facebookresearch/hanabi_SAD) repo.



## Requirements and Installation
The build process is tested with Python 3.7,  PyTorch 1.5.1, CUDA 10.1, cudnn 7.6, and nccl 2.4

```bash
# clone the repo
git clone --recursive git@github.com:chandar-lab/Lifelong-Hanabi.git
cd Lifelong-Hanabi

# create new conda env
conda create -n lifelong_hanabi python=3.7
conda activate lifelong_hanabi
pip install -r requirements.txt

# build 
mkdir build
cd build
cmake ..
make
mv hanalearn.cpython-37m-x86_64-linux-gnu.so ..
mv rela/rela.cpython-37m-x86_64-linux-gnu.so ..
mv hanabi-learning-environment/libpyhanabi.so ../hanabi-learning-environment/

```
Once the building is done and the `.so` files are moved to their required places as mentioned above, every subsequent time you just need to run:
```bash
conda activate lifelong_hanabi
export PYTHONPATH=/path/to/lifelong_hanabi:$PYTHONPATH
export OMP_NUM_THREADS=1
```
## Run

### 1- Pre-Trained Agents

Run the following command to download the pre-trained agents used in the paper.
```bash
pip install gdown
gdown --id 1rpmTPIT-g026pdQfAwHoE4i8tP7Qj2vI
```
You can find a detailed description of each agent's configs and architectures here:
`results/Pre-trained agents pool for Continual Hanabi.xlsx`

`all_pretrained_pool.zip` contains the pre-trained agents we used in our experiments (this can be extended by further training more expert Hanabi players).

To run any `.sh` file, update `<path-to-pretrained-model-pool-dir>` and `<save-dir>`, accordingly.
Important flags are:
|Flags | Description|
|:-------------|:-------------|
| `--sad`                      |enables Simplified Action Decoder|
| `--pred_weight`            |weight for auxiliary task (typically 0.25)|
| `--shuffle_color`          |enable other-play|
| `--seed`          |seed|

For details of other hyperparameters refer code and/or paper. 

#### * Pre-train a new agent through self-play:
A sample script is provided in `pyhanabi/tools/pretrain.sh` that can be run:
```bash
cd pyhanabi
sh tools/pretrain.sh
```

#### * Reproduce the cross-play matrix:
To evaluate all the agents with each other, run:
```bash
cd pyhanabi
sh generate_cp.sh
```
Cross-play matrix from our runs can be found in `results/scores_data_100_nrun5.csv` (`results/sem_data_100_nrun5.csv` contains s.e.m)

### 2- Continual Training
To train the learner with a set of 5 partners using for eg. [ER](https://arxiv.org/abs/1902.10486) method, run:
```bash
cd pyhanabi
sh tools/continual_learning_scripts/ER_easy_interactive.sh
```
Zero-shot and few-shot checkpoints will be stored in `<save-dir>`. 
Similar scripts are available for all the other algorithms described in paper. 

In order to log the continual training results (from the above checkpoints stored in `<save-dir>`), run:

```bash
cd pyhanabi
sh tools/continual_evaluation.sh
```

#### * Add your lifelong algorithm:
In order to implement a new lifelong learning algorithm, depending on the type of the algorithm you can modify one of the following:

**Memory based methods:** [episodic_memory](https://github.com/chandar-lab/Lifelong-Hanabi/blob/1c79a5349e70419f45b34e13b90fb003109e85ec/pyhanabi/continual_training.py#L378) is a list of the replay buffers from previous tasks. You can change the way the batch is collected like [here](https://github.com/chandar-lab/Lifelong-Hanabi/blob/1c79a5349e70419f45b34e13b90fb003109e85ec/pyhanabi/utils.py#L264) or the way this replayed batch constrains the current gradients [code](https://github.com/chandar-lab/Lifelong-Hanabi/blob/1c79a5349e70419f45b34e13b90fb003109e85ec/pyhanabi/continual_training.py#L567).

**Regularization based methods:** [Here](https://github.com/chandar-lab/Lifelong-Hanabi/blob/1c79a5349e70419f45b34e13b90fb003109e85ec/pyhanabi/continual_training.py#L387) is where the fisher information matrix at the end of each task is estimated. You can modify the way corresponding regularization loss is calculated and added to the original loss [here](https://github.com/chandar-lab/Lifelong-Hanabi/blob/1c79a5349e70419f45b34e13b90fb003109e85ec/pyhanabi/continual_training.py#L561). 

**Training regimes:** These are a list of hyper-parameters which has been shown [here](https://arxiv.org/abs/2006.06958) that have high impact on the performance of the lifelong learning algorithms.
|Flags | Description|
|:-------------|:-------------|
| `--optim_name`                      |optimizer|
| `--batchsize`            |batch size|
| `--decay_lr`          |learning rate decay|
| `--initial_lr`          |initial learning rate|

### 3- Testing
To evaluate the learner against a set of unseen agents, run:
```bash
cd pyhanabi
sh tools/testing.sh
```
Logging continual training results and testing requires a [wandb](https://wandb.ai/home) account to plot the results. 

## Plot results
All the plots and experiment details are available at [wandb report](https://wandb.ai/akileshbadrinaaraayanan/ContPlay_Hanabi_complete/reports/Lifelong-Hanabi-Experiments--VmlldzozOTk2NjY).

* Other code used to reproduce figures in the paper can be found in `results`

## Citation:

If you found this work useful, please consider citing our [paper](https://arxiv.org/abs/2103.03216).
```
@misc{nekoei2021continuous,
      title={Continuous Coordination As a Realistic Scenario for Lifelong Learning},
      author={Hadi Nekoei and Akilesh Badrinaaraayanan and Aaron Courville and Sarath Chandar},
      year={2021},
      eprint={2103.03216},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
