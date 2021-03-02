# Lifelong Hanabi

## Introduction

This repo contains code and models for [Continuous Coordination As a Realistic Scenario for Lifelong Learning](), a multi-agent lifelong learning testbed that supports both zero-shot and few-shot settings. Our setup is based on [hanabi](https://github.com/deepmind/hanabi-learning-environment) — a partially-observable, fully cooperative multi-agent game. 


<br/>

![LifelongHanabi_setup](https://user-images.githubusercontent.com/43013139/107289273-c4f17680-6a32-11eb-93c2-0a70a9e342f3.png)

<br/>



The code is built on top of the [Other-Play & Simplified Action Decoder in Hanabi](https://github.com/facebookresearch/hanabi_SAD) repo.



## Requirements and Installation
The build process is tested with Python 3.7,  PyTorch 1.5.1, CUDA 10.1, cudnn 7.6, and nccl 2.4

```
clone the repo
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


Lifelong Hanabi consists of 3 phases: 1- Pre-training, 2- Continual training, 3- Testing 

### 1- Pre-Trained Agents

Run the following command to download the pre-trained agents used in the paper.
```bash
pip install gdown
gdown --id 1SHnPa5TkE9WuQPq7lCvshkDVBhAG_QNm
```
You can find a detailed description of each agent's configs and architectures here:
`misc/Pre-trained agents pool for Continual Hanabi.xlsx`

To run any `.sh` file, update `<path-to-pretrained-model-pool-dir>` and `<save-dir>`, accordingly.

#### Reproduce the Cross-Play matrix:
To evaluate all the agents with each other, simply run:
```bash
cd pyhanabi
sh generate_cp.sh
```

### 2- Continual Training
To train the learner with a set of 5 partners using [ER](https://arxiv.org/abs/1902.10486) method, run:
```bash
cd pyhanabi
sh tools/continual_learning_scripts/ER_easy_interactive.sh
```
Zero-shot and few-shot checkpoints will be stored in `<save-dir>`. 

To log the continual training results, run:

```bash
cd pyhanabi
sh tools/continual_evaluation.sh
```
This step needs a [wandb](https://wandb.ai/home) account to plot the results. 

### 3- Testing
To evaluate the learner against a set of unseen agents, run:
```bash
cd pyhanabi
python final_evaluation.py
```
This step also needs a [wandb](https://wandb.ai/home) account to plot the results. 
## Plot results
All the plots and experiment details are available at [wandb report](https://wandb.ai/akileshbadrinaaraayanan/ContPlay_Hanabi_complete/reports/Lifelong-Hanabi-Experiments--VmlldzozOTk2NjY).


