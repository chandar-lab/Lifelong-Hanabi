# Lifelong Hanabi

## Introduction

This repo contains code and models for [Continuous coordination as A Realistic Scenario for Lifelong Learning]().


<br/>

![LifelongHanabi_setup](https://user-images.githubusercontent.com/43013139/107289273-c4f17680-6a32-11eb-93c2-0a70a9e342f3.png)

<br/>



The code is built on top of the [Other-Play & Simplified Action Decoder in Hanabi](https://github.com/facebookresearch/hanabi_SAD) repo.

If you found this work useful, please consider citing our [paper](). 
```

```

## Requirements and Installation
Detailed description of installation steps is given [here](https://docs.google.com/document/d/1mYGzWU_5ELupcNe2YsWrFunSBhTVXXy9Fx694qCL2pA/edit?usp=sharing). 

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

!!! To run any .sh file, update `<path-to-pretrained-model-pool-dir>` , accordingly.

#### Reproduce the Cross-Play matrix:
To evaluate all the agents with each other, simply run:
```bash
cd pyhanabi
sh generate_cp.sh
```

### 2- Continual Training

```bash
cd pyhanabi
sh tools/continual_learning_scripts/ER_easy_interactive.sh
```
This step creates a folder called ... which contains the zero-shot and few-shot model checkpoints. 

To plot the continual training results, run
```bash
cd pyhanabi
sh tools/continual_evaluation.sh
```

### 3- Testing

```bash
cd pyhanabi
python final_evaluation.py
```

## Plot results
We used wandb for plotting the results. But results are also stored as `.csv` file which can be found here: `results/` 


