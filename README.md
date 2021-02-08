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
We have been using `pytorch-1.5.1`, `cuda-10.1`, and `cudnn-v7.6.5` in our development environment.
```bash
# create new conda env
conda create -n hanabi python=3.7
conda activate hanabi
```
## Run
Lifelong Hanabi consists of 3 phases: 1- Pre-training, 2- Continual training, 3- Testing 

### 1- Pre-Trained Agents

Run the following command to download the pre-trained agents used in the paper.
```bash
cd model
sh download.sh
```
Or run the following command to download all 100 pre-trained agents to create the full Cross-Play matrix.

To evaluate all the agents with each other, simply run:
```bash
cd pyhanabi
python tools/eval_model.py --weight ../models/sad_2p_10.pthw --num_player 2
```

### 2- Continual Training

```bash
cd pyhanabi
sh tools/dev.sh
```


### 3- Testing


```bash
cd pyhanabi
python final_evaluation.py
```


