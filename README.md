# RSCFed: Random Sampling Consensus Federated Semi-supervised Learning

## Introduction

This is the official PyTorch implementation of CVPR 2022 paper "RSCFed: Random Sampling Consensus Federated Semi-supervised Learning".
![RSCFed: pipeline](https://github.com/XMed-Lab/RSCFed/blob/main/figure/pipeline_final.png)
## Preparation
1. Create conda environment:

		conda create -n RSCFed python=3.8
		conda activate RSCFed

2. Install dependencies:

		pip install -r requirements.txt
		
SVHN and CIFAR-100 dataset will be downloaded automatically once training started.

## Run the code
3. Train model for each dataset. To produce the claimed results for SVHN dataset:
```
python train_main.py --dataset=SVHN \
	--model=simple-cnn \
	--unsup_num=9 \
	--batch_size=64 \
	--lambda_u=0.02 \
	--opt=sgd \
	--base_lr=0.03 \
	--unsup_lr=0.021 \
	--max_grad_norm=5 \
	--resume \
	--from_labeled \
	--rounds=1000 \
	--meta_round=3 \
	--meta_client_num=5 \
	--w_mul_times=6 \
	--sup_scale=100 \
	--dist_scale=1e4 \
```
## Parameters
Parameter     | Description
-------- | -----
dataset  | dataset used
model | backbone structure
unsup_num  | number of unlabeled clients
batch_size | batch size
lambda_u | ratio of loss on unlabeled clients
opt | optimizer
base_lr | lr on labeled clients
unsup_lr | lr on unlabeled clients
max_grad_norm | limit maximum gradient
resume | resume
from_labeled | whether resume from warm-up model
rounds | maximum global communication rounds
meta_round | number of sub-consensus models
meta_client_num | number of clients in each subset
w_mul_times | scaling times for labeled clients
sup_scale | scaling weights for labeled clients when computing model distance
dist_scale | scaling weights when computing model distance

## Evaluation
For SVHN and CIFAR-100 dataset, the best model is placed in [final_model](https://github.com/XMed-Lab/RSCFed/tree/main/final_model). For ISIC2018 dataset, please find the best model [here](https://drive.google.com/drive/folders/1FJKwRI2MTv0SGedVP61AFDgE6ixXdb0q?usp=sharing).

Use the following command to generate the claimed results:
```
python test.py --dataset=SVHN \
	--batch_size=5 \
	--model=simple-cnn \
```
For different datasets, please modify file path, arguments "dataset" and "model" correspondingly.
