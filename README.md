# RSCFed: Random Sampling Consensus Federated Semi-supervised Learning

## Introduction

This is the official PyTorch implementation of CVPR 2022 paper "[RSCFed: Random Sampling Consensus Federated Semi-supervised Learning](https://arxiv.org/abs/2203.13993)".
![RSCFed: pipeline](https://github.com/XMed-Lab/RSCFed/blob/main/figure/pipeline_final.png)
## Preparation
1. Create conda environment:

		conda create -n RSCFed python=3.8
		conda activate RSCFed

2. Install dependencies:

		pip install -r requirements.txt
		
SVHN and CIFAR-100 dataset will be downloaded automatically once training started.

## Run the code
3. Train model for each dataset. 
To produce the claimed results for SVHN dataset:
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
For CIFAR-100 dataset:
```
python train_main.py --dataset=cifar100 \
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
For ISIC 2018 dataset, please find the warm-up model [here](https://drive.google.com/drive/folders/1FJKwRI2MTv0SGedVP61AFDgE6ixXdb0q?usp=sharing). To produce the claimed result:
```
python train_main.py --dataset=skin \
	--model=resnet18 \
	--unsup_num=9 \
	--batch_size=12 \
	--lambda_u=0.02 \
	--opt=sgd \
	--base_lr=2e-3 \
	--unsup_lr=1e-3 \
	--max_grad_norm=5 \
	--rounds=800 \
	--meta_round=3 \
	--meta_client_num=5 \
	--w_mul_times=200 \
	--pre_sz=250 \
	--input_sz=224 \
	--dist_scale=0.01 \
	--sup_scale=0.01 \
	--resume \
	--from_labeled \
```
To produce all the claimed results, please modify the path of warm-up model accordingly. Warm-up models are trained only on labeled clients. 
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
For CIFAR-100:
```
python test.py --dataset=cifar100 \
	--batch_size=5 \
	--model=simple-cnn \
```
For ISIC 2018:
```
python test.py --dataset=skin \
	--batch_size=5 \
	--model=resnet18 \
	--pre_sz=250 \
	--input_sz=224 \
```
For different datasets, please modify file path, arguments "dataset" and "model" correspondingly.
