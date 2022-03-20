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

## Run the code
3. Train model for each dataset. Example:
```
		python -u train_main.py --dataset=SVHN
		 --model=simple-cnn
		 --unsup_num=9
		 --batch_size=64
		 --lambda_u=0.02
		 --opt=sgd 
		 --base_lr=0.03 
		 --unsup_lr=0.021
		 --max_grad_norm=5 
		 --resume 
		 --from_labeled 
		 --rounds=1000 
		 --meta_round=3 
		 --meta_client_num=5 
		 --w_mul_times=6
		 --un_dist=avg
		 --sup_scale=100
		 --dist_scale=1e4
```
