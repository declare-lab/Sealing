#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python run.py with data_root=DataSet num_gpus=2 \
	num_nodes=1 \
	num_frames=3 \
	per_gpu_batchsize=8 task_finetune_tgifqa \
	load_path="pretrained/all-in-one-base.ckpt"
