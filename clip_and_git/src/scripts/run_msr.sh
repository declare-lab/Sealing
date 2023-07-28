#!/bin/bash

gpu=$1

rm -rf ./saved_models/msrvtt_qa_001
CUDA_VISIBLE_DEVICES=$gpu python tasks/run_video_qa.py --task msrvtt_qa --config ./configs/msrvtt_qa_base.json