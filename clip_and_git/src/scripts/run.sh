#!/bin/bash

gpu=$1

rm -rf ./saved_models/msvd_qa_001
CUDA_VISIBLE_DEVICES=$gpu python tasks/run_video_qa.py --task msvd_qa --config ./configs/msvd_qa_base.json --ans2label_path ../dataset/msvd_qa/processed/ans2label.json

