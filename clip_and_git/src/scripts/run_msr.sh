mode='train'
gpu=$1


if [[ $mode = 'train' ]];then
    rm -rf ./saved_models/msrvtt_qa_001
    CUDA_VISIBLE_DEVICES=$gpu python tasks/run_video_qa.py --task msrvtt_qa --config ./configs/msrvtt_qa_base.json
else
    CUDA_VISIBLE_DEVICES=1 python tasks/run_video_qa.py --task msrvtt_qa --config ./configs/msrvtt_qa_base.json --do_inference 1
fi