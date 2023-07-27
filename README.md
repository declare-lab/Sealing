# SAS-VQA

## Introduction
This repository the official implementation code of the paper "[Self-adaptive Sampled Video Question Answering]()". In this work we introduce two sampling strategies (__MDF__ and __MIF__) applied during the time of preparing the input data to pretrained image--text models. 
Once running complete, sampled frames will be saved in a h5 file for fast loading during training and test time.

We test our methods totally on three models (__CLIP__, __GIT__ and __All-in-one__) and three datasets.
The implementation on CLIP (including our refined structure which remarkably enhances the accuracy on __raw-CLIP__) and GIT are in the folder "clip_and_git", while the implementation on All-in-one are under the folder "all_in_one".

## Usage
### 1. Downloading Datasets
To download MSVD-QA and MSRVTT-QA, please refer to this [repository](https://github.com/xudejing/video-question-answering). For TGIF-QA, please visit this [repository](https://github.com/YunseokJANG/tgif-qa) for specific downloading guidance.

The suggested path to store these datasets is "model/dataset/<dataset_name>" 

### 2. Preprocessing
The code to do sampling for all three models is same, under the folder "clip_and_git/src/preprocessing". 

* To sample via MDF strategy, run the python script as follows:
    ```
    python extract_features.py --dataset=<dataset_name> --dataset_root=<root_path> --sampling_strategy='repr' --model_name=<vlm_model_name>
    ```
    If your code prompts an out-of-memory exception, please using a smaller chunksize (default=512) to shrink the input size per computation.

* To sample via MIF strategy, first run a uniform sampling with large K to obtain a sparse video sequence

    ```
    python extract_features.py --sampling_strategy='uni' ...
    ```
    Then run the python script to capture and start sampling
    ```
    python gen_sample.py --dataset=<dataset_name> --dataset_root=<root_path> --sampling_strategy='repr' --vlm_model=<vlm_model_name> --sim_model=<sim_model_name> --task='gen_cap'

    python gen_sample.py --dataset=<dataset_name> --dataset_root=<root_path> --sampling_strategy='repr' --vlm_model=<vlm_model_name> --sim_model=<sim_model_name> --task='gen_inds'
    ```

### 3. How to run
For experiments on CLIP and GIT, please modify our provided reference scripts (in 'src/scripts'). For all-in-one, please check its attached README file for more details.

## Results
### CLIP-Dec
|Sampling|MSVD-QA|MSRVTT-QA|TGIF-Frame|
|---|---|---|---|
|noDec|27.7|30.3|42.8|
|Uniform|33.8|33.7|47.2|
|MDF|__35.0__|35.2|__63.2__|
|MIF|__35.0__|__35.4__|61.8|

### GIT-Base
|Sampling|MSVD-QA|MSRVTT-QA|TGIF-Frame|
|---|---|---|---|
|Report|51.2|41.0|__69.1__|
|Uniform|52.2|41.1|47.0|
|MDF|__55.3__|42.0|68.8|
|MIF|46.7|__42.3__|67.5|

### AllInOne-Base
|Sampling|MSVD-QA|MSRVTT-QA|TGIF-Frame|
|---|---|---|---|
|Report|46.5|42.9|64.2|
|Uniform|46.1|42.7|64.0|
|MDF|__46.9__|43.8|__66.2__|
|MIF|46.7|__44.0__|65.9|

## Citation
Please cite our paper if you find this project is related to your work
```
@misc{han2023sas,
      title={SAS Video-QA: Self-Adaptive Sampling for Efficient Video Question-Answering}, 
      author={Wei Han and Hui Chen and Min-Yen Kan and Soujanya Poria},
      year={2023},
      eprint={2307.04192},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Feel free to contact me at henryhan88888@gmail.com if you have any problems with the paper and code.
