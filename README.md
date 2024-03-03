# Learning-free Self-adaptive Sampling in Video Question Ansering

## Introduction
This repository contains the official implementation code of the paper "[Self-adaptive Sampling for Efficient Video Question Answering](https://arxiv.org/pdf/2307.04192.pdf)". In this work we introduce two sampling strategies (__MDF__ and __MIF__) applied during the time of preparing the input data to pretrained image--text models. 
Once running complete, sampled frames will be saved in a h5 file for fast loading during training and test time.

<p align="center">
    <image src="MDF.png" width="324"> 
    <image src="MIF.png" width="432">
</p>

Once running completes, sampled frames will be saved in a hdf5 (.h5) file as a "dataset" for fast loading during training and test time.
We test our methods on three models (__CLIP__, __GIT__ and __All-in-one__) and three datasets (**MSVD-QA**, **MSRVTT-QA**, **TGIF-Frame**).
The implementation on CLIP (including our refined structure **CLIP-Dec** which significantly enhances the performance on **raw-CLIP**) and GIT are in the folder "clip_and_git", while the implementation on All-in-one are under the folder "all_in_one".

## Usage
### 1. Downloading Datasets
To download MSVD-QA and MSRVTT-QA, please refer to this [repository](https://github.com/xudejing/video-question-answering). For TGIF-QA, please visit this [repository](https://github.com/YunseokJANG/tgif-qa) for specific downloading guidance.

The suggested path to store these datasets is "model/dataset/<dataset_name>" 

### 2. Preprocessing
The code to do sampling for all three models is same, under the folder "clip_and_git/src/preprocessing". 

* To sample via MDF strategy, run the python script as follows:
    ```
    python extract_features.py --dataset=<dataset_name> --dataset_root=<root_path> --sampling_strategy='repr' --model_name=<vlm_model_name> ... (other hps)
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
The following displayed digits are prediction accuracy, whose definition can be found in our paper.

### CLIP-Dec (3 Frame)
|Sampling|MSVD-QA|MSRVTT-QA|TGIF-Frame|
|---|---|---|---|
|noDec|27.7|30.3|42.8|
|Uniform|33.8|33.7|47.2|
|MDF|__35.0__|35.2|__63.2__|
|MIF|__35.0__|__35.4__|61.8|

### GIT-Base (6 Frame)
|Sampling|MSVD-QA|MSRVTT-QA|TGIF-Frame|
|---|---|---|---|
|Report|51.2|41.0|__69.1__|
|Uniform|52.2|41.1|67.5|
|MDF|__55.3__|42.0|__69.9__|
|MIF|54.5|__42.3__|69.6|

### AIO-Base (3 Frame)
|Sampling|MSVD-QA|MSRVTT-QA|TGIF-Frame|
|---|---|---|---|
|Report|46.5|42.9|64.2|
|Reprd.|46.1|42.7|64.0|
|MDF|__46.9__|43.8|__66.2__|
|MIF|46.7|__44.0__|65.9|

### AIO-Base+ on Next-QA (3 Frame)
|Method|Val|Test|
|---|---|---|
|Base|48.4|48.1|
|MIF|49.7|49.5|
|MDF|50.2|49.8|


### BLIP2-T5XXL on Next-QA (3 Frame)
|Method|Val|Test|
|---|---|---|
|Base|60.1|59.7|
|MIF|61.5|__61.2__|
|MDF|__61.8__|61.1|

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
