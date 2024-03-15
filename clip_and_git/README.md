# CLIP and GIT with MDF and MIF
This reporsitory contains code of sampling preprocessing (MDF and MIF, under the folder [src/preprocessing](src/preprocessing/)) and implementation of CLIP, CLIP-Dec, and GIT that read from the ".h5" file created during the sampling stage.

## Usage
The usage of the sampling code has been included in the upper-level [README.md](../README.md) file so we do not cover that part here anymore. Here we only illustrate how to run the latter part below.

### Setup
We recommend to create a new conda environment to run this code. 
```
conda create -n <your_env_name>
```
After successfully creating the environment, install all the needed packages with __pip__

```bash
pip install -r requirements.txt
```

### Configuration
The configuration files are in `src/configs/`. Here are some important items that at the most chance you need to modify
```json
{
    "train/val_datasets": [
        "name": "", // dataset name
        "txt": "",  // txt data path
        "img": ""   // img data path
    ],
    "model": {
        ...
        "pretrained_model": "", // the name of pretrained model you want to run
        "img_len": "", // number of images as input to the model
        ...
    }
    "inference_txt/img_db": "", // the location that your TEST text and image data to save
}
```

### Start to run
We provide many example scripts in ```scripts/run_<task_name>.py```. You can simply start training and evaluation via the command line
```bash
bash scripts/run_<task_name>.sh $gpu_id
```

## Misc.
This code is developed based on [CLIP-BERT](https://github.com/jayleicn/ClipBERT) and [huggingface-transformers](https://github.com/huggingface/transformers). Credits to their great contribution!
