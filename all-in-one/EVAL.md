# Evaluation

We provide two types of logs: _tensor log_ or _txt file_.

## 1. Modify Configuration File
The configuration file for data I/O in each task is in the folder [AllInOne/datasets](AllInOne/datasets). 
For different download sources, the file name of each text/video data may change a bit.
Please carefully examine and modify the path variables in "_load_metadata" and "\_\_getitem\_\_" (hdf5 dataset) before starting your own run.
If there is a File_Not_Found error

## 2. Running Command
The data I/O files for these datasets are in [AllInOne/datasets/](AllInOne/datasets/). You need to modify the path of self.metada_dir and name in the "split_files" dictionary to point to the correct files.

### Evaluate TGIF

### TGIF-QA FrameQA
```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=8 task_finetune_tgifqa \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 65.0  | 64.0 | [Google Driver](https://drive.google.com/file/d/164UwQsl99zU1O81U014ihimLl5HNBSZT/view?usp=sharing) |

For tensor log, simple using 
```bash
mkdir tensor_log
cp [path_to_provided_logs] tensor_log/
tensorboard --logdir tensor_log
```
As below:

![](figures/tensorboard_ft_example.png)


Notice msrvtt_vqa is a loss name which is equal to open-set VQA in the final code.

### TGIF-QA Action/Transition

Modify line 19 in [`tgifqa`](AllInOne/datasets/tgifqa.py) for transition/action.

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_tgif_action_trans \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 93.0  | 92.5 | [google driver](https://drive.google.com/file/d/1GQLvIKpEC_flfOFx9GA7c7Ks26cfcvcK/view?usp=sharing) |


### MSRVTT-QA

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_msrvttqa \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 42.9  | 42.5 | [google driver](finetune_msrvtt_qa_seed0_from_last2022_2_25) |

### MSVD-QA
```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_msvdqa \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 46.1  | 46.5 | [google driver](https://drive.google.com/file/d/1f-vSnS1I7vu6Z7eiimGGY8B1vRbnNn0W/view?usp=sharing) |



