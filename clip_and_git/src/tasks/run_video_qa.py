import torch
import os
import time
import random, math
import json
from transformers import AutoTokenizer, AutoProcessor

import sys
sys.path.append('..')
from src.modeling.modeling import CLIPForSeqClassification, GITBaseModel
from src.modeling.clip_model import CLIPModelforFinetune
from src.datasets.dataset_video_qa import VideoQADataset, VideoQACollator,  GITVideoQACollator
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import load_jsonl, load_json, save_json, get_rounded_percentage
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from os.path import join
from collections import Counter
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict


def mk_tgif_qa_dataloader(task_type, anno_path, ans2label, img_hdf5_dir, cfg, tokenizer,
                          is_train=True, return_label=True):
    """
    Returns:
        list(dict), each dict is
            frameqa: {
                "gif_name": "tumblr_no73q2fm0I1uuf348o1_250",
                "question": "what is being placed in the white ice cream cone ?",
                "answer": "cookie",
                "answer_type": "object"
                }
            msrvtt_qa: {
                "question": "what are three people sitting on?",
                "answer": "couch",
                "video_id": "video6513",
                "answer_type": "what"
                }
            msvd_qa{
                "question": "what is chewing on a nut?",
                "answer": "animal",
                "video_id": "-4wsuPCjDBc_5_15.avi",
                "answer_type": "what"
            }
    """
    raw_datalist = json.load(open(anno_path, 'r'))
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    datalist = []
    if task_type == 'msvd_qa':  # msvd-qa
        for qid, raw_d in enumerate(raw_datalist):
            d = dict(
                question=raw_d["question"],
                answer=raw_d["answer"],
                video_id=raw_d["video"].split('.')[0], # <id>.avi -> ['<id>', 'avi]
                answer_type=raw_d["answer_type"],
                question_id=qid,
                sampled_inds=raw_d["sampled_inds"] if "sampled_inds" in raw_d.keys() else None
            )
            datalist.append(d)
    elif task_type == 'msrvtt_qa':
        for qid, raw_d in enumerate(raw_datalist):
            question = raw_d["question"]
            answer_type = question.split()[0]
            d = dict(
                question=question,
                answer=raw_d["answer"],
                video_id='video' + str(raw_d["video_id"]), # <id>.avi -> ['<id>', 'avi]
                answer_type=answer_type,
                question_id=qid
            )
            if 'sampled_inds' in raw_d:
                d['sampled_inds']=raw_d["sampled_inds"]
            datalist.append(d)
            
    elif task_type == 'tgif_qa':
        for qid, raw_d in enumerate(raw_datalist):
            question = raw_d["question"]
            d = dict(
                question=question,
                answer=raw_d["answer"],
                gif_name=raw_d["gif_name"],
                answer_type=raw_d["category_id"],
                question_id=qid
            )
            if 'sampled_inds' in raw_d:
                d['sampled_inds']=raw_d["sampled_inds"]
            datalist.append(d)
    else:
        raw_datalist = load_jsonl(anno_path)
        LOGGER.info(f"Loaded data size {len(raw_datalist)}")
        if cfg.data_ratio != 1.0:
            random.shuffle(raw_datalist)
            raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]
            LOGGER.info(f"Use {100 * cfg.data_ratio}% of the loaded data: {len(raw_datalist)}")

        datalist = []
        qid = 0
        for raw_d in raw_datalist:
            d = dict(
                question=raw_d["question"],
                vid_id=raw_d["gif_name"] if "gif_name" in raw_d else raw_d["video_id"],
                answer=raw_d["answer"],  # int or str
                question_id=qid  # be careful, it is not unique across splits
            )
            qid += 1

            d["answer_type"] = raw_d["answer_type"]

            datalist.append(d)
    LOGGER.info(f"datalist {len(datalist)}")

    # examples grouped by image/video id
    grouped = defaultdict(list)
    for d in datalist:
        if "video_id" in d.keys():
            grouped[d["video_id"]].append(d)
        elif "gif_name" in d.keys():
            grouped[d["gif_name"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")
    vidmapping = load_json(cfg.vid_mapping)
    dataset = VideoQADataset(
        task_type=cfg.task,
        datalist=group_datalist,
        tokenizer=tokenizer,
        img_hdf5_dir=img_hdf5_dir,
        ans2label=ans2label,
        vid2id=vidmapping,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        ensemble_n_clips=cfg.train_n_clips if is_train else cfg.inference_n_clips,
        return_label=return_label,
        is_train=is_train
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    
    if 'git' in cfg.model.pretrained_model:
        vqa_collator = GITVideoQACollator(processor=tokenizer,
                                    max_length=cfg.max_txt_len,
                                    task_type=cfg.task,
                                    nframe=cfg.nframe,
                                    samp_policy=cfg.samp_policy,
                                    img_size=cfg.img_size,
                                    add_ans=is_train,
                                    pretrained_model=cfg.model.pretrained_model)
    elif 'clip' in cfg.model.pretrained_model.lower():
        vqa_collator = VideoQACollator(tokenizer=tokenizer,
                                    max_length=cfg.max_txt_len,
                                    task_type=cfg.task,
                                    nframe=cfg.nframe,
                                    samp_policy=cfg.samp_policy)
        
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=is_train,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vqa_collator.collate_batch)
    return dataloader

def build_common_answer_dict(anno_files, k=1500):
    answer_list = []

    for file in anno_files:
        with open(file, 'r') as f:
            qa_list = json.load(f)
            answer_list += list(map(lambda qa: qa['answer'], qa_list))

    answer_occ = Counter(answer_list)
    top_k_answer = answer_occ.most_common(k)
    answer_dict = {val: i for i, (val, _) in enumerate(top_k_answer)}
    return answer_dict

def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    if cfg.task in ['msvd_qa', 'msrvtt_qa']:
        anno_files = (cfg.train_datasets[0].txt,)
        ans2label = build_common_answer_dict(anno_files, 1000)  # all included
        
        if not os.path.exists(cfg.ans2label_path):
            with open(os.path.join(cfg.ans2label_path), 'w') as f:
                json.dump(ans2label, f)
                
        train_loader = mk_tgif_qa_dataloader(
            task_type=cfg.task,
            anno_path=cfg.train_datasets[0].txt,
            ans2label=ans2label,
            img_hdf5_dir=cfg.train_datasets[0].img,
            cfg=cfg, tokenizer=tokenizer, is_train=True
        )
        val_loader = mk_tgif_qa_dataloader(
            task_type=cfg.task,
            anno_path=cfg.val_datasets[0].txt,
            ans2label=ans2label,
            img_hdf5_dir=cfg.train_datasets[0].img,
            cfg=cfg, tokenizer=tokenizer, 
            is_train=False, 
            return_label=True
        )
        test_loader = mk_tgif_qa_dataloader(
            task_type=cfg.task,
            anno_path=cfg.inference_txt_db,
            ans2label=ans2label,
            img_hdf5_dir=cfg.inference_img_db,
            cfg=cfg, tokenizer=tokenizer,
            is_train=False,
            return_label=True
        )
    elif cfg.task == 'tgif_qa':
        anno_files = (cfg.train_datasets[0].txt,)
        ans2label = build_common_answer_dict(anno_files, 2000)  # all included
        train_loader = mk_tgif_qa_dataloader(
            task_type=cfg.task, anno_path=cfg.train_datasets[0].txt,
            ans2label=ans2label, img_hdf5_dir=cfg.train_datasets[0].img,
            cfg=cfg, tokenizer=tokenizer, is_train=True
        )
        val_loader = None   # tgif_qa does not have a val_loader
        test_loader = mk_tgif_qa_dataloader(
            task_type=cfg.task,
            anno_path=cfg.inference_txt_db,
            ans2label=ans2label,
            img_hdf5_dir=cfg.inference_img_db,
            cfg=cfg, tokenizer=tokenizer,
            is_train=False,
            return_label=True
        )
    return train_loader, val_loader, test_loader

def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    # add downstream model config
    add_attr_list = [
        "num_labels", "classifier", "cls_hidden_scale",
        "loss_type",
    ]
    for k in add_attr_list:
        setattr(cfg.model, k, cfg[k])
    
    if 'clip' in cfg.model.pretrained_model.lower():
        vlm_model_cls = CLIPForSeqClassification
    elif 'git' in cfg.model.pretrained_model.lower():
        vlm_model_cls = GITBaseModel

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = CLIPModelforFinetune(
        cfg.model, 
        vlm_cls=vlm_model_cls
    )
    model.to(device)
    LOGGER.info("Setup model done!")
    return model

def forward_step(model, batch, cfg):
    # move to gpu
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.cuda()

    outputs = model(batch)  # dict
    return outputs

@torch.no_grad()
def validate(model, val_loader, cfg, eval_score=True, processor=None, ans2label=None, flag_prtr=0):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    model.eval()
    loss = 0.
    n_ex = 0
    qa_results = []
    st = time.time()
    debug_step = 5
    pbar = tqdm(total=len(val_loader))
    
    for val_step, batch in enumerate(val_loader):
        # forward pass
        question_ids = batch["question_ids"]
        del batch["question_ids"]
        # add visual part into the mini batch and perform inference
        mini_batch = dict()
        for k, v in batch.items():
            if k != "visual_inputs":
                mini_batch[k] = v

        n_ex += len(question_ids)
        losses = []

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = forward_step(model, batch, cfg)
            
        if flag_prtr in [0, 1]:
            # cross entropy
            logits = outputs["logits"].cpu()
            _loss = outputs["loss"].sum().item() if isinstance(
                outputs["loss"], torch.Tensor) else 0
            losses.append(_loss)
            pred_labels = logits.argmax(dim=-1).tolist()
        elif flag_prtr == 2:
            # mapback to task-specific vocabulary ids
            pred_labels_str = processor.batch_decode(outputs["generated_ids"], skip_special_tokens=True)
            pred_labels = [ans2label.get(pred.split()[-1], -1) for pred in pred_labels_str]
        
        for i, (qid, pred_label) in enumerate(zip(question_ids, pred_labels)):
            qa_results.append(dict(
                question_id=qid,
                answer=pred_label,
                answer_str=pred_labels_str[i] if flag_prtr == 2 else None,
                data=val_loader.dataset.qid2data[qid]
            ))
        pbar.update(1)
        if cfg.debug and val_step >= debug_step:
            break
    
    if cfg.debug:
        LOGGER.info(qa_results[:10])
    val_log = {f'valid/loss': float(loss / n_ex)}
    
    if eval_score:
        LOGGER.info(f"QA Task [{cfg.task}], "
                    f"{len(qa_results)} qa_results,"
                    f"3 examples here: {qa_results[:3]}")
        vqa_scores = val_loader.dataset.evaluate_tgif_qa(qa_results)

        # Gather scores
        scores_per_rank = vqa_scores
        gathered_scores = {}
        if "ratios" in scores_per_rank:
            gathered_ratios = {
                k: [0, 0] for k, _ in scores_per_rank["ratios"].items()}
            # Gather ratios
            current_ratios = scores_per_rank["ratios"]
            for k, v in current_ratios.items():
                gathered_ratios[k][1] += v[1]
            for k, v in gathered_ratios.items():
                gathered_ratios[k][0] = get_rounded_percentage(
                    1. * v[1] / n_ex)
            gathered_scores["ratios"] = gathered_ratios

        # FIXME: Gather scores become complicated due to np.mean and dict format.
        for scores_k, gathered_v in vqa_scores.items():
            if "ratio" in scores_k:
                continue
            if cfg.task in ["msvd_qa", "tgif_qa", "msrvtt_qa"]:
                gathered_scores[scores_k] = get_rounded_percentage(
                    gathered_v)
            else:
                gathered_scores[scores_k] = round(gathered_v, 2)

        for k, v in gathered_scores.items():
            if "ratio" not in k:
                val_log[f'valid/{k}'] = v
    else:
        LOGGER.info("eval_score = False, no scores are calculated.")
        gathered_scores = 0

    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds."
                f"{gathered_scores}")

    model.train()
    return gathered_scores['overall_acc']

def start_training(cfg):
    set_random_seed(cfg.seed)
    # n_gpu = hvd.size()
    cfg.n_gpu = n_gpu = 1
    device = torch.device("cuda")
    best_test = 0.0
    
    if 'git' in cfg.model.pretrained_model.lower():
        flag_prtr = 2
        anno_files = (cfg.train_datasets[0].txt,)
        ans2label = build_common_answer_dict(anno_files, 1000)
    elif 'clip' in cfg.model.pretrained_model.lower():
        flag_prtr = 1
        anno_files = (cfg.train_datasets[0].txt,)
        ans2label = build_common_answer_dict(anno_files, 2000)
        
    # prepare data
    if cfg.task in ['msvd_qa', 'msrvtt_qa', 'tgif_qa']:
        tokenizer = AutoProcessor.from_pretrained(cfg.model.pretrained_model)
    else:
        raise ValueError
    
    # setup dataset and model
    train_loader, val_loader, test_loader = setup_dataloaders(cfg, tokenizer)
    model = setup_model(cfg, device=device)
    model.train()
    all_params =  [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, cfg.optim)(params = all_params, lr = cfg.learning_rate)
    if getattr(cfg, 'decay', 'constant') == 'constant':
        scheduler = None 
    elif cfg.decay == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=cfg.step_decay_epochs, gamma=cfg.gamma)

    # compute the number of steps and update cfg
    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group
    total_train_batch_size = int(n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    cfg.num_train_steps = int(math.ceil(1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))
    cfg.valid_steps = int(math.ceil(1. * cfg.num_train_steps / cfg.num_valid / cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    TB_LOGGER.create(join(cfg.output_dir, 'log'))
    pbar = tqdm(total=cfg.num_train_steps)
    add_log_to_file(join(cfg.output_dir, "log", "log.txt"))

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

    optimizer.zero_grad()
    if global_step == 0:
        optimizer.step()
    debug_step = 3
    running_loss = RunningMeter('train_loss')
    
    if getattr(cfg, 'zero_eval', False):
        validate(model, test_loader, cfg, flag_prtr=flag_prtr, processor=tokenizer, ans2label=ans2label)
    
    total_correct = total_preds = 0
    
    scaler = torch.cuda.amp.GradScaler()
    valid_score = 0.0
    for step, batch in enumerate(InfiniteIterator(train_loader)):
        del batch["question_ids"]
        with torch.cuda.amp.autocast():
            outputs = forward_step(model, batch, cfg)
        
        if flag_prtr <= 1:
            loss = outputs["loss"].mean()
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch["labels"]).sum()
            total_preds += len(preds)
        elif flag_prtr == 2:
            loss = outputs["loss"].mean()

        scaler.scale(loss).backward()
        # Scales the loss, and calls backward()
        running_loss(loss.item())

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            acc = total_correct / (total_preds + 1e-6)
            pbar.set_description(str(running_loss)+' acc: {:2.3f}'.format(acc * 100))
            global_step += 1
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            pbar.update(1)

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                total_correct = total_preds = 0
                LOGGER.info(f'Step {global_step}: start validation')
                # tgif-qa does not have a validation set
                if cfg.task != 'tgif_qa':
                    valid_score = validate(model, val_loader, cfg, flag_prtr=flag_prtr, \
                        processor=tokenizer, ans2label=ans2label)
                
                if valid_score > 41.0 or cfg.task == 'tgif_qa':
                    test_score = validate(model, test_loader, cfg, flag_prtr=flag_prtr, \
                        processor=tokenizer, ans2label=ans2label)
                    best_test = max(best_test, test_score)
                
                LOGGER.info(f'Step {global_step}: Best Test Score Till Now is {best_test}')

                if scheduler is not None:
                    scheduler.step()
                
        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

if __name__ == '__main__':
    # Initialize Horovod
    input_cfg = shared_configs.get_video_qa_args()
    if input_cfg.do_inference:
        pass
    else:
        start_training(input_cfg)
