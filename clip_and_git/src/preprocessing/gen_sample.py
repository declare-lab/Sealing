import argparse, os
import h5py, json
from tqdm import tqdm
import torch
from torch import nn
import random
import numpy as np
from datautils.utils import Timer
from torch.nn.functional import softmax
import sys
sys.path.append('./')
from src.utils.basic_utils import load_jsonl, load_json, save_json, get_rounded_percentage
# from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
# from datautils import svqa
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter


def get_cap(processor, cap_model, frms):
    bsz = frms.shape[0]
    input_ids = processor(text=['[CLS] ']*bsz, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
    pixel_values = torch.Tensor(frms.reshape(bsz, 3, 224, 224)).cuda()
    generated_captions = processor.batch_decode(cap_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=30), skip_special_tokens=True, max_length=50)
    return generated_captions

def generate_cap(processor, caption_model, anno_path, h5_outfile):
    caption_model.eval().cuda()
    if isinstance(caption_model, torch.nn.DataParallel):
        caption_model = caption_model.module
    
    gened_caps = {}
    with h5py.File(h5_outfile, 'r') as fd:
        sampled_frames = fd['sampled_frames']
        i = 0
        for frms in tqdm(sampled_frames):
            # frms = sampled_frames[i]
            caps = get_cap(processor, caption_model, frms)
            gened_caps[i] = caps
            i += 1

    new_anno_file = os.path.join(anno_path, 'frame_captions.json')
    save_json(gened_caps, new_anno_file)

def move_to_cuda(input):
    return {k:v.cuda() for k, v in input.items()}

def generate_inds(tokenizer, model, anno_path, vid_map_file, args):
    model.eval().cuda()
    ds_rate = args.ds_rate
    cap_src_file = os.path.join(anno_path, 'frame_captions.json')

    vid2id = load_json(vid_map_file)
    all_captions = load_json(cap_src_file)

    if args.dataset == 'msvd_qa':
        vid_name = 'video'
        qid_temp = 'video{}'
    elif args.dataset == 'msrvtt_qa':
        vid_name = 'video_id'
        qid_temp = '{}'
    else:
        raise ValueError('Invalid dataset name! Current supported dataset msvd_qa, msrvtt_qa')

    for split in ['train', 'val', 'test']:
        read_file = os.path.join(anno_path, 'qa_{}.json'.format(split))
        ds = load_json(read_file)
        new_ds = []
        for sample in tqdm(ds):
            question = sample['question']
            
            # calculate the score between question and captions
            vid = sample[vid_name]
            
            # msrvtt does not need this transformation
            query_id =  qid_temp.format(vid)
            if args.dataset == 'msvd_qa':
                vid = vid.split('.')[0]
                query_id = str(vid2id[vid])
            captions = all_captions[query_id] 
            
            bsz = len(captions)
            inputs = tokenizer(text=[question]*bsz, text_pair=captions, padding=True, truncation=True, return_tensors='pt')
            inputs = move_to_cuda(inputs)
            output = model(**inputs)
            scores = softmax(output[0][:,0], dim=-1) # logits
            
            # get highest scores
            # FIXME: Downsample 1/2
            inds = scores[::ds_rate].topk(args.K)[1].detach().cpu().tolist()
            inds = [i*ds_rate for i in inds]
            # do not sort here so that we can retrive the top-K important frames
            sample['sampled_inds'] = inds
            
            new_ds.append(sample)
        save_file = os.path.join(anno_path, 'qa_winds_{}.json'.format(split))
        save_json(new_ds, save_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset info
    parser.add_argument('--dataset', default='msvd_qa', choices=['msvd_qa', 'msrvtt_qa', 'svqa'], type=str)
    parser.add_argument('--dataset_root', default='./dataset', type=str)
    parser.add_argument('--anno_path', default='annotations', type=str)
    # output
    parser.add_argument('--out', dest='outfile', help='output filepath', default="{}_video_feat.h5", type=str)

    # feature extraction hps
    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size for computing feature similarity')
    parser.add_argument('--img_size', type=int, default=224, help='image size of extracted frames')
    parser.add_argument('--intv', type=int, default=1, help='sampling interval between video frames')
    parser.add_argument('--K', type=int, default=32, help='number of frames to be sampled (esp. uniform sampling)')

    # network params
    parser.add_argument('--vlm_model', type=str, default="microsoft/git-base-coco")
    parser.add_argument('--sim_model', type=str, default="iarfmoose/bert-base-cased-qa-evaluator")
    parser.add_argument('--h5_path', type=str, default="processed")
    parser.add_argument('--task', type=str, choices=['gen_cap', 'gen_inds'], default='gen_cap')
    parser.add_argument('--ds_rate', type=int, default=1)
    args = parser.parse_args()
    args.seed = 666

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = os.path.join(args.dataset_root, args.dataset)

    if args.task == 'gen_cap':
    # initialize caption model and processor (default GIT)
        processor = AutoProcessor.from_pretrained(args.vlm_model)
        if 'git' in args.vlm_model.lower():
            caption_model = AutoModelForCausalLM.from_pretrained(args.vlm_model)
            LOGGER.info('git pretrained model loaded')
        else:
            raise ValueError('No such captioning model implementations')

        # annotation files

        if args.dataset in ['msvd_qa', 'msrvtt_qa']:
            args.annotation_file = os.path.join(dataset_path, 'annotations/qa_{}.json')
            
            # read H5 file, mapping file
            dataset_path = os.path.join(args.dataset_root, args.dataset)
            h5_path = os.path.join(dataset_path, args.h5_path)
            vid_map_file = os.path.join(h5_path, 'vidmapping.json')
            h5_outfile = os.path.join(h5_path, args.outfile.format(args.dataset))
            anno_path = os.path.join(dataset_path, args.anno_path)
            
            # generate h5 file
            generate_cap(processor, caption_model, anno_path, h5_outfile, vid_map_file, args)

        elif args.dataset == 'svqa':
            args.annotation_file = './data/SVQA/questions.json'
            args.video_dir = './data/SVQA/useVideo/'
            video_paths = svqa.load_video_paths(args)
            random.shuffle(video_paths)
            # load model
            generate_cap(processor, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

    elif args.task == 'gen_inds':    # text-only models
        tokenizer = AutoTokenizer.from_pretrained(args.sim_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.sim_model)
        
        if args.dataset in ['msvd_qa', 'msrvtt_qa']:
            # annotation file, h5 file, mapping file
            dataset_path = os.path.join(args.dataset_root, args.dataset)
            anno_path = os.path.join(dataset_path, args.anno_path)

            h5_path = os.path.join(dataset_path, args.h5_path)
            vid_map_file = os.path.join(h5_path, 'vidmapping.json')
            # generate captions
            generate_inds(tokenizer, model, anno_path, vid_map_file, args)

