import argparse, os
import h5py, json
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
import torchvision
import random
import numpy as np
from datautils.utils import sample_representative_frames, sample_frames_uniform, Timer
# from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
# from datautils import svqa
from transformers import CLIPImageProcessor, CLIPVisionModel, GitVisionModel
# from transformers import BLIPImageProcessor, BLIPVisionModel
from transformers import AutoProcessor
from prefetch_loader import *
from queue import Queue, Empty, Full
from threading import Thread
from collections import Counter


def generate_vidid_json(video_paths, json_outfile):
    mapping_dict = {}
    for i, video_path in enumerate(video_paths):
        video_id = video_path.split('/')[-1].split('.')[0]
        mapping_dict[video_id] = i
    json.dump(mapping_dict, open(json_outfile, 'w'))

def sample_frame_indices(video_frms, clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    assert start_idx >= 0
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return video_frms[indices]

def generate_h5_parallel(processor, model, video_paths, args, h5_outfile):
    if not os.path.exists('data/{}'.format(args.dataset)):
        os.makedirs('data/{}'.format(args.dataset))
    
    if model is not None:
        model.eval()
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # cpu video queue
    memory_video_queue = Queue(maxsize=8)
    # cuda processing queue
    cuda_video_queue = Queue(maxsize=4)
    
    # video frame generator
    frm_generator = InputGen(video_paths, processor, args.intv)
    load_thread_killer = thread_killer()
    load_thread_killer.set_tokill(False)
    preprocess_workers = 1
    
    # launch 4 threads to do load && pre-process the input video frames
    for _ in range(preprocess_workers):
        t = Thread(target=threaded_batches_feeder, args=(load_thread_killer, memory_video_queue, frm_generator))
        t.start()
    
    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)

    cudathread = Thread(target=threaded_cuda_batches, \
                args=(cuda_transfers_thread_killer, cuda_video_queue, memory_video_queue))
    cudathread.start()
    # let queue get filled
    time.sleep(8)
    IMG_HW = args.img_size
    with h5py.File(h5_outfile, 'w') as fd:    
        fd.create_dataset("sampled_frames", (len(video_paths), args.K, 3 * IMG_HW * IMG_HW))
        sampled_frames_h5 = fd["sampled_frames"]
        for i in range(len(video_paths)):
            # read video frames out of the queue
            _, video_frms = cuda_video_queue.get(block=True)
            
            if len(video_frms.size()) > 4:  # should be (N, 3, H, W)
                video_frms = video_frms.squeeze(0)
            # extract special representative frames
            if args.sampling_strategy == 'repr':
                # move model to cuda, set it to eval mode
                # FIXME: remove the counter
                exted_frms = sample_representative_frames(video_frms, model, K=args.K, W=args.W, alpha=float(args.alpha))
            elif args.sampling_strategy == 'uni':
                exted_frms = sample_frames_uniform(video_frms, K=args.K)
            elif args.sampling_strategy == 'git6':  # same as GIT-VideoQA implementation
                exted_frms = sample_frame_indices(video_frms, args.K, 4, len(video_frms))

            frms_to_store = exted_frms.cpu().reshape(args.K, -1)
            sampled_frames_h5[i] = frms_to_store
    
    load_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread to shut down
            memory_video_queue.get(block=True, timeout=1)
            cuda_video_queue.get(block=True, timeout=1)
        except Empty:
            pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # dataset info
    parser.add_argument('--dataset', default='msvd_qa', choices=['msvd_qa', 'msrvtt_qa', 'svqa'], type=str)
    parser.add_argument('--dataset_root', default='./dataset', type=str)
    parser.add_argument('--anno_path', default='annotations', type=str)
    parser.add_argument('--question_type', default='none', choices=['none'], type=str)
    # output
    parser.add_argument('--out', dest='outfile', help='output filepath', default="{}_{}_feat.h5", type=str)

    # feature extraction hps
    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size for computing feature similarity')
    parser.add_argument('--img_size', type=int, default=224, help='image size of extracted frames')
    parser.add_argument('--intv', type=int, default=1, help='sampling interval between video frames')
    parser.add_argument('--sampling_strategy', default='uni', choices=['uni', 'repr', 'git6'], type=str)
    parser.add_argument('--K', type=int, default=16, help='number of frames to be sampled (esp. uniform sampling)')
    parser.add_argument('--W', type=int, default=-1, help='interval length to sample 2 points')
    parser.add_argument('--alpha', type=str, default='2.5', help='width-adjust hp to control W, activated only when W=-1')

    # network params
    parser.add_argument('--vlm_model', type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument('--h5_fname', type=str, default="processed")
    args = parser.parse_args()
    args.seed = 666
    args.feature_type = 'video'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize clip processors
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    if args.sampling_strategy == 'repr':
        vision_model = GitVisionModel.from_pretrained(args.vlm_model)
    else:
        vision_model = None
    dataset_path = os.path.join(args.dataset_root, args.dataset)

    # annotation files
    if args.dataset == 'tgif-qa':
        args.annotation_file = './data/tgif-qa/csv/Total_{}_question.csv'
        args.video_dir = './data/tgif-qa/gifs'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model

    elif args.dataset == 'msrvtt_qa':
        args.annotation_file = os.path.join(dataset_path, args.anno_path, '{}_qa.json')
        args.video_dir = os.path.join(dataset_path, 'video')
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)

        # load model
        outpath = os.path.join(dataset_path, args.h5_fname)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        h5_outfile = os.path.join(outpath, args.outfile.format(args.dataset, args.feature_type))
        json_outfile = os.path.join(outpath, 'vidmapping.json')
        
        # generate mapping dict
        if not os.path.exists(json_outfile):
            generate_vidid_json(video_paths, json_outfile)
        # generate h5 file
        generate_h5_parallel(processor, vision_model, video_paths, args,
                    h5_outfile)

    elif args.dataset == 'msvd_qa':
        args.annotation_file = os.path.join(dataset_path, args.anno_path, 'qa_{}.json')
        args.video_dir = os.path.join(dataset_path, 'video')
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        
        outpath = os.path.join(dataset_path, args.h5_fname)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        h5_outfile = os.path.join(outpath, args.outfile.format(args.dataset, args.feature_type))
        json_outfile = os.path.join(outpath, 'vidmapping.json')
        
        # generate mapping dict
        if not os.path.exists(json_outfile):
            generate_vidid_json(video_paths, json_outfile)
        # generate h5 file
        generate_h5_parallel(processor, vision_model, video_paths, args,
                    h5_outfile)

