import time
from heapq import heappush, heappop
import torch
from torch.nn.functional import normalize

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)

CHUNK_SIZE = 256
INTERVAL = 20
def sample_representative_frames(frames, model, K=16, W=8, debug_counter=None):
    if W == -1: # adaptive width
        W = len(frames) // INTERVAL
    
    feat_chunks = []
    num_frames = frames.size(0)
    num_chunks = num_frames // CHUNK_SIZE + (1 if (num_frames % CHUNK_SIZE) > 0 else 0)
    
    for i in range(num_chunks):
        model_out = model(frames[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE])
        if hasattr(model_out, 'pooler_output'):
            chunk_feats = model_out.pooler_output
        else:
            chunk_feats = model_out.last_hidden_state.mean(dim=1)    # pooling

        chunk_feats = chunk_feats.detach()
        chunk_feats = normalize(chunk_feats)
        feat_chunks.append(chunk_feats)
    
    if len(feat_chunks) == 0:   # empty input
        debug_counter['Zeros'] += 1
        return frames.new_zeros(K, 3, 224, 224)

    all_feats = torch.cat(feat_chunks, dim=0) # (N, 768)
    all_sims = all_feats @ all_feats.transpose(0, 1)  # (N, N)
    
    # filter frames
    lcl_avg = torch.zeros(all_sims.shape[0])
    for i in range(W, all_sims.shape[0]-W):
        subsim = all_sims[i][i-W:i+W]
        lcl_avg[i] = (subsim.sum()-1) / (len(subsim) - 1)
    
    # dfs-based search
    top_idx = lcl_avg.argmax()  # the one has top lcl_avg shoule be preserved
    res = [top_idx.item()]
    intvs = []
    if top_idx - W > 0:
        v, idx = lcl_avg[0:top_idx-W].topk(1)
        # intvs.append(((0, top-W), v, idx))
        heappush(intvs, (-v, (0, top_idx-W), idx))
    if top_idx + W < len(lcl_avg):
        v, idx = lcl_avg[top_idx+W:].topk(1)
        heappush(intvs, (-v, (top_idx+W, len(lcl_avg)), top_idx+W+idx))
        
    # while len(intvs) > 0 and len(res) < K:
    while len(res) < K and len(intvs) > 0:
        top = heappop(intvs)   # pick the next dominant frame
        top_idx = top[2][0].item()
        res.append(top_idx)
        l, r = top[1]
        left = top_idx - W
        right = top_idx + W
        if left > l:
            v, idx = lcl_avg[l:left].topk(1)
            heappush(intvs, (-v, (l, left), l+idx))
        if right < r:
            v, idx = lcl_avg[right:r].topk(1)
            heappush(intvs, (-v, (right, r), right+idx))
    
    # res.sort()
    if len(res) < K:
        res = lcl_avg.topk(K)[1]
        debug_counter['Failure'] += 1
    return frames[res]

def sample_frames_uniform(frames, K=8):
    num_frames = len(frames)
    if num_frames <= K:
        print(num_frames)
    intv = num_frames / K
    
    cur_idx = int(intv // 2)
    sampled_frames = []
    for _ in range(K):
        sampled_frames.append(frames[cur_idx])
        cur_idx = int(cur_idx + intv)
    
    assert len(sampled_frames) == K
    return torch.stack(sampled_frames)

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff