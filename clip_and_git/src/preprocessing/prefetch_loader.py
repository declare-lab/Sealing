import threading
import numpy as np
import cv2, time
import random
import torch
from tqdm import tqdm

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = iter(it)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def get_path_i(paths_count):
    """Cyclic generator of paths indice
    """
    for current_path_id in range(paths_count):
        yield current_path_id

class InputGen:
    def __init__(self, paths, processor, intv, toolkit='cv2'):
        self.paths = paths
        self.index = 0
        self.init_count = 0
        self.lock = threading.Lock() #mutex for input path
        self.yield_lock = threading.Lock() #mutex for generator yielding of batch
        self.path_id_generator = LockedIterator(get_path_i(len(self.paths))) 
        # self.path_id_generator = get_path_i(len(self.paths))
        self.processor = processor  # clip vision processor
        self.prced_frms = None
        self.intv = intv
    
    def get_samples_count(self):
        """ Returns the total number of images needed to train an epoch """
        return len(self.paths)

    def get_batches_count(self):
        """ Returns the total number of batches needed to train an epoch """
        return int(self.get_samples_count() / self.batch_size)

    def next(self):
        return self.__iter__()

    def __iter__(self):
        #Iterates through the input paths in a thread-safe manner
        for path_id in tqdm(self.path_id_generator, total=len(self.paths)):
            video_file = self.paths[path_id]
            video_data = []
            try:
                frame_count = 0
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    rval, frm = cap.read()
                    while rval:
                        b, g, r = cv2.split(frm)
                        frm = cv2.merge([r, g, b])
                        if frame_count % self.intv == 0:
                            video_data.append(frm)
                        frame_count += 1
                        rval, frm = cap.read()
                cap.release()
            except:
                print('file {} error'.format(video_file))
                raise ValueError
            
            with self.yield_lock:
                # (N, 3, H, W)
                self.prced_frms = np.stack(self.processor(images = video_data)['pixel_values'])
                yield torch.from_numpy(self.prced_frms)
                self.prced_frms = None

    def __call__(self):
        return self.__iter__()

class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """
    def __init__(self):
        self.to_kill = False
    
    def __call__(self):
        return self.to_kill
    
    def set_tokill(self, tokill):
        self.to_kill = tokill

def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for idx, prced_frms in enumerate(dataset_generator):
            # We fill the queue with new fetched batch until we reach the max size.
            batches_queue.put((idx, prced_frms), block=True)
            if tokill() == True:
                return

def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch, video_frms = batches_queue.get(block=True)
        video_frms = video_frms.cuda()
        cuda_batches_queue.put((batch, video_frms), block=True)
        if tokill() == True:
            return
