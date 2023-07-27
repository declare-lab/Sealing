import numpy as np
from .video_base_dataset import BaseDataset
import os
import pandas as pd
import torch, h5py
import json, torch

class MSVDQADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["msvd_qa_train"]
        elif split == "val":
            names = ["msvd_qa_test"]  # test: directly output test result
            # ["msvd_qa_val"]
        elif split == "test":
            names = ["msvd_qa_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        self._load_metadata()

    def _load_metadata(self):
        self.metadata_dir = metadata_dir = './DataSet/msvd'
        split_files = {
            'train': 'msvd_train_qa_encode.json',
            'val': 'msvd_val_qa_encode.json',
            'test': 'msvd_test_qa_encode.json'
        }
        # read ans dict
        self.ans_lab_dict = {}
        # answer_fp = os.path.join(metadata_dir, 'msvd_ans
        answer_fp = os.path.join(metadata_dir, 'ans2label.json')
        self.youtube_mapping_dict = dict()
        mapping_fp = os.path.join(metadata_dir, './processed/vidmapping.json')
        
        
        self.ans_lab_dict = json.load(answer_fp)
        self.vidmapping = json.load(mapping_fp)
        for name in self.names:
            split = name.split('_')[-1]
            target_split_fp = split_files[split]
            metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
            if self.metadata is None:
                self.metadata = metadata
            else:
                self.metadata.update(metadata)
        print("total {} samples for {}".format(len(self.metadata), self.names))

    def _get_video_path(self, sample):
        rel_video_fp = self.youtube_mapping_dict['vid' + str(sample["video_id"])] + '.avi'
        # print(rel_video_fp)
        full_video_fp = os.path.join(self.data_dir, 'YouTubeClips', rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_text(self, sample):
        text = sample['question']
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return (text, encoding)

    def get_answer_label(self, sample):
        text = sample['answer']
        ans_total_len = len(self.ans_lab_dict) + 1  # one additional class
        try:
            ans_label = self.ans_lab_dict[text]  #
        except KeyError:
            ans_label = -100  # ignore classes
            # ans_label = 1500 # other classes
        scores = np.zeros(ans_total_len).astype(int)
        scores[ans_label] = 1
        return text, ans_label, scores
        # return text, ans_label_vector, scores

    def __getitem__(self, index):
        with h5py.File(os.path.join(self.metadata_dir, 'processed/msvd_qa_video_feat.h5')) as f:
            sample = self.metadata[index].iloc[0]
            qid = index
            vid = self.vidmapping[sample['video_id']]
            
            frames = f['sampled_frames'][vid].reshape(1, 3, 3, 224, 224)
            image_tensor = torch.Tensor(frames)
            text = self.get_text(sample)
            if self.split != "test":
                answers, labels, scores = self.get_answer_label(sample)
            else:
                answers = list()
                labels = list()
                scores = list()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }

    def __len__(self):
        return sum(1 for line in self.metadata)  # count # json lines