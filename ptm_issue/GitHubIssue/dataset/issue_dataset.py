import json
import os
import pickle
import re
from typing import Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
import transformers
from GitHubIssue.tokenizer.allennlp_tokenizer import AllennlpTokenizer
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer


def concat_str(tokenizer, text_list):
    final_str = ""
    for idx, text in enumerate(text_list):
        if text is None:
            continue
        if idx == 0:
            if isinstance(tokenizer, T5Tokenizer):
                final_str += "task: classify issue type. context: " + text
            else:
                final_str += text
        else:
            if isinstance(tokenizer, BertTokenizer):
                final_str += " [SEP] " + text
            elif isinstance(tokenizer, T5Tokenizer):
                # final_str += " </s> " + text
                final_str += " . " + text
            elif isinstance(tokenizer, GPT2Tokenizer):
                # final_str += " <eos> " + text
                final_str += " " + text
            else:
                final_str += " . " + text
    return final_str

class IssueDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: Union[str, Sequence], all_labels: Sequence, tokenizer=None, lazy=False, is_gpt=True):
        self.data = []
        if isinstance(dataset, str):
            with open(dataset, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            self.data = dataset

        self.text_list = []
        self.label_list = []

        # id to label and label to id
        self.label_to_id = {}
        for i, c in enumerate(all_labels):
            self.label_to_id[c] = i

        self.id_to_label = {}
        for key, value in self.label_to_id.items():
            self.id_to_label[value] = key

        # convert data to matrices
        for obj in self.data:
            # text = obj['title'] + ' ' + obj['description']
            title = "Title: "+ obj['title']
            description = "Details: " + obj['description']
            if obj.get("commment_concat_str") is not None:
                # text += " " + obj["commment_concat_str"]
                comments_list = obj['commment_concat_str'].split("concatcommentsign")
                if len(comments_list) != 0:
                    comments_list[0] = "Comments: " + comments_list[0]
                text = concat_str(tokenizer, [title, description] + comments_list)
            else:
                text = concat_str(tokenizer, [title, description])
            # text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')['input_ids']
            if isinstance(tokenizer, AllennlpTokenizer):
                _text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')
                _text_ids['input_ids'] = torch.tensor(_text_ids['input_ids'], dtype=torch.long)
                text_ids = {}
                for k, v in _text_ids.items():
                    if isinstance(v, torch.Tensor):
                        text_ids[k] = v.squeeze(0)
                    else:
                        text_ids[k] = v
            else:
                _text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
                # 清除batch_size 维度，数据集会自动添加该维度
                text_ids = {}
                for k, v in _text_ids.items():
                    if isinstance(v, torch.Tensor):
                        text_ids[k] = v.squeeze(0)
                    else:
                        text_ids[k] = v

            labels = obj['labels']
            labels_ids = np.zeros((len(all_labels),))

            label_id = self.label_to_id[labels]
            labels_ids[label_id] = 1
        
            # for c in labels:
            #     label_id = self.label_to_id[c]
            #     labels_ids[label_id] = 1

            self.text_list.append(text_ids)
            self.label_list.append(labels_ids)

    def __getitem__(self, i):
        # return (
        #     torch.tensor(self.text_list[i], dtype=torch.long),
        #     torch.tensor(self.label_list[i], dtype=torch.long)
        # )
        return (
            self.text_list[i],
            torch.tensor(self.label_list[i], dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
