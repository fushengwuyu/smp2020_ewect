# author: sunshine
# datetime:2021/8/5 下午3:16
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import json
from harvesttext import HarvestText
import re
from functools import partial


def load_data(path):
    data = json.load(open(path, 'r', encoding='utf-8'))
    D = []
    ht = HarvestText()
    remove_fnc = lambda x: re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', x, flags=re.MULTILINE)
    for d in data:
        content = remove_fnc(ht.clean_text(d['content'], emoji=False))
        if content:
            D.append((content, d['label']))
    return D


class SMPDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len, task='all'):
        super(SMPDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.task = task

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def create_collate_fn(self):
        def collate(examples):
            if self.task == 'all':
                sample1 = [e[0] for e in examples]
                sample2 = [e[1] for e in examples]

                inputs_1 = self.tokenizer([s[0] for s in sample1], padding='longest', max_length=self.max_len,
                                          truncation='longest_first')
                input_ids_1 = torch.tensor(inputs_1['input_ids'], dtype=torch.long)
                attention_mask_1 = torch.tensor(inputs_1['attention_mask'], dtype=torch.long)
                token_type_ids_1 = torch.tensor(inputs_1['token_type_ids'], dtype=torch.long)

                inputs_2 = self.tokenizer([s[0] for s in sample2], padding='longest', max_length=self.max_len,
                                          truncation='longest_first')
                input_ids_2 = torch.tensor(inputs_2['input_ids'], dtype=torch.long)
                attention_mask_2 = torch.tensor(inputs_2['attention_mask'], dtype=torch.long)
                token_type_ids_2 = torch.tensor(inputs_2['token_type_ids'], dtype=torch.long)

                label1 = torch.tensor([self.label2id[l[1]] for l in sample1], dtype=torch.long)
                label2 = torch.tensor([self.label2id[l[1]] for l in sample2], dtype=torch.long)
                return [input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2,
                        token_type_ids_2, label1, label2]
            else:
                inputs = self.tokenizer([s[0] for s in examples], padding='longest', max_length=self.max_len,
                                        truncation='longest_first')
                input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
                attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
                token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

                label = np.array([self.label2id[l[1]] for l in examples])
                return [input_ids, attention_mask, token_type_ids, label, self.task]

        return partial(collate)

    def get_data_loader(self, batch_size=16, num_workers=0, shuffle=True):
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=self.create_collate_fn())


if __name__ == '__main__':
    # data = load_data('../dataset/train/virus_train.txt')
    # print(len(data))
    ...
