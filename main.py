# author: sunshine
# datetime:2021/8/5 下午4:58

from argparse import Namespace
from transformers import BertTokenizer
from src.data_loader import load_data, SMPDataset
from src.train import Trainer
import numpy as np
import random
import torch


def get_args():
    params = dict(
        max_len=128,
        batch_size=4,
        drop=0.3,
        epoch_num=10,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        seed=2333,
        data_path='dataset',
        output='models',
        bert_path='/home/sunshine/pre_models/pytorch/bert-base-chinese/',
        train_mode='train'
    )
    return Namespace(**params)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(args, tokenizer, debug=False):
    """生成数据迭代器
    """

    train_usual_data = load_data(args.data_path + '/train/usual_train.txt')
    train_virus_data = load_data(args.data_path + '/train/virus_train.txt')

    pad_virus_data = train_virus_data * (len(train_usual_data) // len(train_virus_data)) + train_virus_data[: len(
        train_usual_data) % len(train_virus_data)]

    assert len(train_usual_data) == len(pad_virus_data)

    train_data = list(zip(train_usual_data, pad_virus_data))

    dev_usual_data = load_data(args.data_path + '/eval/usual_eval_labeled.txt')
    dev_virus_data = load_data(args.data_path + '/eval/virus_eval_labeled.txt')

    if debug:
        train_data = train_data[:20]
        dev_usual_data = dev_usual_data[:5]
        dev_virus_data = dev_virus_data[:5]
    # neural（无情绪）、happy（积极）、angry（愤怒）、sad（悲伤）、fear（恐惧）、surprise（惊奇）。
    label2id = {"neutral": 0, "happy": 1, "angry": 2, "sad": 3, "fear": 4, "surprise": 5}

    train_data_loader = SMPDataset(
        data=train_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=256,
        task='all'
    ).get_data_loader(batch_size=args.batch_size)

    valid_usual_data_loader = SMPDataset(
        data=dev_usual_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=256,
        task='usual'
    ).get_data_loader(batch_size=args.batch_size)

    valid_virus_data_loader = SMPDataset(
        data=dev_virus_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=256,
        task='virus'
    ).get_data_loader(batch_size=args.batch_size)

    return (train_data_loader, valid_usual_data_loader, valid_virus_data_loader), label2id


def main():
    args = get_args()
    # 设置随机种子
    set_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    # 处理数据
    data_loader, label2id = build_dataset(args, tokenizer, debug=True)

    # 构建trainer

    trainer = Trainer(
        args=args,
        data_loaders=data_loader,
        tokenizer=tokenizer,
        id2label={v: k for k, v in label2id.items()}
    )

    trainer.train()


if __name__ == '__main__':
    main()
