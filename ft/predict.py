'''
注意：此文件仅能根据输入文件，输出句子对应的向量，不能测试准确率！
'''

import argparse
import numpy as np
import os
import torch

from data import test_collate_fn, TestDataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

parser = argparse.ArgumentParser()

# 参数：测试集文件名
parser.add_argument('--test-file', type=str, required=True)

# 参数：输出文件名
parser.add_argument('--output-file', type=str, required=True)

# 测试时有关的超参数
parser.add_argument('--batch-size', type=int, default=8)

# 测试时加载的checkpoint
parser.add_argument('--checkpoint', type=str, required=True)

args = parser.parse_args()
print(args)

# 目前是单卡训练
device = torch.device('cuda')
torch.cuda.set_device(0)

config = AutoConfig.from_pretrained(args.checkpoint, num_labels=1, cache_dir=None)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, cache_dir=None, use_fast=False)

# 加载模型（目前是加载XLM模型，因为只有XLM支持language embedding，能够标明每个单词属于哪种语言）
model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, config=config)
model.to(device)
print('Model loaded from', args.checkpoint)

# 构建dataset
test_dataset = TestDataset(args.test_file, 512, tokenizer)

# 构建dataloader
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=test_collate_fn
)

fout = open(args.output_file, 'w')
model.eval()
with torch.no_grad():
    # 逐个batch输出句子的句向量
    for inputs in test_dataloader:
        # 获取句向量
        sent_emb = model(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            output_hidden_states=True
        )[1][-1][ : , 0, : ]
        sent_emb = list(sent_emb.cpu().numpy())

        # 将句向量输出到文件
        for emb in sent_emb:
            fout.write(' '.join([str(x) for x in emb]) + '\n')
fout.close()
