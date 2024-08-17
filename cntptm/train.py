import argparse
import os
import pandas as pd
import torch

from data import DataCollatorForSimALM, SimALMDataset
from model import SimALMModel
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

parser = argparse.ArgumentParser()

parser.add_argument('--train-query', type=str, required=True)
parser.add_argument('--train-label', type=str, required=True)

parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--update-cycle', type=int, default=2)
parser.add_argument('--num-epochs', type=int, default=3)
parser.add_argument('--warmup-steps', type=int, default=0)
parser.add_argument('--learning-rate', type=float, default=5e-6)
parser.add_argument('--alm-loss-weight', type=float, default=0.1)
parser.add_argument('--max-length', type=int, default=128)

parser.add_argument('--pretrained-model-path', type=str, required=True)
parser.add_argument('--save-model-path', type=str, required=True)
parser.add_argument('--is-continue', action='store_true')

args = parser.parse_args()
print(args)

device = torch.device('cuda')
torch.cuda.set_device(0)

config = AutoConfig.from_pretrained(args.pretrained_model_path, cache_dir=None)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, cache_dir=None, use_fast=False)

if args.is_continue:
    checkpoint = '%s/checkpoint_last' % args.save_model_path
    fstep = open('%s/checkpoint_last/step.txt' % args.save_model_path, 'r')
    last_step = int(fstep.read().strip())
    fstep.close()
else:
    checkpoint = args.pretrained_model_path
    last_step = 0

model = SimALMModel.from_pretrained(checkpoint, config=config)
model.to(device)
print('Model loaded from', checkpoint)

train_dataset = SimALMDataset({
    'query': args.train_query,
    'label': args.train_label},
    tokenizer, args.max_length
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=DataCollatorForSimALM(tokenizer)
)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            0.01
    },
    {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
total_batch_size = args.batch_size * args.update_cycle
total_steps = ((len(train_dataset) - 1) // total_batch_size + 1) * args.num_epochs
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

print('Number of Examples:', len(train_dataset))
print('Total Batch Size:', total_batch_size)
print('Number of Epochs:', args.num_epochs)
print('Total Optimization Steps:', total_steps)

files_to_copy = os.listdir(args.pretrained_model_path)
d = 'checkpoint_last'
os.system('mkdir -p %s' % os.path.join(args.save_model_path, d))
for f in files_to_copy:
    if f != 'pytorch_model.bin':
        os.system('cp %s %s' % (
            os.path.join(args.pretrained_model_path, f),
            os.path.join(args.save_model_path, d, f)
        ))
print('Configuration files copied')

def save_model(model, save_dir):
    print('Saving Model to', save_dir)
    if os.path.exists(save_dir):
        print('%s already exists. Removing it...' % save_dir)
        os.remove(save_dir)
        print('%s removed successfully.' % save_dir)
    
    torch.save(model.state_dict(), save_dir)
    print('%s saved successfully.' % save_dir)

total_minibatches = len(train_dataloader)
num_steps = 1
model.train()
model.zero_grad()

ranking_loss_func = nn.CrossEntropyLoss()

total_loss = 0.0
for epoch in range(args.num_epochs):
    for i, inputs in enumerate(train_dataloader):
        n_minibatches = i + 1

        if num_steps > last_step:
            query_mlm_loss, query_sent_emb = model(
                input_ids=inputs['input_ids_query'].to(device),
                attention_mask=inputs['attention_mask_query'].to(device),
                mlm_labels=inputs['mlm_labels_query'].to(device),
            )

            label_mlm_loss, label_sent_emb = model(
                input_ids=inputs['input_ids_label'].to(device),
                attention_mask=inputs['attention_mask_label'].to(device),
                mlm_labels=inputs['mlm_labels_label'].to(device),
            )

            alm_loss = query_mlm_loss + label_mlm_loss

            query_sent_emb = query_sent_emb.unsqueeze(1)
            label_sent_emb = label_sent_emb.unsqueeze(0)
            dot_product = torch.sum(query_sent_emb * label_sent_emb, dim=-1)
            dist_query_sent_emb = torch.sum(query_sent_emb ** 2, dim=-1) ** 0.5
            dist_label_sent_emb = torch.sum(label_sent_emb ** 2, dim=-1) ** 0.5
            cos_sim = dot_product / (dist_query_sent_emb * dist_label_sent_emb)

            actual_batch_size = int(inputs['input_ids_query'].shape[0])
            ranking_labels = torch.tensor(list(range(0, actual_batch_size))).to(device)
            ranking_loss = ranking_loss_func(cos_sim / 0.05, ranking_labels)

            print('epoch = %d, step = %d, alm_loss = %.6f, ranking_loss = %.6f, total_loss = %.6f' % (epoch, num_steps, float(alm_loss), float(ranking_loss), float(args.alm_loss_weight * alm_loss + ranking_loss)))

            loss = (args.alm_loss_weight * alm_loss + ranking_loss) / float(args.update_cycle)
            loss.backward()
        
        if (n_minibatches == total_minibatches) or (n_minibatches % args.update_cycle == 0):
            if num_steps <= last_step:
                lr_scheduler.step()
            else:
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

            num_steps += 1
    
    if num_steps > last_step:
        save_model(model, os.path.join(args.save_model_path, 'checkpoint_last/pytorch_model.bin'))
        fstep = open(os.path.join(args.save_model_path, 'checkpoint_last/step.txt'), 'w')
        fstep.write('%d' % num_steps)
        fstep.close()
