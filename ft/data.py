import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

def make_input(input_sentence, max_length, tokenizer):
    '''
        根据输入句子生成input_ids, attention_mask。
        其中input_ids是每个单词在词表中的编号。
        attention_mask是训练时用的注意力掩码。

        函数参数input_sentence是输入的原始句子，无需进行处理（或者是简单的codemix）。
        例如：输入句子是“My delivery status says failed”（没有codemix）
        或者“My delivery 状态 says 失败”（有codemix）
    '''

    input_tokens = [tokenizer.cls_token] + tokenizer.tokenize(input_sentence) + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    if len(input_ids) > max_length:
        input_ids = input_ids[ : max_length] # 若句子长度超过最大长度限制则截断
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask


class SentenceRetrievalDataset(Dataset):
    def __init__(self, input_filenames, max_length, tokenizer):
        '''
            从文件加载预训练数据集。这里query和label分在两个不同的文件中。
            input_filenames是一个字典，其格式为{'query': query文件的路径, 'label': label文件的路径}。
            tokenizer是预训练时使用的tokenizer。
        '''

        # 首先从文件读入query和label的句子
        inputs = {}
        for key in ['query', 'label']:
            if input_filenames[key] is not None:
                fin = open(input_filenames[key], 'r', encoding='utf-8')
                inputs[key] = [x.strip() for x in fin]
                fin.close()
        
        n_examples = len(inputs['query'])
        self.examples = []
        for i in range(n_examples):
            # 对query和label分别构建input_ids, attention_mask
            input_ids_query, attention_mask_query = make_input(inputs['query'][i], max_length, tokenizer)
            input_ids_label, attention_mask_label = make_input(inputs['label'][i], max_length, tokenizer)
            
            self.examples.append({
                'input_ids_query': input_ids_query,
                'attention_mask_query': attention_mask_query,
                'input_ids_label': input_ids_label,
                'attention_mask_label': attention_mask_label,
            })
            
            # 显示数据加载进度
            if i % 1000 == 0:
                print('%d examples loaded' % i)
        
        # 加载完毕后，输出一些数据示例
        for i in range(5):
            print('Example ID:', i)
            print('Input Tokens (Query):', ' '.join(tokenizer.convert_ids_to_tokens(self.examples[i]['input_ids_query'])))
            print('Input Token IDs (Query):', ' '.join([str(x) for x in self.examples[i]['input_ids_query']]))
            print('Attention Mask (Query):', ' '.join([str(x) for x in self.examples[i]['attention_mask_query']]))
            print('Input Tokens (Label):', ' '.join(tokenizer.convert_ids_to_tokens(self.examples[i]['input_ids_label'])))
            print('Input Token IDs (Label):', ' '.join([str(x) for x in self.examples[i]['input_ids_label']]))
            print('Attention Mask (Label):', ' '.join([str(x) for x in self.examples[i]['attention_mask_label']]))
        
    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


class TestDataset(Dataset):
    def __init__(self, input_filename, max_length, tokenizer):
        '''
            从文件加载预测试数据集。测试数据集的格式是每行一个句子。
            input_filename是测试数据的文件名。
            tokenizer是预训练时使用的tokenizer。
        '''

        # 首先从文件读入query和label的句子
        inputs = {}
        fin = open(input_filename, 'r', encoding='utf-8')
        inputs = [x.strip() for x in fin]
        fin.close()
        
        n_examples = len(inputs)
        self.examples = []
        for i in range(n_examples):
            # 对query和label分别构建input_ids, attention_mask
            input_ids, attention_mask = make_input(inputs[i], max_length, tokenizer)
            
            self.examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            })
            
            # 显示数据加载进度
            if i % 1000 == 0:
                print('%d examples loaded' % i)
        
        # 加载完毕后，输出一些数据示例
        for i in range(5):
            print('Example ID:', i)
            print('Input Tokens:', ' '.join(tokenizer.convert_ids_to_tokens(self.examples[i]['input_ids'])))
            print('Input Token IDs:', ' '.join([str(x) for x in self.examples[i]['input_ids']]))
            print('Attention Mask:', ' '.join([str(x) for x in self.examples[i]['attention_mask']]))
        
    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def _collate_batch(examples, key):
    '''
        将一个batch中的多个输入序列转换为padded tensor的过程。
        其中examples是从dataset中随机抽取的数据。key是待转换的字段的名称。
    '''

    examples = [torch.tensor(e[key], dtype=torch.long) for e in examples]
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], 0)
    for i, example in enumerate(examples):
        result[i, : example.shape[0]] = example
    return result


def collate_fn(examples):
    '''
        从dataset加载一个batch的数据的过程。
        examples是从dataset中随机抽取的数据。
    '''

    # 将query和label的input_ids、attention_mask都转换成padded tensor的形式
    batch = {
        'input_ids_query': _collate_batch(examples, 'input_ids_query'),
        'attention_mask_query': _collate_batch(examples, 'attention_mask_query'),
        'input_ids_label': _collate_batch(examples, 'input_ids_label'),
        'attention_mask_label': _collate_batch(examples, 'attention_mask_label'),
    }
    return batch


def test_collate_fn(examples):
    '''
        从dataset加载一个batch的数据的过程。（用于测试集）
        examples是从dataset中抽取的数据。
    '''

    # 将query和label的input_ids、attention_mask都转换成padded tensor的形式
    batch = {
        'input_ids': _collate_batch(examples, 'input_ids'),
        'attention_mask': _collate_batch(examples, 'attention_mask'),
    }
    return batch
