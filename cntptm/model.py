from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import XLMRobertaConfig
from torch import nn


class RobertaLMHead(nn.Module):
    '''
        RobertaLMHead这个类是从Transformers库中复制过来的。不修改Transformer库的代码的话，似乎无法直接调用此类。
        这个类的作用是输出MLM的loss和logits。
    '''

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x



class SimALMModel(RobertaPreTrainedModel):
    '''
        此类为预训练模型的类。
    '''
    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(self, input_ids, attention_mask, mlm_labels):
        """
            input_ids是句子中每个单词在词表中的编号。
            lang_ids是句子中每个单词所属语言的编号。
            attention_mask是句子对应的注意力掩码。
            mlm_labels是MLM需要预测的句子中的单词。
        """

        # outputs是句子在输出层的表示(hidden state)。
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        # 这里是计算MLM loss的过程（模仿Transformers库中的写法）
        prediction_scores = self.lm_head(outputs)
        mlm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        # sent_emb是句子在[CLS]位置处的表示（作为句向量）
        sent_emb = outputs[ : , 0, : ] # shape: [batch_size, hidden_size]

        # 返回ALM loss和句向量
        return mlm_loss, sent_emb
