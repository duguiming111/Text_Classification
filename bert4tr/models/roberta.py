# Author: dgm
# Description: roberta模型
# Date: 2020-12-09
import os
import torch
import torch.nn as nn

from transformers import BertTokenizer, RobertaConfig, RobertaForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification
"""
这里使用roberta的模型来自于https://github.com/brightmart/roberta_zh
只能用transformers中的BertConfig, BertForSequenceClassification
"""

from bert4tr.models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    name = 'roberta'
    batch_size = 64
    num_epochs = 10
    pad_size = 32
    learning_rate = 5e-5
    hidden_size = 768
    require_improvement = 1000

    roberta_path = os.path.join(base_dir, 'models/roberta_pretrain')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(roberta_path)
    best_model_dir = os.path.join(base_dir, 'result/best_model/roberta/') + name + '.ckpt'


class RoBERTaModel(nn.Module):
    def __init__(self, config):
        super(RoBERTaModel, self).__init__()
        self.roberta_config = BertConfig.from_pretrained(config.roberta_path, num_labels=config.num_classes)
        self.roberta = BertForSequenceClassification.from_pretrained(config.roberta_path, config=self.roberta_config)
        for param in self.roberta.parameters():
            param.requires_grad = True

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        pooled = self.roberta(context, attention_mask=mask)
        out = pooled[0]
        return out