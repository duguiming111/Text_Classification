# Author: dgm
# Description: albert模型
# Date: 2020-12-08
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AlbertConfig, AlbertForSequenceClassification

from bert4tr.models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    name = 'albert'
    batch_size = 64
    num_epochs = 10
    pad_size = 32
    learning_rate = 5e-5
    hidden_size = 768
    require_improvement = 1000

    bert_path = os.path.join(base_dir, 'models/albert_pretrain')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    best_model_dir = os.path.join(base_dir, 'result/best_model/albert/') + name + '.ckpt'


class ALBERTModel(nn.Module):
    def __init__(self, config):
        super(ALBERTModel, self).__init__()
        self.albert_config = AlbertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.albert = AlbertForSequenceClassification.from_pretrained(config.bert_path, config=self.albert_config)
        for param in self.albert.parameters():
            param.requires_grad = True

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        pooled = self.albert(context, attention_mask=mask)
        out = pooled[0]
        return out