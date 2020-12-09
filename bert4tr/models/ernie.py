# Author: dgm
# Description: ernie模型
# Date: 2020-12-09
import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
"""
将ernie的模型转成pytorch可以加载的模型
然后使用BERT加载模型
"""

from bert4tr.models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    name = 'ernie'
    batch_size = 64
    num_epochs = 10
    pad_size = 32
    learning_rate = 5e-5
    hidden_size = 768
    require_improvement = 1000

    ernie_path = os.path.join(base_dir, 'models/ernie_pretrain')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(ernie_path)
    best_model_dir = os.path.join(base_dir, 'result/best_model/ernie/') + name + '.ckpt'


class ERNIEModel(nn.Module):
    def __init__(self, config):
        super(ERNIEModel, self).__init__()
        self.ernie_config = BertConfig.from_pretrained(config.ernie_path, num_labels=config.num_classes)
        self.ernie = BertForSequenceClassification.from_pretrained(config.ernie_path, config=self.ernie_config)
        for param in self.ernie.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        pooled = self.ernie(context, attention_mask=mask)
        out = pooled[0]
        return out