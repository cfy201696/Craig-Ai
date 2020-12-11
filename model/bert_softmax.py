# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
from pytorch_transformers import BertModel
import torch
import torch.nn as nn
from model.base_model import base_model
import numpy as np

sys.path.append("../")
from layers.utensil import _generate_mask

class bert_softmax(base_model):
    def __init__(self, pretrain_model_path = None, pretrain_output_size = 768, lable_num = 4, device = "cpu"):
        super(bert_softmax, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain_model_path)

        self.linear = nn.Linear(pretrain_output_size, lable_num)

        self.loss = nn.CrossEntropyLoss()

        self.device = device

        print("模型加载完成")

    def forward(self, x):
        x = x.to(self.device).long()

        segments_ids = torch.zeros(x.shape, dtype=torch.long).to(self.device)

        emb_outputs = self.bert(x, token_type_ids=segments_ids)

        linear_output = self.linear(emb_outputs[1])

        return linear_output

    def get_loss(self, linear_output, y = None):
        y = y.to(self.device).long()
        loss = self.loss(linear_output, torch.squeeze(y))
        return loss

    @torch.no_grad()
    def decode(self, dev_x, dev_y = None):
        dev_x = torch.tensor(dev_x, dtype=torch.long).to(self.device)
        output = self.forward(x = dev_x)
        loss = None
        if dev_y != None:
            dev_y = torch.tensor(dev_y, dtype=torch.long).to(self.device)
            loss = self.get_loss(output, y = dev_y)
        output = np.vstack(torch.argmax(output, dim=1).cpu().numpy()).tolist()

        return output, loss