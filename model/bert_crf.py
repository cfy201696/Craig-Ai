# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
from pytorch_transformers import BertModel
import torch
import torch.nn as nn
from model.base_model import base_model
from torchcrf import CRF

sys.path.append("../")
from layers.utensil import _generate_mask

class bert_crf(base_model):
    def __init__(self, pretrain_model_path = None, pretrain_output_size = 768,
                 batch_first = True, lable_num = 4, device = "cpu"):
        super(bert_crf, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain_model_path)

        self.linear = nn.Linear(pretrain_output_size, lable_num)

        self.crf = CRF(lable_num, batch_first = batch_first)

        self.device = device

    def forward(self, x):

        segments_ids = torch.zeros(x.shape, dtype=torch.long).to(self.device)

        emb_outputs = self.bert(x, token_type_ids=segments_ids)

        linear_output = self.linear(emb_outputs[0])

        return linear_output

    def get_loss(self, linear_output, max_len, sen_len, y = None, use_cuda = True):
        log_likelihood = self.crf(linear_output, y,
                                  mask=_generate_mask(sen_len, max_len, use_cuda),
                                  reduction='mean')
        return -log_likelihood

    @torch.no_grad()
    def decode(self, dev_x, max_len = None, sen_len = None, use_cuda = None, dev_y = None):
        output = self.forward(x = dev_x)
        loss = None
        if dev_y != None:
            loss = self.get_loss(output, max_len, sen_len, y = dev_y, use_cuda = use_cuda)
        return self.crf.decode(output, mask=_generate_mask(sen_len, max_len, use_cuda)), loss