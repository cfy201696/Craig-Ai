# -*- coding:utf-8 -*-
import sys

from pytorch_transformers import BertModel
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.base_model import base_model
from torchcrf import CRF
sys.path.append("../")
from layers.utensil import _generate_mask
import copy
import numpy as np

class pa_lstm_crf(base_model):
    def __init__(self, pretrain_model_path = None, pretrain_output_size = 768, lstm_hidden_size = 20,
                 num_layers = 1, dropout_ratio = 0.5, batch_first = True, bidirectional = True, label_num = 4, device = "cpu"):
        super(pa_lstm_crf, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain_model_path)

        self.bilstm = nn.LSTM(
            input_size=pretrain_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_ratio,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

        self.linear = nn.Linear(lstm_hidden_size * 2, label_num)

        self.W_H = nn.Linear(label_num, label_num)

        self.W_P = nn.Linear(label_num, label_num)

        self.W_h = nn.Linear(label_num, label_num)

        self.V = torch.randn(label_num, requires_grad = True).to(device)

        # self.dropout = SpatialDropout(drop_p)
        # self.layer_norm = LayerNorm(hidden_size * 2)
        self.tanh = nn.Tanh()  # 从nn模块中调入Tanh()层，在nn.functional中有对应的函数

        self.crf = CRF(label_num, batch_first = batch_first)

        self.device = device

        print("模型加载完成")

    def forward(self, x):
        '''
        :param x: (batch_size, l)
        :return:
        '''
        x = x.to(self.device).long()

        segments_ids = torch.zeros(x.shape, dtype=torch.long).to(self.device)

        # batch_size, sentence_length->batch_size, sentence_length, 768
        emb_outputs = self.bert(x, token_type_ids=segments_ids)

        # batch_size, sentence_length,768->batch_size, sentence_length, 400
        Bilstm_H, _ = self.bilstm(emb_outputs[0])

        # batch_size, sentence_length, 400 -> batch_size, sentence_length, label_num
        H = self.linear(Bilstm_H)

        # batch_size, sentence_length, label_num->batch_size, sentence_length, label_num
        w_H = self.W_H(H)

        # batch_size, sentence_length, label_num->batch_size, sentence_length, label_num
        w_P = self.W_P(H)

        # batch_size,sentence_length,label_num->batch_size,sentence_length,label_num
        w_h = self.W_h(H)

        batch, l, n  = H.size()

        # batch_size * sentence_max_len * sentence_max_len * num_logits_output
        b_p_empty = torch.empty(batch, l, l, n).to(self.device)

        # batch_size, sentence_length, label_num->
        # batch_size, sentence_max_len, sentence_max_len, num_logits_output
        for b in range(batch):
            H_copy = w_H[b] # sentence_length,label_num
            for p in range(l):
                p_copy = w_P[b][p] # label_num
                l_n_empty = torch.empty(l, n).to(self.device) # sentence_max_len, label_num
                for t in range(l):
                    t_copy = w_h[b][t] # label_num
                    mid_atten = F.softmax(self.tanh(H_copy + p_copy.unsqueeze(0).expand_as(H_copy)
                                                    + t_copy.unsqueeze(0).expand_as(H_copy)) @
                                          self.V.T, dim=0) @ H[b]
                    l_n_empty[t] = mid_atten
                b_p_empty[b][p] = l_n_empty

        return b_p_empty # batch_size, sentence_max_len, sentence_max_len, num_logits_output

    def get_loss(self, b_p_empty, max_len, sen_len, y = None, use_cuda = True):
        '''
        :param b_p_empty: batch_size, sentence_max_len(P), sentence_max_len, num_logits_output
        :param y:batch_size, sentence_max_len(P), sentence_max_len
        :param use_cuda:
        :return:
        '''
        y = y.to(self.device).long()
        be_ = 1 == torch.Tensor([False]).to(self.device)
        for idx, p_out in enumerate(b_p_empty):
            index = ~(y[idx] == 0).all(1).to(self.device)
            length = len(index[index == True])
            if idx == 0:
                log_likelihood = -self.crf(p_out[torch.cat((be_, index),0)[:-1]], y[idx][index], mask=_generate_mask([sen_len[idx]] * length, max_len, use_cuda), reduction='mean')
            else:
                log_likelihood -= self.crf(p_out[torch.cat((be_, index),0)[:-1]], y[idx][index], mask=_generate_mask([sen_len[idx]] * length, max_len, use_cuda), reduction='mean')
        return log_likelihood

    @torch.no_grad()
    def decode(self, dev_x, max_len, sen_len, use_cuda, dev_y = None):
        '''
        :param dev_x:[8, 100]
        :param max_len:
        :param sen_len:
        :param use_cuda:
        :param dev_y:[8, 100, 100, 49]
        :return:
        '''
        dev_x = torch.tensor(dev_x, dtype=torch.long).to(self.device)
        b_p_empty = self.forward(x = dev_x)
        result = []
        loss = None
        if dev_y != None:
            dev_y = torch.tensor(dev_y, dtype=torch.long).to(self.device)
            loss = self.get_loss(copy.deepcopy(b_p_empty), max_len, sen_len, y = dev_y, use_cuda = use_cuda)
        for idx, p_out in enumerate(b_p_empty):
            length = sen_len[idx]
            result.append(self.crf.decode(p_out[1:length+1], mask=_generate_mask([sen_len[idx]] * length, max_len, use_cuda)))

        return result, loss



