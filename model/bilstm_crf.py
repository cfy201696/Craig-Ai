# -*- coding:utf-8 -*-
import sys
from gensim.models import KeyedVectors
import torch
import numpy as np
import torch.nn as nn
from model.base_model import base_model
from torchcrf import CRF
sys.path.append("../")
from layers.utensil import _generate_mask

class bilstm_crf(base_model):
    def __init__(self, pretrain_model_path = None, word2id = None, pretrain_output_size = 768, lstm_hidden_size = 384,
                 num_layers = 1, dropout_ratio = 0.5, batch_first = True, bidirectional = True, lable_num = 4, device = "cpu"):
        super(bilstm_crf, self).__init__()

        if word2id:
            self.word2id = word2id
            self.emb = nn.Embedding(len(self), pretrain_output_size)
        else:
            self.emb = KeyedVectors.load(pretrain_model_path) # 加载保存的word vectors
            self.vocab = self.emb.vocab.keys()

        self.pretrain_output_size = pretrain_output_size

        self.bilstm = nn.LSTM(
                input_size=pretrain_output_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_layers,
                dropout=dropout_ratio,
                batch_first=batch_first,
                bidirectional=bidirectional
            )

        # self.dropout = SpatialDropout(drop_p)
        # self.layer_norm = LayerNorm(hidden_size * 2)

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.linear = nn.Linear(lstm_hidden_size * 2, lable_num)

        self.crf = CRF(lable_num, batch_first = batch_first)

        self.device = device

    def forward(self, x):
        # print(type(x))
        # print(x)
        # print(type(x[0]))
        x_ = []
        for bs in x:
            sen = []
            for w in bs:
                if w in self.vocab:
                    sen.append(self.emb[w].tolist())
                else:
                    sen.append([0.0]*self.pretrain_output_size)
            x_.append(sen)

        x_input = torch.tensor(x_, device=self.device)
        x_input = x_input.float()


        # segments_ids = torch.zeros(x.shape, dtype=torch.long).to(self.device)
        #
        # emb_outputs = self.bert(x, token_type_ids=segments_ids)

        bilstm_output, _ = self.bilstm(x_input)

        drop_out = self.dropout(bilstm_output)

        linear_output = self.linear(drop_out)

        return linear_output

    def get_loss(self, linear_output, max_len, sen_len, y = None, use_cuda = True):
        y = y.to(self.device).long()
        log_likelihood = self.crf(linear_output, y,
                                  mask=_generate_mask(sen_len, max_len, use_cuda),
                                  reduction='mean')
        return -log_likelihood

    @torch.no_grad()
    def decode(self, dev_x, max_len = None, sen_len = None, use_cuda = None, dev_y = None):
        output = self.forward(x = dev_x)
        loss = None
        if dev_y != None:
            dev_y = torch.tensor(dev_y, dtype=torch.long).to(self.device)
            loss = self.get_loss(output, max_len, sen_len, y = dev_y, use_cuda = use_cuda)
        return self.crf.decode(output, mask=_generate_mask(sen_len, max_len, use_cuda)), loss


