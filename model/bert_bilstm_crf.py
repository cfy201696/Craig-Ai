# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
from pytorch_transformers import BertModel
import torch
import torch.nn as nn
from torchcrf import CRF
from layers.utensil import _generate_mask

class bert_bilstm_crf(nn.Module):
    def __init__(self, pretrain_model_path = None, pretrain_output_size = 768, lstm_hidden_size = 384,
                 num_layers = 1, dropout_ratio = 0.5, batch_first = True, bidirectional = True, lable_num = 4, device = "cpu"):
        super(bert_bilstm_crf, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain_model_path)

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

        segments_ids = torch.zeros(x.shape, dtype=torch.long).to(self.device)

        emb_outputs = self.bert(x, token_type_ids=segments_ids)

        bilstm_output, _ = self.bilstm(emb_outputs[0])

        drop_out = self.dropout(bilstm_output)

        linear_output = self.linear(drop_out)

        return linear_output

    def get_loss(self, linear_output, max_len, sen_len, y = None, use_cuda = True):
        log_likelihood = self.crf(linear_output, y,
                                  mask=_generate_mask(sen_len, max_len, use_cuda),
                                  reduction='mean')
        return -log_likelihood

    def save_model(self, path):
        print("saving model")
        # self.file_record.write("saving model\n")
        torch.save(self.state_dict(), path)

    def model_structure(self):
        for name, parameters in self.named_parameters():
            print(name, '------->', parameters.size(), '------->', parameters.requires_grad)
            # self.file_record.write('{}------->{}------->{}\n'.format(name, str(parameters.size()), parameters.requires_grad))
        print("-" * 100)

    @torch.no_grad()
    def decode(self, dev_x, max_len = None, sen_len = None, use_cuda = None, dev_y = None):
        output = self.forward(x = dev_x)
        loss = None
        if dev_y != None:
            loss = self.get_loss(output, max_len, sen_len, y = dev_y, use_cuda = use_cuda)
        return self.crf.decode(output, mask=_generate_mask(sen_len, max_len, use_cuda)), loss


