from pytorch_transformers import BertModel
import torch
import torch.nn as nn
from torchcrf import CRF

class bert_bilstm_crf(nn.Module):
    def __init__(self, pretrain_model_path = None, pretrain_output_size = 768, lstm_hidden_size = 384,
                 num_layers = 1, batch_first = True, bidirectional = True, lable_num = 4):
        super(bert_bilstm_crf, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain_model_path)

        self.bilstm = nn.LSTM(
                input_size=pretrain_output_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional
            )

        # self.dropout = SpatialDropout(drop_p)
        # self.layer_norm = LayerNorm(hidden_size * 2)

        self.linear = nn.Linear(hidden_size * 2, lable_num)

        self.crf = CRF(lable_num, batch_first = batch_first)

    def forward(self, x, max_len, sen_len, y = None, use_cuda = True):

        segments_ids = torch.zeros(x.shape, dtype=torch.long)

        emb_outputs = self.bert(x, token_type_ids=segments_ids)

        bilstm_output = self.bilstm(emb_outputs[0])

        linear_output = self.linear(bilstm_output)

        if train_y:
            log_likelihood = self.crf(linear_output, y,
                                        mask=_generate_mask(sen_len, max_len, use_cuda),
                                        reduction='mean')
            return log_likelihood
        else:
            output = self.crf.decode(x, mask=_generate_mask(sen_len, max_len, use_cuda))

    @torch.no_grad()
    def decode(self, dev_x, max_len, sen_len, use_cuda):
        return self.forward(x = dev_x, max_len = max_len, sen_len = sen_len, use_cuda = use_cuda)


