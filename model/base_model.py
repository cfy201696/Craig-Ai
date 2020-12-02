# -*- coding:utf-8 -*-
import sys
import abc
import torch.nn as nn
import torch

class base_model(nn.Module):

    @abc.abstractmethod
    def get_loss(self, linear_output, max_len, sen_len, y = None, use_cuda = True):
        pass

    def save_model(self, path):
        print("saving model")
        # self.file_record.write("saving model\n")
        torch.save(self.state_dict(), path)

    def model_structure(self):
        for name, parameters in self.named_parameters():
            print(name, '------->', parameters.size(), '------->', parameters.requires_grad)
            # self.file_record.write('{}------->{}------->{}\n'.format(name, str(parameters.size()), parameters.requires_grad))
        print("-" * 100)

    @abc.abstractmethod
    def decode(self, dev_x, max_len = None, sen_len = None, use_cuda = None, dev_y = None):
        pass
