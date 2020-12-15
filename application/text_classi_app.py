# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
import os
import json
import numpy as np
import copy
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from pytorch_transformers import AdamW, WarmupLinearSchedule
from process_data.text_classification_process import text_classification_data_process_machine
from model.bert_softmax import bert_softmax
from torch.utils import data
from tensorboardX import SummaryWriter
from evaluation.evaluation import evaluation


class text_classi_app:
    def __init__(self, model_file_path, model_name):
        super(text_classi_app, self).__init__()
        self.model_file_path = model_file_path
        self.model_name = model_name

        with open(self.model_file_path + self.model_name + "_model_config.json",
                  encoding="utf-8") as f:
            self.config = json.load(f)

        if self.config["model_structure"] == "bert_softmax":
            self.begin, self.end = 1, 1

        print(self.config)

        if self.config["model_structure"] == "bert_softmax":
            self.model = bert_softmax(
                pretrain_model_path=self.config["model_path"],
                lable_num=len(self.config["label2id"]),
                device=device)

        self.model.load_state_dict(torch.load(self.config["model_file_path"] + self.config["model_name"]))
        self.model.to(device)
        self.model.eval()

    def decode(self, data_list):
        '''
        data_list = [{"text":,"label":}]
        '''
        data_list_copy = []
        for d in data_list:
            d["label"] = list(self.config["label2id"])[0]
            data_list_copy.append(d)
        test_dataset = text_classification_data_process_machine(
            data=data_list_copy,
            sentence_max_len=self.config["sentence_max_length"],
            model_structure=self.config["model_structure"],
            label2id=self.config["label2id"],
            tokenizer_path=self.config["model_path"])
        test_text_list, test_x, test_y = test_dataset.extract_data(copy.deepcopy(test_dataset.get_data()))
        test_dataloader = data.DataLoader(test_dataset,
                                          batch_size=self.config["batch_size"], shuffle=False, num_workers=6)

        test_output = []
        test_loss = []
        for ide, batch_data_dev in enumerate(test_dataloader):
            d_output, d_loss = self.model.decode(copy.deepcopy(batch_data_dev[1]), dev_y=copy.deepcopy(batch_data_dev[2]))
            test_output += d_output
            test_loss += [d_loss.item()]

        source_test_data, predict_test_data = \
            test_dataset.transform_data_back(
                test_dataset.uni_data(test_text_list, test_x, test_output))
        return predict_test_data





