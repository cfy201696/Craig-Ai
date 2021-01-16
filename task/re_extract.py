# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
import os
import argparse
import json
import torch.optim as optim
import numpy as np
import copy
import torch
import shutil
device = "cuda" if torch.cuda.is_available() else "cpu"

from pytorch_transformers import AdamW, WarmupLinearSchedule
from process_data.re_process import re_data_process_machine
from model.pa_lstm_crf import pa_lstm_crf
from torch.utils import data
from tensorboardX import SummaryWriter
from evaluation.re_evaluation import f1_eva_re, f1_eva_re_predicate_level
import time

parser = argparse.ArgumentParser()
# 数据集参数
parser.add_argument("-tr","--train_file_path", help="训练文件路径，如果指定此参数则代表训练模型，并需要指定验证集", default=None)
parser.add_argument("-de","--dev_file_path", help="验证集路径", default=None)
parser.add_argument("-te","--test_file_path", help="测试文件路径，如果指定此参数，需要指定模型路径参数", default=None)
parser.add_argument("-mfp","--model_file_path", default="data/")
parser.add_argument("-mn","--model_name", default="v1")


# 模型结构参数
parser.add_argument("-ms","--model_structure", help="模型结构", default="pa_lstm_crf")
parser.add_argument("-mp","--model_path", default="../pretrained_model_file/bert/chinese_L-12_H-768_A-12")
parser.add_argument("-ft","--fine_tuning", help="微调", default=True)
parser.add_argument("-pos","--pretrain_output_size", help="预训练模型的输出维度", default=768)


# lstm的参数
parser.add_argument("-lhs","--lstm_hidden_size", help="lstm的隐藏单元数", default=200)
parser.add_argument("-nl","--num_layers", help="lstm的层数", default=1)
parser.add_argument("-bid","--bidirectional", help="lstm是否双向", default=True)
parser.add_argument("-dr","--dropout_ratio", help="dropout_ratio", default=0.5)


# 数据构造参数
parser.add_argument("-bs","--batch_size", help="batch_size", default=512)
parser.add_argument("-lr","--learning_rate", help="learning_rate", default=1e-5)
parser.add_argument("-sml","--sentence_max_length", help="sentence_max_length", default=300)

# 训练参数
parser.add_argument("-e","--epochs", help="epochs", default=20)

def str_to_bool(string):
    return True if str(string).lower() == 'true' else False


args = parser.parse_args()
model_config = {
    "train_file_path":args.train_file_path,
    "dev_file_path" :args.dev_file_path,
    "test_file_path" : args.test_file_path,
    "model_file_path" : args.model_file_path,
    "model_name" : args.model_name,
    "model_structure" : args.model_structure,
    "model_path":args.model_path,
    "fine_tuning":str_to_bool(args.fine_tuning),
    "pretrain_output_size":int(args.pretrain_output_size),
    "lstm_hidden_size" : int(args.lstm_hidden_size),
    "num_layers" : int(args.num_layers),
    "bidirectional" : str_to_bool(args.bidirectional),
    "dropout_ratio" : float(args.dropout_ratio),
    "batch_size" : int(args.batch_size),
    "learning_rate" : float(args.learning_rate),
    "sentence_max_length" : int(args.sentence_max_length),

    "epochs" : int(args.epochs)}


if model_config["train_file_path"]:

    if os.path.exists(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"])):
        # os.rmdir(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"]))
        shutil.rmtree(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"]))

    summary_writer = SummaryWriter(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"]))

    train_dataset = re_data_process_machine(file_path = model_config["train_file_path"],
                                             model_structure = model_config["model_structure"],
                                             sentence_max_len = model_config["sentence_max_length"],
                                             tokenizer_path = model_config["model_path"])
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=model_config["batch_size"], shuffle=True, num_workers=4)

    model_config["label2id"] = train_dataset.get_label2id()

    with open(model_config["model_file_path"] + model_config["model_name"] + "_model_config.json", "w+", encoding="utf-8") as f:
        f.write(json.dumps(model_config, ensure_ascii=False))

    dev_dataset = re_data_process_machine(
        file_path = model_config["dev_file_path"],
        sentence_max_len = model_config["sentence_max_length"],
        model_structure=model_config["model_structure"],
        label2id = model_config["label2id"],
        tokenizer = train_dataset.get_tokenizer())
    dev_dataloader = data.DataLoader(dev_dataset,
                                       batch_size=model_config["batch_size"], shuffle=False, num_workers=4)

    # 得到验证集的loss和F1，p，r
    dev_text_list, dev_split_text_list, dev_x, dev_y, dev_start_index, dev_sen_len = \
        dev_dataset.extract_data(copy.deepcopy(dev_dataset.get_data()))

    source_dev_data, transform_back_dev_data = dev_dataset.transform_data_back(
            dev_dataset.uni_data(copy.deepcopy(dev_text_list),copy.deepcopy(dev_split_text_list), copy.deepcopy(dev_x),
                                 copy.deepcopy(dev_y), copy.deepcopy(dev_start_index), copy.deepcopy(dev_sen_len)))


    print("验证集上限P R F1：", f1_eva_re(transform_back_dev_data, source_dev_data, save=True, save_path=os.path.join(model_config["model_file_path"],'Result',model_config["model_name"])))

    if model_config["model_structure"] == "pa_lstm_crf":
        model = pa_lstm_crf(
            pretrain_model_path = model_config["model_path"],
            lstm_hidden_size = model_config["lstm_hidden_size"],
            num_layers = model_config["num_layers"],
            dropout_ratio = model_config["dropout_ratio"],
            bidirectional = model_config["bidirectional"],
            label_num = len(model_config["label2id"]),
            device = device)

    model.to(device)

    num_total_steps = model_config["epochs"] * len(train_dataset) / model_config["batch_size"]
    num_warmup_steps = int(0.1 * num_total_steps)

    if model_config["fine_tuning"]:
        warm_up_params = []
    non_warm_up_params = []

    for name, param in model.named_parameters():
        if "bert" in name:
            if model_config["fine_tuning"]:
                warm_up_params.append(param)
            else:
                param.requires_grad = False
        else:
            non_warm_up_params.append(param)

    model.model_structure()

    if model_config["fine_tuning"]:
        warm_up_optimizer = AdamW(warm_up_params, lr=model_config["learning_rate"],
                                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = WarmupLinearSchedule(warm_up_optimizer, warmup_steps=num_warmup_steps,
                                         t_total=num_total_steps)  # PyTorch scheduler
    non_warm_up_optimizer = optim.Adam(non_warm_up_params, lr=model_config["learning_rate"] * 10)

    best_f1 = 0

    for i in range(model_config["epochs"]):
        loss_list = []
        model.train()
        epoch_start = time.time()
        for batch_data in train_dataloader:
            start = time.time()
            batch_data_array = np.array(batch_data)
            _, __, x, y, ___, sen_len = \
                batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5]

            # x_input = torch.tensor(x, dtype=torch.long).to(device)
            # y = torch.tensor(y, 0.T, dtype=torch.long).to(device)

            output = model(x)
            start2 = time.time()
            print("前向传播用时：", start2 - start)
            loss = model.get_loss(output, model_config["sentence_max_length"], sen_len, y = y)
            start3 = time.time()
            print("计算loss用时：", start3 - start2)

            loss.backward()
            start4 = time.time()
            print("反向传播用时：", start4 - start3)

            if model_config["fine_tuning"]:
                warm_up_optimizer.step()
                scheduler.step()
                warm_up_optimizer.zero_grad()
            non_warm_up_optimizer.step()
            non_warm_up_optimizer.zero_grad()

            loss_list.append(loss.item())
            start5 = time.time()
            print("一个batch用时：", start5 - start)
        print("一个epoch用时：", time.time() - epoch_start)

        ## 验证模型
        model.eval()
        dev_output = []
        dev_loss = []
        for ide, batch_data_dev in enumerate(dev_dataloader):
            # batch_data_array = np.array(batch_data)
            _, __, x, y, ___, sen_len = \
                batch_data_dev[0], batch_data_dev[1], batch_data_dev[2], batch_data_dev[3], batch_data_dev[4], batch_data_dev[5]

            d_output, d_loss = model.decode(copy.deepcopy(x), max_len = model_config["sentence_max_length"],
                                            sen_len = sen_len, use_cuda = True, dev_y=copy.deepcopy(y))
            dev_output += d_output
            dev_loss += [d_loss.item()]
            print("the epoch {}, evaluation step--------------->{}".format(i, ide))

        summary_writer.add_scalars("loss",{"train_loss" : np.mean(loss_list), "dev_loss": np.mean(dev_loss)}, i)

        source_dev_data, transform_back_dev_data = dev_dataset.transform_data_back(
            dev_dataset.uni_data(copy.deepcopy(dev_text_list), copy.deepcopy(dev_split_text_list), copy.deepcopy(dev_x),
                                 copy.deepcopy(dev_output), copy.deepcopy(dev_start_index), copy.deepcopy(dev_sen_len)))

        p, r, f1 = f1_eva_re(transform_back_dev_data, source_dev_data)

        summary_writer.add_scalars("dev_evaluation_total", {"p": p, "r": r, "f1": f1}, i)
        ans_dict = f1_eva_re_predicate_level(transform_back_dev_data, source_dev_data)
        for key, v in ans_dict.items():
            summary_writer.add_scalars(key, v, i)

        if f1 > best_f1:
            best_f1 = f1
            model.save_model(model_config["model_file_path"]+model_config["model_name"])
            # print("错误类型和个数：", sta_error_type(source_dev_data, predict_dev_data))
            print("label lavel:", ans_dict)
        print("训练完成：{}， 训练集loss：{}，验证集loss：{}，验证集P：{}，验证集R：{}，验证集F1：{}，最高F1：{}".format(
            i, np.mean(loss_list), np.mean(dev_loss), p, r, f1, best_f1))
        print("-"*100)

if model_config["test_file_path"]:
    with open(model_config["model_file_path"] + model_config["model_name"] + "_model_config.json", encoding="utf-8") as f:
        config = json.load(f)

    config["test_file_path"] = model_config["test_file_path"]
    print(config)

    if model_config["model_structure"] == "pa_lstm_crf":
        model = pa_lstm_crf(
            pretrain_model_path=config["model_path"],
            lstm_hidden_size=config["lstm_hidden_size"],
            num_layers=config["num_layers"],
            dropout_ratio=config["dropout_ratio"],
            bidirectional=config["bidirectional"],
            label_num=len(config["label2id"]),
            device=device)

    model.load_state_dict(torch.load(config["model_file_path"]+config["model_name"]))
    model.to(device)
    model.eval()

    test_dataset = re_data_process_machine(
        file_path=config["test_file_path"],
        sentence_max_len=config["sentence_max_length"],
        model_structure=config["model_structure"],
        label2id=config["label2id"],
        tokenizer_path=config["model_path"])

    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=model_config["batch_size"], shuffle=False, num_workers=4)

    # 得到验证集的loss和F1，p，r
    test_text_list, test_split_text_list, test_x, test_y, test_start_index, test_sen_len = \
        dtest_dataset.extract_data(copy.deepcopy(test_dataset.get_data()))

    source_test_data, transform_back_test_data = dev_dataset.transform_test_back(
        test_dataset.uni_data(copy.deepcopy(test_text_list), copy.deepcopy(test_split_text_list), copy.deepcopy(test_x),
                             copy.deepcopy(test_y), copy.deepcopy(test_start_index), copy.deepcopy(test_sen_len)))

    print("验证集上限P R F1：", f1_eva_re(transform_back_test_data, source_test_data, save=True,
                                    save_path=os.path.join(model_config["model_file_path"], 'Result',
                                                           model_config["model_name"])))

    test_output = []
    test_loss = []
    for ide, batch_data_test in enumerate(test_dataloader):
        _, __, x, y, ___, sen_len = \
            batch_data_test[0], batch_data_test[1], batch_data_test[2], batch_data_test[3], batch_data_test[4], \
            batch_data_test[5]
        d_output, d_loss = model.decode(copy.deepcopy(x), max_len=model_config["sentence_max_length"],
                                        sen_len=sen_len, use_cuda=True, dev_y=copy.deepcopy(y))
        test_output += d_output
        test_loss += [d_loss.item()]
        print("test step--------------->{}".format(ide))
    #
    # test_output, test_loss = model.decode(test_x)

    source_test_data, predict_test_data = \
        test_dataset.transform_data_back(
            test_dataset.uni_data(copy.deepcopy(test_text_list), copy.deepcopy(test_split_text_list), copy.deepcopy(test_x),
                             copy.deepcopy(test_output), copy.deepcopy(test_start_index), copy.deepcopy(test_sen_len)))

    p, r, f1 = f1_eva_re(transform_back_dev_data, source_dev_data)
    print("测试集结果P:{} R:{} F1:{}".format(p, r, f1))