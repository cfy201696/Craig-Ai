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
from process_data.text_classification_process import text_classification_data_process_machine
from model.bert_softmax import bert_softmax
from torch.utils import data
from tensorboardX import SummaryWriter
from evaluation.evaluation import evaluation

parser = argparse.ArgumentParser()
# 数据集参数
parser.add_argument("-tr","--train_file_path", help="训练文件路径，如果指定此参数则代表训练模型，并需要指定验证集", default=None)
parser.add_argument("-de","--dev_file_path", help="验证集路径", default=None)
parser.add_argument("-te","--test_file_path", help="测试文件路径，如果指定此参数，需要指定模型路径参数", default=None)
parser.add_argument("-mfp","--model_file_path", default="data/")
parser.add_argument("-mn","--model_name", default="v1")


# 模型结构参数
parser.add_argument("-ms","--model_structure", help="模型结构", default="bert_softmax")
parser.add_argument("-mp","--model_path", default="../pretrained_model_file/bert/chinese_L-12_H-768_A-12")
parser.add_argument("-ft","--fine_tuning", help="微调", default=False)
parser.add_argument("-pos","--pretrain_output_size", help="预训练模型的输出维度", default=768)


# lstm的参数
parser.add_argument("-lhs","--lstm_hidden_size", help="lstm的隐藏单元数", default=384)
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

    # 制作训练集
    train_dataset = text_classification_data_process_machine(file_path = model_config["train_file_path"],
                                             model_structure = model_config["model_structure"],
                                             sentence_max_len = model_config["sentence_max_length"],
                                             tokenizer_path = model_config["model_path"])
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=model_config["batch_size"], shuffle=True, num_workers=6)

    # 保存模型配置
    model_config["label2id"] = train_dataset.get_label2id()
    with open(model_config["model_file_path"] + model_config["model_name"] + "_model_config.json", "w+", encoding="utf-8") as f:
        f.write(json.dumps(model_config, ensure_ascii=False))

    # 制作验证集
    dev_dataset = text_classification_data_process_machine(
        file_path = model_config["dev_file_path"],
        sentence_max_len = model_config["sentence_max_length"],
        model_structure=model_config["model_structure"],
        label2id = model_config["label2id"],
        tokenizer = train_dataset.get_tokenizer())
    dev_dataloader = data.DataLoader(dev_dataset,
                                       batch_size=model_config["batch_size"], shuffle=False, num_workers=6)

    # 得到验证集的loss和F1，p，r
    dev_text_list, dev_x, dev_y = \
        dev_dataset.extract_data(copy.deepcopy(dev_dataset.get_data()))
    source_dev_data, transform_back_dev_data = dev_dataset.transform_data_back(
            dev_dataset.uni_data(copy.deepcopy(dev_text_list), copy.deepcopy(dev_x),
                                 copy.deepcopy(dev_y)))

    evaluation_result_dev = evaluation([line["label"] for line in source_dev_data],
                                       [line["label"] for line in transform_back_dev_data],
                                       dev_dataset.get_id2label())
    print("验证集上限P:{} R:{} F1:{} micro_f1:{}".format(evaluation_result_dev[0],
                                         evaluation_result_dev[1], evaluation_result_dev[2], evaluation_result_dev[3]))

    print("model_config:",model_config)

    # 加载模型
    if model_config["model_structure"] == "bert_softmax":
        model = bert_softmax(
            pretrain_model_path = model_config["model_path"],
            lable_num = len(model_config["label2id"]),
            device = device)

    model.to(device)

    # 设置模型参数
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

    # 打印模型参数
    model.model_structure()

    # 设置优化器
    num_total_steps = model_config["epochs"] * len(train_dataset) / model_config["batch_size"]
    num_warmup_steps = int(0.1 * num_total_steps)
    if model_config["fine_tuning"]:
        warm_up_optimizer = AdamW(warm_up_params, lr=model_config["learning_rate"],
                                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = WarmupLinearSchedule(warm_up_optimizer, warmup_steps=num_warmup_steps,
                                         t_total=num_total_steps)  # PyTorch scheduler
    non_warm_up_optimizer = optim.Adam(non_warm_up_params, lr=model_config["learning_rate"] * 10)

    best_f1 = 0
    # 训练
    for i in range(model_config["epochs"]):
        loss_list = []
        model.train()
        for idx, batch_data in enumerate(train_dataloader):
            batch_data_array = np.array(batch_data)
            _, x, y, __, sen_len = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4]
            output = model(x)
            loss = model.get_loss(output, y = y)
            loss.backward()
            if model_config["fine_tuning"]:
                warm_up_optimizer.step()
                scheduler.step()
                warm_up_optimizer.zero_grad()
            non_warm_up_optimizer.step()
            non_warm_up_optimizer.zero_grad()
            loss_list.append(loss.item())
            print("train epoch: {}, step--------------->{}".format(i, idx))


        ## 验证模型
        model.eval()
        dev_output = []
        dev_loss = []
        for ide, batch_data_dev in enumerate(dev_dataloader):
            d_output, d_loss = model.decode(copy.deepcopy(batch_data_dev[1]), dev_y = copy.deepcopy(batch_data_dev[2]))
            dev_output += d_output
            dev_loss += [d_loss.item()]
            print("the epoch {}, evaluation step--------------->{}".format(i, ide))

        # 计算推理验证指标
        source_dev_data, predict_dev_data = \
            dev_dataset.transform_data_back(
                dev_dataset.uni_data(dev_text_list, dev_x, dev_output, dev_start_index, dev_sen_len))

        evaluation_result = evaluation([line["label"] for line in source_dev_data],
                                       [line["label"] for line in predict_dev_data], dev_dataset.get_label2id())

        # 制作tensorboardx图
        summary_writer.add_scalars("loss", {"train_loss": np.mean(loss_list), "dev_loss": np.mean(dev_loss)}, i)
        summary_writer.add_scalars("dev_evaluation_total",
                                   {"p": evaluation_result[0], "r": evaluation_result[1], "f1": evaluation_result[2]}, i)
        ans_dict = evaluation_result[-1]
        if len(ans_dict) < 15:
            for key, v in ans_dict.items():
                summary_writer.add_scalars(key, v, i)

        # 结尾工作，保存模型和控制台打印
        if evaluation_result[2] > best_f1:
            best_f1 = evaluation_result[2]
            model.save_model(model_config["model_file_path"]+model_config["model_name"])
            print("label lavel:", ans_dict)
        print("训练完成：{}， 训练集loss：{}，验证集loss：{}，验证集P：{}，验证集R：{}，验证集F1：{}，最高F1：{}".format(
            i, np.mean(loss_list), np.mean(dev_loss), evaluation_result[0], evaluation_result[1], evaluation_result[2], best_f1))
        print("-"*100)

if model_config["test_file_path"]:

    with open(model_config["model_file_path"] + model_config["model_name"] + "_model_config.json", encoding="utf-8") as f:
        config = json.load(f)

    config["test_file_path"] = model_config["test_file_path"]
    print(config)

    if model_config["model_structure"] == "bert_softmax":
        model = bert_softmax(
            pretrain_model_path=config["model_path"],
            lable_num=len(config["label2id"]),
            device=device)

    model.load_state_dict(torch.load(config["model_file_path"]+config["model_name"]))
    model.to(device)
    model.eval()

    test_dataset = text_classification_data_process_machine(
        file_path=config["test_file_path"],
        sentence_max_len=config["sentence_max_length"],
        model_structure=config["model_structure"],
        label2id=config["label2id"],
        tokenizer_path=config["model_path"])
    test_dataloader = data.DataLoader(test_dataset,
                                     batch_size=config["batch_size"], shuffle=False, num_workers=6)
    # 得到测试集的loss和F1，p，r
    test_text_list, test_x, test_y = test_dataset.extract_data(copy.deepcopy(test_dataset.get_data()))
    source_test_data, transform_back_test_data = test_dataset.transform_data_back(
        test_dataset.uni_data(copy.deepcopy(test_text_list), copy.deepcopy(test_x), copy.deepcopy(test_y)))

    source_label = [line["label"] for line in source_test_data]
    pre_label = [line["label"] for line in transform_back_test_data]
    evaluation_result_dev = evaluation(source_label, pre_label, config["label2id"])
    print("测试集上限P:{} R:{} F1:{} micro_f1:{}".format(evaluation_result_dev[0],
                                                    evaluation_result_dev[1], evaluation_result_dev[2],
                                                    evaluation_result_dev[3]))

    test_output = []
    test_loss = []
    for ide, batch_data_dev in enumerate(test_dataloader):
        d_output, d_loss = model.decode(copy.deepcopy(batch_data_dev[1]), dev_y=copy.deepcopy(batch_data_dev[2]))
        test_output += d_output
        test_loss += [d_loss.item()]
        print("test step--------------->{}".format(ide))
    #
    # test_output, test_loss = model.decode(test_x)

    source_test_data, predict_test_data = \
        test_dataset.transform_data_back(
            test_dataset.uni_data(test_text_list, test_x, test_output))

    source_label = [line["label"] for line in source_test_data]
    pre_label = [line["label"] for line in predict_test_data]
    evaluation_result_dev = evaluation(source_label, pre_label, config["label2id"])
    print("测试集上限P:{} R:{} F1:{} micro_f1:{}".format(evaluation_result_dev[0],
                                                    evaluation_result_dev[1], evaluation_result_dev[2],
                                                    evaluation_result_dev[3]))