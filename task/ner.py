# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
import argparse

from process_dataner_process import ner_data_process_machine
from model.bert_bilstm_crf import bert_bilstm_crf

parser = argparse.ArgumentParser()
# 数据集参数
parser.add_argument("-tr","--train_file_path", help="训练文件路径，如果指定此参数则代表训练模型，并需要指定验证集", default=None)
parser.add_argument("-de","--dev_file_path", help="验证集路径", default=None)
parser.add_argument("-te","--test_file_path", help="测试文件路径，如果指定此参数，需要指定模型路径参数", default=None)
parser.add_argument("-mfp","--model_file_path", default="data/model_config.json")

# 模型结构参数
parser.add_argument("-ms","--model_structure", help="模型结构", default="bert_bilstm_crf")
parser.add_argument("-bmp","--bert_model_path", default="chinese_L-12_H-768_A-12")

# lstm的参数
parser.add_argument("-lhs","--lstm_hidden_size", help="lstm的隐藏单元数", default=384)
parser.add_argument("-nl","--num_layers", help="lstm的层数", default=1)
parser.add_argument("-bid","--bidirectional", help="lstm是否双向", default=True)

# 数据构造参数
parser.add_argument("-bs","--batch_size", help="batch_size", default=512)
parser.add_argument("-lr","--learning_rate", help="learning_rate", default=1e-5)
parser.add_argument("-sml","--sentence_max_length", help="sentence_max_length", default=300)

# 训练参数
parser.add_argument("-e","--epochs", help="epochs", default=20)

def str_to_bool(string):
    return True if str(string).lower() == 'true' else False

args = parser.parse_args()
train_file_path = args.train_file_path
dev_file_path = args.dev_file_path
test_file_path = args.test_file_path
model_file_path = args.model_file_path

model_structure = args.model_structure
bert_model_path = args.bert_model_path

lstm_hidden_size = args.lstm_hidden_size
num_layers = args.num_layers
bidirectional = str_to_bool(args.bidirectional)

batch_size = int(args.batch_size)
learning_rate = float(args.learning_rate)
sentence_max_length = int(args.sentence_max_length)

epochs = int(args.epochs)

