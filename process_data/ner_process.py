import json
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np

class ner_data_process_machine(Dataset):
    def __init__(self, file_path, sentence_max_len, tokenizer_path = None, label2id = None, tokenizer = None):
        '''
        :param file_path:文件路径
        :param sentence_max_len:最大句子长度
        :param tokenizer_path: 分词器的路径
        :param label2id: 标签到id的映射，当是训练集时，无需传入此参数，根据训练集生成此映射，当是验证集或测试集时传入根据训练集构建的此映射
        '''
        self.file_path = file_path
        self.sentence_max_len = sentence_max_len
        if not tokenizer:
            print("加载词典")
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=True)
        else:
            self.tokenizer = tokenizer

        # self.data = [] #[原句, start_index, x, y, 小句长度]

        if label2id:
            self.label2id = label2id
        else:
            self.label2id = {"O":0}
            id = 1
            with open(self.file_path, encoding="utf-8") as f:
                for line in f:
                    for entity in json.loads(line)["entity_list"]:
                        entity_type = entity["entity_type"]
                        if "B-" + entity_type not in self.label2id:
                            self.label2id["B-" + entity_type] = id
                            id += 1
                        if "I-" + entity_type not in self.label2id:
                            self.label2id["I-" + entity_type] = id
                            id += 1

        self.id2label = {}
        for k, v in self.label2id.items():
            self.id2label[v] = k

        self.transform_data()

    def __getitem__(self, item):
        return self.data[item][0], np.array(self.data[item][1]).T, np.array(self.data[item][2]).T, self.data[item][3], self.data[item][4]

    def __len__(self):
        return len(self.data)

    def get_label2id(self):
        return self.label2id

    def get_tokenizer(self):
        return self.tokenizer

    def get_data(self):
        return self.data

    def transform_data(self):
        '''
        :return:file - > [原句, x, y, start_index, 小句长度]
        '''

        source_data = []
        # self.text_list = []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line_json = json.loads(line)
                text = line_json["text"]
                label = ["O"] * len(text)
                for entity in line_json["entity_list"]:
                    entity_type = entity["entity_type"]
                    label[entity["entity_index"]["begin"]] = "B-" + entity_type
                    for i in range(entity["entity_index"]["begin"] + 1, entity["entity_index"]["end"]):
                        label[i] = "I-" + entity_type
                source_data.append([text, label])
                # self.text_list.append(text)

        self.data = []
        for line_data in source_data:
            text, y = line_data[0], line_data[1]
            for i in range(len(line_data[0]) // (self.sentence_max_len - 2) + 1):
                if i * (self.sentence_max_len - 2) < min(len(text), (i + 1) * (self.sentence_max_len - 2)):
                    tokens_str = ["[CLS]"] + list(text[i * (self.sentence_max_len - 2):min(len(text), (i + 1) * (self.sentence_max_len - 2))]) + ["[SEP]"]
                    tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
                    y_ = [0] + [self.label2id[x] for x in y[i * (self.sentence_max_len - 2):min(len(text), (i + 1) * (self.sentence_max_len - 2))]] + [0]
                    tokens += [0] * (self.sentence_max_len - len(tokens))
                    y_ += [0] * (self.sentence_max_len - len(y_))
                    self.data.append([text, tokens, y_, i * (self.sentence_max_len - 2),
                                      min(len(text), (i + 1) * (self.sentence_max_len - 2)) - i * (self.sentence_max_len - 2)])

    def get_entity(self, text, y):
        entity_list = []
        start = -1
        for idx, id in enumerate(y):
            if self.id2label[id].startswith("B"):
                if start != -1:
                    entity_list.append({"entity":text[start:idx], "entity_type":self.id2label[y[start]][2:],
                                        "entity_index":{"begin":start, "end":idx}})
                start = idx
            elif self.id2label[id].startswith("O"):
                if start != -1:
                    entity_list.append({"entity":text[start:idx], "entity_type":self.id2label[y[start]][2:],
                                        "entity_index":{"begin":start, "end":idx}})
                start = -1
        if self.id2label[y[-1]].startswith("I") and start != -1:
            entity_list.append({"entity": text[start:], "entity_type": self.id2label[y[start]][2:],
                                "entity_index": {"begin": start, "end": len(y)}})

        return entity_list

    def transform_data_back(self, data):
        '''
        :param data: [原句, x, y, start_index, 小句长度]
        :return: {“text”：。。。，“entity_list”:...}
        '''
        # print("data:",data)
        # sort_by_start_index = sorted(data, key=lambda x: x[3])
        # sort_by_text = sorted(sort_by_start_index, key=lambda x:x[0])

        # [原句, x, y, start_index, 小句长度] -> [原句, x, y]
        predict_data = []
        # if sort_by_text:
        text = data[0][0]
        # x = data[0][1]
        y = data[0][2][1:data[0][4] + 1]
        # len_sen = data[0][4]
        for line_data in data[1:]:
            if line_data[0] == text:
                y += line_data[2][1:line_data[4] + 1]
            else:
                predict_data.append({"text":text, "entity_list": self.get_entity(text, y)})
                text = line_data[0]
                y = line_data[2][1:line_data[4] + 1]

        source_data = []
        with open(self.file_path) as f:
            for line in f:
                source_data.append(json.loads(line))

        # print(predict_data)
        return source_data, predict_data

    def extract_data(self, data):
        batch_data_array = np.array(data)
        text_list = batch_data_array[:, 0].tolist()
        x = batch_data_array[:, 1].tolist()
        y = batch_data_array[:, 2].tolist()
        start_index = batch_data_array[:, 3].tolist()
        sen_len = batch_data_array[:, 4].tolist()
        return text_list, x, y, start_index, sen_len

    def uni_data(self,dev_text_list, dev_x, dev_y, dev_start_index, dev_sen_len):
        data = []
        for d_t, d_x, d_y, d_s_i, d_s_l in zip(dev_text_list, dev_x, dev_y, dev_start_index, dev_sen_len):
            data.append([d_t, d_x, d_y, d_s_i, d_s_l])
        return data









