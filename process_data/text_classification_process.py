import json
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
# from gensim.models import word2vec
import numpy as np

class text_classification_data_process_machine(Dataset):
    def __init__(self, file_path, sentence_max_len, model_structure = "bert_softmax",
                 tokenizer_path = None, label2id = None, tokenizer = None):
        '''
        :param file_path:文件路径
        :param sentence_max_len:最大句子长度
        :param tokenizer_path: 分词器的路径
        :param label2id: 标签到id的映射，当是训练集时，无需传入此参数，根据训练集生成此映射，当是验证集或测试集时传入根据训练集构建的此映射
        '''
        self.file_path = file_path
        self.sentence_max_len = sentence_max_len
        self.model_structure = model_structure

        if self.model_structure == "bert_softmax":
            self.begin, self.end = 1, 1
        # elif self.model_structure == "w2v_bilstm_crf" or self.model_structure == "bilstm_crf":
        #     self.begin, self.end = 0, 0

        if not tokenizer:
            print("加载词典")
            if self.model_structure == "bert_softmax":
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=True)
        else:
            self.tokenizer = tokenizer

        # self.data = [] #[原句, start_index, x, y, 小句长度]

        if label2id:
            self.label2id = label2id
        else:
            self.label2id = {}
            id = 0
            with open(self.file_path, encoding="utf-8") as f:
                for line in f:
                    line_json = json.loads(line)
                    label = line_json["label"]
                    if label not in self.label2id:
                        self.label2id[label] = id
                        id += 1

        self.id2label = {}
        for k, v in self.label2id.items():
            self.id2label[v] = k

        self.transform_data()

    def __getitem__(self, item):

        # if self.model_structure == "bert_bilstm_crf" or self.model_structure == "bert_crf":
        #     return self.data[item][0], np.array(self.data[item][1]).T, np.array(self.data[item][2]).T, self.data[item][
        #         3], self.data[item][4]
        # elif self.model_structure == "w2v_bilstm_crf" or self.model_structure == "bilstm_crf":
        #     return self.data[item][0], self.data[item][1], np.array(self.data[item][2]).T, self.data[item][
        #         3], self.data[item][4]
        #
        # print(self.data[item][0], np.array(self.data[item][1]).T, np.array(self.data[item][2]).T, self.data[item][3], self.data[item][4])
        #
        return self.data[item][0], np.array(self.data[item][1]).T, np.array(self.data[item][2]).T, self.data[item][3], self.data[item][4]

    def __len__(self):
        return len(self.data)

    def get_label2id(self):
        return self.label2id

    def get_id2label(self):
        return self.id2label

    def get_tokenizer(self):
        return self.tokenizer

    def get_data(self):
        return self.data

    def get_word2vec(self):
        return self.word2id

    def transform_data(self):
        '''
        :return:file - > [原句, x, y, start_index, 小句长度]
        '''

        source_data = []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line_json = json.loads(line)
                text = line_json["text"]
                label = line_json["label"]
                source_data.append([text, label])

        self.data = []
        for line_data in source_data:
            text, y = line_data[0], line_data[1]
            tokens_str = ["[CLS]"] * self.begin + list(text) + ["[SEP]"] * self.end
            tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
            tokens += [0] * (self.sentence_max_len - len(tokens))
            self.data.append([text, tokens, [self.label2id[y]], 0, len(text)])
                              # min(len(text), (i + 1) * (self.sentence_max_len - self.begin - self.end)) - i * (self.sentence_max_len - self.begin - self.end)])

            # for i in range(len(line_data[0]) // (self.sentence_max_len - self.begin - self.end) + 1):
                # if i * (self.sentence_max_len - 2) < min(len(text), (i + 1) * (self.sentence_max_len - 2)):
                #     tokens_str = ["[CLS]"] * self.begin + list(text[i * (self.sentence_max_len - 2):min(len(text), (i + 1) * (self.sentence_max_len - 2))]) + ["[SEP]"] * self.end
                #
                #     if self.model_structure == "bert_softmax":
                #         tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
                #         tokens += [0] * (self.sentence_max_len - len(tokens))
                #     # elif self.model_structure == "w2v_bilstm_crf":
                #     #     tokens = tokens_str
                #     #     tokens = tokens + ["U"] * (self.sentence_max_len - len(tokens))
                #     #     # tokens = [self.tokenizer(w) for w in tokens]
                #     #     tokens = ''.join(tokens)
                #     # elif self.model_structure == "bilstm_crf":
                #     #     tokens = [self.word2id[x] for x in tokens_str + ["U"] * (self.sentence_max_len - len(tokens))]
                #     #     tokens = ''.join(tokens)
                #     # y_ = [0] * self.begin + [self.label2id[x] for x in y[i * (self.sentence_max_len - self.begin - self.end):min(len(text), (i + 1) * (self.sentence_max_len - self.begin - self.end))]] + [0] * self.end
                #     # y_ += [0] * (self.sentence_max_len - len(y_))
                #     self.data.append([text, tokens, [self.label2id[y]], i * (self.sentence_max_len - self.begin - self.end),
                #                       min(len(text), (i + 1) * (self.sentence_max_len - self.begin - self.end)) - i * (self.sentence_max_len - self.begin - self.end)])
        print(self.file_path,"数据处理完成")

    def transform_data_back(self, data):
        '''
        :param data: [原句, x, y, start_index, 小句长度]
        :return: {“text”：。。。，“entity_list”:...}
        '''

        # [原句, x, y, start_index, 小句长度] -> [原句, x, y]
        predict_data = []

        text = data[0][0]

        y = data[0][2][0]

        for line_data in data[1:]:
            # if line_data[0] == text:
            #     pass
            # else:
            predict_data.append({"text":text, "label": self.id2label[y]})
            text = line_data[0]
            y = line_data[2][0]
        predict_data.append({"text": text, "label": self.id2label[y]})

        source_data = []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                source_data.append(json.loads(line))

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









