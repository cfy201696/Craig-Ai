import json
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
# from gensim.models import word2vec
import numpy as np
from collections import defaultdict

class re_data_process_machine(Dataset):
    def __init__(self, file_path, sentence_max_len, model_structure = "pa_lstm_crf",
                 tokenizer_path = None, label2id = None, tokenizer = None):
        '''
        :param file_path:文件路径:
        {"text": "目前候审的调味品公司还包括江苏井神盐化、安记食品、火锅底料等细分领域具
        有领头羊优势。", "spo_list": [{"object_index": {"begin": 20, "end": 24}, "subject_index": {"begin": 13, "end": 19},
         "predicate": "unknown", "object": "安记食品", "subject": "江苏井神盐化"}]}
        :param sentence_max_len:最大句子长度
        :param tokenizer_path: 分词器的路径
        :param label2id: 标签到id的映射，当是训练集时，无需传入此参数，根据训练集生成此映射，当是验证集或测试集时传入根据训练集构建的此映射
        '''
        self.file_path = file_path
        self.sentence_max_len = sentence_max_len
        self.model_structure = model_structure

        if self.model_structure == "pa_lstm_crf":
            self.begin, self.end = 1, 1
        else:
            raise NotImplementedError()

        if not tokenizer:
            print("加载词典")
            if self.model_structure == "pa_lstm_crf":
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=True)
            else :
                self.tokenizer  = "None"
        else:
            self.tokenizer = tokenizer

        if label2id:
            self.label2id = label2id
        else:
            # 制作标签体系：{BIES-re，O)}
            self.label2id = {"O":0}
            id_ = 1
            with open(self.file_path, encoding="utf-8") as f:
                for line in f:
                    line_json = json.loads(line)
                    for re in line_json["spo_list"]:
                        re_type = re["predicate"]
                        if "B-sub-" + re_type not in self.label2id:
                            self.label2id["B-sub-" + re_type] = id_
                            id_ += 1
                        if "I-sub-" + re_type not in self.label2id:
                            self.label2id["I-sub-" + re_type] = id_
                            id_ += 1
                        if "E-sub-" + re_type not in self.label2id:
                            self.label2id["E-sub-" + re_type] = id_
                            id_ += 1
                        if "S-sub-" + re_type not in self.label2id:
                            self.label2id["S-sub-" + re_type] = id_
                            id_ += 1

                        if "B-obj-" + re_type not in self.label2id:
                            self.label2id["B-obj-" + re_type] = id_
                            id_ += 1
                        if "I-obj-" + re_type not in self.label2id:
                            self.label2id["I-obj-" + re_type] = id_
                            id_ += 1
                        if "E-obj-" + re_type not in self.label2id:
                            self.label2id["E-obj-" + re_type] = id_
                            id_ += 1
                        if "S-obj-" + re_type not in self.label2id:
                            self.label2id["S-obj-" + re_type] = id_
                            id_ += 1

        self.id2label = {}
        for k, v in self.label2id.items():
            self.id2label[v] = k

        self.transform_data()

    def __getitem__(self, item):
        '''
        [原句, 小句, x, [sentence_max_len(P), sentence_max_len], start_index, 小句长度]
        '''
        # print(np.array(self.data[item][3]).shape)
        # print(self.data[item][5])
        c = np.vstack((self.data[item][3], np.zeros((self.sentence_max_len - len(self.data[item][3]), 100))))
        return self.data[item][0], self.data[item][1], np.array(self.data[item][2]).T, \
               c, self.data[item][4], self.data[item][5]

    def __len__(self):
        return len(self.data)

    def get_label2id(self):
        return self.label2id

    def get_tokenizer(self):
        return self.tokenizer

    def get_data(self):
        return self.data

    def get_word2vec(self):
        return self.word2id

    def _get_spo_list(self, spo_list, begin, end):
        '''
        {"text": "但事实上,在<N>年和<N>年,南纺股份已先后两次分别收购了南泰国展<N>%和<N>%的股权,后者已成为南纺股份的全资子公司;",
        "spo_list": [{"object_index": {"begin": 30, "end": 34}, "subject_index": {"begin": 16, "end": 20},
        "predicate": "收购", "object": "南泰国展", "subject": "南纺股份"}]}
        '''
        spo_list_ = []
        for spo in spo_list:
            if spo["object_index"]["begin"] >= begin and spo["object_index"]["end"] <= end and \
                    spo["subject_index"]["begin"] >= begin and spo["subject_index"]["end"] <= end:
                spo_list_.append({"object_index": {"begin": spo["object_index"]["begin"] - begin, "end": spo["object_index"]["end"] - begin},
                                  "subject_index":{"begin": spo["subject_index"]["begin"] - begin, "end": spo["subject_index"]["end"] - begin},
                                "predicate": spo["predicate"], "object": spo["object"], "subject": spo["subject"]})
        return spo_list_

    def _get_pa_pos(self, text, spo_list):
        '''
        text:"小句"
        spo_list:[{"object_index": {"begin": 30, "end": 34}, "subject_index": {"begin": 16, "end": 20},
        "predicate": "收购", "object": "南泰国展", "subject": "南纺股份"}]
        return tokens, [sentence_max_len(P), sentence_max_len]
        '''
        head_entity_index_dict = defaultdict(list)
        pa_pos_list = []
        for spo in spo_list:
            head_entity_index_dict[spo["subject_index"]["begin"]] += [spo]
        for idx, c in enumerate(text):
            label = [0] * (self.sentence_max_len - 1)
            if idx in head_entity_index_dict:
                for idx_, spo_ in enumerate(head_entity_index_dict[idx]):
                    if idx_ == 0: # 头实体标注一次即可
                        if spo_["subject_index"]["end"] - spo_["subject_index"]["begin"] == 1:
                            label[spo_["subject_index"]["begin"]] = self.label2id["S-sub-" + spo_["predicate"]]
                        else:
                            label[spo_["subject_index"]["begin"]] = self.label2id["B-sub-" + spo_["predicate"]]
                            label[spo_["subject_index"]["begin"] + 1:spo_["subject_index"]["end"] - 1] = \
                                [self.label2id["I-sub-" + spo_["predicate"]]] * (spo_["subject_index"]["end"] - spo_["subject_index"]["begin"] - 2)
                            label[spo_["subject_index"]["end"]-1] = self.label2id["E-sub-" + spo_["predicate"]]
                    if spo_["object_index"]["end"] - spo_["object_index"]["begin"] == 1:
                        label[spo_["object_index"]["begin"]] = self.label2id["S-obj-" + spo_["predicate"]]
                    else:
                        label[spo_["object_index"]["begin"]] = self.label2id["B-obj-" + spo_["predicate"]]
                        label[spo_["object_index"]["begin"] + 1:spo_["object_index"]["end"] - 1] = \
                            [self.label2id["I-obj-" + spo_["predicate"]]] * (
                                        spo_["object_index"]["end"] - spo_["object_index"]["begin"] - 2)
                        label[spo_["object_index"]["end"] - 1] = self.label2id["E-obj-" + spo_["predicate"]]

            pa_pos_list.append([0] + label)

        tokens_str = ["[CLS]"] + list(text) + ["[SEP]"]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
        tokens += [0] * (self.sentence_max_len - len(tokens))

        return tokens, pa_pos_list

    def transform_data(self):
        '''
        :return:file -> [原句,小句, x, [sentence_max_len(P), sentence_max_len], start_index, 小句长度]
        '''
        # [source text, split text, start index, spo_list]
        source_data = []
        # self.text_list = []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line_json = json.loads(line)
                text = line_json["text"]
                spo_list = line_json["spo_list"]
                for i in range(len(text) // (self.sentence_max_len - self.begin - self.end) + 1):
                    if i * (self.sentence_max_len - self.begin - self.end) < len(text):
                        begin, end = i * (self.sentence_max_len - 2), min(len(text), (i + 1) * (self.sentence_max_len - 2))
                        spo_list_ = self._get_spo_list(spo_list, begin, end)
                        source_data.append([text, text[begin:end], begin, spo_list_])

        # [原句, x, [sentence_max_len(P), sentence_max_len], start_index, 小句长度]
        self.data = []
        for line_data in source_data:
            text, text_, start_index, spo_list = line_data
            x, pa_pos = self._get_pa_pos(text_, spo_list)
            self.data.append([text, text_, x, pa_pos, start_index, len(text_)])
        print(self.file_path,"数据处理完成")

    def _get_re_list(self, text, label_list, before_length):
        '''
        {"text": "但事实上,在<N>年和<N>年,南纺股份已先后两次分别收购了南泰国展<N>%和<N>%的股权,后者已成为南纺股份的全资子公司;",
        "spo_list": [{"object_index": {"begin": 30, "end": 34}, "subject_index": {"begin": 16, "end": 20},
        "predicate": "收购", "object": "南泰国展", "subject": "南纺股份"}]}
        '''
        # print(self.id2label)
        spo_list = []
        for label in label_list:
            obj_list = []
            label = [self.id2label[i] for i in label[1:]]
            obj_type, obj_begin = None, None
            sub_type, sub_begin, sub_end = None, None, None
            # print("label:", label)
            for idx, l in enumerate(label):

                if l.startswith("B-obj"):
                    obj_begin = idx
                    obj_type = l[6:]
                if l.startswith("I-obj"):
                    if obj_begin == None or l[6:] != obj_type:
                        obj_begin, obj_type = None, None
                if l.startswith("E-obj"):
                    if obj_begin != None and obj_type == l[6:]:
                        obj_list.append((l[6:], obj_begin, idx + 1))
                    obj_type, obj_begin = None, None
                if l.startswith("O"):
                    obj_type, obj_begin = None, None
                if l.startswith("S-obj"):
                    obj_list.append((l[6:], idx, idx + 1))
                    obj_type, obj_begin = None, None

                if l.startswith("S-sub"):
                    sub_type = l[6:]
                    sub_begin = idx
                    sub_end = idx + 1
                if l.startswith("O"):
                    if sub_begin == None or sub_end == None:
                        sub_type, sub_begin, sub_end = None, None, None
                if l.startswith("B-sub"):
                    if sub_begin == None or sub_end == None:
                        sub_type, sub_begin = l[6:], idx
                if l.startswith("I-sub"):
                    if sub_end == None and sub_begin and l[6:] != sub_type:
                        sub_type, sub_begin, sub_end = None, None, None
                if l.startswith("E-sub"):
                    if sub_begin != None:
                        if sub_end:
                            pass
                        else:
                            if sub_type != l[6:]:
                                sub_type, sub_begin, sub_end = None, None, None
                            else:
                                sub_end = idx + 1

            # print("obj_list:", obj_list)

            if sub_begin != None and sub_end != None:
                for t, o_b, o_e in obj_list:
                    spo_list.append({"object_index": {"begin": o_b + before_length, "end": o_e + before_length},
                                     "subject_index": {"begin": sub_begin + before_length, "end": sub_end + before_length},
                "predicate": t, "object": text[o_b:o_e], "subject": text[sub_begin:sub_end]})

            # if "波音中国方面透露,<N>月<N>日,波音公司旗" in text:
            #     print("label:",label)
            #     print("spo_list:",spo_list)
            #     print("obj_list:",obj_list)
            #     print("sub:", [sub_begin, sub_end, sub_type])
        return spo_list

    def transform_data_back(self, data):
        '''
        :param data:[原句, 小句, x, [sentence_max_len(P), sentence_max_len], start_index, 小句长度]
        :return: {“text”：。。。，“spo_list”:...}
        '''

        # [原句, 小句, x, [sentence_max_len(P), sentence_max_len], start_index, 小句长度] ->
        # [原句, 小句, x, spo_list, start_index, 小句长度]
        predict_data = []
        for idx, row in enumerate(data):
            # print(row)
            if idx == 0:
                tem_text = row[0]
                before_length = 0
                spo_list = self._get_re_list(row[1], row[3], before_length)
            else:
                if row[0] == tem_text:
                    before_length += len(row[1])
                    spo_list += self._get_re_list(row[1], row[3], before_length)
                else:
                    predict_data.append({"text":tem_text, "spo_list":spo_list})
                    # print({"text":row[0], "spo_list":spo_list})

                    tem_text = row[0]
                    before_length = 0
                    spo_list = self._get_re_list(row[1], row[3], before_length)

        predict_data.append({"text": tem_text, "spo_list": spo_list})

        source_data = []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                source_data.append(json.loads(line))

        return source_data, predict_data

    def extract_data(self, data):
        batch_data_array = np.array(data)
        text_list = batch_data_array[:, 0].tolist()
        split_text = batch_data_array[:, 1].tolist()
        x = batch_data_array[:, 2].tolist()
        y = batch_data_array[:, 3].tolist()
        start_index = batch_data_array[:, 4].tolist()
        sen_len = batch_data_array[:, 5].tolist()
        return text_list, split_text, x, y, start_index, sen_len

    def uni_data(self, dev_text_list, split_text, dev_x, dev_y, dev_start_index, dev_sen_len):
        data = []
        for d_t, d_st, d_x, d_y, d_s_i, d_s_l in zip(dev_text_list, split_text, dev_x, dev_y, dev_start_index, dev_sen_len):
            data.append([d_t, d_st, d_x, d_y, d_s_i, d_s_l])
        return data