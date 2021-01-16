import json
import csv
import random
import os
from collections import defaultdict
import numpy as np

def csv_to_json(source_file, target_file):
    '''
    序号	品类	描述	SKU
    '''
    with open(source_file, encoding="utf-8") as f, open(target_file, "w+", encoding="utf-8") as g:
        reader = csv.reader(f)
        text_length = []
        for row in reader:
            text = row[1]
            entity_list = []
            for entity in row[4].strip().split(";"):
                begin, end = entity.split()
                begin = int(begin.split(":")[1])
                end = int(end.split(":")[1])
                entity_list.append({"entity_index":{"begin":begin, "end": end}, "entity_type": "本体", "entity": row[3]})

            begin, end = row[6].strip().split()
            begin = int(begin.split(":")[1])
            end = int(end.split(":")[1])
            entity_list.append({"entity_index": {"begin": begin, "end": end}, "entity_type": "品牌", "entity": row[5]})

            if row[7].strip():
                begin, end = row[8].strip().split()
                begin = int(begin.split(":")[1])
                end = int(end.split(":")[1])
                entity_list.append(
                    {"entity_index": {"begin": begin, "end": end}, "entity_type": "规格型号", "entity": row[7]})
            g.write(json.dumps({"text":text, "entity_list":entity_list}, ensure_ascii=False) + "\n")
            text_length.append(len(text))
        print("最长句子长度：", max(text_length))

# csv_to_json("JD/total.csv", "JD/total.json")
# csv_to_json("JD/test_5.txt", "JD/test_5.json")

def text_classification(source_file, target_file):
    '''
    item_third_cate_name	item_id	item_name	barndname_en	barndname_cn    item_first_cate_name	item_second_cate_name	item_third_cate_name	item_type	size	item_num	rn
    '''
    length = []
    with open(source_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        label_dict = defaultdict(list)
        for idx, row in enumerate(reader):
            if idx != 0 and row[2].strip() and row[0].strip():
                if np.random.randint(2) == 0:
                    label_dict[row[0].strip()].append({"text":row[2].strip(), "label":row[0].strip()}) # 不拼接品牌和规格类别...
                else:
                    label_dict[row[0].strip()].append(
                        {"text": row[2].strip() + row[3].strip() + row[7].strip() + row[8].strip() + row[9].strip() + row[10].strip(), "label": row[0].strip()})  # 拼接品牌和规格类别...
                length.append(len(row[2].strip()))
        min_len = float("inf")
        for item in label_dict:
            if len(label_dict[item]) < min_len:
                min_len = len(label_dict[item])
            random.shuffle(label_dict[item])

        train_data = []
        dev_data = []
        for label, d in label_dict.items():
            # if len(d) >= 50:
            train_data += d[:round(len(d)*0.8)]
            dev_data += d[round(len(d)*0.8):]

    random.shuffle(train_data)
    random.shuffle(dev_data)
    with open(os.path.join(target_file, "train.txt"), "w+", encoding="utf-8") as g:
        for line in train_data:
            g.write(json.dumps(line, ensure_ascii=False)+"\n")

    with open(os.path.join(target_file, "dev.txt"), "w+", encoding="utf-8") as g:
        for line in dev_data:
            g.write(json.dumps(line, ensure_ascii=False)+"\n")
    print("最大句子长度：", max(length))
    print("最小类别样例数：", min_len)
    print("训练集：", len(train_data))
    print("验证集：", len(dev_data))

text_classification("./text_classification/JD/5000数据excel数据_total.csv", "./text_classification/JD/")
def text_classification_v2(source_file, target_file):
    '''
    "序号 （必填）"	一级类目	二级类目	三级类目	物资名称（必填）	要求品牌	型号/规格/其他参数
    '''
    text_list = []
    with open(source_file, encoding="utf-8") as f, open(target_file, "w+", encoding="utf-8") as g:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            print(row)
            if idx != 0:
                text = ' '.join([x for x in row[4:] if x.strip()])
                if text not in text_list and text:
                    text_list.append(text)
                    g.write(json.dumps({"text":text, "label":row[3].strip()}, ensure_ascii=False)+"\n")
# text_classification_v2("./text_classification/JD/推理文件.csv", "./text_classification/JD/test.txt")