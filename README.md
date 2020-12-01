# 说明
本项目实现各NLP任务，不断完善
# 目录
```
├── data 存放项目数据的目录
│   ├── ResumeNER 简历实体识别数据
│   │   ├── dev_1000_filter.txt 1000验证集
│   │   ├── train_20_filter.txt 20训练集
│   └── yidu-s4k 依渡云实体识别数据
│       ├── dev_100.txt 100验证集 
│       ├── train_10.txt 10训练集
├── evaluation 评价函数
│   ├── f1_evaluation.py 实体识别评价文件
├── layers 模型的层
│   └── utensil.py mask函数
├── LICENSE 证书
├── model 各个模型
│   ├── bert_bilstm_crf.py Bert_bilstm_crf模型
├── pretrained_model_file 预训练模型路径
│   └── bert Bert的预训模型
├── process_data 数据的预处理函数
│   ├── ner_process.py 实体识别数据预处理
│   └── sta_error_type.py 实体识别测试的错误类型统计函数
├── README.md 
├── requirements.txt
└── task 各个任务的启动代码
    └── ner.py 实体识别启动代码
```
# 环境配置
- python3.7
- ```pip install -r requirements.txt```

# 实体识别
pytorch版的Bert-bilstm-crf模型
## 环境配置
- 下载Bert预训练模型解压到pretrained_model_file/bert下，预训练模型下载:链接：https://pan.baidu.com/s/1KauLJeiJUErWu4YdYEuKiA ,提取码：18z2 
- 数据集放到data目录下
## 数据格式（每个样本占一行，每行格式如下）
```{"text": "现任长春大学管理学院教授、长春高新技术产业(集团)股份有限公司董事会外部董事。", "entity_list": [{"entity_index": {"begin": 10, "end": 12}, "entity_type": "TITLE", "entity": "教授"}, {"entity_index": {"begin": 31, "end": 38}, "entity_type": "TITLE", "entity": "董事会外部董事"}]}```
## 训练（task目录下）
```python -u ner.py -tr ../data/yidu-s4k/train_10.txt -de ../data/yidu-s4k/dev_100.txt -mfp ../data/yidu-s4k/v5 -mn v5 -lhs 200 -bs 5 -lr 1e-5 -sml 512 -e 2000```
- -tr 为训练集的路径
- -de 为验证集的路径
- -mfp 为训练的模型和模型配置保存路径
- -mn 保存的模型名称
- -lhs 为Bilstm的输出单元//2（前向和后向concat）
- -bs batch size
- -lr 学习率
- -sml 最大句子长度
- -e 迭代训练次数
## 测试（task目录下）
```python -u ner.py -te ../data/yidu-s4k/dev_100.txt -mfp ../data/yidu-s4k/v5 -mn v5 -lhs 200```
- -te 测试集路径
- -mfp 需要加载的模型和模型配置路径
- -mn 加载模型名称
- -lhs 为Bilstm的输出单元//2（前向和后向concat）
