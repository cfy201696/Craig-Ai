# 1 说明
本项目基于pytorch深度学习框架实现各NLP任务，不断完善中.
# 2 目录
```.
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
├── model 各个模型,如bert_bilstm_crf
├── pretrained_model_file 预训练模型路径
│   └── bert Bert的预训模型
├── process_data 数据的预处理函数
├── README.md 
├── requirements.txt
└── task 各个任务的启动文件
```
# 3 环境配置
- Linux(建议在Linux下使用,Windows 10目前已经兼容)
- python3.7
- 使用Bert模型：下载Bert预训练模型解压到pretrained_model_file/bert下(可新建此目录)，预训练模型下载:链接：https://pan.baidu.com/s/1KauLJeiJUErWu4YdYEuKiA ,提取码：18z2 
- 使用word2vec模型：下载模型https://disk.pku.edu.cn:443/link/DB6CB39A363911F14D28B949604D16C5 有效期限：2021-01-04 23:59 ，放到到pretrained_model_file/word2vec下(可新建此目录)
- 数据集放到data目录下
- ```pip install -r requirements.txt```

# 4 实体识别
通过task文件夹下ner.py文件进行实体识别的训练和推理,目前已经实现bert bilstm crf、bert crf和word2vec_bilstm_crf模型.
## 4.1 数据格式（每个样本占一行，每行格式如下）
```{"text": "现任长春大学管理学院教授、长春高新技术产业(集团)股份有限公司董事会外部董事。", "entity_list": [{"entity_index": {"begin": 10, "end": 12}, "entity_type": "TITLE", "entity": "教授"}, {"entity_index": {"begin": 31, "end": 38}, "entity_type": "TITLE", "entity": "董事会外部董事"}]}```
## 4.2 参数和使用
- -tr 训练集路径
- -de 验证集路径
- -te 测试集路径
- -mfp 模型保存路径
- -mn 模型名称
- -ms 模型结构,默认为bert_bilstm_crf,可选bert_crf、word2vec
- -mp 预训练模型路径
- -ft 是否微调预训练模型,默认不进行微调
- -pos 预训练模型的输出维度
- -lhs 为Bilstm的输出单元//2（前向和后向concat）
- -nl lstm的层数,默认为1
- -bid lstm是否双向
- -dr dropout_ratio
- -bs batch size
- -lr 学习率
- -sml 最大句子长度
- -e 

**使用说明：** yes代表需要，no代表不需要，optional代表可选

|模型参数|tr|de|te|mfp|mn|ms|mp|ft|pos|lhs|nl|bid|dr|bs|lr|sml|e|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|bert_bilstm_crf(训练)|yes|yes|optional|yes|yes|optional|optional|optional|no|optional|optional|optional|optional|optional|optional|optional|optional|
|bert_crf(训练)|yes|yes|optional|yes|yes|yes|optional|optional|pos|no|no|no|no|optional|optional|optional|optional|
|w2v_bilstm_crf|yes|yes|optional|yes|yes|yes|yes|no|yes|no|no|no|no|optional|optional|optional|optional|
|预测|no|no|yes|yes|yes|no|no|no|no|no|no|no|no|no|no|no|no|


```
# 训练（task目录下，使用Bert_Bilsmt_crf模型）
python -u ner.py -tr ../data/yidu-s4k/train_10.txt -de ../data/yidu-s4k/dev_100.txt -mfp ../data/yidu-s4k/ -mn v1 -lhs 200 -bs 5 -lr 1e-5 -sml 512 -e 2000
# 训练（task目录下，使用w2v_Bilsmt_crf模型）
python -u ner.py -tr ../data/yidu-s4k/train_10.txt -de ../data/yidu-s4k/dev_100.txt -mfp ../data/yidu-s4k/ -mn v1 -ms w2v_bilstm_crf -mp ../pretrained_model_file/word2vec/baike_26g_news_13g_novel_229g_chinese.wordvectors -pos 128 -lhs 200 -bs 5 -lr 1e-5 -sml 512 -e 2000# 测试（task目录下）

# 测试
python -u ner.py -te ../data/yidu-s4k/dev_100.txt -mfp ../data/yidu-s4k/ -mn v1
```
