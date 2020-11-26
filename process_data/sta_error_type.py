import json
import traceback
#统计ner的错误类型
def sta_error_type(label_data, pre_data, save_path = None, title = None):
    '''
    :param label_data:
    :param pre_data:
    :return:
    错误类型1：边界对但是类型错误
    错误类型2：边界错但类型对
    错误类型3：边界错类型也错
    错误类型4：预测少了（是实体没有预测成实体）
    错误类型5：预测多了（不是实体预测成实体）
    '''

    correct_border_error_type = 0
    error_border_correct_type = 0
    error_border_error_type = 0
    less = 0
    more = 0
    entity_sum = 0 # 统计label总的实体数

    # 保存五种错误类型的数据
    correct_border_error_type_file = []
    error_border_correct_type_file = []
    error_border_error_type_file = []
    less_file = []
    more_file = []
    entity_sum = 0  # 统计label总的实体数



    def get_text(item):
        return item["text"]


    # label_data_list_ = []
    # with open(label_data) as f:
    #     for line in f:
    #         label_data_list_.append(json.loads(line))
    label_data_list_ = label_data

    # label_data_list.sort(key=get_text)

    # with open(pre_data) as g:
    #     pre_data_list = json.load(g)
    pre_data_list = pre_data

    label_data_list = []
    for item in pre_data_list:
        for _ in label_data_list_:
            if item["text"] == _["text"]:
                label_data_list.append(_)
                break
    # pre_data_list.sort(key=get_text)
    #
    # print(len(label_data_list))
    # print(len(pre_data_list))
    print("len(label_data_list):",len(label_data_list))
    print("len(pre_data_list):",len(pre_data_list))
    assert len(label_data_list) == len(pre_data_list)

    for label_single, pre_single in zip(label_data_list, pre_data_list):

        c_b_e_t_f = []
        e_b_c_t_f = []
        e_b_e_t_f = []
        l_f = []
        m_f = []

        try:

            # 标注结果反标注
            label_sen_tag = ["O"] * len(label_single["text"])
            for entity in label_single["entity_list"]:
                e_index = entity["entity_index"]
                for i in range(e_index["begin"], e_index["end"]):
                    label_sen_tag[i] = str(e_index["begin"]) + "-" + entity["entity_type"] + "-" + str(e_index["end"])

            # 统计实体数
            s_e = 0
            while s_e < len(label_sen_tag):
                if label_sen_tag[s_e] != "O":
                    ss, tt, ee = label_sen_tag[s_e].split("-")
                    s_e = int(ee)
                    entity_sum += 1
                else:
                    s_e += 1

            # 预测结果反标注
            pre_sen_tag = ["O"] * len(pre_single["text"])
            for entity in pre_single["entity_list"]:
                e_index = entity["entity_index"]
                for i in range(e_index["begin"], e_index["end"]):
                    pre_sen_tag[i] = str(e_index["begin"]) + "-" + entity["entity_type"] + "-" + str(e_index["end"])
            # print(len(label_sen_tag))
            # print(len(pre_sen_tag))
            # print(label_single["text"])
            # print(pre_single["text"])
            # print(len(label_single["text"]))
            # print(len(pre_single["text"]))

            assert len(label_sen_tag) == len(pre_sen_tag)

            start = 0
            while start < len(label_sen_tag):
                if label_sen_tag[start] == "O" and pre_sen_tag[start] == "O":
                    start += 1
                elif label_sen_tag[start] == "O" and pre_sen_tag[start].strip() != "O":
                    s, t, e = pre_sen_tag[start].split("-")
                    s, e = int(s), int(e)
                    s_reserve = s
                    while s < e:
                        if label_sen_tag[s] != "O":
                            break
                        s += 1
                    if s != e:
                        #边界错误
                        s_, t_, e_ = label_sen_tag[s].split("-")
                        s_, e_ = int(s_), int(e_)
                        if t_ != t:
                            # 边界错误和类型错误
                            error_border_error_type += 1
                            e_b_e_t_f.append({"label_entity": pre_single["text"][s_:e_], "label_entity_type":t_,
                                              "pre_entity": pre_single["text"][s_reserve: e], "pre_entity_type":t})
                        else:
                            # 边界错误类型对
                            error_border_correct_type += 1
                            e_b_c_t_f.append({"label_entity": pre_single["text"][s_:e_], "label_entity_type":t_,
                                              "pre_entity": pre_single["text"][s_reserve: e], "pre_entity_type":t})
                        start = max(e, int(e_))
                    else:
                        #不是实体预测成了实体（预测多了）
                        more += 1
                        start = e
                        m_f.append(
                            {"pre_entity": pre_single["text"][s_reserve: e], "pre_entity_type":t})

                elif label_sen_tag[start] != "O" and pre_sen_tag[start] == "O":
                    s, t, e = label_sen_tag[start].split("-")
                    s, e = int(s), int(e)
                    s_reserve = s
                    while s < e:
                        if pre_sen_tag[s] != "O":
                            break
                        s += 1
                    if s!= e:
                        #边界错误
                        s_ ,t_, e_ = pre_sen_tag[s].split("-")
                        s_, e_ = int(s_), int(e_)
                        if t_ != t:
                            # 边界错误类型错误
                            error_border_error_type += 1
                            e_b_e_t_f.append({"label_entity": pre_single["text"][s_reserve: e], "label_entity_type":t,
                                              "pre_entity": pre_single["text"][s_:e_], "pre_entity_type":t_})
                        else:
                            # 边界错误类型对
                            error_border_correct_type += 1
                            e_b_c_t_f.append({"label_entity": pre_single["text"][s_reserve: e], "label_entity_type":t,
                                              "pre_entity": pre_single["text"][s_:e_], "pre_entity_type":t_})
                        start = max(e, int(e_))
                    else:
                        #是实体没有预测成实体（预测少了）
                        less += 1
                        start = e
                        l_f.append({"label_entity": pre_single["text"][s_reserve: e], "label_entity_type":t})
                elif label_sen_tag[start] != "O" and pre_sen_tag[start] != "O":
                    s, t, e = pre_sen_tag[start].split("-")
                    s, e = int(s), int(e)

                    s_, t_, e_ = label_sen_tag[start].split("-")
                    s_, e_ = int(s_), int(e_)

                    if t == t_ and e != e_:
                        #边界错误类型对
                        error_border_correct_type += 1
                        e_b_c_t_f.append({"label_entity": pre_single["text"][s_: e_],"label_entity_type":t_,
                                          "pre_entity": pre_single["text"][s:e], "pre_entity_type":t})
                    elif t != t_ and e != e_:
                        # 边界错误类型错误
                        error_border_error_type += 1
                        e_b_e_t_f.append({"label_entity": pre_single["text"][s_: e_], "label_entity_type":t_,
                                          "pre_entity": pre_single["text"][s:e], "pre_entity_type":t})
                    elif t != t_ and e == e_:
                        #边界对类型错误
                        correct_border_error_type += 1
                        c_b_e_t_f.append({"label_entity": pre_single["text"][s_: e_],"label_entity_type":t_,
                                          "pre_entity": pre_single["text"][s:e], "pre_entity_type":t})
                    start = max(e, e_)
        except:
            print("error------------------->统计错误类型时出错")
            print(traceback.format_exc())
            pass

        if c_b_e_t_f:
            correct_border_error_type_file.append({"text":pre_single["text"], "error":c_b_e_t_f})
        if e_b_c_t_f:
            error_border_correct_type_file.append({"text":pre_single["text"], "error":e_b_c_t_f})
        if e_b_e_t_f:
            error_border_error_type_file.append({"text":pre_single["text"], "error":e_b_e_t_f})
        if l_f:
            less_file.append({"text":pre_single["text"], "error":l_f})
        if m_f:
            more_file.append({"text":pre_single["text"], "error":m_f})

    # 保存错误类型结果文件
    if save_path:
        with open(save_path + "/" + title + '_error_type_file.txt', "w+", encoding="utf-8") as f:
            json.dump({"边界对类型错": correct_border_error_type_file, "边界错类型对": error_border_correct_type_file,
                       "边界错类型错": error_border_error_type_file, "预测少了": less_file, "预测多了": more_file}, f, ensure_ascii=False)

    print("边界对但是类型错误有%d个,共有实体数：%d,占总的实体比例为：%.2f"%(correct_border_error_type, entity_sum, correct_border_error_type / entity_sum))
    print("边界错但类型对有%d个,共有实体数：%d,占总的实体比例为：%.2f" % (error_border_correct_type, entity_sum, error_border_correct_type / entity_sum))
    print("边界错类型也错有%d个,共有实体数：%d,占总的实体比例为：%.2f" % (error_border_error_type, entity_sum, error_border_error_type / entity_sum))
    print("预测少了（是实体没有预测成实体）有%d个,共有实体数：%d,占总的实体比例为：%.2f" % (less, entity_sum, less / entity_sum))
    print("预测多了（不是实体预测成实体）有%d个,共有实体数：%d,占总的实体比例为：%.2f" % (more, entity_sum, more / entity_sum))

    return {"边界对类型错": correct_border_error_type / entity_sum, "边界错类型对": error_border_correct_type / entity_sum,
            "边界错类型错": error_border_error_type / entity_sum, "预测少": less / entity_sum, "预测多了": more / entity_sum}
