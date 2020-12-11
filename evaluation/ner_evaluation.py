from collections import defaultdict
import json
import os

def f1_eva_ner(predict_list,label_list,indexed = True,save = False,save_path = None):
    result_dict = defaultdict(dict)

    for single_label in label_list:
        text = single_label["text"]
        result_dict[text]["label"] = single_label

    for single_predict in predict_list:
        text = single_predict["text"]
        if text in result_dict:
            result_dict[text]["predict"] = single_predict

    tp = 0
    p_tp = 0
    r_tp = 0
    p_count = 0
    r_count = 0
    p_error_list = []
    r_error_list = []
    match_list = []
    for text in result_dict:
        if "predict" not in result_dict[text]:
            r_count += len(result_dict[text]["label"]["entity_list"])
        else:
            single_predict = result_dict[text]["predict"]

            single_label = result_dict[text]["label"]
            r_count += len(single_label["entity_list"])
            p_count += len(single_predict["entity_list"])

            single_label_set = set()
            for j in single_label["entity_list"]:
                if indexed:
                    single_label_set.add(json.dumps({"entity":j["entity"],"entity_type":j["entity_type"],"entity_index":j["entity_index"]},ensure_ascii=False))
                else:
                    single_label_set.add(json.dumps({"entity":j["entity"],"entity_type":j["entity_type"]},ensure_ascii=False))

            single_predict_set = set()
            for j in single_predict["entity_list"]:
                if indexed:
                    single_predict_set.add(json.dumps(
                        {"entity": j["entity"], "entity_type": j["entity_type"], "entity_index": j["entity_index"]},
                        ensure_ascii=False))
                else:
                    single_predict_set.add(
                        json.dumps({"entity": j["entity"], "entity_type": j["entity_type"]}, ensure_ascii=False))


            for j in single_predict["entity_list"]:
                if indexed:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"], "entity_index": j["entity_index"]},
                            ensure_ascii=False) in single_label_set:
                        p_tp += 1
                    else:
                        p_error_list.append(json.dumps({"text": text, "error":j, "true_list":list(single_label_set)},ensure_ascii=False))
                else:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"]},
                            ensure_ascii=False) in single_label_set:
                        p_tp += 1
                    else:
                        p_error_list.append(json.dumps({"text": text, "error":j, "true_list":list(single_label_set)},ensure_ascii=False))

            for j in single_label["entity_list"]:
                if indexed:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"], "entity_index": j["entity_index"]},
                            ensure_ascii=False) in single_predict_set:
                        r_tp += 1
                    else:
                        r_error_list.append(json.dumps({"text": text, "error":j, "true_list":list(single_predict_set)},ensure_ascii=False))
                else:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"]},
                            ensure_ascii=False) in single_predict_set:
                        r_tp += 1
                    else:
                        r_error_list.append(json.dumps({"text": text, "error":j, "true_list":list(single_predict_set)},ensure_ascii=False))

    precision = round(p_tp / (p_count + 1e-8), 4)
    recall = round(r_tp / (r_count + 1e-8), 4)
    f1 = round(2 * precision * recall / (precision + recall + 1e-8), 4)

    if save:
        p_err_path = os.path.join(save_path, "p_evaluation_error.log")
        r_err_path = os.path.join(save_path, "r_evaluation_error.log")
        match_path = os.path.join(save_path, "evaluation_match.log")
        with open(p_err_path, "w") as f:
            for item in p_error_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(r_err_path, "w") as f:
            for item in r_error_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(match_path, "w") as f:
            for item in match_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # print(r_error_list)

    return precision, recall, f1

def f1_eva_ner_label_level(predict_list,label_list,indexed = True,save = False,save_path = None):
    result_dict = defaultdict(dict)

    for single_label in label_list:
        text = single_label["text"]
        result_dict[text]["label"] = single_label

    for single_predict in predict_list:
        text = single_predict["text"]
        if text in result_dict:
            result_dict[text]["predict"] = single_predict

    p_tp_predicate = defaultdict(int)
    r_tp_predicate = defaultdict(int)
    p_count_predicate = defaultdict(int)
    r_count_predicate = defaultdict(int)
    for text in result_dict:
        if "predict" not in result_dict[text]:
            single_label = result_dict[text]["label"]
            for item in single_label["entity_list"]:
                r_count_predicate[item["entity_type"]] += 1
        else:
            single_predict = result_dict[text]["predict"]

            single_label = result_dict[text]["label"]

            for item in single_label["entity_list"]:
                r_count_predicate[item["entity_type"]] += 1
            for item in single_predict["entity_list"]:
                p_count_predicate[item["entity_type"]] += 1

            single_label_set = set()
            for j in single_label["entity_list"]:
                if indexed:
                    single_label_set.add(json.dumps({"entity":j["entity"],"entity_type":j["entity_type"],"entity_index":j["entity_index"]},ensure_ascii=False))
                else:
                    single_label_set.add(json.dumps({"entity":j["entity"],"entity_type":j["entity_type"]},ensure_ascii=False))

            single_predict_set = set()
            for j in single_predict["entity_list"]:
                if indexed:
                    single_predict_set.add(json.dumps(
                        {"entity": j["entity"], "entity_type": j["entity_type"], "entity_index": j["entity_index"]},
                        ensure_ascii=False))
                else:
                    single_predict_set.add(
                        json.dumps({"entity": j["entity"], "entity_type": j["entity_type"]}, ensure_ascii=False))


            for j in single_predict["entity_list"]:
                if indexed:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"], "entity_index": j["entity_index"]},
                            ensure_ascii=False) in single_label_set:
                        p_tp_predicate[j["entity_type"]] += 1
                else:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"]},
                            ensure_ascii=False) in single_label_set:
                        p_tp_predicate[j["entity_type"]] += 1

            for j in single_label["entity_list"]:
                if indexed:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"], "entity_index": j["entity_index"]},
                            ensure_ascii=False) in single_predict_set:
                        r_tp_predicate[j["entity_type"]] += 1
                else:
                    if json.dumps({"entity": j["entity"], "entity_type": j["entity_type"]},
                            ensure_ascii=False) in single_predict_set:
                        r_tp_predicate[j["entity_type"]] += 1


    ans_dict = {}
    for predicate in r_count_predicate.keys():
        p_tp = p_tp_predicate[predicate]
        r_tp = r_tp_predicate[predicate]
        p_count = p_count_predicate[predicate]
        r_count = r_count_predicate[predicate]

        precision = round(p_tp / (p_count + 1e-8), 4)
        recall = round(r_tp / (r_count + 1e-8), 4)
        f1 = round(2 * precision * recall / (precision + recall + 1e-8), 4)

        ans_dict[predicate] = {"F1":f1,"P":precision,"R":recall}

    return ans_dict

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

    label_data_list_ = label_data

    pre_data_list = pre_data

    label_data_list = []
    for item in pre_data_list:
        for _ in label_data_list_:
            if item["text"] == _["text"]:
                label_data_list.append(_)
                break

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

