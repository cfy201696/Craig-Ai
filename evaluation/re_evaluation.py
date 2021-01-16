from collections import defaultdict
import json
import os

def f1_eva_re(predict_list,label_list,indexed = True,save = False,save_path = None):
    result_dict = defaultdict(dict)

    for single_label in label_list:
        text = single_label["text"]
        result_dict[text]["label"] = single_label

    for single_predict in predict_list:
        text = single_predict["text"]
        if text in result_dict:
            result_dict[text]["predict"] = single_predict

    tp = 0
    p_count = 0
    r_count = 0
    error_list = []
    match_list = []
    for text in result_dict:
        if "predict" not in result_dict[text]:
            r_count += len(result_dict[text]["label"]["spo_list"])
        else:
            ## 有 bug !!!!!!!
            single_predict = result_dict[text]["predict"]

            predicts_set = set()
            for item in single_predict["spo_list"]:
                str_item = json.dumps(item,ensure_ascii=False)
                if str_item not in predicts_set:
                    predicts_set.add(str_item)
                else:
                    continue
            single_predict["spo_list"] = [json.loads(item) for item in predicts_set]


            single_label = result_dict[text]["label"]
            r_count += len(single_label["spo_list"])
            p_count += len(single_predict["spo_list"])

            for i in single_predict["spo_list"]:
                if_not_error = False
                for j in single_label["spo_list"]:
                    if "object" in i and "subject" in i:
                        if indexed:
                            try:
                                if i["object"] == j["object"] and i["subject"] == j["subject"] and i["predicate"] == j[
                                    "predicate"] and str(i["object_index"]) == str(j["object_index"]) and str(i["subject_index"]) == str(j["subject_index"]):
                                    tp += 1
                                    if_not_error = True
                            except:
                                pass
                        else:
                            if i["object"] == j["object"] and i["subject"] == j["subject"] and i["predicate"] == j[
                                "predicate"]:
                                tp += 1
                                if_not_error = True
                                break
                if if_not_error:
                    match_list.append({"text":text,"spo":i})
                else:
                    error_list.append({"text":text,"spo":i,"spo_list":single_label["spo_list"]})

    precision = round(tp/(p_count+1e-8),4)
    recall = round(tp/(r_count+1e-8),4)
    f1 = round(2*precision*recall/(precision+recall+1e-8),4)

    if save:
        err_path = os.path.join(save_path,"evaluation_error.log")
        match_path = os.path.join(save_path, "evaluation_match.log")
        with open(err_path,"w") as f:
            for item in error_list:
                f.write(json.dumps(item,ensure_ascii=False)+"\n")
        with open(match_path,"w") as f:
            for item in match_list:
                f.write(json.dumps(item,ensure_ascii=False)+"\n")

    return precision,recall,f1


def f1_eva_re_predicate_level(predict_list, label_list, indexed = True):
    result_dict = defaultdict(dict)

    for single_label in label_list:
        text = single_label["text"]
        result_dict[text]["label"] = single_label

    for single_predict in predict_list:
        text = single_predict["text"]
        if text in result_dict:
            result_dict[text]["predict"] = single_predict

    tp_predicate = defaultdict(int)
    p_count_predicate = defaultdict(int)
    r_count_predicate = defaultdict(int)

    for text in result_dict:
        if "predict" not in result_dict[text]:
            single_label = result_dict[text]["label"]
            for item in single_label["spo_list"]:
                r_count_predicate[item["predicate"]] += 1
        else:
            ## 有 bug !!!!!!!
            single_predict = result_dict[text]["predict"]
            predicts_set = set()
            for item in single_predict["spo_list"]:
                str_item = json.dumps(item,ensure_ascii=False)
                if str_item not in predicts_set:
                    predicts_set.add(str_item)
                else:
                    continue
            single_predict["spo_list"] = [json.loads(item) for item in predicts_set]

            single_label = result_dict[text]["label"]

            for item in single_label["spo_list"]:
                r_count_predicate[item["predicate"]] += 1
            for item in single_predict["spo_list"]:
                p_count_predicate[item["predicate"]] += 1

            for i in single_predict["spo_list"]:
                for j in single_label["spo_list"]:
                    if "object" in i and "subject" in i:
                        if indexed:
                            try:
                                if i["object"] == j["object"] and i["subject"] == j["subject"] and i["predicate"] == j[
                                    "predicate"] and str(i["object_index"]) == str(j["object_index"]) and str(i["subject_index"]) == str(j["subject_index"]):
                                    tp_predicate[i["predicate"]] += 1
                            except:
                                pass
                        else:
                            if i["object"] == j["object"] and i["subject"] == j["subject"] and i["predicate"] == j[
                                "predicate"]:
                                tp_predicate[i["predicate"]] += 1
                                break

    ans_dict = {}
    for predicate in r_count_predicate.keys():
        tp = tp_predicate[predicate]
        p_count = p_count_predicate[predicate]
        r_count = r_count_predicate[predicate]

        precision = round(tp / (p_count + 1e-8), 4)
        recall = round(tp / (r_count + 1e-8), 4)
        f1 = round(2 * precision * recall / (precision + recall + 1e-8), 4)

        ans_dict[predicate] = {"F1":f1,"P":precision,"R":recall}

    return ans_dict
