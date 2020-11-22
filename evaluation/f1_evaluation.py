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

    precision = round(p_tp / (p_count + 1e-8), 3)
    recall = round(r_tp / (r_count + 1e-8), 3)
    f1 = round(2 * precision * recall / (precision + recall + 1e-8), 3)

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

        precision = round(p_tp / (p_count + 1e-8), 3)
        recall = round(r_tp / (r_count + 1e-8), 3)
        f1 = round(2 * precision * recall / (precision + recall + 1e-8), 3)

        ans_dict[predicate] = {"F1":f1,"P":precision,"R":recall}

    return ans_dict

