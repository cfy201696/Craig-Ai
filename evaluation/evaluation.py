def TP_FP_FN(label, pre, true_label=1):
    TP, FP, FN = 0, 0, 0

    for l, p in zip(label, pre):
        if l == p and l == true_label:
            TP += 1
        if l != p and p == true_label:
            FP += 1
        if l != p and l == true_label:
            FN += 1
    return TP, FP, FN

def evaluation(label, pre, label2id=None):
    print(label[:20])
    print("-"*100)
    print(pre[:20])

    assert len(label) == len(pre)
    label_leval = {}
    total_TP, total_FP, total_FN = 0, 0, 0
    total_p, total_r, total_f1 = 0, 0, 0


    for type, id in label2id.items():
        label_, pre_ = [], []
        for l, p in zip(label, pre):
            label_.append(1 if l == type else 0)
            pre_.append(1 if p == type else 0)
        TP, FP, FN = TP_FP_FN(label_, pre_)

        p = round(TP / (TP + FP + 1e-8),4)
        r = round(TP / (TP + FN + 1e-8),4)
        f1 = round(2 * p * r / (p + r + 1e-8),4)

        label_leval[type] = {"P":p, "r":r, "F1":f1}

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_p += p
        total_r += r
        total_f1 += f1


    # micro_f1
    micro_p = total_TP / (total_TP + total_FP + 1e-8)
    micro_r = total_TP / (total_TP + total_FN + 1e-8)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)

    return round(total_p / len(label2id),4), round(total_r / len(label2id),4), round(total_f1 / len(label2id),4),\
           round(micro_f1,4), label_leval







