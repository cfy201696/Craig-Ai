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

    assert len(label) == len(pre)
    label_leval = {}
    total_TP, total_FP, total_FN = 0, 0, 0
    total_f1 = 0

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
        total_f1 += f1

    # macro_f1
    macro_p = total_TP / (total_TP + total_FP + 1e-8)
    macro_r = total_TP / (total_TP + total_FN + 1e-8)
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r + 1e-8)

    return round(macro_p,4), round(macro_r,4), round(macro_f1,4), round(total_f1 / len(label2id),4), label_leval








