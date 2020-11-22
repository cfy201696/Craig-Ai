import numpy as np
import torch



def _generate_mask( sen_len,max_len, use_cuda=True):
    mask = np.zeros([len(sen_len), max_len])
    for idx, length in enumerate(sen_len):
        mask[idx][:length] = 1
    if use_cuda:
        return torch.ByteTensor(mask).cuda()
    else:
        return torch.ByteTensor(mask)

def _array_from_mask(output,sen_len ):
    result = []
    for single_output, single_length in zip(output,sen_len):
        result.append(single_output.cpu().detach().numpy().tolist()[:single_length])
    return result