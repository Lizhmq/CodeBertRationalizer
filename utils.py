import torch
from allennlp.nn import util


def normalize_attentions(attentions, starting_offsets, ending_offsets):
    cum_attn = attentions.cumsum(1)
    start_v = cum_attn.gather(1, starting_offsets - 1)
    end_v = cum_attn.gather(1, ending_offsets - 1)
    return end_v - start_v


def get_start_idxs_batched(b_tokens, b_sub_tokens, bpe_indicator='Ġ'):
    
    # print(b_sub_tokens[0])

    from copy import deepcopy as cp
    ss = []
    es = []
    batch_size = len(b_tokens)
    for i in range(batch_size):
        starts = [1]
        starts += list(filter(lambda j: b_sub_tokens[i][j][0] == bpe_indicator, range(len(b_sub_tokens[i]))))
        assert len(starts) == len(b_tokens[i])
        ends = cp(starts)
        ends.append(len(b_sub_tokens[i]) - 1)
        ends = ends[1:]
        # starts = [0] + starts + [len(b_sub_tokens[i]) - 1]
        # ends = [1] + ends + [len(b_sub_tokens[i])]
        ends = list(filter(lambda x: x <= 512, ends))       # less than 512
        ss.append(starts[:len(ends)])
        es.append(ends)
    return ss, es