import torch
from torch.nn import Module
import math
import numpy as np
from thretholder.base_thresholder import Thresholder


class ContiguousThresholder(Thresholder):
    def __init__(self, _max_length_ratio):
        self._max_length_ratio = _max_length_ratio
        super().__init__()

    def forward(self, attentions, as_one_hot=False):
        """
        attentions is normalized. (B * L), on cpu
        """
        lens = (attentions != 0.0).sum(axis=1)
        cumsumed_attention = attentions.cumsum(-1)
        rationales = []
        for b in range(cumsumed_attention.shape[0]):
            attn = cumsumed_attention[b]
            l = lens[b]
            max_length = math.ceil(l * self._max_length_ratio)
            best_v = np.zeros((l, ))
            for i in range(0, l - max_length + 1):
                j = i + max_length
                best_v[i] = attn[j - 1] - (attn[i - 1] if i >= 1 else 0)
            index = np.argmax(best_v)
            i, j, v = index, index + max_length, best_v[index]
            top_idx = list(range(i, j))
            if as_one_hot:
                rationales.append([1 if i in top_idx else 0 for i in range(attentions.shape[1])])
            else:
                rationales.append([{'span': (i, j), 'value': float(v)}])
        return rationales