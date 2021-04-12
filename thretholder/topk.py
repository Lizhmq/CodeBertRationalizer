import torch
from torch.nn import Module
import math
import numpy as np

from thretholder.base_thresholder import Thresholder


class TopKThresholder(Thresholder):
    def __init__(self, _max_length_ratio):
        self._max_length_ratio = _max_length_ratio
        super().__init__()

    def forward(self, attentions, as_one_hot=False):
        """
        attentions is normalized. (B * L)
        """
        lens = (attentions != 0.0).sum(axis=1)
        assert(lens.shape[0] == attentions.shape[0])
        # attentions = attentions.cpu().data.numpy()
        rationales = []
        for b in range(attentions.shape[0]):
            attn = attentions[b][:lens[b]]
            max_length = math.ceil(len(attn) * self._max_length_ratio)
            max_length = 1
            top_ind, top_vals = np.argsort(attn)[-max_length:], \
                                np.sort(attn)[-max_length:]
            if as_one_hot:
                rationales.append([1 if i in top_ind else 0 for i in range(attentions.shape[1])])
            else:
                rationales.append([{"span": (i, i + 1), "value": float(v)} \
                                    for i, v in zip(top_ind, top_vals)])
        return rationales
