import torch
import torch.nn as nn
import numpy as np
from utils import gettensor



class SaliencyScorer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, inputs, ls):
        _, attentions = self.model(inputs, ls, need_attn=True)
        return attentions.cpu().data.numpy()