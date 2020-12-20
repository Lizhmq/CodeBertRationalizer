import torch
import torch.nn as nn
import numpy as np
from utils import gettensor
from models.Transformer import TransformerClassifier



class SaliencyScorer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, batch):
        inputs, labels, ls = gettensor(batch, self.model)
        _, attentions = self.model(inputs, ls, need_attn=True)
        return attentions.cpu().data.numpy()