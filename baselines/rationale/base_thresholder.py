import torch
from torch.nn import Module


class Thresholder(Module):
    def __init__(self):
        super().__init__()

    def forward(self, attentions):
        raise NotImplementedError
    
    def extract_rationale(self, **kwargs):
        raise NotImplementedError
