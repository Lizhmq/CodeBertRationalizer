import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from models.transformer import *



class TransformerClassifier(nn.Module):
    
    def __init__(self, vocab_size=5000, num_classes=104, d_model=512, d_ff=1024, h=8, N=3, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.c = copy.deepcopy
        self.d_model = d_model
        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Sequential(Embeddings(d_model, vocab_size), self.c(self.position))
        self.enc = Encoder(EncoderLayer(d_model, self.c(self.attn), self.c(self.ff), dropout), N)
        self.classify = nn.Linear(d_model, num_classes)
        self.padding = 0
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, ls, need_attn=False):
        x = x.permute([1, 0])    # convert to batch first
        x_zero_mask = (x == self.padding).unsqueeze(-1)
        x_mask = (x != self.padding).unsqueeze(-2)
        x = self.embedding(x)
        outputs = self.enc(x, x_mask)
        # outputs = outputs.permute([1, 0, 2])
        outputs = outputs.masked_fill(x_zero_mask, 0)
        ls = torch.sum(~x_zero_mask, dim=1)
        outputs = torch.sum(outputs, dim=1) / ls

        logits = self.classify(outputs)
        if need_attn:
            return logits, None         # todo
        else:
            return logits
    
    def prob(self, inputs):
        
        logits = self.forward(inputs)
        prob = nn.Softmax(dim=0)(logits)
        return prob