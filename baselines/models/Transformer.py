import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from transformers import BertModel, BertConfig


class TransformerClassifier(nn.Module):

    def __init__(self, vocab_size=30000, num_classes=2, d_model=512, d_ff=1024, h=8, N=3, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.padding = 0
        self.config = BertConfig(vocab_size=vocab_size, hidden_size=d_model, num_hidden_layers=N, \
                        num_attention_heads=h, intermediate_size=d_ff, hidden_dropout_prob=dropout)
        self.model = BertModel(self.config)
        self.classify = nn.Linear(d_model, num_classes)
        
    def forward(self, x, ls, need_attn=False):
        x_mask = (x != self.padding).unsqueeze(-2)
        outputs = self.model(x, x_mask, output_attentions=True)
        outputs, attentions = outputs[0], outputs[-1]
        outputs = outputs[:, 0, :]
        if need_attn:
            attentions = attentions[-1]
            attentions = torch.mean(attentions, dim=1)[:, 0, :]

        logits = self.classify(outputs)
        if need_attn:
            return logits, attentions
        else:
            return logits
    
    def prob(self, inputs):
        logits = self.forward(inputs)
        prob = nn.Softmax(dim=0)(logits)
        return prob