import torch
import torch.nn as nn
import numpy as np


class LSTMEncoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_layers,
                 drop_prob=0.5, brnn=True):
        
        super(LSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = brnn
        self.m = nn.LSTM(embedding_dim, hidden_dim, \
                         n_layers, dropout=drop_prob, bidirectional=brnn)
        
    def forward(self, input, hidden=None):
        return self.m(input, hidden)


class LSTMClassifier(nn.Module):
    
    def __init__(self, vocab_size, encoder, num_class, device=None, verbose=False):
        
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = encoder.embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_dim * 2 if self.encoder.bidirectional \
                            else encoder.hidden_dim
        self.query = nn.Linear(self.hidden_dim, 1)
        self.attn_softmax = nn.Softmax(dim=0)
        self.n_channel = self.hidden_dim
        self.n_class = num_class
        self.classify = nn.Linear(self.n_channel, self.n_class)
        self.pred_softmax = nn.Softmax(dim=1)
        size = 0
        for p in self.parameters():
            size += p.nelement()
        if verbose:
            print('Total param size: {}'.format(size))
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            
    def get_hidden(self, inputs):
        
        emb = self.embedding(inputs)        
        hidden_states, _ = self.encoder(emb)
        return hidden_states
    
    def get_mask(self, hidden_states, ls):
        
        mask = (torch.arange(hidden_states.shape[0]).to(self.device)[None, :] < ls[:, None])
        return mask.permute([1, 0])
    
    def get_attention(self, hidden_states, mask):
        
        alpha_logits = self.query(hidden_states.reshape([-1, self.hidden_dim]))
        alpha_logits = alpha_logits.reshape(hidden_states.shape[:2])
        alpha_logits[~mask] = float("-inf")
        alpha = self.attn_softmax(alpha_logits)
        return alpha
        
    def forward(self, inputs, ls, need_attn=False):
  
        hidden_states = self.get_hidden(inputs)
        mask = self.get_mask(hidden_states, ls)
        alpha = self.get_attention(hidden_states, mask)
        
        # [l, bs, nch] => [bs, nch] => [bs, ncl]
        _alpha = torch.stack([alpha for _ in range(self.n_channel)], dim=2)
        logits = self.classify(torch.sum(hidden_states * _alpha, dim=0))
        if need_attn:
            return logits, alpha.permute([1, 0])
        else:
            return logits

    def prob(self, inputs, ls):
        
        logits = self.forward(inputs, ls)
        prob = self.pred_softmax(logits)
        return prob