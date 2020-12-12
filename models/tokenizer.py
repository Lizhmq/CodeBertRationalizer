import os
from numpy.core.shape_base import block
import torch
import numpy as np
from transformers import RobertaTokenizer


class Tokenizer(RobertaTokenizer):

    def __init__(self) -> None:
        pass
        
    def from_pretrained(model_path, do_lower_case=False):
        return RobertaTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)

    def tokenize_with_special(self, text, masked_idxs=None):
        tokenized_text = self.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.mask_token
        tokenized = [self.cls_token] + tokenized_text + [self.sep_token]
        return tokenized

    def subword_tokenize(self, tokens, block_size=512):
        subwords = list(map(self.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        
        flatten = lambda t: [item for sublist in t for item in sublist]
        subwords = list(flatten(subwords))
        subwords = subwords[:block_size-2]
        subwords = [self.cls_token] + subwords + [self.sep_token]

        if len(subwords) < block_size:
            subwords += [self.pad_token] * (block_size - len(subwords))

        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_end_idxs = np.cumsum(subword_lengths) + 1
        return subwords, token_start_idxs, token_end_idxs

    def subword_tokenize_to_ids(self, tokens, block_size=512):
        subwords, token_start_idxs, token_end_idxs = self.subword_tokenize(tokens, block_size=block_size)
        subword_ids = self.convert_tokens_to_ids(subwords)
        # token_starts = torch.zeros(1, max_len).to(subword_ids)
        # token_starts[0, token_start_idxs] = 1
        return subword_ids, token_start_idxs, token_end_idxs



