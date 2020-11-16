from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
# from utils import generate_embeddings_for_pooling

import numpy as np
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaForMaskedLM, RobertaTokenizer

logger = logging.getLogger(__name__)

class codebert(object):
    def __init__(self, model_type, model_path, device):
        self.model_type = model_type
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        if model_type == "mlm":
            self.model = RobertaForMaskedLM.from_pretrained(model_path)
        elif model_type == "cls":
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.block_size = 512
        self.device = device

    def tokenize(self, inputs, cut_and_pad=False, ret_id=False):
        rets = []
        if isinstance(inputs, str):
            inputs = [inputs]
        for sent in inputs:
            if cut_and_pad:
                tokens = self.tokenizer.tokenize(sent)[:self.block_size-2]
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                padding_length = self.block_size - len(tokens)
                tokens += [self.tokenizer.pad_token] * padding_length
            else:
                tokens = self.tokenizer.tokenize(sent)
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            if not ret_id:
                rets.append(tokens)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                rets.append(ids)
        return rets

    def _run_batch(self, batch, need_attn=False):
        self.model.eval()
        batch_max_length = batch.ne(self.tokenizer.pad_token_id).sum(-1).max().item()
        inputs = batch[:, :batch_max_length]
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), output_attentions=True)
            logits, attentions = outputs[0], outputs[-1]
        if need_attn:
            return logits, attentions
        return logits
    
    def run(self, inputs, batch_size=16):
        input_ids = self.tokenize(inputs, cut_and_pad=True, ret_id=True)
        outputs = []
        batch_num = (len(input_ids) - 1) // batch_size + 1
        for step in range(batch_num):
            batch = torch.tensor(input_ids[step*batch_size: (step+1)*batch_size])
            output = self._run_batch(batch)
            outputs.append(output)
        outputs = torch.stack(outputs, 0).squeeze(0)
        return outputs

    def run_info(self, inputs, batch_size=16, need_attn=False):
        input_ids = self.tokenize(inputs, cut_and_pad=True, ret_id=True)
        outputs = []
        attns = []
        batch_num = (len(input_ids) - 1) // batch_size + 1
        for step in range(batch_num):
            batch = torch.tensor(input_ids[step*batch_size: (step+1)*batch_size])
            output = self._run_batch(batch, need_attn)
            if need_attn:
                output, attn = output
            outputs.append(output)
            if need_attn:
                attns.append(attn[-1])      # get last layer attn
        logits = torch.stack(outputs, 0).squeeze(0)
        if need_attn:
            attns = torch.stack(attns, 0).squeeze(0)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}
        attentions = attns[:, :, 0, :].mean(1)  # get mean on multi-heads

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["attentions"] = attentions if need_attn else None
        output_dict["predicted_labels"] = probs.argmax(-1)
        return output_dict

    # def nor


class codebert_mlm(codebert):
    def __init__(self, model_path, device):
        super().__init__("mlm", model_path, device)

class codebert_cls(codebert):
    def __init__(self, model_path, device):
        super().__init__("cls", model_path, device)

