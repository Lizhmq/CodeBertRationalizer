from __future__ import absolute_import, division, print_function

import logging
import torch

from models.tokenizer import Tokenizer
from utils import get_start_idxs_batched
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaForMaskedLM, RobertaTokenizer

logger = logging.getLogger(__name__)

class codebert():
    def __init__(self, model_type, model_path, device):
        self.model_type = model_type
        self.tokenizer = Tokenizer.from_pretrained(model_path)
        self.tokenizer.__class__ = Tokenizer         # not perfect convert
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
            if isinstance(sent, list):
                sent = " ".join(sent)
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
        # self.model.eval()
        batch_max_length = batch.ne(self.tokenizer.pad_token_id).sum(-1).max().item()
        inputs = batch[:, :batch_max_length]
        inputs = inputs.to(self.device)
        # with torch.no_grad():
        outputs = self.model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id).to(inputs), output_attentions=True)
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

    def run_info(self, inputs, batch_size=16, need_attn=True):
        '''
            inputs is list of token list
        '''
        # compute start_idxs and end_idxs
        input_subs = self.tokenize(inputs, cut_and_pad=False, ret_id=False)
        b_start_idxs, b_end_idxs = get_start_idxs_batched(inputs, input_subs)
        input_ids = list(map(lambda x: self.tokenizer.convert_tokens_to_ids(x), input_subs))
        def pad(seq):
            seq = seq[:self.block_size - 2]
            seq += [self.tokenizer.pad_token_id] * (self.block_size - len(seq))
            return seq
        input_ids = list(map(pad, input_ids))
        input_ids = torch.LongTensor(input_ids).to(self.device)

        max_len = max(map(len, b_start_idxs))
        for i in range(len(b_start_idxs)):
            pad_to = lambda lst: list(lst) + [b_end_idxs[i][-1]] * (max_len - len(lst))
            b_start_idxs[i] = pad_to(b_start_idxs[i])
            b_end_idxs[i] = pad_to(b_end_idxs[i])
        b_start_idxs = torch.LongTensor(b_start_idxs).to(input_ids)
        b_end_idxs = torch.LongTensor(b_end_idxs).to(input_ids)
        

        outputs = []
        attns = []
        batch_num = (len(input_ids) - 1) // batch_size + 1
        for step in range(batch_num):
            batch = input_ids[step*batch_size: (step+1)*batch_size]
            output = self._run_batch(batch, need_attn)
            if need_attn:
                output, attn = output
            outputs.append(output)
            if need_attn:
                attns.append(attn[-1])     # get last layer attn   
        logits = torch.stack(outputs).squeeze(0)
        if need_attn:
            attns = torch.stack(attns).squeeze(0)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}
        if need_attn:
            attentions = attns[:, :, 0, :].mean(1)  # get mean on multi-heads
        else:
            attentions = None
        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["attentions"] = attentions
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["start_idxs"] = b_start_idxs
        output_dict["end_idxs"] = b_end_idxs
        return output_dict




class codebert_mlm(codebert):
    def __init__(self, model_path, device):
        super().__init__("mlm", model_path, device)

class codebert_cls(codebert):
    def __init__(self, model_path, device):
        super().__init__("cls", model_path, device)

