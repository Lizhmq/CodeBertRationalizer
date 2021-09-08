import logging
import os

import torch
import torch.nn as nn
from transformers import RobertaModel


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class NBLModel(nn.Module):

    def __init__(self, hidden, classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.classes = classes
        self.dropout = nn.Dropout(0.1)
        self.embdim = 32
        self.embedding = nn.Embedding(231, self.embdim)
        self.classifier = nn.Linear(768 + self.embdim, classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden + self.embdim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, classes)
        # )
        self.bert = RobertaModel.from_pretrained(self.args.pretrain_dir)
        # self._init_cls_weight()


    def _init_cls_weight(self, initializer_range=0.02):
        for layer in (self.classifier, self.embedding):
            layer.weight.data.normal_(mean=0.0, std=initializer_range)
            if layer is not self.embedding:
                if layer.bias is not None:
                    layer.bias.data.zero_()
    

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
        torch.save(self.classifier.state_dict(), os.path.join(path, "cls.bin"))
        torch.save(self.embedding.state_dict(), os.path.join(path, "emb.bin"))
        
    def from_pretrained(self, path):
        self.bert = RobertaModel.from_pretrained(path)
        self.classifier.load_state_dict(torch.load(os.path.join(path, "cls.bin"), map_location=self.device))
        self.embedding.load_state_dict(torch.load(os.path.join(path, "emb.bin"), map_location=self.device))
        return self

    def forward(self, input_ids, input_mask, tids, y=None):
        sequence_output = self.bert(input_ids, input_mask)[0]
        output0 = self.dropout(sequence_output[:, 0, :])
        embedded = self.embedding(tids)
        rep = torch.cat((output0, embedded), dim=1)
        # batch_size, max_len, feat_dim = sequence_output.shape

        logits0 = self.classifier(rep)
        if y != None:
            loss_fct0 = nn.CrossEntropyLoss()
            loss0 = loss_fct0(logits0, y)
            return loss0
        else:
            return logits0
    
    def run_info(self, input_ids, input_mask, tids):
        outputs = self.bert(input_ids, input_mask, output_attentions=True)
        output0, attns = outputs[0], outputs[-1][-1]
        output0 = self.dropout(output0[:, 0, :])
        embedded = self.embedding(tids)
        rep = torch.cat((output0, embedded), dim=1)
        logits0 = self.classifier(rep)
        pred = torch.argmax(logits0, dim=-1)
        output_dict = dict()
        output_dict["logits"] = logits0
        output_dict["probs"] = torch.softmax(logits0, dim=1)
        output_dict["predictions"] = pred
        output_dict["attentions"] = attns[:, :, 0, :].mean(1)
        return output_dict


def build_model(args, load_path=None):
    model = NBLModel(768, 2, args, args.device)
    if load_path is not None:
        model = model.from_pretrained(load_path).to(args.device)
    return model
