import argparse
import os
import pickle

from dataset import Java, Dataset
from models.lstm_cls import LSTMEncoder, LSTMClassifier
from models.Transformer import TransformerClassifier

import torch
import torch.nn as nn
from torch import optim
import numpy as np
from utils import gettensor, myDataParallel
from sklearn.metrics import precision_recall_fscore_support

                
def evaluate(classifier, device, dataset, batch_size=128):

    classifier.eval()
    dataset.reset_epoch()
    predict = []
    label = []
    while True:
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, labels, lens = gettensor(batch, classifier)
        with torch.no_grad():
            outputs = classifier(inputs, lens)        
            outputs = torch.argmax(outputs, dim=1).cpu().data.numpy()
            predict += list(outputs)
            label += list(labels.cpu().data.numpy())

    precision, recall, _, _ = precision_recall_fscore_support(label, predict)
    precision, recall = precision[1], recall[1]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)     # prevent from zero division
    print("Evaluation:\n\tPrecision: %.3f\n\tRecall: %.3f\n\tF1: %.3f\n" % (precision, recall, f1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='-1')
    parser.add_argument('-model', type=str, default='LSTM')
    parser.add_argument('-bs_eval', type=int, default=64)
    parser.add_argument('-data', type=str, default='java')
    parser.add_argument('-save_name', type=str, default='java-lstm')
    parser.add_argument('--load_dataset', action='store_true')
    
    opt = parser.parse_args()

    _save = os.path.join("./save", opt.save_name)

    data_dict = {"JAVA": "../../bigJava/datasets"}
    _data = data_dict[opt.data.upper()]
    _load_dataset = opt.load_dataset
    _bs_eval = opt.bs_eval

    if opt.gpu == "-1":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    _model = opt.model
    vocab_size = 30000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    n_channel = -1
    n_class_dict = {"JAVA": 2}
    n_class = n_class_dict[opt.data.upper()]
    max_len = 400
    bidirection = True

    ## WARNING: dataset is pre saved, if parameters are changed, please run dataset.py first
    if _load_dataset:
        print("Loading dataset.")
        with open(os.path.join(_data, "Java.pkl"), "rb") as f:
            dataset = pickle.load(f)
        print("Done.")
    else:
        print("Creating dataset.")
        dataset = Java(path=_data,
                    max_len=max_len,
                    vocab_size=vocab_size)
        print("Done.")
    training_set = dataset.train
    valid_set = dataset.dev
    test_set = dataset.test
    
    if _model == 'LSTM':
        enc = LSTMEncoder(embedding_dim=embedding_size,
                    hidden_dim=hidden_size,
                    n_layers=n_layers,
                    drop_prob=0,
                    brnn=bidirection)
        classifier = LSTMClassifier(vocab_size=vocab_size,
                                encoder=enc,
                                num_class=n_class,
                                device=device).to(device)
    else:       # Transformer
        classifier = TransformerClassifier(vocab_size + 1, n_class, hidden_size, d_ff=1024, h=6, N=n_layers, dropout=0).to(device)
        # classifier = myDataParallel(classifier).to(device)
    try:
        classifier.load_state_dict(torch.load(_save))
    except:
        classifier = myDataParallel(classifier).to(device)
        classifier.load_state_dict(torch.load(_save))

    classifier.eval()
    evaluate(classifier, device, test_set, _bs_eval)
