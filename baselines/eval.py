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

 
class myDataParallel(nn.DataParallel):
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

 
def gettensor(batch, model):
    ''' Batch second '''
    device = model.classify.weight.device
    inputs, labels, lens = batch['x'], batch['y'], batch['l']
    with_cls = isinstance(model.module, TransformerClassifier)
    batch_first = with_cls
    if with_cls:    # add <CLS> to inputs
        # cls = vocab_size
        cls = model.model.embeddings.word_embeddings.weight.shape[0] - 1
        inputs = np.insert(inputs, 0, [cls] * inputs.shape[0], axis=1)
        lens = lens + 1     # numpy supported
    inputs, labels, lens = torch.tensor(inputs, dtype=torch.long).to(device), \
                           torch.tensor(labels, dtype=torch.long).to(device), \
                           torch.tensor(lens, dtype=torch.long).to(device)
    if not batch_first:
        inputs = inputs.permute([1, 0])
    return inputs, labels, lens

                
def evaluate(classifier, device, dataset, batch_size=128):

    classifier.eval()
    testnum = 0
    testcorrect = 0
    dataset.reset_epoch()
    while True:
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, labels, lens = gettensor(batch, classifier)
        with torch.no_grad():
            outputs = classifier(inputs, lens)        
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
    print('eval_acc:  %.2f%%' % (float(testcorrect) * 100.0 / testnum))


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
    hidden_size = 384
    n_layers = 6
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
        classifier = myDataParallel(classifier).to(device)
        classifier.load_state_dict(torch.load(_save))
    else:       # Transformer
        classifier = TransformerClassifier(vocab_size + 1, n_class, hidden_size, d_ff=1024, h=6, N=n_layers, dropout=0).to(device)
        classifier = myDataParallel(classifier).to(device)
        classifier.load_state_dict(torch.load(_save))

    classifier.eval()
    evaluate(classifier, device, test_set, _bs_eval)
