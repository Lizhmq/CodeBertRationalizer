import argparse
import os
import pickle

from dataset import Java, Dataset
from models.lstm_cls import LSTMEncoder, LSTMClassifier
from models.transformer_cls import TransformerClassifier
from models.transformer import NoamOpt, get_std_opt

import torch
import torch.nn as nn
from torch import optim
import numpy

 
def gettensor(batch, device):
    ''' Batch second '''
    inputs, labels, lens = batch['x'], batch['y'], batch['l']
    inputs, labels, lens = torch.tensor(inputs, dtype=torch.long).to(device), \
                           torch.tensor(labels, dtype=torch.long).to(device), \
                           torch.tensor(lens, dtype=torch.long).to(device)
    inputs = inputs.permute([1, 0])
    return inputs, labels, lens
    

def trainEpochs(device, epochs, training_set, valid_set, criterion, opt, batch_size=32,
                batch_size_eval=64, print_each=100, saving_path='./', lrdecay=1):
    
    classifier.train()
    epoch = 0
    i = 0
    print_loss_total = 0
    n_batch = int(training_set.get_size() / batch_size)
    print('start training epoch ' + str(epoch + 1) + '....')
    
    while True:
        batch = training_set.next_batch(batch_size)
        if batch['new_epoch']:
            epoch += 1
            print_loss_total = 0
            evaluate(device, valid_set, batch_size_eval)
            classifier.train()
            torch.save(classifier.state_dict(),
                       os.path.join(saving_path, str(epoch)+'.pt'))
            if lrdecay < 1:
                adjust_learning_rate(optimizer, lrdecay)
            if epoch == epochs:
                break
            i = 0
            print('start training epoch ' + str(epoch + 1) + '....')
        inputs, labels, lens = gettensor(batch, device)
        
        optimizer.zero_grad()
        outputs= classifier(inputs, lens)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print_loss_total += loss.item()
        if (i + 1) % print_each == 0: 
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            print('\tEp %d %d/%d, loss = %.6f' \
                  % (epoch + 1, (i + 1), n_batch, print_loss_avg))
        i += 1

def adjust_learning_rate(optimizer, decay_rate=0.8):
    if type(optimizer) == NoamOpt:      # adjust lr only for simple Adam optimizer
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
                
def evaluate(device, dataset, batch_size=128):

    classifier.eval()
    testnum = 0
    testcorrect = 0
    dataset.reset_epoch()
    while True:
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, labels, lens = gettensor(batch, device)
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
    parser.add_argument('-lr', type=float, default=0.003)
    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-bs', type=int, default=32)
    parser.add_argument('-bs_eval', type=int, default=64)
    parser.add_argument('-l2p', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lrdecay', type=float, default=1)
    parser.add_argument('-data', type=str, default='java')
    parser.add_argument('-save_name', type=str, default='java-lstm')
    parser.add_argument('--load_dataset', action='store_true')
    
    opt = parser.parse_args()

    _save = os.path.join("./save", opt.save_name)
    if not os.path.isdir(_save):
        os.mkdir(_save)

    data_dict = {"JAVA": "../../bigJava/datasets"}
    _data = data_dict[opt.data.upper()]
    _drop = opt.dropout
    _lr = opt.lr
    _l2p = opt.l2p
    _lrdecay = opt.lrdecay
    _load_dataset = opt.load_dataset
    
    _bs = opt.bs
    _bs_eval = opt.bs_eval
    _ep = opt.epoch

    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

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
                    drop_prob=_drop,
                    brnn=bidirection)
        classifier = LSTMClassifier(vocab_size=vocab_size,
                                encoder=enc,
                                num_class=n_class,
                                device=device).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=_lr, weight_decay=_l2p)
    else:       # Transformer
        classifier = TransformerClassifier(vocab_size, n_class, hidden_size, d_ff=1024, h=8, N=n_layers, dropout=_drop).to(device)
        optimizer = NoamOpt(classifier.embedding[0].d_model, 1, 2000,
                    torch.optim.Adam(classifier.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    criterion = nn.CrossEntropyLoss()
    
    batch = test_set.next_batch(1)
    inputs, labels, lens = gettensor(batch, device)
    optimizer.zero_grad()
    classifier.eval()
    outputs = classifier(inputs, lens)
    print(outputs)
    outputs, attn = classifier(inputs, lens, need_attn=True)
    print(outputs, attn.shape if attn else (0, 0))
    
    trainEpochs(device, _ep, training_set, valid_set, criterion, opt, saving_path=_save,
                batch_size=_bs, batch_size_eval=_bs_eval, lrdecay=_lrdecay)