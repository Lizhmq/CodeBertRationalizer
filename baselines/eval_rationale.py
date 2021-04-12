import os
import torch
import numpy as np
import pickle
from models.lstm_cls import LSTMEncoder, LSTMClassifier
from tqdm import tqdm
from rationale.scorer import SaliencyScorer
from rationale.contiguous import ContiguousThresholder
from rationale.topk import TopKThresholder
from rationale.evaluator import Evaluator
from dataset import Dataset, Java
from utils import set_seed

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    set_seed(2333)
    device = torch.device("cuda", 0)
    model_path = "./save/lstm/11.pt"
    # model_path = "./save/py-op/9.pt"
    vocab_size = 30000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    n_channel = -1
    n_class_dict = {"JAVA": 2}
    n_class = n_class_dict["JAVA"]
    # max_len = 400
    max_len = 512
    bidirection = True

    enc = LSTMEncoder(embedding_dim=embedding_size, hidden_dim=hidden_size,
                    n_layers=n_layers, drop_prob=0, brnn=bidirection)
    classifier = LSTMClassifier(vocab_size=vocab_size, encoder=enc,
                    num_class=n_class, device=device).to(device)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    # classifier = myDataParallel(classifier).to(device)

    data_path = "../../bigJava/datasets/test_tp_lstm.pkl"
    # data_path = '../../CuBert/wrong_op/test_tp_lstm.pkl'

    data = load_data(data_path)


    scorer1 = SaliencyScorer(classifier)
    extractor1 = TopKThresholder(1)
    # extractor2 = ContiguousThresholder(0.0001)


    for i in range(5):
        print("Attention - TopK:")
        evaluator = Evaluator(scorer1, extractor1, data)
        evaluator.shuffle()
        extractor1._max_length_ratio = 1
        print(evaluator.evaluate())

        extractor1._max_length_ratio = 2
        print(evaluator.evaluate())

        extractor1._max_length_ratio = 3
        print(evaluator.evaluate())
        
        extractor1._max_length_ratio = 4
        print(evaluator.evaluate())

        extractor1._max_length_ratio = 5
        print(evaluator.evaluate())

        extractor1._max_length_ratio = 6
        print(evaluator.evaluate())

        extractor1._max_length_ratio = 7
        print(evaluator.evaluate())

        extractor1._max_length_ratio = 8
        print(evaluator.evaluate())



        # print("Attention - Contiguous:")
        # evaluator = Evaluator(scorer1, extractor2, data)
        # print(evaluator.evaluate())



if __name__ == '__main__':
    main()
