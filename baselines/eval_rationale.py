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


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    device = torch.device("cuda", 0)

    vocab_size = 30000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    n_channel = -1
    n_class_dict = {"JAVA": 2}
    n_class = n_class_dict["JAVA"]
    max_len = 400
    bidirection = True

    enc = LSTMEncoder(embedding_dim=embedding_size, hidden_dim=hidden_size,
                    n_layers=n_layers, drop_prob=0, brnn=bidirection)
    classifier = LSTMClassifier(vocab_size=vocab_size, encoder=enc,
                    num_class=n_class, device=device).to(device)
    classifier.eval()
    # classifier = myDataParallel(classifier).to(device)

    scorer1 = SaliencyScorer(classifier)
    extractor1 = TopKThresholder(0.05)
    extractor2 = ContiguousThresholder(0.05)

    data_path = "../../bigJava/datasets/test_tp_lstm.pkl"
    
    data = load_data(data_path)

    print("Attention - TopK:")
    evaluator = Evaluator(scorer1, extractor1, data)
    print(evaluator.evaluate())

    print("Attention - Contiguous:")
    evaluator = Evaluator(scorer1, extractor2, data)
    print(evaluator.evaluate())



if __name__ == '__main__':
    main()