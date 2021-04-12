import torch
import numpy as np

from rationale.contiguous import ContiguousThresholder
from rationale.topk import TopKThresholder
from rationale.scorer import SaliencyScorer
from models.lstm_cls import LSTMClassifier, LSTMEncoder
from dataset import Dataset, Java
from utils import gen_score_img, gettensor, autopad, get_java

import os
import pickle


def main():
    gpu_num = -1
    model_path = "./save/lstm/11.pt"
    data_path = '../../bigJava/datasets/test_tp_lstm.pkl'
    output_path = '.././pics/'
    output_name = 'attn-lstm'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, output_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    if gpu_num < 0:
        device = torch.device("cpu")        # cpu supported only now
    else:
        device = torch.device("cuda", gpu_num)
    
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
    classifier.load_state_dict(torch.load(model_path))
    classifier = classifier.to(device)
    classifier.eval()

    scorer = SaliencyScorer(classifier)
    # scorer = SaliencyScorer(cls_model)
    # thresholder = ContiguousThresholder(0.1)
    thresholder = ContiguousThresholder(0.1)

    java = get_java()
    txt2idx = java.get_txt2idx()
    vocab_size = java.get_vocab_size()
    idx_func = lambda x: list([txt2idx["<unk>"] if tok >= vocab_size else tok for tok in java.raw2idxs(x) ])

    # N = 100
    N = len(test_data["norm"])
    for i in range(N):
        inputs = [test_data['norm'][i]]
        label = test_data['label'][i]
        idx = test_data["idx"][i]
        span = test_data['span'][i]
        
        batch_in, ls = autopad(inputs, maxlen=100)
        batch_in = list(map(idx_func, batch_in))
        y = [1] * len(ls)
        dic = {"x": batch_in, "y": y, "l": ls}
        inputs, labels, lens = gettensor(dic, classifier)

        attentions = scorer(inputs, lens)
        rationale = thresholder.forward(attentions)
        # print(span, idx, rationale[0])
        for dic in rationale[0]:
            span__ = dic["span"]
            inputs = [test_data['norm'][i]]
            print(inputs[0][span__[0]:span__[1]], sep=" ")
        
        path = os.path.join(output_path, str(i))
        l = min(len(inputs[0]), len(attentions[0]))
        gen_score_img(inputs[0][:l], attentions[0][:l], label, path)
        attentions[0] = np.array([0.] * l)
        if label > 0:
            attentions[0][span[0]:span[1]] = np.array([1.] * (span[1] - span[0])) / (span[1] - span[0])
            gen_score_img(inputs[0][:l], attentions[0][:l], label, path + "_truth")


if __name__ == '__main__':
    main()