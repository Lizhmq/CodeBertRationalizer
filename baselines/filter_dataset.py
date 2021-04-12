import torch
import numpy as np

import pickle
from tqdm import tqdm
from models.lstm_cls import LSTMEncoder, LSTMClassifier
from utils import myDataParallel, gettensor
from dataset import Dataset, Java


def main():
    gpu_num = 1
    _data = "PY_OP"
    model_path = "./save/py-op/9.pt"
    data_path = '../../CuBert/wrong_op/Python.pkl'
    out_path = '../../CuBert/wrong_op/test_tp_lstm.pkl'

    if gpu_num < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu_num)
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    test_set = data.test

    vocab_size = 30000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    n_channel = -1
    n_class_dict = {"JAVA": 2, "PY150": 2, "PY_OP": 2}
    n_class = n_class_dict[_data]
    max_len = {"JAVA": 400, "PY150": 250, "PY_OP": 512}[_data]
    bidirection = True

    enc = LSTMEncoder(embedding_dim=embedding_size, hidden_dim=hidden_size,
                    n_layers=n_layers, drop_prob=0, brnn=bidirection)
    classifier = LSTMClassifier(vocab_size=vocab_size, encoder=enc,
                    num_class=n_class, device=device).to(device)
    classifier.load_state_dict(torch.load(model_path))
    classifier.to(device)
    classifier.eval()
    # classifier = myDataParallel(classifier).to(device)

    example_num = 0
    batch = test_set.next_batch(1)
    while not batch["new_epoch"]:
        example_num += 1
        batch = test_set.next_batch(1)

    raws, norms, idxs, spans, labels = [], [], [], [], []
    for i in tqdm(range(example_num)):
    # for i in tqdm(range(5)):
        batch = test_set.next_batch(1)
        inputs, labs, lens = gettensor(batch, classifier)
        output = classifier(inputs, lens).cpu().data.numpy()
        predicted_label = np.argmax(output, 1)
        if labs.cpu().data[0] == predicted_label[0] and predicted_label[0] == 1:
            raws.append(batch["raw"][0])
            norms.append(batch["norm"][0])
            idxs.append(batch["pos"][0])
            spans.append(batch["span"][0])
            labels.append(batch["y"][0])

    out_dict = {
        "raw": raws,
        "norm": norms,
        "idx": idxs,
        "span": spans,
        "label": labels
    }
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)

    print("Original size: %d" % (test_set.get_size()))
    print("Current size: %d" % (len(out_dict["idx"])))
    print(len(out_dict["idx"]) / test_set.get_size())

if __name__ == '__main__':
    main()