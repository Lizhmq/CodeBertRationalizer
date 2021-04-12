import os
import torch
import numpy as np
import pickle
from sklearn.metrics import precision_recall_fscore_support
from models.codebert import codebert_mlm, codebert_cls
from tqdm import tqdm

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    device = torch.device("cuda", 2)
    # cls_model = codebert_cls("./save/calls-128-64/checkpoint-44000-0.9401", device)
    # cls_model = codebert_cls("./save/varmis-512/checkpoint-36000-0.9514", device, attn_head=-1)
    # cls_model = codebert_cls("./save/java0/checkpoint-16000-0.9311", device, attn_head=-1)
    # cls_model = codebert_cls("./save/python-op2/checkpoint-38000-0.9305", device, attn_head=-1)
    cls_model = codebert_cls("./save/python-op2/checkpoint-12000-0.9395", device, attn_head=-1)
    cls_model.model = cls_model.model.to(device)
    cls_model.model.eval()
    # data_path = "../DeepBugs/data/pkl/calls/test.pkl"
    data_path = "../CuBert/wrong_op/test.pkl"
    # data_path = "../great/test.pkl"
    # data_path = "../bigJava/datasets/test.pkl"
    batch_size = 12
    data = load_data(data_path)


    test_size = len(data["norm"])
    import math
    batch_num = math.ceil(test_size / batch_size) // 10


    test_num = 0
    test_true = 0
    predict = []
    label = []

    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            batch_in = data["norm"][i * batch_size:(i + 1) * batch_size]
            batch_in = [" ".join(tokens) for tokens in batch_in]
            batch_out = data["label"][i * batch_size:(i + 1) * batch_size]
            batch_size = len(batch_in)
            predicted = cls_model.run(batch_in, batch_size=batch_size).cpu().data.numpy()
            predicted = np.argmax(predicted, axis=1)
            test_num += batch_size
            test_true += np.sum(predicted == batch_out)
            predict += list(predicted)
            label += list(batch_out)
    precision, recall, _, _ = precision_recall_fscore_support(label, predict)
    precision, recall = precision[1], recall[1]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)     # prevent from zero division
    print("Precision: %.4f\nRecall: %.4f\nF1: %.4f\n" % (precision, recall, f1))
    print("Accuracy: %.4f\n" % (test_true / test_num))
    print("Test true: %d, Test num: %d" % (test_true, test_num))

if __name__ == '__main__':
    main()