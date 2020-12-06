import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import torch
import numpy as np
import pickle
from models.codebert import codebert_mlm, codebert_cls
from tqdm import tqdm

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    # device = torch.device("cuda", 0)
    device = torch.device("cuda", 1)
    cls_model = codebert_cls("./save/java-new/checkpoint-39000-0.9505", device)
    cls_model.model = cls_model.model.to(device)
    cls_model.model.eval()
    data_path = "../bigJava/datasets/valid.pkl"
    batch_size = 16
    data = load_data(data_path)


    test_size = len(data["norm"])
    import math
    batch_num = math.ceil(test_size / batch_size)


    test_num = 0
    test_true = 0

    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            batch_in = data["norm"][i * batch_size:(i + 1) * batch_size]
            batch_in = [" ".join(tokens) for tokens in batch_in]
            batch_size = len(batch_in)
            batch_out = data["label"][i * batch_size:(i + 1) * batch_size]
            predicted = cls_model.run(batch_in).cpu().data.numpy()
            predicted = np.argmax(predicted, axis=1)
            test_num += batch_size
            test_true += np.sum(predicted == batch_out)

    print("Acc:", test_true / test_num)


if __name__ == '__main__':
    main()