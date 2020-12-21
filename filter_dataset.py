import torch
import numpy as np
from models.codebert import codebert_cls

import pickle
from tqdm import tqdm

def main():
    gpu_num = 0
    model_path = "./save/java-new/checkpoint-39000-0.9505"
    data_path = '../bigJava/datasets/test.pkl'
    out_path = '../bigJava/datasets/test_tp_ts.pkl'

    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    if gpu_num < 0:
        device = torch.device("cpu")        # cpu supported only now
    else:
        device = torch.device("cuda", gpu_num)
    cls_model = codebert_cls(model_path, device)
    cls_model.model = cls_model.model.to(device)

    raws, norms, idxs, spans, labels = [], [], [], [], []
    N = len(test_data["norm"])
    # for i in tqdm(range(10)):
    for i in tqdm(range(N)):
        inputs = [test_data['norm'][i]]
        label = test_data['label'][i]
        span = test_data['span'][i]
        predicted_label = int(cls_model.run_info(inputs)['predicted_labels'][0])
        if label == predicted_label and label > 0:
            raws.append(test_data["raw"][i])
            norms.append(test_data["norm"][i])
            idxs.append(test_data["idx"][i])
            spans.append(test_data["span"][i])
            labels.append(test_data["label"][i])
    out_dict = {
        "raw": raws,
        "norm": norms,
        "idx": idxs,
        "span": spans,
        "label": labels
    }
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)

if __name__ == '__main__':
    main()