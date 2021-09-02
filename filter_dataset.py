import torch
import math
import numpy as np
from models.codebert import codebert_cls
from sklearn.metrics import classification_report

import pickle
from tqdm import tqdm

def main():
    gpu_num = 2
    model_path = "./save/java1/checkpoint-20000-0.9243"
    # model_path = "./save/varmis-512/checkpoint-36000-0.9514"
    # model_path = "./save/python-op2/checkpoint-38000-0.9305"
    data_path = '../bigJava/datasets/test.pkl'
    # data_path = "../great/test.pkl"
    # data_path = "../CuBert/wrong_op/test.pkl"
    out_path = '../bigJava/datasets/new_test_tp_ts20.pkl'
    # out_path = "../great/test_ts_tp.pkl"
    # out_path = "../CuBert/wrong_op/test_tp_ts.pkl"

    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    if gpu_num < 0:
        device = torch.device("cpu")        # cpu supported only now
    else:
        device = torch.device("cuda", gpu_num)
    cls_model = codebert_cls(model_path, device)
    cls_model.model = cls_model.model.to(device)
    cls_model.block_size = 512

    ospans = []
    ofunctions = []
    oraws, onorms, oidxs, olabels, ocandidates, otargets = [], [], [], [], [], []
    N = len(test_data["norm"])
    batch_size = 16
    batch_num = math.ceil(N / batch_size)
    preds, golds = [], []
    for i in tqdm(range(batch_num)):
    # for i in tqdm(range(100)):
        inputs = test_data['norm'][i*batch_size:(i+1)*batch_size]
        raws = test_data['raw'][i*batch_size:(i+1)*batch_size]
        # errors = test_data['error'][i*batch_size:(i+1)*batch_size]
        errors = test_data['idx'][i*batch_size:(i+1)*batch_size]
        labels = test_data['label'][i*batch_size:(i+1)*batch_size]
        spans = test_data['span'][i*batch_size:(i+1)*batch_size]
        # functions = test_data['function'][i*batch_size:(i+1)*batch_size]
        # candidates = test_data['candidates'][i*batch_size:(i+1)*batch_size]
        # targets = test_data['targets'][i*batch_size:(i+1)*batch_size]

        predicted_labels = cls_model.run_info(inputs, batch_size=batch_size)['predicted_labels'].cpu().data.numpy()
        preds.extend(predicted_labels)
        golds.extend(labels)
        pred_true = predicted_labels == labels
        pos = predicted_labels > 0
        for j, boole in enumerate(np.logical_and(pred_true, pos)):
            if boole:
                oraws.append(raws[j])
                onorms.append(inputs[j])
                oidxs.append(errors[j])
                ospans.append(spans[j])
                olabels.append(labels[j])
                # ofunctions.append(functions[j])
                # ocandidates.append(candidates[j])
                # otargets.append(targets[j])
    out_dict = {
        "raw": oraws,
        "norm": onorms,
        "idx": oidxs,
        "span": ospans,
        "label": olabels,
        # "functions": ofunctions,
        # "candidates": ocandidates,
        # "targets": otargets
    }
    print(classification_report(golds, preds, digits=3))
    # java: raw, norm, idx, span, label
    # varmis: raw, norm, error, label, candidates, targets
    # python: raw, norm idx, span, label, functions
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)

if __name__ == '__main__':
    main()