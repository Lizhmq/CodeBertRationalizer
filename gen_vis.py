from os import name
import torch
import numpy as np
from models.codebert import codebert_cls
from scorer.base_scorer import SaliencyScorer
from scorer.gradient_scorer import GradientSaliency
from utils import gen_score_img

import os
import pickle

def main():
    gpu_num = -1
    model_path = "./save/java-classifier2/checkpoint-20000-0.9311"
    data_path = '../bigJava/datasets/test.pkl'
    output_path = './pics/'
    output_name = 'grad'
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
    cls_model = codebert_cls(model_path, device)
    cls_model.model = cls_model.model.to(device)
    scorer = GradientSaliency(cls_model)
    # scorer = SaliencyScorer(cls_model)

    N = 100
    for i in range(N):
        inputs = [test_data['norm'][i]]
        label = test_data['label'][i]
        span = test_data['span'][i]
        predicted_label = int(cls_model.run_info(inputs)['predicted_labels'][0])
        if label != predicted_label:
            print("Skip input %d, wrong prediction." % (i))
        attentions = scorer(inputs)
        path = os.path.join(output_path, str(i))
        l = min(len(inputs[0]), len(attentions[0]))
        gen_score_img(inputs[0][:l], attentions[0][:l], label, path)
        attentions[0] = np.array([0.] * l)
        if label > 0:
            attentions[0][span[0]:span[1]] = np.array([1.] * (span[1] - span[0])) / (span[1] - span[0])
            gen_score_img(inputs[0][:l], attentions[0][:l], label, path + "_truth")

if __name__ == '__main__':
    main()