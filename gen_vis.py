from os import name
import torch
from models.codebert import codebert_cls
from scorer.base_scorer import SaliencyScorer
from scorer.gradient_scorer import GradientSaliency
from utils import gen_score_img

import os
import pickle

def main():
    gpu_num = -1
    model_path = "./save/java-classifier4/checkpoint-36000-0.9365"
    data_path = '../bigJava/datasets/test.pkl'
    output_path = './pics/'
    output_name = 'attn'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, output_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    if gpu_num < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu_num)
    cls_model = codebert_cls(model_path, device)
    # scorer = GradientSaliency(cls_model)
    scorer = SaliencyScorer(cls_model)

    N = 100
    for i in range(N):
        inputs = [test_data['norm'][i]]
        label = test_data['label'][i]
        attentions = scorer(inputs)
        path = os.path.join(output_path, str(i))
        l = min(len(inputs[0], len(attentions[0])))
        gen_score_img(inputs[0][:l], attentions[0][:l], label, path)

if __name__ == '__main__':
    main()