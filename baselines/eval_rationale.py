import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import torch
import numpy as np
import pickle
from models.codebert import codebert_mlm, codebert_cls
from tqdm import tqdm
from scorer.base_scorer import SaliencyScorer
from scorer.gradient_scorer import GradientSaliency
from thretholder.contiguous import ContiguousThresholder
from thretholder.topk import TopKThresholder
from scorer.evaluator import Evaluator


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    # device = torch.device("cuda", 0)
    device = torch.device("cuda", 0)
    cls_model = codebert_cls("./save/java0/checkpoint-16000-0.9311", device)
    cls_model.model = cls_model.model.to(device)
    cls_model.model.eval()

    scorer1 = SaliencyScorer(cls_model)
    scorer2 = GradientSaliency(cls_model)
    extractor1 = TopKThresholder(0.05)
    extractor2 = ContiguousThresholder(0.05)

    data_path = "../bigJava/datasets/valid_tp.pkl"
    
    data = load_data(data_path)

    print("Attention - TopK:")
    evaluator = Evaluator(scorer1, extractor1, data)
    print(evaluator.evaluate())

    print("Attention - Contiguous:")
    evaluator = Evaluator(scorer1, extractor2, data)
    print(evaluator.evaluate())
    
    print("Gradient - TopK:")
    evaluator = Evaluator(scorer2, extractor1, data)
    print(evaluator.evaluate())

    print("Gradient - Contiguous:")
    evaluator = Evaluator(scorer2, extractor2, data)
    print(evaluator.evaluate())



if __name__ == '__main__':
    main()