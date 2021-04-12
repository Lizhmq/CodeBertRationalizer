import os
import torch
import numpy as np
import pickle
import random
from models.codebert import codebert_mlm, codebert_cls
from tqdm import tqdm
from scorer.base_scorer import SaliencyScorer
from scorer.gradient_scorer import GradientSaliency
from thretholder.contiguous import ContiguousThresholder
from thretholder.contiguous_mask import ContiguousMaskThresholder
from thretholder.topk import TopKThresholder
from scorer.evaluator import Evaluator
from utils import set_seed

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    # set_seed(2333)
    device = torch.device("cuda", 3)
    # device = torch.device("cuda", 1)
    # cls_model = codebert_cls("./save/java0/checkpoint-28000-0.9366", device, attn_head=6)
    # cls_model = codebert_cls("./save/java0/checkpoint-16000-0.9311", device, attn_head=-1)
    cls_model = codebert_cls("./save/varmis-512/checkpoint-36000-0.9514", device, attn_head=-1)
    # cls_model = codebert_cls("./save/python-op2/checkpoint-38000-0.9305", device, attn_head=-1)
    cls_model.model = cls_model.model.to(device)
    cls_model.model.eval()

    scorer1 = SaliencyScorer(cls_model)
    # scorer2 = GradientSaliency(cls_model)
    extractor1 = TopKThresholder(0.0001)
    # extractor1 = ContiguousMaskThresholder(0.0001)
    # extractor2 = ContiguousThresholder(0.0001)

    # data_path = "../bigJava/datasets/test_tp_ts.pkl"
    # data_path = "../CuBert/wrong_op/test_tp_ts.pkl"
    data_path = "../great/test_tp_36.pkl"

    data = load_data(data_path)

    idxs = list(range(12))
    # idxs = [-1, 9]
    # idxs = [-1, 6]
    for iterations in range(5):
        print("\n\n\n")
        for idx in idxs:
            # idx = -1
            cls_model.attn_head = idx
            print(idx)
            print("Attention - TopK:", sep="\t")
            # evaluator = Evaluator(scorer1, extractor1, data)
            # print(evaluator.evaluate())
            ls = 2000
            ls = list(range(len(data["norm"])))
            random.shuffle(ls)
            ls = ls[:2000]
            pos = 0
            pos2 = 0
            pos3 = 0
            print("Test Dataset Size: %d" % (len(ls)))
            for i in tqdm(ls):
                input = [data["norm"][i]]
                scores = scorer1(input)
                
                # cands = data["candidates"][i]
                # rationale = extractor1(scores, [cands])[0]
                rationale = extractor1(scores)[0]
                
                # print(rationale)
                rationale = [k["span"] for k in rationale]
                # print(data["idx"][i])
                # print(len(cands))
                # assert(rationale[0][0] in cands)
                if data["idx"][i] in range(*rationale[0]):
                    pos += 1
                if data["idx"][i] in range(*rationale[0]) or (data["idx"][i] + 1) in range(*rationale[0]):
                    pos2 += 1
                # set1 = set(range(*data["span"][i]))
                # set2 = set(range(*rationale[0]))
                # if len(set1.intersection(set2)) > 0:
                #     pos3 += 1
                # else:
                    # print("Error idx: %d, wrong location: %d" % (i, rationale[0][0]))
            print(pos / len(ls), pos2 / len(ls))#, pos3 / len(ls))


    # print("Attention - Contiguous:")
    # # evaluator = Evaluator(scorer1, extractor2, data)
    # # print(evaluator.evaluate())
    # # ls = 10
    # ls = len(data["norm"])
    # pos = 0
    # for i in range(ls):
    #     input = [data["norm"][i]]
    #     print(data["raw"][i])
    #     scores = scorer1(input)
    #     rationale = extractor2(scores)[0]
    #     rationale = [k["span"] for k in rationale]
    #     if data["idx"][i] in range(*rationale[0]):
    #         pos += 1
    # print(pos / ls)
    # print("Gradient - TopK:")
    # evaluator = Evaluator(scorer2, extractor1, data)
    # print(evaluator.evaluate())

    # print("Gradient - Contiguous:")
    # evaluator = Evaluator(scorer2, extractor2, data)
    # print(evaluator.evaluate())



if __name__ == '__main__':
    main()