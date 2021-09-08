import os
import torch
import numpy as np
import pickle
import random
from models.codebert import codebert_mlm, codebert_cls
from tqdm import tqdm
from scorer.base_scorer import SaliencyScorer
from scorer.gradient_scorer import GradientSaliency
from scorer.integrad_scorer import IntegradSaliency
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
    set_seed(2333)
    # device = torch.device("cpu")
    device = torch.device("cuda", 2)
    # model_path = "./save/java-head-1/checkpoint-40000-0.9328"
    # model_path = "./save/varmis-head-1/checkpoint-176000-0.9568"
    model_path = "./save/pyop-head-1/checkpoint-28000-0.9126"
    cls_model = codebert_cls(model_path, device, attn_head=-1)

    cls_model.model = cls_model.model.to(device)
    cls_model.model.eval()

    scorer1 = SaliencyScorer(cls_model)
    scorer2 = GradientSaliency(cls_model)
    scorer3 = IntegradSaliency(cls_model)
    extractor1 = TopKThresholder(0.0001)
    # extractor1 = ContiguousMaskThresholder(0.0001)
    # extractor2 = ContiguousThresholder(0.0001)

    # data_path = "../well_datasets/javaop/testtp.pkl"
    # data_path = "../well_datasets/great/testtp.pkl"
    data_path = "../well_datasets/pyop/testtp28.pkl"


    data = load_data(data_path)
    run = "attention"
    sample_num = 2000

    if run == "attention":
        # idxs = [-1] + list(range(12))
        idxs = [-1, 3, 5, 8]
        ls = list(range(len(data["norm"])))
        random.shuffle(ls)
        ls = ls[:sample_num]
        for iterations in range(1):         # repeat to counteract randomness
            for idx in idxs:
                processed = 0
                cls_model.attn_head = idx
                print(idx)
                print("Attention - TopK:", sep="\t")
                pos = 0
                print("Test Dataset Size: %d" % (len(ls)))
                for i in tqdm(ls):
                    if data["label"][i] == 0:
                        continue
                    input = [data["norm"][i]]
                    output_dict = cls_model.run_info(input)
                    if output_dict["predicted_labels"][0] == 0:
                        continue
                    processed += 1
                    scores = scorer1(input)
                    # cands = data["candidates"][i]
                    # rationale = extractor1(scores, [cands])[0]     # for mask selection
                    rationale = extractor1(scores)[0]
                    rationale = [k["span"] for k in rationale]
                    if data["idx"][i] in range(*rationale[0]) or data["idx"][i] - 1 in range(*rationale[0]):
                        pos += 1
                    # see is not???
                    # if "wrong_op" in data_path:
                    #     if data["idx"][i] in range(*rationale[0]) or (data["idx"][i] + 1) in range(*rationale[0]):
                    #         pos += 1
                print(pos / processed)
            print("\n\n\n")
    if run == "gradient" or run == "integrad":
        if run == "gradient":
            scorer = scorer2
            print("Gradient - TopK:")
        else:
            scorer = scorer3
            print("Integral Gradient - TopK:")
        ls = list(range(len(data["norm"])))
        random.shuffle(ls)
        ls = ls[:sample_num]
        pos = 0
        print("Test Dataset Size: %d" % (len(ls)))
        for i in tqdm(ls):
            input = [data["norm"][i]]
            scores = scorer(input)
            rationale = extractor1(scores)[0]
            rationale = [k["span"] for k in rationale]
            if data["idx"][i] in range(*rationale[0]):
                pos += 1
        print(pos / len(ls))
        
    # evaluator = Evaluator(scorer, extractor1, data)
    # print(evaluator.evaluate())



if __name__ == '__main__':
    main()