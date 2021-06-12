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
from utils import gen_masked_seq, get_mask_idx, get_masked_logits, merge_top_k


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def fill_mask(new_s, idxs, mlm_model, mask_num=1, topk=5):
    masked_seqs = gen_masked_seq(new_s, idxs=idxs, n_mask=mask_num)
    masked_strs = [" ".join(seq) for seq in masked_seqs]
    subs = mlm_model.tokenize(masked_strs, cut_and_pad=True, ret_id=True)
    masked_idxs = get_mask_idx(subs, mask=mlm_model.tokenizer.mask_token_id)
    logits = mlm_model.run(masked_strs)
    masked_logits = get_masked_logits(logits, masked_idxs)
    topk = merge_top_k(masked_logits, topk, mlm_model.tokenizer)
    return topk

def f_op(op):
    dic = {
        ">=": ">",
        "<=": "<",
        ">": ">=",
        "<": "<=",
    }
    return dic[op]

def dmap(tok):
    dic = {"<int>": "0", "<str>": '"str"', "<char>": "'c'", "<fp>": "1.0"}
    if tok in dic:
        return dic[tok]
    return tok


def main():
    set_seed(2333)
    # model_path1 = "./save/java0/checkpoint-16000-0.9311"
    model_path1 = "./save/varmis-512/checkpoint-36000-0.9514"
    model_path = "/var/data/zhanghz/codebert-base-mlm"
    device = torch.device("cuda", 2)
    cls_model = codebert_cls(model_path1, device, attn_head=6)
    cls_model.model = cls_model.model.to(device)
    cls_model.model.eval()
    mlm_model = codebert_mlm(model_path, device)
    mlm_model.model = mlm_model.model.to(device)
    mlm_model.model.eval()

    scorer = SaliencyScorer(cls_model)
    extractor = TopKThresholder(0.0001)

    # data_path = "../bigJava/datasets/test_tp_ts.pkl"
    data_path = "../great/test_tp_36.pkl"

    data = load_data(data_path)

    idx = -1
    cls_model.attn_head = idx
    print(idx)
    ls = list(range(len(data["norm"])))
    random.shuffle(ls)
    tot_l = 100
    ls = ls[:tot_l]
    pos, fix = 0, 0

    print("Test Dataset Size: %d" % (len(ls)))
    for i in tqdm(ls):
        input = [data["norm"][i]]
        scores = scorer(input)
        
        rationale = extractor(scores)[0]

        rationale = [k["span"] for k in rationale][0]
        if data["idx"][i] in range(*rationale):
            pos += 1
            masked_ = list(data["norm"][i])
            masked_[data["idx"][i]] = "<mask>"
            masked_ = list(map(dmap, masked_))
            predicted = []
            for k in range(1, 6):
                cands = fill_mask([masked_], [data["idx"][i]], mlm_model, mask_num=k)
                cands = cands[0][:k]     # first 0 for batch
                predicted.extend(cands)
            # targets = [f_op(data["norm"][i][rationale[0]])]
            # targets = ['Ġ' + data["norm"][i][IDX] for IDX in data["targets"][i]]
            if len(data["targets"][i]) == 0 or data["targets"][i][0] >= len(data["norm"][i]):
                targets = None
            else:
                targets = 'Ġ' + data["norm"][i][data["targets"][i][0]]
            print(predicted)
            print(targets)
            if targets in predicted:
                fix += 1
                # print("fix")

        else:
            pass
            # print(f"Wrong loc: sample index {i}.")
    print(f"Loc acc: {pos / tot_l}")
    print(f"Fix acc: {fix / tot_l}")


if __name__ == '__main__':
    main()