import torch
import torch.nn as nn
import math
import random
from tqdm import tqdm
from copy import deepcopy as cp
from utils import gettensor, autopad, get_java


class Evaluator(object):

    def __init__(self, scorer, extractor, dataset):
        self.scorer = scorer
        self.dataset = dataset
        self.extractor = extractor
        self.java = get_java()

    def shuffle(self):
        idxs = list(range(len(self.dataset["norm"])))
        random.shuffle(idxs)
        self.dataset["norm"] = [self.dataset["norm"][i] for i in idxs]
        self.dataset["span"] = [self.dataset["span"][i] for i in idxs]
        self.dataset["idx"] = [self.dataset["idx"][i] for i in idxs]
        

    def evaluate(self):
        test_num = 0
        batch_size = 32
        ls = len(self.dataset["norm"])
        batch_num = math.ceil(ls / batch_size) // 7
        hit, hitexp, iou = 0, 0, 0.0
        
        java = self.java
        txt2idx = java.get_txt2idx()
        vocab_size = java.get_vocab_size()
        idx_func = lambda x: list([txt2idx["<unk>"] if tok >= vocab_size else tok for tok in java.raw2idxs(x) ])

        for i in tqdm(range(batch_num)):
            batch_in = self.dataset["norm"][i*batch_size:(i+1)*batch_size]
            span_in = self.dataset["span"][i*batch_size:(i+1)*batch_size]
            idx_in = self.dataset["idx"][i*batch_size:(i+1)*batch_size]
            size = len(batch_in)

            batch_in, ls = autopad(batch_in, maxlen=512)
            batch_in = list(map(idx_func, batch_in))
            y = [1] * len(ls)
            dic = {"x": batch_in, "y": y, "l": ls}
            inputs, labels, lens = gettensor(dic, self.scorer.model)

            scores = self.scorer(inputs, lens)
            rationales = self.extractor(scores)

            test_num += size
            for j in range(size):
                dic = self.metrics(span_in[j], idx_in[j], rationales[j])
                hit += dic["Hit"]
                hitexp += dic["Hitexp"]
                iou += dic["IOU"]
            # break
        hit = hit / test_num
        hitexp = hitexp / test_num
        iou = iou / test_num

        return {"Hit": hit, "Hitexp": hitexp, "IOU": iou}

    def metrics(self, oracle_span, oracle_idx, rationale):
        output_dict = {}
        output_dict["Hit"] = False
        output_dict["Hitexp"] = False
        output_dict["IOU"] = 0
        
        rationale = [k["span"] for k in rationale]
        oracle_set = set().union(range(*oracle_span))
        assert(oracle_idx in oracle_set)
        all_set = cp(oracle_set)
        inters_set = set()
        hit = False
        hitexp = False
        for span in rationale:
            new_set = set().union(range(*span))
            all_set = all_set.union(new_set)
            cur_inters = oracle_set.intersection(new_set)
            inters_set = inters_set.union(cur_inters)
            if oracle_idx in new_set:
                hit = True
            if len(cur_inters) > 0:
                hitexp = True
        
        output_dict["Hit"] = hit
        output_dict["Hitexp"] = hitexp
        output_dict["IOU"] = len(inters_set) / len(all_set)
        
        return output_dict
