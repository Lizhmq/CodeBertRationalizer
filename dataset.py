# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import os, json
import pickle

import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from utils import get_start_idxs_batched


class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=512, split_rate=0.05):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_langs_%s"%(args.langs)+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels, self.idxs = datas["inputs"], datas["labels"], datas["idxs"]

        else:
            self.inputs = []
            self.labels = []
            self.idxs = []
            if file_type != "test":
                if file_type == "train":
                    datafile = os.path.join(args.data_dir, "train.pkl")
                else:
                    datafile = os.path.join(args.data_dir, "valid.pkl")
            else:
                datafile = os.path.join(args.data_dir, "test.pkl")
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
            datas = pickle.load(open(datafile, "rb"))
            labels = datas["label"]
            inputs = datas["norm"]
            # poss = datas["idx"]
            poss = datas["error"]

            # train_len = len(inputs)
            # split_l = int(train_len * split_rate)
            length = len(inputs)

            for idx, (data, label, pos) in enumerate(zip(inputs, labels, poss)):
                if idx % world_size == local_rank:
                    keeppos = pos
                    code = " ".join(data)
                    code_tokens = tokenizer.tokenize(code)
                    sts, eds = get_start_idxs_batched([data], [code_tokens])
                    if sts == -1:
                        continue
                    sts, eds = sts[0], eds[0]
                    if len(sts) <= pos:
                        st, ed = 0, 0
                        # print(code)
                        # print(code_tokens)
                        # print(pos)
                    else:
                        st, ed = sts[pos] + 1, eds[pos] + 1     # [CLS] inserted to the front, so add 1
                    pos = [0 for _ in range(block_size)]
                    if random.random() < args.prob:         # probability of using localization supervision
                        pos[st:ed] = [1 for _ in range(st, ed)]
                    code_tokens = code_tokens[:block_size-2]
                    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
                    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
                    padding_length = block_size - len(code_ids)
                    code_ids += [tokenizer.pad_token_id] * padding_length
                    # print(data)
                    # print(data[keeppos])
                    # print(code_tokens)
                    # print(pos)
                    # print("\n\n\n")
                    self.inputs.append(code_ids)
                    self.labels.append(label)
                    self.idxs.append(pos)

                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))

            if file_type == 'train':
                logger.warning("Rank %d Training %d samples"%(local_rank, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump({"inputs": self.inputs, "labels": self.labels, "idxs": self.idxs}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item]), torch.tensor(self.idxs[item])


class NBLDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_name='xxx.pkl', block_size=512):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_name[:-4] +"_blocksize_%d" %
                                   (block_size)+"_wordsize_%d" % (world_size)+"_rank_%d" % (local_rank))
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels, self.tids = datas["inputs"], datas["labels"], datas["tids"]
        else:
            self.inputs = []
            self.labels = []
            self.tids = []
            datafile = os.path.join(args.data_dir, file_name)
            datalst = []
            with open(datafile, "r") as f:
                for line in f.readlines():
                    datalst.append(json.loads(line))
            with open(os.path.join(args.data_dir, "tidmap.json"), "r") as f:
                mapdic = json.loads(f.read())
            
            inputs = [dic["x"] for dic in datalst]
            labels = [dic["y"] for dic in datalst]
            tids = [mapdic[dic["tid"]] for dic in datalst]

            length = len(inputs)

            for idx, (data, label, tid) in enumerate(zip(inputs, labels, tids)):
                if idx % world_size == local_rank:
                    code = " ".join(data)
                    code_tokens = tokenizer.tokenize(code)
                    code_tokens = code_tokens[:block_size-2]
                    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
                    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
                    padding_length = block_size - len(code_ids)
                    code_ids += [tokenizer.pad_token_id] * padding_length
                    # print(data)
                    # print(data[keeppos])
                    # print(code_tokens)
                    # print(pos)
                    # print("\n\n\n")
                    self.inputs.append(code_ids)
                    self.labels.append(label)
                    self.tids.append(tid)

                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))

            if 'train' in file_name:
                logger.warning("Rank %d Training %d samples"%(local_rank, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)

            with open(cached_file, 'wb') as handle:
                pickle.dump({"inputs": self.inputs, "labels": self.labels, "tids": self.tids}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item]), torch.tensor(self.tids[item])