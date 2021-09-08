#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import math
import os, json
import numpy as np
from models.nblmodel import NBLModel, build_model
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer

import pickle
from tqdm import tqdm
from copy import deepcopy as cp
from utils import normalize_attentions, helper
from copy import deepcopy as cp


# In[2]:


model_path = "./save/nbl/checkpoint-8000-0.7714"
data_path = "../c-treesitter/nbl"
out_path = "../c-treesitter/nbl/testft.json"


# In[3]:


data = []
with open(os.path.join(data_path, "test.json"), "r") as f:
	for line in f.readlines():
		data.append(json.loads(line))
print(len(data))
# pids = set()
# for dic in data:
# 	pids.add(dic["id"])
# print(len(pids))


# In[5]:


gpu_num = 0
if gpu_num < 0:
	device = torch.device("cpu")
else:
	device = torch.device("cuda", gpu_num)
class Args:
	def __init__(self) -> None:
		self.device = None
		self.pretrain_dir = None
args = Args()
args.device = device
args.pretrain_dir = model_path
model = build_model(args, model_path)
# args.pretrain_dir = "microsoft/codebert-base"
# model = build_model(args)
model.to(device)
model.eval()
tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_dir)


# In[9]:


with open(os.path.join(data_path, "line.pkl"), "rb") as f:
	label2line = pickle.load(f)
with open(os.path.join("../nbl/prepare_dataset/id2prog.pkl"), "rb") as f:
	id2prog = pickle.load(f)
with open(os.path.join("../nbl/prepare_dataset/err2corr.pkl"), "rb") as f:
	err2corr = pickle.load(f)
with open(os.path.join(data_path, "tidmap.json"), "r") as f:
	mapdic = json.loads(f.read())


# In[6]:


save_dic, save_pred, save_prob = dict(), dict(), dict()
neg_progs = set()
for __i in tqdm(range(len(data))):
# for __i in range(1000):
	prog_id = data[__i]["id"]
	test_id = data[__i]["tid"]
	y = data[__i]["y"]
	if y == 0:
		neg_progs.add(prog_id)
	code_ids, tid, starts, ends = helper(data[__i], tokenizer, mapdic, device)
	
	output_dict = model.run_info(code_ids, code_ids.ne(tokenizer.pad_token_id).to(code_ids), tid)
	attns = output_dict["attentions"]
	pred = int(output_dict["predictions"][0])
	probs = output_dict["probs"].detach().cpu().numpy()[0]
	scores = normalize_attentions(attns, starts, ends).detach().cpu().numpy()[0]
	if prog_id not in save_pred:
		save_pred[prog_id] = dict()
	save_pred[prog_id][test_id] = pred
	if prog_id not in save_dic:
		save_dic[prog_id] = dict()
	save_dic[prog_id][test_id] = scores
	if prog_id not in save_prob:
		save_prob[prog_id] = dict()
	save_prob[prog_id][test_id] = probs


# In[11]:


# thred = 0.9
# golds, preds = [], []
# for dic in data:
# 	prog_id = dic["id"]
# 	test_id = dic["tid"]
# 	y = dic["y"]
# 	pred = int(save_prob[prog_id][test_id][1] > thred)
# 	golds.append(y)
# 	preds.append(pred)
# print(classification_report(golds, preds, digits=4))


# In[12]:


print(len(neg_progs))
print(len(save_dic))
for prog_id in save_dic:
	if prog_id not in neg_progs:
		del save_dic[prog_id]
print(len(save_dic))


# In[7]:


target = dict()
for pid in label2line:
	if pid not in target:
		target[pid] = set()
	for tid, lines in label2line[pid].items():
		target[pid] = target[pid].union(lines)


# In[8]:


thred = 0.5
keepids = set()
for prog_id in save_prob:
	for tid, vals in save_prob[prog_id].items():
		if vals[1] < thred:
			keepids.add(prog_id)
print(len(save_dic))
print(len(keepids))


# In[11]:


import random
testids = random.sample(list(neg_progs), 1449)


# In[12]:


k_val = {1: 0, 5: 0, 10: 0}
for prog_id in testids:
	lst = []
	for tid in save_dic[prog_id]:
		lst.append(save_dic[prog_id][tid])
	if len(lst) == 0:
		continue
	lst = np.mean(lst, axis=0)
	idxs = np.argsort(lst)
	for k in (1, 5, 10):
		cands = idxs[-k:]
		cands = [v + 1 for v in cands]	# cands index from 0 and targets index from 1
		if set(cands).intersection(target[prog_id]):
			k_val[k] += 1
print(k_val)
for v in k_val:
	print(k_val[v] / len(testids))

