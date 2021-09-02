import random
import numpy as np
import copy
import pickle
import torch
import matplotlib.pyplot as plt
from matplotlib import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_pkl(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def normalize_attentions(attentions, starting_offsets, ending_offsets):
    cum_attn = attentions.cumsum(1)
    start_v = cum_attn.gather(1, starting_offsets - 1)
    end_v = cum_attn.gather(1, ending_offsets - 1)
    return end_v - start_v


def get_start_idxs_batched(b_tokens, b_sub_tokens, bpe_indicator='Ġ'):
    
    # print(b_sub_tokens[0])

    from copy import deepcopy as cp
    ss = []
    es = []
    batch_size = len(b_tokens)
    for i in range(batch_size):
        starts = [1]
        starts += list(filter(lambda j: b_sub_tokens[i][j][0] == bpe_indicator, range(len(b_sub_tokens[i]))))
        # assert len(starts) == len(b_tokens[i]), print(len(starts), len(b_tokens[i]), b_tokens[i], b_sub_tokens[i])
        if len(starts) != len(b_tokens[i]):
            print("Warning in get_start_idxs_batched.")
            return -1, -1
        ends = cp(starts)
        ends.append(len(b_sub_tokens[i]) - 1)
        ends = ends[1:]
        # starts = [0] + starts + [len(b_sub_tokens[i]) - 1]
        # ends = [1] + ends + [len(b_sub_tokens[i])]
        ends = list(filter(lambda x: x <= 512 - 2, ends))       # less than 512, remove CLS SEP
        ss.append(starts[:len(ends)])
        es.append(ends)
    return ss, es


def rainbow_text(x, y, strings, colors, bgcolors, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas
    for s, c, bgc in zip(strings, colors, bgcolors):
        text = ax.text(x, y, s+" ", color=c, backgroundcolor=bgc,
                       transform=t, **kwargs)
        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(
            text.get_transform(), x=ex.width, units='dots')


def gen_score_img(raw, score, label, path=None):
    
    plt.figure()
    plt.axis('off')
    size = 12
    ntab = 0
    nline = 1
    sep = 0.105
    xinit = 0.02
    yinit = 0.98
    rainbow_text(xinit, yinit, ["Class %d" % (label)],
                    ["black"], ["white"], size=size)
    words, colors, bgcolors = [], [], []
    nonewline, nbrace, flush = False, 0, False
    assert len(raw) == len(score)
    for t, h, i in zip(raw, score, range(len(score))):
        if flush:
            flush = False
            if t != "{":
                rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                             colors, bgcolors, size=size)
                nline += 1
                words, colors, bgcolors = [], [], []
        words.append(t)
        bgc = 1-float(h)
        bgcolors.append((bgc, bgc, bgc))
        if bgc > 0.5:
            colors.append((0, 0, 0))
        else:
            colors.append((1, 1, 1))
        if t == '}':
            ntab -= 1
            rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                         colors, bgcolors, size=size)
            nline += 1
            words, colors, bgcolors = [], [], []
        elif t == "{":
            rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                         colors, bgcolors, size=size)
            ntab += 1
            nline += 1
            words, colors, bgcolors = [], [], []
        elif t == ";" and (not nonewline):
            rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                         colors,  bgcolors, size=size)
            nline += 1
            words, colors, bgcolors = [], [], []
        elif t in ["for", "while", "if"]:
            nonewline = True
            nbrace = 0
        elif t == "(" and nonewline:
            nbrace += 1
        elif t == ")" and nonewline:
            nbrace -= 1
            if nbrace <= 0:
                flush = True
                nonewline = False
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
    plt.close()



    
def gen_masked_seq(tokens, idxs, n_mask, mask="<mask>"):
    
    ret = copy.deepcopy(tokens)
    for t, i in zip(ret, idxs):
        if i < len(t):
            t.pop(i)
        for j in range(n_mask):
            t.insert(i, mask)
    return ret

def get_mask_idx(subtokens, mask="<mask>"):
    
    ret = []
    for t in subtokens:
        ret.append([])
        for i in range(len(t)):
            if t[i] == mask:
                ret[-1].append(i)
    return ret

def get_masked_logits(logits, idxs):
    
    idxs = torch.stack([torch.tensor(idxs) for _ in range(logits.shape[2])], 2).to(logits.device)
    return torch.gather(logits, 1, idxs)

def merge_top_k(logits, k, tokenizer, bpe_indicator='Ġ'):
    
    logits, idxs = torch.sort(logits, -1, True)
    # Logits.shape = [batch_size, n_mask, vocab_size]
    
    probs, tokens = None, None
    for l, idx in zip(logits.unbind(1), idxs.unbind(1)):
        # l.shape = [batch_size, vocab_size]
        tmp_probs, tmp_tokens = [], []
        for i in range(l.shape[0]):
            tmp_probs.append([])
            tmp_tokens.append([])
            for j in range(l.shape[1]):
                subtoken = tokenizer.convert_ids_to_tokens(idx[i, j].cpu().item())
                if (probs is not None and subtoken[0] != bpe_indicator) or ((probs is None) and subtoken[0] == bpe_indicator):
                    tmp_probs[-1].append(l[i, j])
                    tmp_tokens[-1].append(subtoken)
                    if len(tmp_probs[-1]) >= k:
                        break
            assert len(tmp_probs[-1]) == k
        tmp_probs = torch.tensor(tmp_probs)
        # The first subtoken -- without merging
        if probs is None:
            probs = tmp_probs
            # probs.shape = [batch_size, k]
            tokens = tmp_tokens
        # Not the first subtoken -- merging
        else:
            new_probs = torch.reshape(probs.unsqueeze(1) + tmp_probs.unsqueeze(2), [probs.shape[0], -1])
            # new_probs.shape = [batch_size, k * k]
            new_probs, new_idxs = torch.topk(new_probs, k, -1)
            # new_probs.shape = [batch_size, k]
            # new_idxs are the combined indices (i-th & j-th) -- i * k + j
            new_tokens = []
            for i in range(new_idxs.shape[0]):
                new_tokens.append([])
                for j in range(new_idxs.shape[1]):
                    new_tokens[-1].append(tokens[i][new_idxs[i][j] // k] + tmp_tokens[i][new_idxs[i][j] % k])
            tokens = new_tokens
            probs = new_probs
            
    return tokens