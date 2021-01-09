import torch
import torch.nn as nn
import numpy as np
import pickle
import random
from models.Transformer import TransformerClassifier
from dataset import Dataset, Java

import matplotlib.pyplot as plt
from matplotlib import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class myDataParallel(nn.DataParallel):
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_java(java_path="../../bigJava/datasets/Java.pkl"):
    with open(java_path, "rb") as f:
        data = pickle.load(f)
    return data


def autopad(inputs, ls=None, maxlen=100):
    ls = list(map(len, inputs))
    maxl = max(ls)
    maxlen = max(maxl, maxlen)
    for i in range(len(inputs)):
        inputs[i] += ["<pad>"] * (maxlen - ls[i])
    return inputs, ls


def gettensor(batch, model):
    ''' Batch second '''
    device = model.classify.weight.device
    inputs, labels, lens = batch['x'], batch['y'], batch['l']
    if isinstance(model, nn.DataParallel):
        with_cls = isinstance(model.module, TransformerClassifier)
    else:
        with_cls = isinstance(model, TransformerClassifier)
    batch_first = with_cls
    if with_cls:    # add <CLS> to inputs
        # cls = vocab_size
        cls = model.model.embeddings.word_embeddings.weight.shape[0] - 1
        inputs = np.insert(inputs, 0, [cls] * inputs.shape[0], axis=1)
        lens = lens + 1     # numpy supported
    inputs, labels, lens = torch.tensor(inputs, dtype=torch.long).to(device), \
                           torch.tensor(labels, dtype=torch.long).to(device), \
                           torch.tensor(lens, dtype=torch.long).to(device)
    if not batch_first:
        inputs = inputs.permute([1, 0])
    return inputs, labels, lens

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