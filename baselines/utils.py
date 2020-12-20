import torch
import torch.nn as nn
import numpy as np
from models.Transformer import TransformerClassifier


class myDataParallel(nn.DataParallel):
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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
