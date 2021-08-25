from utils import normalize_attentions
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy as cp


class IntegradSaliency(nn.Module):

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._model.model.eval()
        self.embedding_layer = model.model.roberta.embeddings
        self.grads = []
        self.embedded = None
        self.iters = 20


    def forward(self, inputs):
        self.grads, self.ret_tensors = [], []
        
        def backward_tensor_hook(grad):
            self.grads.append(grad)

        def get_forward_hook_func(alpha=1.):
            def forward_hook(module, input, embedded):
                if alpha == 0:
                    self.embedded = cp(embedded.detach())
                embedded = embedded * alpha
                embedded.register_hook(backward_tensor_hook)
                return embedded
            return forward_hook

        with torch.enable_grad():

            for i in range(self.iters):
                alpha = 1. * i / (self.iters - 1)
                forward_hook = get_forward_hook_func(alpha)
                forward_handle = self.embedding_layer.register_forward_hook(forward_hook)
                self._model.model.zero_grad()
                output_dict = self._model.run_info(inputs)
                # predicted_class_probs = torch.log(output_dict["probs"][:, 1])
                predicted_class_probs = output_dict["probs"][
                    torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"].squeeze().detach()
                ]  # (B, C)
                predicted_class_probs.sum().backward(retain_graph=True)
                forward_handle.remove()
            self.grads = torch.stack(self.grads, 0).mean(0)
            attentions = torch.einsum("ijk,ijk->ij", self.grads, self.grads)
            attentions = torch.abs(attentions)
            # print(attentions.shape)
            s_idxs = output_dict["start_idxs"]
            e_idxs = output_dict["end_idxs"]
            assert attentions is not None

            attentions = normalize_attentions(attentions, s_idxs, e_idxs)
        
        return attentions.cpu().data.numpy()
