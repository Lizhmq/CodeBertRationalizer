from scorer.base_scorer import SaliencyScorer
from utils import normalize_attentions
import torch
import torch.nn as nn


class GradientSaliency(nn.Module):

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._model.model.eval()
        self.embedding_layer = model.model.roberta.embeddings


        # _embedding_layer = [x for x in model.model.modules() if any(x == y for y in self.embedding_layers)]
        # assert len(_embedding_layer) == 1

        # self.embedding_layer = _embedding_layer[0]


    def forward(self, inputs):
        with torch.enable_grad():
            # self._model.model.train()
            for param in self.embedding_layer.parameters():
                param.requires_grad = True

            embeddings_list = []
            def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
                embeddings_list.append(output)
                output.retain_grad()

            hook = self.embedding_layer.register_forward_hook(forward_hook)
            output_dict = self._model.run_info(inputs)

            hook.remove()

            assert(len(embeddings_list) == 1)
            embeddings = embeddings_list[0]

            predicted_class_probs = output_dict["probs"][
                torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"].squeeze().detach()
            ]  # (B, C)


            self._model.model.zero_grad()
            predicted_class_probs.sum().backward(retain_graph=True)

            gradients = ((embeddings.grad * embeddings.grad).sum(-1).detach()).abs()
            # gradients = (embeddings * embeddings.grad).sum(-1).detach()
            gradients = gradients / gradients.sum(-1, keepdim=True)

            attentions = gradients
            s_idxs = output_dict["start_idxs"]
            e_idxs = output_dict["end_idxs"]
            assert attentions is not None

            attentions = normalize_attentions(attentions, s_idxs, e_idxs)
        
        return attentions.cpu().data.numpy()
