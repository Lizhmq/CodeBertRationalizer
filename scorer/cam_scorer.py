from scorer.base_scorer import SaliencyScorer
from utils import normalize_attentions
import torch
import torch.nn as nn


class CamSaliency(nn.Module):

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._model.model.eval()
        self._dense = model.model.classifier.dense

    def forward(self, inputs):
        with torch.no_grad():
            self._model.model.eval()
            output_dict = self._model.run_info(inputs)

            predicted_class_probs = output_dict["probs"][
                torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"].squeeze().detach()
            ]  # (B, C)


            predicted_class_probs.sum().backward(retain_graph=True)
            
            attentions = gradients
            s_idxs = output_dict["start_idxs"]
            e_idxs = output_dict["end_idxs"]
            assert attentions is not None

            attentions = normalize_attentions(attentions, s_idxs, e_idxs)
        
        return attentions.cpu().data.numpy()
