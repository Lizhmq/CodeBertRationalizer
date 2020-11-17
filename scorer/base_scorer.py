import torch
import torch.nn as nn
from utils import normalize_attentions

class SaliencyScorer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._model.model.eval()

    def forward(self, inputs) :
        output_dict = self._model.run_info(inputs, need_attn=True)
        attentions = output_dict["attentions"]
        s_idxs = output_dict["start_idxs"]
        e_idxs = output_dict["end_idxs"]
        assert attentions is not None
        attentions = normalize_attentions(attentions, s_idxs, e_idxs)
        return attentions.cpu().data.numpy()

    # def make_output_human_readable(self, output_dict) :
    #     assert "attentions" in output_dict
    #     assert "metadata" in output_dict

    #     new_output_dict = {k:[] for k in output_dict['metadata'][0].keys()}
    #     for example in output_dict['metadata'] :
    #         for k, v in example.items() :
    #             new_output_dict[k].append(v)

    #     tokens = [example.split() for example in new_output_dict['document']]

    #     attentions = output_dict['attentions'].cpu().data.numpy()

    #     assert len(tokens) == len(attentions)
    #     assert max([len(s) for s in tokens]) == attentions.shape[-1]

    #     new_output_dict['saliency'] = [[round(float(x), 5) for x in list(m)[:len(tok)]] for m, tok in zip(attentions, tokens)]
            
    #     return new_output_dict

    def score(self, **inputs) :
        raise NotImplementedError

    def init_from_model(self) :
        pass