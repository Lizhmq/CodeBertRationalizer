import torch

class SaliencyScorer():

    def __init__(self, model):
        self._model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, **inputs) :
        logits, attns = self.run(**inputs, need_attn=True)
        return attns.cpu().data.numpy()

    def make_output_human_readable(self, output_dict) :
        assert "attentions" in output_dict
        assert "metadata" in output_dict

        new_output_dict = {k:[] for k in output_dict['metadata'][0].keys()}
        for example in output_dict['metadata'] :
            for k, v in example.items() :
                new_output_dict[k].append(v)

        tokens = [example.split() for example in new_output_dict['document']]

        attentions = output_dict['attentions'].cpu().data.numpy()

        assert len(tokens) == len(attentions)
        assert max([len(s) for s in tokens]) == attentions.shape[-1]

        new_output_dict['saliency'] = [[round(float(x), 5) for x in list(m)[:len(tok)]] for m, tok in zip(attentions, tokens)]
            
        return new_output_dict

    def score(self, **inputs) :
        raise NotImplementedError

    def init_from_model(self) :
        pass