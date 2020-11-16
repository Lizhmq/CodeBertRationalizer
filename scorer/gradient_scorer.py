from scorer.base_scorer import SaliencyScorer
from models.utils import normalize_attentions
from models.utils import generate_embeddings_for_pooling
import torch
import logging


class GradientSaliency(SaliencyScorer):

    def __init__(self, model):
        super().__init__(model)
        self.init_from_model()

    def init_from_model(self) :
        logging.info("Initialising from Model .... ")
        model = self.model
        self.embedding_layer = model.embeddings

    def score(self, **kwargs) :
        with torch.enable_grad() :
            self.model.train()

            for param in self.embedding_layer.parameters():
                param.requires_grad = True

            embeddings_list = []
            def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
                embeddings_list.append(output)
                output.retain_grad()

            hook = self.embedding_layer.register_forward_hook(forward_hook)
            output_dict = self.model.forward(**kwargs)

            hook.remove()

            assert(len(embeddings_list) == 1)
            embeddings = embeddings_list[0]

            predicted_class_probs = output_dict["probs"][
                torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"].detach()
            ]  # (B, C)


            predicted_class_probs.sum().backward(retain_graph=True)

            gradients = ((embeddings * embeddings.grad).sum(-1).detach()).abs()
            gradients = gradients / gradients.sum(-1, keepdim=True)

            output_dict['attentions'] = gradients

        output_dict = self.normalize_attentions(output_dict)

        return output_dict

    def normalize_attentions(self, output_dict):
        attentions = output_dict['attentions'].unsqueeze(-1)
        document_token_starts = output_dict['document-starting-offsets']
        document_token_ends = output_dict['document-ending-offsets']
        
        token_attentions, token_mask = generate_embeddings_for_pooling(attentions, document_token_starts, document_token_ends)

        token_attentions = (token_attentions * token_mask.unsqueeze(-1)).squeeze(-1).sum(-1)
        output_dict["attentions"] = token_attentions / token_attentions.sum(-1, keepdim=True)

        return output_dict