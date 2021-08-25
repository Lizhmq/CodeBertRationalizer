import torch
from models.codebert import codebert_mlm, codebert_cls
from scorer.base_scorer import SaliencyScorer
from scorer.gradient_scorer import GradientSaliency
from scorer.integrad_scorer import IntegradSaliency

device = torch.device("cuda", 2)
# device = torch.device("cpu")
cls_model = codebert_cls("./save/java0/checkpoint-28000-0.9366", device)
cls_model.model = cls_model.model.to(device)
# cls_model.model.eval()

inputs = [
    "for ( int i = 0 ; i <= length ; ++ i ) { }",
    # "int main ( ) { int <mask>, i ; <mask> = 1 ; return 0 }",  # tokenizer fails on <mask>
    # "for ( int i = 0 ; i <= a . length ; ++ i ) { }",
]
inputs = [st.split() for st in inputs]

scorer = SaliencyScorer(cls_model)
scores = scorer(inputs)
print(scores)

scorer = GradientSaliency(cls_model)
scores = scorer(inputs)
print(scores)


scorer = IntegradSaliency(cls_model)
scores = scorer(inputs)
print(scores)