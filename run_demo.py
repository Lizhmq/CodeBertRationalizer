import torch
from models.codebert import codebert_mlm, codebert_cls
from scorer.base_scorer import SaliencyScorer
from scorer.gradient_scorer import  GradientSaliency

# device = torch.device("cuda", 0)
device = torch.device("cpu")
cls_model = codebert_cls("./save/java-classifier4/checkpoint-36000-0.9365", device)
cls_model.model = cls_model.model.to(device)

inputs = [
    "int main ( ) { int n , i ; n <= 11 ; return 0 }",
    # "int main ( ) { int <mask>, i ; <mask> = 1 ; return 0 }",  # tokenizer fails on <mask>
    "void main ( ) { ALongVarName x ; }",
]
inputs = [st.split() for st in inputs]
scorer = GradientSaliency(cls_model)
attentions = scorer(inputs)
# print(attentions)
print(attentions)

# print(output_dict["predicted_labels"])
