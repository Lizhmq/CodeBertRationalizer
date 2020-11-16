import torch
from models.codebert import codebert_mlm, codebert_cls

# device = torch.device("cuda", 0)
device = torch.device("cpu")
cls_model = codebert_cls("./save/java-classifier4/checkpoint-36000-0.9365", device)
# cls_model = codebert_cls("/var/data/lushuai/bertvsbert/save/poj-classifier/checkpoint-51000-0.986", device)

inputs = [
    "int main ( ) { int n , i ; n < 11 ; return 0 }",
    "int main ( ) { int <mask>, i ; <mask> = 1 ; return 0 }", 
    "void main ( ) { double x ; }",
]
# print(cls_model.tokenize(inputs))
# print(cls_model.run(inputs))
output_dict = cls_model.run_info(inputs, need_attn=True)
# print(output_dict["predicted_labels"])
print(output_dict["attentions"].shape)
