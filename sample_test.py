from torch.functional import norm
import tree_sitter
from tree_sitter import Language, Parser
import pickle
import torch
import numpy as np
from models.codebert import codebert_cls
from scorer.base_scorer import SaliencyScorer
from thretholder.contiguous import ContiguousThresholder

def get_method(node, method_list):
    if node.type == 'method_declaration':
        method_list.append(node)
        return
    for child in node.children:
        get_method(child, method_list)
    return

def node2tokens(node, code_bytes):
    seq = []
    if node.type == 'comment':
        return seq
    if not node.children:
        return [code_bytes[node.start_byte:node.end_byte].decode()]
    for child in node.children:
        a = node2tokens(child, code_bytes)
        seq += a
    return seq

def mask_special_value(tokens):
    def trans(token):
        if len(token) <= 1:     # keeps 0-9 and other 1 size char
            return token
        elif token[0] == '"' and token[-1] == '"':
            return "<str>"
        elif token[0] == "'" and token[-1] == "'":
            return "<char>"
        elif token[0] in "0123456789":
            if 'e' in token.lower() or '.' in token:
                return "<fp>"
            else:
                return "<int>"
        else:
            return token
    return list(map(trans, tokens))


def main():
    java_path = "./sample.java"
    with open(java_path, "r") as f:
        code_str = f.read()
    print("Code:")
    print(code_str)
    Language.build_library(
        # Store the library in the `build` directory
        '../bigJava/code/build/java.so',

        # Include one or more languages
        [
        '../bigJava/tree-sitter-java-master'
        ]
    )
    JAVA_LANGUAGE = Language('../bigJava/code/build/java.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    byted = bytes(code_str.encode('utf-8'))
    tree = parser.parse(byted)
    root = tree.root_node
    method_list = []
    get_method(root, method_list)
    method = method_list[0]
    tokens = node2tokens(method, byted)
    print(tokens)

    norm = mask_special_value(tokens)
    print(norm)
    print(len(norm))

    device = torch.device("cuda", 2)
    cls_model = codebert_cls("./save/java0/checkpoint-16000-0.9311", device, attn_head=-1)
    cls_model.model = cls_model.model.to(device)
    cls_model.model.eval()

    inputs = [norm]
    predicted = cls_model.run(inputs).cpu().data.numpy()
    predicted = np.argmax(predicted, axis=1)[0]
    print(predicted)

    scorer = SaliencyScorer(cls_model)
    extractor = ContiguousThresholder(0.0001)

    scores = scorer(inputs)
    rationale = extractor(scores)[0]
    rationale = [k["span"] for k in rationale][0]
    print(rationale[0])
    print(norm[rationale[0]])
    


if __name__ == "__main__":
    main()