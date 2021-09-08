### WEakly supervised bug LocaLization (WELL)

Repository for holding source code of paper "WELL: Applying Bug Detectors to Bug Localization via Weakly Supervised Learning".

#### Hierachy

```
.
├── baselines							# baseline models - lstm & transformer
├── dataset.py
├── eval_rationale.py					# evaluate localization accuracy
├── filter_dataset.py
├── fix.py								# WELL for bug fixing
├── gen_vis.py
├── models								# CodeBERT model
│   ├── codebert.py
│   ├── nblmodel.py
│   └── tokenizer.py
├── nbl.py								# Evaluate NBL model
├── run_classifier.py
├── run_nbl.py							# Train NBL model
├── scorer								# attention scorer and gradient scorer
│   ├── base_scorer.py
│   ├── evaluator.py
│   ├── gradient_scorer.py
│   └──integrad_scorer.py
├── thretholder							# extractor - e.g. top-k/contiguous extractor
│   ├── base_thresholder.py
│   ├── contiguous_mask.py
│   ├── contiguous.py
│   └── topk.py
├── train_classifier.sh					# Training script
├── train_nbl.sh
├── train_reg.sh
└── utils.py

```

#### Environment

Python 3.7 with package requirements:

```
torch==1.7.1
transformers==3.4.0
```

#### Acknowledgement

The repository is built based on [FRESH](https://github.com/successar/FRESH). Thanks for their well-written open-sourced project.

Thanks for Huggingface for their transformers package and detailed [documentations](https://huggingface.co/transformers/).