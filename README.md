# Dataless Knowledge Fusion by Merging Weights of Language Models

This repository contains the experimental code to reproduce the results in [Dataless Knowledge Fusion by Merging Weights of Language Models](https://openreview.net/forum?id=FCnohuR6AnM), a paper to be published during the [Eleventh International Conference on Learning Representations (ICLR 2023)](https://iclr.cc/), to be held May 1-5, 2023 in Kigali, Rwanda.

```
@inproceedings{
    jin2023dataless,
    title={Dataless Knowledge Fusion by Merging Weights of Language Models},
    author={Xisen Jin and Xiang Ren and Daniel Preotiuc-Pietro and Pengxiang Cheng},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=FCnohuR6AnM}
}
```

## Requirements
We used PyTorch 1.13.1. See [requirements.txt](requirements.txt) for other requirements.

## Quick Demo
If you are just interested in the Regresssion Mean (RegMean) algorithm, please check [regmean_demo.ipynb](regmean_demo.ipynb).

This is a standalone Jupyter notebook that merges two Hugging Face transformer models fine-tuned on GLUE. This file does not import files under `src/`.

## Reproducing Results
### Preparing Emotion Classification Datasets
Please download the unified emotion dataset in this [repo](https://github.com/sarnthil/unify-emotion-datasets). The files should be placed under `PROJECT_ROOT/resources/emotion_splits` in the following structure.

```
.
├── crowdflower
│   ├── dev.jsonl
│   ├── full.jsonl
│   ├── test.jsonl
│   └── train.jsonl
├── dailydialog
│   ├── dev.jsonl
│   ├── full.jsonl
│   ├── test.jsonl
│   └── train.jsonl
├── electoraltweets
│   ├── dev.jsonl
│   ├── full.jsonl
│   ├── test.jsonl
│   └── train.jsonl
├── emobank
│   ├── dev.jsonl
│   ├── full.jsonl
│   ├── test.jsonl
│   └── train.jsonl
...
```

### Preparing NER Datasets
Please prepare CoNLL2003, OntoNotes, and Twitter NER datasets and place them under `PROJECT_ROOT/resources/ner`.
```
.
├── conll2003
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── ontonotes
│   ├── onto.development.bc.ner
│   ├── onto.development.bn.ner
│   ├── onto.development.mz.ner
│   ├── onto.development.nw.ner
│   ├── onto.development.tc.ner
│   ├── onto.development.wb.ner
│   ├── onto.test.bc.ner
│   ├── onto.test.bn.ner
│   ├── onto.test.mz.ner
│   ├── onto.test.nw.ner
│   ├── onto.test.tc.ner
│   ├── onto.test.wb.ner
│   ├── onto.train.bc.ner
│   ├── onto.train.bn.ner
│   ├── onto.train.mz.ner
│   ├── onto.train.nw.ner
│   ├── onto.train.tc.ner
│   └── onto.train.wb.ner
└── twitter
    ├── annotated.twitter-ner-20-21-tweet-dev-withcleaned.json
    ├── annotated.twitter-ner-20-21-tweet-test-withcleaned.json
    └── annotated.twitter-ner-20-21-tweet-train-withcleaned.json
```

Here, CoNLL and OntoNotes datasets contain entries in the CoNLL format.

```
CRICKET	O	Conll
-	O	Conll
LEICESTERSHIRE	B-ORG	Conll
TAKE	O	Conll
OVER	O	Conll
AT	O	Conll
TOP	O	Conll
AFTER	O	Conll
INNINGS	O	Conll
VICTORY	O	Conll
.	O	Conll

LONDON	B-LOC	Conll
1996-08-30	O	Conll
...
```

Twitter NER contains 1 JSON dict per line.

```
{"text": "Spectacular skies over #Clonmel tonight http://t.co/OxclQkuyTp /via @niallodonovan #lastdayofautumn", "id": "539106999980797952", "entities": [{"startCharOffset": 24, "endOffset": 31, "endCharOffset": 31, "surface": "Clonmel", "startOffset": 24, "type": "LOC"}, {"startCharOffset": 69, "endOffset": 82, "endCharOffset": 82, "surface": "niallodonovan", "startOffset": 69, "type": "PER"}], "labels": ["O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "B-PER", "O", "O"], "tokens": ["Spectacular", "skies", "over", "#", "Clonmel", "tonight", "http://t.co/OxclQkuyTp", "/", "via", "@niallodonovan", "#", "lastdayofautumn"], "domain": "TWT"}
```

### Preparing GLUE datasets
GLUE datasets will be downloaded and loaded with Hugging Face's `datasets` library.

### Preparing Pretrained LMs
Please download pretrained models (e.g., RoBERTa-base) from the Hugging Face models repository and place them under `PROJECT_ROOT/resources` (e.g., `PROJECT_ROOT/resources/roberta-base`).

### Usage
- `--config_files`: See under `src/configs`. The training module (`src.run_experiments`) requires three config files defining default arguments (`src/defaults.yaml`), data config (under `src/configs/datasets`), and exp config (under `src/configs/exps`).

- `--filter_model`: Useful when merging only a subset of individual models specificed in data config, e.g., `--filter_model model0 model1` will perform pairwaise merging of model0 and model1 (see the definition of alias like model0, model1 in the data config).

- `--templates`: config files may contain templates like `{seed}`. The values of templates should be specified in command lines (e.g., `--templates seed=1`).

Individual models (before merging) will be trained and stored under `local_zoo_dir` specified in the config. If none of the individual models in the zoo match the given model type and `zoo_filter` arguments in the config, then the program will automatically train new individual models and store them under `local_zoo_dir`. If individual models are found in `local_zoo_dir`, they will be loaded without re-training.

Example: *RegMean, Emotion, Same Head Init, Merginging Model0 (dailydialogue) and Model1 (crowdflower)*

```
HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m src.run_experiments --config src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/roberta-base/roberta-base-emotion.yaml --templates seed=1 --filter_model model0 model1
```

### Scripts
#### Pairwise Merging
Merging two emotion classification models trained on different datasets (domains).
- Emotion, RoBERTa-base: `scripts/roberta/pairwise_emotion.py`
- Emotion, T5-base: `scripts/t5/pairwise_emotion.py`
- Emotion, RoBERTa-base: `scripts/t5/pairwise_emotion.py`

Merging two models trained on different GLUE tasks. Task-specific classification heads are not merged.
- GLUE, DistilBERT-base: `scripts/distilbert/pairwise_glue_difftask.py`
- GLUE, RoBERTa-base: `scripts/roberta/pairwise_glue_difftask.py`

Merging two models trained on two non-IID partitions of the same GLUE task
- GLUE, DistilBERT-base: `scripts/distilbert/pairwise_glue_subset.py`
- GLUE, RoBERTa-base: `scripts/roberta/pairwise_glue_subset.py`

#### Greedy Merging
Greedily merging multiple (two to all) models in the order of OOD performance of individual models.
- Emotion, RoBERTa-base: `scripts/roberta/incremental_emotion.py`
- Emotion, T5-base: `scripts/t5/incremental_emotion.py`
- Emotion, DeBERTa-large: `scripts/deberta/incrementale_emotion.py`
- NER, RoBERTa-base: `scripts/roberta/incremental_ner.py`
- NER, DeBERTa-large: `scripts/deberta/incremental_ner.py`

Please note these scripts run inference on both in-domain and out-of-domain test sets.

Each script above will run Simple, Fisher, and RegMean averaging. They also run the Multi-Task Learning (MTL), model ensemble, and the performance of individual models (before merging) as comparators. You can comment out lines inside these scripts to just run part of each one.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Code of Conduct

This project has adopted a [Code of Conduct](https://github.com/bloomberg/.github/blob/master/CODE_OF_CONDUCT.md).
If you have any concerns about the Code, or behavior which you have experienced in the project, please
contact us at opensource@bloomberg.net.

## Security Vulnerability Reporting

If you believe you have identified a security vulnerability in this project, please send an email to the project
team at opensource@bloomberg.net detailing the suspected issue and any methods you've found to reproduce it.

Please do NOT open an issue in the GitHub repository, as we'd prefer to keep vulnerability reports private until
we've had an opportunity to review and address them.
