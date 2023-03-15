# Copyright 2023 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from torch.utils.data import Dataset

from .data_utils import DataCollatorForTokenClassificationExcludeStr
from .glue_data_manager import GLUEDataManager
from .metrics.ner import get_ner_metrics_func
from .ner_data_utils import (
    JsonlDatasetReader,
    TabDatasetReader,
    convert_instances_to_feature_tensors,
)

NER_FLAGS = ["PER", "LOC", "ORG"]
NER_LABELS = ["O"] + ["B-" + x for x in NER_FLAGS] + ["I-" + x for x in NER_FLAGS]


class SimpleDataset(Dataset):
    def __init__(self, examples) -> None:
        super().__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


class NERDataManager(GLUEDataManager):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

    def update_model_config(self, model_config):
        model_config.num_labels = len(NER_LABELS)
        model_config.label2id = {v: i for i, v in enumerate(NER_LABELS)}
        model_config.id2label = {i: label for label, i in model_config.label2id.items()}

    def load_ood_eval_dataset(self, ood_data_config):
        if self.config.mtl:
            sample_model_config = next(
                iter(vars(self.config.local_models._models).values())
            )
        else:
            sample_model_config = next(
                iter(vars(self.config.local_models.models).values())
            )

        ood_data_config.label2id = sample_model_config.label2id
        ood_data_config.max_seq_length = sample_model_config.max_seq_length
        _, _, test_dataset = self.load_dataset(ood_data_config)

        return test_dataset

    def load_all_ood_eval_datasets(self):
        ret = {}
        for name, ood_data_config in vars(self.config.ood_datasets).items():
            ret[name] = self.load_ood_eval_dataset(ood_data_config)
        return ret

    def load_dataset(self, model_config):
        # the format of the dataset name for ner datasets would be "ds@domain"
        # domain is optional
        x = model_config.dataset_name.split("@")
        ds, domain = x[0], x[1] if len(x) == 2 else None
        splits = ["train", "dev", "test"]

        if ds == "ontonotes":
            reader = TabDatasetReader(
                dir_format=os.path.join(
                    self.config.resource_dir,
                    "ner/ontonotes",
                    "onto.{split}.{domain}.ner",
                ),
                label_space=NER_FLAGS,
                split_mapping={"train": "train", "dev": "development", "test": "test"},
            )
        elif ds == "conll":
            reader = TabDatasetReader(
                dir_format=os.path.join(
                    self.config.resource_dir, "ner/conll2003", "{split}.conll"
                ),
                label_space=NER_FLAGS,
            )
        elif ds == "twitter":
            reader = JsonlDatasetReader(
                dir_format=os.path.join(
                    self.config.resource_dir,
                    "ner/twitter",
                    "annotated.twitter-ner-20-21-tweet-{split}-withcleaned.json",
                ),
                label_space=NER_FLAGS,
            )

        else:
            raise NotImplementedError

        split_examples = [reader.load_domain_split(domain, split) for split in splits]
        split_features = [
            convert_instances_to_feature_tensors(
                examples,
                self.tokenizer,
                model_config.label2id,
            )
            for examples in split_examples
        ]
        train_ds, dev_ds, test_ds = [SimpleDataset(x) for x in split_features]

        return train_ds, dev_ds, test_ds

    def get_collator_cls(self):
        return DataCollatorForTokenClassificationExcludeStr

    def get_metrics_func_single(self, model_config):
        func = get_ner_metrics_func(model_config.id2label)
        return func
