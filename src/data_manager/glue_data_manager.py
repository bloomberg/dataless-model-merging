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

import logging
import os

import numpy as np
from datasets import load_dataset
from torch.utils.data import Subset

from ..utils.config import get_component_configs
from .data_utils import (
    ConcatDatasetWithID,
    DataCollatorWithPaddingExcludeStr,
    get_dataset_digest,
    get_random_subset,
    niid_partition_by_label,
)
from .metrics.glue import Glue as GlueMet
from .simple_data_manager import SimpleDataManager

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

key_met_map = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "stsb": "pearson",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
}


class GLUEDataManager(SimpleDataManager):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

    def update_model_config(self, model_config):
        raw_datasets = load_dataset(
            "glue",
            model_config.dataset_name,
            cache_dir=os.path.join(self.config.hf_datasets_cache_dir, "datasets"),
        )
        is_regression = model_config.is_regression
        if is_regression:
            model_config.num_labels = 1
            model_config.label2id = model_config.id2label = None
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            model_config.num_labels = len(label_list)
            model_config.label2id = {v: i for i, v in enumerate(label_list)}
            model_config.id2label = {
                i: label for label, i in model_config.label2id.items()
            }

    def load_mtl_dataset(self, mtl_config):
        model_configs = get_component_configs(self.config, mtl_config.components)
        datasets = [self.load_dataset(model_config) for model_config in model_configs]
        train_dss, eval_dss, test_dss = (
            [x[0] for x in datasets],
            [x[1] for x in datasets],
            [x[2] for x in datasets],
        )
        dataset_names = [model_config.dataset_name for model_config in model_configs]
        concat_train_ds = ConcatDatasetWithID(dataset_names, train_dss)
        concat_eval_dss = ConcatDatasetWithID(dataset_names, eval_dss)
        if test_dss[0] is not None:
            concat_test_dss = ConcatDatasetWithID(dataset_names, test_dss)
        return dataset_names, concat_train_ds, concat_eval_dss, concat_test_dss

    def load_dataset(self, model_config):
        raw_datasets = load_dataset(
            "glue",
            model_config.dataset_name,
            cache_dir=os.path.join(self.config.hf_datasets_cache_dir, "datasets"),
        )

        is_regression = model_config.is_regression
        if is_regression:
            self.num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            self.num_labels = len(label_list)
        self.config.num_labels = self.num_labels
        max_seq_length = model_config.max_seq_length
        sentence1_key, sentence2_key = task_to_keys[model_config.dataset_name]

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if model_config.label2id is not None and "label" in examples:
                result["label"] = [
                    (model_config.label2id[lbl] if lbl != -1 else -1)
                    for lbl in examples["label"]
                ]
            return result

        arr_cache_names = {
            k: os.path.join(
                self.config.hf_datasets_cache_dir,
                "{}-{}-{}".format(
                    k, self.tokenizer.name_or_path.split("/")[-1], get_dataset_digest(v)
                ),
            )
            for k, v in raw_datasets.items()
        }

        raw_datasets = raw_datasets.map(
            preprocess_function, batched=True, cache_file_names=arr_cache_names
        )
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets[
            "validation_matched"
            if model_config.dataset_name == "mnli"
            else "validation"
        ]
        test_dataset = None

        if model_config.train_subset_n != -1:
            if len(train_dataset) > model_config.train_subset_n:
                rng = (
                    np.random.default_rng(self.config.seed)
                    if model_config.train_subset_seed is None
                    else np.random.default_rng(model_config.train_subset_seed)
                )
                idxs = rng.choice(
                    len(train_dataset), model_config.train_subset_n, replace=False
                ).tolist()
                train_dataset = Subset(train_dataset, idxs)
                logging.info(
                    "Subsampled {} examples for {}, {}".format(
                        model_config.train_subset_n, model_config.dataset_name, idxs[:5]
                    )
                )
            else:
                logging.info(
                    "Not subsampled {} ({}<{})".format(
                        model_config.dataset_name,
                        len(train_dataset),
                        model_config.train_subset_n,
                    )
                )

        if model_config.partition != -1 and self.config.partition.method is not None:
            if self.config.partition.method == "iid":
                train_dataset = self.get_partition_subset_iid(
                    model_config, train_dataset
                )
            else:
                train_dataset = self.get_partition_subset_label_niid(
                    model_config, train_dataset
                )

        return train_dataset, eval_dataset, test_dataset

    def get_partition_seed(self, model_config):
        return (
            self.config.seed
            if model_config.train_subset_seed is None
            else model_config.train_subset_seed
        )

    def get_partition_subset_iid(self, model_config, train_dataset):
        seed = self.get_partition_seed(model_config)
        rng = np.random.default_rng(seed)
        total = self.config.partition.n_total_examples
        if total > len(train_dataset):
            total = len(train_dataset)

        full_idxs = rng.choice(len(train_dataset), total, replace=False).tolist()

        start = int(
            (model_config.partition / self.config.partition.n_partition) * total
        )
        stop = int(
            ((1 + model_config.partition) / self.config.partition.n_partition) * total
        )
        idxs = full_idxs[start:stop]
        subset = Subset(train_dataset, idxs)
        logging.info(
            "IID parition: subsampled {} examples from {}...".format(
                len(idxs), full_idxs[:5]
            )
        )
        return subset

    def get_partition_subset_label_niid(self, model_config, train_dataset):
        partition_id = model_config.partition
        seed = self.get_partition_seed(model_config)
        if self.partitions is None:
            # prepare partitions
            train_ds_subset = get_random_subset(
                train_dataset, seed, self.config.partition.n_total_examples
            )
            self.partitions = niid_partition_by_label(
                train_ds_subset,
                seed,
                self.config.partition.n_partition,
                main_portion=self.config.partition.niid_label_alpha,
                is_regression=model_config.is_regression,
            )
        return self.partitions[partition_id]

    def get_metrics_func_single(self, model_config):
        def compute_metrics(p):
            metric = GlueMet(config_name=model_config.dataset_name)
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = (
                np.squeeze(preds)
                if model_config.is_regression
                else np.argmax(preds, axis=1)
            )

            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            result["key_score"] = result[key_met_map[model_config.dataset_name]]
            return result

        return compute_metrics

    def get_metrics_func(self, model_config):
        if hasattr(model_config, "components"):
            component_configs = get_component_configs(
                self.config, model_config.components
            )
            return {
                x.dataset_name: self.get_metrics_func_single(x)
                for x in component_configs
            }
        else:
            return self.get_metrics_func_single(model_config)

    def extract_main_metrics(
        self, met_dict, local_model_names, local_model_configs, prefix=""
    ):
        scores = []
        for model_name, local_model_config in zip(
            local_model_names, local_model_configs
        ):
            ds_name = local_model_config.dataset_name
            key_met = prefix + key_met_map[ds_name]
            scores.append(met_dict[model_name][key_met])
        return scores

    def get_collator_cls(self):
        return DataCollatorWithPaddingExcludeStr
