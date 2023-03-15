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
import random as _random
import re

import numpy as np
from datasets import Dataset

from .data_utils import get_dataset_digest
from .emotion_data_manager import (
    EMOTION_LABELS,
    IGNORE_ID,
    ConcatDataset,
    EmotionDataManager,
    f1_score,
    precision_score,
    recall_score,
)


class EmotionGenDataManager(EmotionDataManager):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.resample_schema = config.resample_schema

    def get_model_type(self, model_config):
        if "t5" in model_config.model_name:
            return "t5"
        elif "gpt2" in model_config.model_name:
            return "gpt2"
        else:
            raise NotImplementedError

    def update_model_config(self, model_config):
        tasks = model_config.dataset_name.split()
        train_examples = EmotionDataManager.load_examples(self, "train", *tasks)
        model_config.freq_dist = self.get_freq_dist_examples(train_examples["label"])
        model_config.label2id = {
            v: i for i, v in enumerate(EMOTION_LABELS) if i in model_config.freq_dist
        }
        model_config.id2label = {
            i: label
            for label, i in model_config.label2id.items()
            if i in model_config.freq_dist
        }
        self.get_prune_idxs(model_config)

    def generate_prompt_examples(
        self, examples, label2id, freq_dist, split_name, model_type
    ):
        if model_type == "t5":
            return self.generate_prompt_examples_t5(
                examples, label2id, freq_dist, split_name
            )
        elif model_type == "gpt2":
            return self.generate_prompt_examples_gpt2(
                examples, label2id, freq_dist, split_name
            )
        else:
            raise NotImplementedError

    def generate_prompt_examples_t5(self, examples, label2id, freq_dist, split_name):
        random = _random.Random(self.config.seed)
        prompt_examples = {"question": [], "answer": []}
        for sent, lb in zip(examples["sentence"], examples["label"]):
            for label, label_id in label2id.items():
                q = f"does the sentence express {label}? <extra_id_0> {sent}"
                y = lb[label_id]
                if type(y) is float:
                    y = 1 if y > 0.1 else 0
                if y == IGNORE_ID:
                    continue
                assert y in [0, 1]

                if (
                    split_name == "train"
                    and self.resample_schema is not None
                    and y == 0
                    and freq_dist is not None
                ):
                    pos_portion = freq_dist[label_id][1] / (
                        freq_dist[label_id][0] + freq_dist[label_id][1]
                    )
                    if self.resample_schema == "sqrt":
                        thres = pos_portion**0.5
                    elif self.resample_schema == "linear":
                        thres = pos_portion
                    else:
                        raise ValueError(self.resample_schema)
                    rng = random.random()
                    if rng > thres:
                        continue

                a = "<extra_id_0> yes" if y else "<extra_id_0> no"
                prompt_examples["question"].append(q)
                prompt_examples["answer"].append(a)
        return prompt_examples

    def generate_prompt_examples_gpt2(self, examples, label2id, freq_dist, split_name):
        random = _random.Random(self.config.seed)
        prompt_examples = {"question": [], "answer": []}
        for sent, lb in zip(examples["sentence"], examples["label"]):
            for label, label_id in label2id.items():
                q = f"{sent} The sentence expresses {label}."
                y = lb[label_id]
                if type(y) is float:
                    y = 1 if y > 0.1 else 0
                if y == IGNORE_ID:
                    continue
                assert y in [0, 1]

                if (
                    split_name == "train"
                    and self.resample_schema is not None
                    and y == 0
                    and freq_dist is not None
                ):
                    pos_portion = freq_dist[label_id][1] / (
                        freq_dist[label_id][0] + freq_dist[label_id][1]
                    )
                    if self.resample_schema == "sqrt":
                        thres = pos_portion**0.5
                    elif self.resample_schema == "linear":
                        thres = pos_portion
                    else:
                        raise ValueError(self.resample_schema)
                    rng = random.random()
                    if rng > thres:
                        continue

                a = " yes" if y else " no"
                prompt_examples["question"].append(q)
                prompt_examples["answer"].append(a)
        return prompt_examples

    def load_ood_eval_dataset(self, ood_data_config):
        if not self.config.mtl:
            sample_model_config = next(
                iter(vars(self.config.local_models.models).values())
            )
        else:
            sample_model_config = next(
                iter(vars(self.config.local_models._models).values())
            )
        tasks = ood_data_config.dataset_name.split()
        _split_examples = self.load_all_splits(*tasks)
        model_type = self.get_model_type(sample_model_config)

        split_examples = {
            split: self.generate_prompt_examples(
                examples,
                sample_model_config.label2id,
                None,
                split,
                model_type=model_type,
            )
            for split, examples in _split_examples.items()
        }

        max_seq_length = sample_model_config.max_seq_length

        full_examples = []
        for split, examples in split_examples.items():
            full_examples.extend(examples)

        split_datasets = {
            split: Dataset.from_dict(examples)
            for split, examples in split_examples.items()
        }
        for split, dataset in split_datasets.items():
            preprocess_function = self.get_preprocess_function(
                max_seq_length,
                model_type,
            )
            dataset = dataset.map(preprocess_function, batched=True)
            split_datasets[split] = dataset

        # in case model_config says all examples belong to test
        all_is_test = self.config.ood_all_is_test
        if hasattr(ood_data_config, "test_only") and ood_data_config.test_only:
            all_is_test = False
        if all_is_test:
            test_dataset = ConcatDataset(
                [split_datasets[x] for x in ["train", "dev", "test"]]
            )
        else:
            test_dataset = split_datasets["test"]

        return test_dataset

    def get_preprocess_function(self, max_seq_length, model_type):
        def preprocess_function_t5(examples):
            # Tokenize the texts
            inputs = self.tokenizer(
                examples["question"], max_length=max_seq_length, truncation=True
            )
            outputs = self.tokenizer(
                examples["answer"], max_length=max_seq_length, truncation=True
            )
            # Map labels to IDs (not necessary for GLUE tasks)
            results = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": outputs["input_ids"],
            }
            return results

        def preprocess_function_gpt2(examples):
            concat_qa = [
                x + y for x, y in zip(examples["question"], examples["answer"])
            ]

            encoded_question = self.tokenizer(
                examples["question"], max_length=max_seq_length - 1, truncation=True
            )
            encoded_inputs = self.tokenizer(
                concat_qa, max_length=max_seq_length, truncation=True
            )

            input_ids = encoded_question["input_ids"]
            attention_mask = encoded_question["attention_mask"]
            attention_mask_qa = encoded_inputs["attention_mask"]
            labels = encoded_inputs["input_ids"]

            for i, (lbs, attn_mask, attn_mask_qa) in enumerate(
                zip(labels, attention_mask, attention_mask_qa)
            ):
                lbs = np.array(lbs, dtype=np.int32)
                attn_mask = np.array(attn_mask, dtype=np.bool)
                attn_mask_qa = np.array(attn_mask_qa, dtype=np.bool)
                attn_mask_ = np.zeros_like(attn_mask_qa)
                attn_mask_[: len(attn_mask)] = attn_mask
                lbs[attn_mask_] = IGNORE_ID
                lbs[~attn_mask_qa] = IGNORE_ID
                labels[i] = lbs.tolist()

            input_ids_ext = [x + [self.tokenizer.pad_token_id] for x in input_ids]
            attention_mask_ext = [x + [0] for x in attention_mask]
            results = {
                "input_ids": input_ids_ext,
                "attention_mask": attention_mask_ext,
                "labels": labels,
            }
            return results

        if model_type == "gpt2":
            return preprocess_function_gpt2
        elif model_type == "t5":
            return preprocess_function_t5
        else:
            raise NotImplementedError

    def load_dataset(self, model_config):
        tasks = model_config.dataset_name.split()
        _split_examples = self.load_all_splits(*tasks)
        model_type = self.get_model_type(model_config)
        split_examples = {
            split: self.generate_prompt_examples(
                examples,
                model_config.label2id,
                model_config.freq_dist,
                split,
                model_type,
            )
            for split, examples in _split_examples.items()
        }

        # for debug
        if self.config.debug:
            split_examples_new = {}
            for split, examples in split_examples.items():
                split_examples_new[split] = {k: v[:100] for k, v in examples.items()}
            split_examples = split_examples_new

        max_seq_length = model_config.max_seq_length

        split_datasets = {
            split: Dataset.from_dict(examples)
            for split, examples in split_examples.items()
        }
        for split, dataset in split_datasets.items():
            preprocess_function = self.get_preprocess_function(
                max_seq_length, model_type
            )
            cache_file_name = os.path.join(
                self.config.hf_datasets_cache_dir,
                "emotion-gen-v1-{}-{}-{}-{}".format(
                    "_".join(tasks),
                    split,
                    self.tokenizer.name_or_path.split("/")[-1],
                    get_dataset_digest(dataset),
                ),
            )
            dataset = dataset.map(
                preprocess_function, batched=True, cache_file_name=cache_file_name
            )
            split_datasets[split] = dataset

        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["dev"]
        test_dataset = split_datasets["test"]

        if model_config.train_subset_n != -1:
            raise NotImplementedError
        if model_config.partition != -1 and self.config.partition.method is not None:
            raise NotImplementedError
        return train_dataset, eval_dataset, test_dataset

    def get_metrics_func_single(self, model_config):
        model_type = self.get_model_type(model_config)

        def compute_metrics_t5(p):
            result = {}
            inputs = p.inputs
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )  # [N, L]
            labels = p.label_ids

            stats = {}
            for i in range(len(preds)):
                x, p, y = inputs[i], preds[i], labels[i]
                class_name = self.extract_class_name(x, model_type)

                gt = self.extract_ans(y[1:])  # get rid of extra_id_1
                pd = self.extract_ans(p[2:])  # additional bos token

                if class_name not in stats:
                    stats[class_name] = {"pred": [], "gt": []}
                stats[class_name]["pred"].append(pd)
                stats[class_name]["gt"].append(gt)

            for class_name in stats:
                f1 = f1_score(stats[class_name]["gt"], stats[class_name]["pred"])
                prec, recall = precision_score(
                    stats[class_name]["gt"], stats[class_name]["pred"]
                ), recall_score(stats[class_name]["gt"], stats[class_name]["pred"])
                (
                    result[f"{class_name}_f1"],
                    result[f"{class_name}_prec"],
                    result[f"{class_name}_recall"],
                ) = (f1, prec, recall)
            result["macro_f1"] = np.mean(
                [v for k, v in result.items() if k.endswith("_f1")]
            )
            result["key_score"] = result["macro_f1"]
            return result

        def compute_metrics_gpt2(p):
            result = {}
            inputs = p.inputs
            logits = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )  # [N, L]
            labels = p.label_ids

            assert logits.shape[-1] == 2  # pruned

            stats = {}
            for i in range(len(inputs)):
                # find the pas token id
                x, p, y = inputs[i], logits[i], labels[i]
                pad_idxs = np.arange(len(x))[x == self.tokenizer.pad_token_id]
                if pad_idxs.size > 0:
                    ans_idx = pad_idxs[0] - 1
                else:
                    raise ValueError
                prob = p[ans_idx]

                class_name = self.extract_class_name(x, model_type)

                if (
                    class_name is None
                ):  # possible because prompt + sentence exceed max length
                    continue  # and state in paper

                gt = self.extract_ans(y)
                # print(prob)
                if prob[1] > prob[0]:
                    pd = 1
                else:
                    pd = 0

                if class_name not in stats:
                    stats[class_name] = {"pred": [], "gt": []}
                stats[class_name]["pred"].append(pd)
                stats[class_name]["gt"].append(gt)

            for class_name in stats:
                f1 = f1_score(stats[class_name]["gt"], stats[class_name]["pred"])
                prec, recall = precision_score(
                    stats[class_name]["gt"], stats[class_name]["pred"]
                ), recall_score(stats[class_name]["gt"], stats[class_name]["pred"])
                (
                    result[f"{class_name}_f1"],
                    result[f"{class_name}_prec"],
                    result[f"{class_name}_recall"],
                ) = (f1, prec, recall)
            result["macro_f1"] = np.mean(
                [v for k, v in result.items() if k.endswith("_f1")]
            )
            result["key_score"] = result["macro_f1"]
            return result

        if model_type == "t5":
            return compute_metrics_t5
        else:
            return compute_metrics_gpt2

    def get_prune_idxs(self, model_config):
        yes_token = self.tokenizer.encode(" yes")
        no_token = self.tokenizer.encode(" no")
        yes_token, no_token = yes_token[0], no_token[0]
        model_config.prune_idxs = [no_token, yes_token]

    def extract_class_name(self, input_ids, model_type):
        if model_type == "t5":
            input_ids = input_ids[input_ids >= 0]
            s = self.tokenizer.decode(input_ids)
            idx1 = len("does the sentence express ")
            idx2 = s.index("?")
            class_name = s[idx1:idx2]
            return class_name
        else:
            input_ids = input_ids[input_ids >= 0]
            s = self.tokenizer.decode(input_ids)
            groups = re.search(r"The sentence expresses (\w+)\.", s)
            if groups is None:
                print("No gt label found in {}".format(s))
                class_name = None
            else:
                class_name = groups.group(1)
            return class_name

    def extract_ans(self, ids):
        ids = ids[ids >= 0]
        s = self.tokenizer.decode(ids).strip()
        if s.startswith("yes"):
            return 1
        return 0

    def get_metrics_func(self, model_config):
        if self.config.mtl:
            model_config = next(iter(vars(self.config.local_models._models).values()))
        return self.get_metrics_func_single(model_config)
