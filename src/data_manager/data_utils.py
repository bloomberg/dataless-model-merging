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

import copy
import hashlib
import json
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers.data.data_collator import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)


def get_dataset_digest(dataset):
    s = json.dumps(
        dataset.to_dict() if not type(dataset) is dict else dataset, sort_keys=True
    ).encode("utf-8")
    md5 = hashlib.md5(s).hexdigest()
    return md5


def pad_tensor_sequence(x, max_len, pad_value):
    # time dimension is at dim 1
    if x.size(1) == max_len:
        return x
    if x.size(1) > max_len:
        raise ValueError(str((x.size(), max_len)))
    target_size = [_ for _ in x.size()]
    target_size[1] = max_len - x.size(1)
    pad = torch.full(target_size, pad_value, dtype=x.dtype).to(x.device)
    ret = torch.cat([x, pad], 1)
    return ret


def remove_feature_str(features):
    # for dict item in features, remove key with value type str
    # self.tokenizer.pad will raise exception otherwise
    cache = {}
    for f in features:
        rm_keys = []
        for k, v in f.items():
            if type(v) is str:
                if k not in cache:
                    cache[k] = []
                cache[k].append(v)
                rm_keys.append(k)
        for k in rm_keys:
            f.pop(k)
    return cache


def add_cached_feature_to_batch(batch, cache):
    for k, v in cache.items():
        batch[k] = v


class DataCollatorWithPaddingExcludeStr(DataCollatorWithPadding):
    def __call__(self, features):
        cache = remove_feature_str(features)

        if "labels" in features[0] and type(features[0]["labels"]) is list:
            labels = [x["labels"] for x in features]
            max_t = max([len(x) for x in labels])
            # is seq2seq, pad with -100
            for i, feat in enumerate(features):
                feat["labels"] += [-100] * (max_t - len(feat["labels"]))

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        add_cached_feature_to_batch(batch, cache)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch


class DataCollatorForTokenClassificationExcludeStr(DataCollatorForTokenClassification):
    def torch_call(self, features):
        import torch

        cache = remove_feature_str(features)

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            add_cached_feature_to_batch(batch, cache)
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        add_cached_feature_to_batch(batch, cache)
        return batch

    def tf_call(self, features):
        import tensorflow as tf

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        cache = remove_feature_str(features)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="tf" if labels is None else None,
        )

        if labels is None:
            add_cached_feature_to_batch(batch, cache)
            return batch

        sequence_length = tf.convert_to_tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        batch = {k: tf.convert_to_tensor(v, dtype=tf.int64) for k, v in batch.items()}
        add_cached_feature_to_batch(batch, cache)
        return batch

    def numpy_call(self, features):
        import numpy as np

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        cache = remove_feature_str(features)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="np" if labels is None else None,
        )

        if labels is None:
            add_cached_feature_to_batch(batch, cache)
            return batch

        sequence_length = np.array(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        add_cached_feature_to_batch(batch, cache)
        return batch


class ConcatDatasetWithID(Dataset):
    def __init__(self, dataset_names, datasets) -> None:
        super().__init__()
        self.names = dataset_names
        self.dss = [
            DatasetWithDsName(ds, name) for ds, name in zip(datasets, self.names)
        ]
        self.is_mtl = True

    def __getitem__(self, index):
        ds_id = 0
        if index < 0 or index >= len(self):
            raise IndexError(
                "Index {} but the length of datasets are {}".format(
                    index, [len(x) for x in self.dss]
                )
            )
        while index >= len(self.dss[ds_id]):
            index -= len(self.dss[ds_id])
            ds_id += 1
        item = self.dss[ds_id][index]
        return item

    def __len__(self):
        return sum([len(x) for x in self.dss])

    def get_dss(self):
        return self.dss

    def get_ds(self, dataset_name):
        return self.dss[self.names.index(dataset_name)]

    def get_ds_names(self):
        return self.names


class DatasetWithDsName(Dataset):
    def __init__(self, dataset, ds_name):
        super().__init__()
        self.dataset = dataset
        self.ds_name = ds_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item["dataset"] = self.ds_name
        return item


class MTLDataloader(DataLoader):
    def __init__(self, dummy_dataset, *args, **kwargs):
        data_loaders = kwargs.pop("data_loaders")
        mtl_bal_sampling = kwargs.pop("mtl_bal_sampling")
        self.task_keys = kwargs.pop("task_keys")
        self.task_num = kwargs.pop("task_num")
        self.pad_token_id = kwargs.pop("pad_token_id")
        self.sqrt = kwargs.pop("sqrt")
        super().__init__(dummy_dataset, *args, **kwargs)
        self.data_loaders = data_loaders
        self.mtl_bal_sampling = mtl_bal_sampling
        self.loader_len = sum([len(x) for x in self.data_loaders])
        # just for random sampling of examples from a task
        self.task_iterators = [iter(x) for x in self.data_loaders]

    def __iter__(self):
        def mtl_data_iterator():
            draws = []
            for i in range(self.task_num):
                draws.extend([i] * len(self.data_loaders[i]))
            iterators = [iter(_) for _ in self.data_loaders]
            random.shuffle(draws)
            self.loader_len = len(draws)
            for loader_id in draws:
                iterator = iterators[loader_id]
                yield next(iterator)

        def mtl_bal_data_iterator():
            draws = []
            max_dataloader_len = max([len(x) for x in self.data_loaders])
            for i in range(self.task_num):
                if self.sqrt:
                    # x : max_dataloader_len = sqrt(len(x)) : sqrt(len(max_dataloader_len))
                    batch_num = int(
                        max_dataloader_len
                        * (len(self.data_loaders[i]) ** 0.5)
                        // (max_dataloader_len**0.5)
                    )
                    draws.extend([i] * batch_num)
                else:
                    draws.extend([i] * max_dataloader_len)
            iterators = [iter(_) for _ in self.data_loaders]
            random.shuffle(draws)
            self.loader_len = len(draws)
            for loader_id in draws:
                task_name = self.task_keys[loader_id]
                iterator = iterators[loader_id]
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterators[loader_id] = iter(self.data_loaders[loader_id])
                    iterator = iterators[loader_id]
                    batch = next(iterator)
                yield (loader_id, task_name), batch

        if self.mtl_bal_sampling:
            return mtl_bal_data_iterator()
        else:
            return mtl_data_iterator()

    def concat_tensors_with_pad(self, tensors, pad_value):
        if len(tensors[0].size()) < 2:
            return torch.cat(tensors, 0)
        max_len = max([x.size(1) for x in tensors])
        padded = []
        for x in tensors:
            pad_x = pad_tensor_sequence(x, max_len, pad_value)
            padded.append(pad_x)
        padded = torch.cat(padded, 0)
        return padded

    def sample_batch_from_task(self, task_id):
        try:
            batch = next(self.task_iterators[task_id])
        except StopIteration:
            self.task_iterators[task_id] = iter(self.data_loaders[task_id])
            batch = next(self.task_iterators[task_id])
        return batch

    def __len__(self):
        return self.loader_len


def get_random_subset(dataset, seed, total):
    rng = np.random.default_rng(seed)
    if total > len(dataset):
        total = len(dataset)
    idxs = rng.choice(len(dataset), total, replace=False).tolist()
    subset = Subset(dataset, idxs)
    return subset


def get_quantify_func(lst, partition_num):
    minx, maxx = np.min(lst), np.max(lst)
    part_size = (maxx - minx) / partition_num

    def f(x):
        if x == maxx:
            return partition_num - 1
        return int((x - minx) / part_size)

    return f


def niid_partition_by_label(
    dataset, seed, n_partitions, main_portion=0.8, is_regression=False
):
    rng = random.Random(seed)

    if is_regression:
        targets = [x["label"] for x in dataset]
        quantify_func = get_quantify_func(targets, n_partitions)
        labels = [quantify_func(x) for x in targets]
        label_set = sorted(list(set(labels)))
    else:
        labels = [x["label"] for x in dataset]
        label_set = sorted(list(set(labels)))

    l2lwidx = {}
    for idx, item in enumerate(dataset):
        label = item["label"] if not is_regression else quantify_func(item["label"])
        if label not in l2lwidx:
            l2lwidx[label] = []
        l2lwidx[label].append((idx, label))
    l2lwidx_ro = copy.copy(l2lwidx)

    all_res = []

    def sample_and_remove(label, n):
        if len(l2lwidx[label]) < n:
            logging.warning(
                "niid partition: sample size {} > {} for label {}".format(
                    n, len(l2lwidx[label]), label
                )
            )
        ret = set(rng.sample(l2lwidx[label], n))
        l2lwidx[label] = [x for x in l2lwidx[label] if x not in ret]
        ret = list(ret)
        return ret

    main_label_order = [_ for _ in label_set]
    rng.shuffle(main_label_order)

    for idx in range(n_partitions - 1):
        items = []
        main_label = main_label_order[idx]
        main_cnt = int(main_portion * len(l2lwidx_ro[main_label]))
        ret = sample_and_remove(main_label, main_cnt)
        items.extend(ret)

        for other_label in [x for x in label_set if x != main_label]:
            other_cnt = int(
                ((1 - main_portion) / (n_partitions - 1)) * len(l2lwidx[other_label])
            )
            ret = sample_and_remove(other_label, other_cnt)
            items.extend(ret)
        rng.shuffle(items)
        all_res.append(items)

    # add remaining items
    items = []
    for label, x in l2lwidx.items():
        items.extend(x)
    all_res.append(items)

    dss = []
    # create datasets
    for i, x in enumerate(all_res):
        idxs = [_[0] for _ in x]
        ds = Subset(dataset, idxs)
        dss.append(ds)
    return dss
