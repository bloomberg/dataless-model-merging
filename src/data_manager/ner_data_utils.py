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

import json
import os
from collections import OrderedDict


def convert_instances_to_feature_tensors(examples, tokenizer, label2idx):
    features = []
    # tokenize the word into word_piece / BPE
    # NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
    # Related GitHub issues:
    #      https://github.com/huggingface/transformers/issues/1196
    #      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
    #      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
    # assert tokenizer.add_prefix_space ## has to be true, in order to tokenize pre-tokenized input
    for idx, inst in enumerate(examples):
        words = inst["words"]
        orig_to_tok_index = []

        res = tokenizer.encode_plus(words, is_split_into_words=True)

        subword_idx2word_idx = res.word_ids(batch_index=0)
        prev_word_idx = -1
        for i, mapped_word_idx in enumerate(subword_idx2word_idx):
            """
            Note: by default, we use the first wordpiece/subword token to represent the word
            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
            """
            if mapped_word_idx is None:  # cls and sep token
                continue
            if mapped_word_idx != prev_word_idx:
                # because we take the first subword to represent the whold word
                orig_to_tok_index.append(i)
                prev_word_idx = mapped_word_idx
        assert len(orig_to_tok_index) == len(words)
        labels = inst["tags"]

        # pad / trunc
        raw_label_ids = [label2idx[label] for label in labels]
        label_ids = [-100] * len(res["input_ids"])

        for idx, label_id in zip(orig_to_tok_index, raw_label_ids):
            label_ids[idx] = label_id

        # leave for data collator

        # if len(res['input_ids']) > max_input_length:
        #     print('@@@@@@@@ Max seq len is too short {} @@@@@@@'.format(len(res['input_ids'])))
        #     res['input_ids'] = res['input_ids'][:max_input_length]
        #     res['attention_mask'] = res['attention_mask'][:max_input_length]
        #     label_ids = label_ids[:max_input_length]

        # if len(res['input_ids']) < max_input_length:
        #     res['input_ids'].extend([tokenizer.pad_token_id] * (max_input_length - len(res['input_ids'])))
        #     res['attention_mask'].extend([0] * (max_input_length - len(res['attention_mask'])))
        #     label_ids.extend([-100] * (max_input_length - len(label_ids)))

        features.append(
            OrderedDict(
                input_ids=res["input_ids"],
                attention_mask=res["attention_mask"],
                labels=label_ids,
            )
        )
    return features


class TabDatasetReader:
    def __init__(self, dir_format, split_mapping=None, label_space=None):
        super().__init__()
        self.dir_format = dir_format
        self.label_space = label_space  # if not None, then replace remaining tags as o
        self.split_mapping = split_mapping
        if self.split_mapping is None:
            self.split_mapping = {"train": "train", "dev": "dev", "test": "test"}

    def load_domain_split(self, domain, split):
        with open(
            self.dir_format.format(domain=domain, split=self.split_mapping[split])
        ) as f:
            lines = f.readlines()

        examples = []
        buffer = []
        for line in lines:
            line = line.strip()
            if not line:
                example = self.process_buffer(buffer)
                examples.append(example)
                buffer = []
            else:
                buffer.append(line)

        if buffer:
            example = self.process_buffer(buffer)
            examples.append(example)

        return examples

    def process_buffer(self, buffer):
        words, tags = [], []
        for line in buffer:
            # print(buffer)im41
            items = line.split()
            words.append(items[0])
            tags.append(items[1])
        words = self.convert_words(words)
        tags = self.convert_tags(tags)
        example = {"words": words, "tags": tags}
        return example

    def convert_words(self, words):
        converted_words = []
        for word in words:
            word = word.replace("/.", ".")
            if not word.startswith("-"):
                converted_words.append(word)
                continue
            tfrs = {
                "-LRB-": "(",
                "-RRB-": ")",
                "-LSB-": "[",
                "-RSB-": "]",
                "-LCB-": "{",
                "-RCB-": "}",
            }
            if word in tfrs:
                converted_words.append(tfrs[word])
            else:
                converted_words.append(word)
        return converted_words

    def convert_tags(self, tags):
        bio_tags = []
        flag = None
        for tag in tags:
            if tag != "O":
                pr, flag = tag.split("-")
                if self.label_space is not None and flag not in self.label_space:
                    bio_tags.append("O")
                else:
                    bio_tags.append(tag)
            else:
                bio_tags.append(tag)
        return bio_tags


class JsonlDatasetReader(TabDatasetReader):
    def __init__(self, dir_format, split_mapping=None, label_space=None):
        super().__init__(
            dir_format, split_mapping=split_mapping, label_space=label_space
        )

    def load_domain_split(self, domain, split):
        with open(
            self.dir_format.format(domain=domain, split=self.split_mapping[split])
        ) as f:
            lines = f.readlines()

        examples = []
        for line in lines:
            item = json.loads(line)
            words, tags = item["tokens"], item["labels"]
            words = self.convert_words(words)
            tags = self.convert_tags(tags)
            examples.append({"words": words, "tags": tags})

        return examples


class OntoNotesDataReader:
    def __init__(self, base_dir, label_space=None):
        super().__init__()
        self.base_dir = base_dir
        self.label_space = label_space  # if not None, then replace remaining tags as o

    def load_domain_split(self, domain, split):
        with open(
            os.path.join(
                self.base_dir, "v12_{}/english".format(domain), "{}.txt".format(split)
            )
        ) as f:
            lines = f.readlines()

        examples = []
        buffer = []
        for line in lines:
            line = line.strip()
            if not line:
                example = self.process_buffer(buffer)
                examples.append(example)
                buffer = []
            else:
                buffer.append(line)

        if buffer:
            example = self.process_buffer(buffer)
            examples.append(example)

        return examples

    def process_buffer(self, buffer):
        words, tags = [], []
        for line in buffer:
            # print(buffer)im41
            items = line.split()
            words.append(items[3])
            tags.append(items[10])
        words = self.convert_word(words)
        tags = self.convert_to_bio(tags)
        example = {"words": words, "tags": tags}
        return example

    def convert_to_bio(self, tags):
        bio_tags = []
        flag = None
        for tag in tags:
            label = tag.strip("()*")
            if "(" in tag:

                flag = label

                # post process, following Wang et al.
                # Multi-Domain Named Entity Recognition with Genre-Aware and Agnostic Inference

                if flag in ["LOC", "FAC", "GPE"]:
                    flag = "LOC"
                if flag in ["PERSON"]:
                    flag = "PER"

                if self.label_space is None or flag in self.label_space:
                    bio_label = "B-" + flag
                else:
                    flag = ""
                    bio_label = "O"
            elif flag:
                bio_label = "I-" + flag
            else:
                bio_label = "O"
            if ")" in tag:
                flag = None
            bio_tags.append(bio_label)
        return bio_tags


if __name__ == "__main__":
    reader = OntoNotesDataReader("OntoNotes-5.0-NER", ["LOC", "PER", "ORG"])
    examples = reader.load_domain_split("bc", "train")

    label_set = set()
    for example in examples:
        for label in example["tags"]:
            label_set.add(label)

    print(label_set)
