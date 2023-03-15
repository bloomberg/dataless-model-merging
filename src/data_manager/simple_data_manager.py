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

from torch.utils.data import ConcatDataset, Dataset


class SimpleDataManager:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.attributes = None
        self.partitions = None

    def join_datasets(self, datasets):
        ds = ConcatDataset(datasets)
        return ds


class FeatureDataset(Dataset):
    def __init__(self, features) -> None:
        super().__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
