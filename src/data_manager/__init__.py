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

from .glue_data_manager import GLUEDataManager
from .emotion_data_manager import EmotionDataManager
from .ner_data_manager import NERDataManager
from .emotion_gen_data_manager import EmotionGenDataManager

DM_CLASS_MAP = {
    "glue": GLUEDataManager,
    "emotion": EmotionDataManager,
    "ner": NERDataManager,
}

DM_CLASS_MAP_GEN = {
    "emotion": EmotionGenDataManager,
}


def get_dm_class(dataset_name, seq2seq):
    if seq2seq:
        return DM_CLASS_MAP_GEN[dataset_name]
    else:
        return DM_CLASS_MAP[dataset_name]
