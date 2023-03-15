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

# same classification head init

for seed in [1, 2, 3, 4, 5]:
    for idx1 in range(0, 6):
        for idx2 in range(idx1 + 1, 6):
            if idx1 != 4 and idx2 != 4:  # this dataset uses a different label space
                # simple
                os.system(
                    f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/deberta-large-emotion.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
                )
                # fisher
                os.system(
                    f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/deberta-large-emotion-fisher.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
                )
                # regmean
                os.system(
                    f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/deberta-large-emotion-regmean.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
                )

# different classification head init

for seed in [1, 2, 3, 4, 5]:
    for idx1 in range(0, 6):
        for idx2 in range(idx1 + 1, 6):
            if idx1 != 4 and idx2 != 4:  # this dataset uses a different label space
                # simple
                os.system(
                    f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/deberta/deberta-large-emotion.yaml --filter_model model{idx1} model{idx2} --templates dseed_generator={seed} seed={seed}"
                )
                # fisher
                os.system(
                    f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/deberta/deberta-large-emotion-fisher.yaml --filter_model model{idx1} model{idx2} --templates dseed_generator={seed} seed={seed}"
                )
                # regmean
                os.system(
                    f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/deberta/deberta-large-emotion-regmean.yaml --filter_model model{idx1} model{idx2} --templates dseed_generator={seed} seed={seed}"
                )
