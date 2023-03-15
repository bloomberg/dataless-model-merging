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

for seed in [1, 2, 3, 4, 5]:
    for idx1 in range(0, 9):
        for idx2 in range(idx1 + 1, 9):
            os.system(
                f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/glue.yaml src/configs/exps/distilbert/distilbert-base.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
            )
            # fisher
            os.system(
                f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/glue.yaml src/configs/exps/distilbert/distilbert-fisher.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
            )
            # regmean
            os.system(
                f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/glue.yaml src/configs/exps/distilbert/distilbert-regmean.yaml --filter_model model{idx1} model{idx2} --templates seed={seed}"
            )
