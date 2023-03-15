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

# same classification head init, model merging

orders = ["model2", "model5", "model1", "model3", "model0"]

for seed in [1, 2, 3, 4, 5]:
    for idx in range(2, len(orders) + 1):
        # simple
        to_merge = " ".join(orders[:idx])
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/ood/deberta-large-emotion-ood.yaml --filter_model {to_merge} --templates seed={seed}"
        )
        # fisher
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/ood/deberta-large-emotion-fisher-ood.yaml --filter_model {to_merge} --templates seed={seed}"
        )
        # regmean
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/ood/deberta-large-emotion-regmean-ood.yaml --filter_model {to_merge} --templates seed={seed}"
        )

# different classification head init, model merging

for seed in [1, 2, 3, 4, 5]:
    for idx in range(2, len(orders) + 1):
        to_merge = " ".join(orders[:idx])
        # simple
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/deberta/ood/deberta-large-emotion-ood.yaml --filter_model {to_merge} --templates dseed_generator={seed} seed={seed}"
        )
        # fisher
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/deberta/ood/deberta-large-emotion-fisher-ood.yaml --filter_model {to_merge} --templates dseed_generator={seed} seed={seed}"
        )
        # regmean
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_diffseed.yaml src/configs/exps/deberta/ood/deberta-large-emotion-regmean-ood.yaml --filter_model {to_merge} --templates dseed_generator={seed} seed={seed}"
        )

for seed in [1, 2, 3, 4, 5]:
    # multi task learning comparator
    to_merge = " ".join(orders)
    os.system(
        f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_mtl.yaml src/configs/exps/deberta/ood/deberta-large-emotion-mtl-ood.yaml --filter_model {to_merge} --templates seed={seed}"
    )

    # ensembling
    to_merge = " ".join(orders)
    os.system(
        f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/ood/deberta-large-emotion-ensemble-ood.yaml --filter_model {to_merge} --templates seed={seed}"
    )

    # individual models,  without merging
    for to_merge in orders:
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion.yaml src/configs/exps/deberta/ood/deberta-large-emotion-ood.yaml --filter_model {to_merge} --templates seed={seed}"
        )
