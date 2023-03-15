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
from collections import Counter

import numpy as np


class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)


class Ensembler:
    # this is not a torch module
    def __init__(self, local_models):
        self.local_models = local_models

    def evaluate_ensemble(self, dataset):
        outputs = []

        label_freqs = []

        for local_model in self.local_models:
            _, output = local_model.evaluate(dataset, return_output=True)
            if self.local_models[0].config.ensembler.handle_missing_label:
                label_freqs.append(local_model.model_config.freq_dist)
            outputs.append(output)

        logging.info("Ensembling predictions of local models")

        if self.local_models[0].config.ensembler.handle_missing_label:
            logging.info("Handling missing label in some local models")
            # print(label_freqs)
            ensemble_preds = np.zeros_like(outputs[0].predictions)
            ns = np.zeros(ensemble_preds.shape[1])
            for label_freq, output in zip(label_freqs, outputs):
                for label_id, _ in label_freq.items():
                    ensemble_preds[:, label_id] += output.predictions[:, label_id]
                    ns[label_id] += 1
            ns += 1e-10
            preds = ensemble_preds / np.tile(
                np.expand_dims(ns, 0), (ensemble_preds.shape[0], 1)
            )
        elif self.local_models[0].config.ensembler.hard_ensemble:
            # print(output.predictions[0])
            ret = np.zeros_like(outputs[0].predictions)
            it = np.nditer(ret, flags=["multi_index"])
            for _ in it:
                items = [output.predictions[it.multi_index] for output in outputs]
                v = most_common(items)
                ret[it.multi_index] = v
            preds = ret
        else:
            preds = np.stack([output.predictions for output in outputs])
            preds = np.mean(preds, 0)

        ensemble_output = Struct(predictions=preds, label_ids=outputs[0].label_ids)

        # print(outputs[0])
        if hasattr(outputs[0], "inputs"):
            ensemble_output.inputs = outputs[0].inputs

        # assume that compute metrics func are the same
        compute_metrics_func = self.local_models[0].trainer.compute_metrics
        met = compute_metrics_func(ensemble_output)

        return met
