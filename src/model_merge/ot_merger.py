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

import numpy as np
import ot
import torch
from torch import nn

from .avg_merger import FedAvgMerger
from .misc import filter_modules_by_regex
from .net import create_model
from .ot_utils.ot_ground_metric import GroundMetric


class TmpLocalModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model


def get_submodule(m, name):
    ns = name.split(".")
    r = m
    for n in ns:
        r = getattr(r, n)
    return r


class OptimalTransportMerger(FedAvgMerger):
    def __init__(
        self, config, merger_config, local_models, global_model, merger_ds=None
    ):
        super().__init__(config, merger_config, local_models, global_model, merger_ds)
        self.ot_params = merger_config.ot_params
        self.tgt_local_model = None

    def merge_to_global(self, **kwargs):
        tgt_local_model, _ = create_model(
            self.config, self.local_models[0].model_config
        )
        self.tgt_local_model = TmpLocalModel(tgt_local_model)
        if torch.cuda.is_available():
            self.tgt_local_model = self.tgt_local_model.cuda()
        self.tgt_local_model.load_state_dict(self.local_models[0].state_dict())

        # match model a into other models
        with torch.no_grad():
            self.match_all_ffns(self.local_models, self.tgt_local_model)

        # replace model a with the matched model
        all_local_models = nn.ModuleList()
        all_local_models.append(self.tgt_local_model)
        for model in self.local_models[1:]:
            all_local_models.append(model)
        self.avg_merge(all_local_models, self.global_model, **kwargs)

    def match_all_ffns(self, local_models, tgt_local_model):
        # we assume the input and outputs of the ffn layers are aligned
        local_modules = []

        ot_patterns = [v for v in vars(self.merger_config.ot_patterns).values()]
        for ot_pattern in ot_patterns:
            for local_model_id, local_model in enumerate(local_models):
                modules = filter_modules_by_regex(
                    local_model, ot_pattern.ot_filter_regex, include_type=None
                )
                local_modules.append(modules)

            tgt_modules = filter_modules_by_regex(
                tgt_local_model, ot_pattern.ot_filter_regex, include_type=None
            )

            for ffn_name in tgt_modules:
                logging.info("Matching {}".format(ffn_name))
                single_ffns = [x[ffn_name] for x in local_modules]
                tgt_ffn = tgt_modules[ffn_name]
                self.match_ffns(single_ffns, tgt_ffn, ot_pattern)

    def match_ffns(self, ffns, tgt_ffn, ot_pattern):
        # input: list of ffns, on for each local model (a and b)
        assert len(ffns) == 2
        eps = 1e-10
        layers = [
            [get_submodule(x, ot_pattern.ot_lin1) for x in ffns]
            + [get_submodule(tgt_ffn, ot_pattern.ot_lin1)],
            [get_submodule(x, ot_pattern.ot_lin2) for x in ffns]
            + [get_submodule(tgt_ffn, ot_pattern.ot_lin2)],
        ]
        ground_metric_object = GroundMetric(self.ot_params)

        T_var = None

        for layer_id, (lina, linb, tgt_lin) in enumerate(layers):
            w_a = lina.weight.data
            w_b = linb.weight.data
            w_tgt = tgt_lin.weight.data

            mu_card, nu_card = w_a.shape[0], w_b.shape[0]

            if layer_id == 0:
                M = ground_metric_object.process(w_a, w_b).to(w_a.device)
                aligned_wt = w_a
            else:
                aligned_wt = torch.matmul(w_a, T_var).to(w_a.device)
                M = ground_metric_object.process(aligned_wt, w_b)

            mu = np.ones(mu_card) / mu_card
            nu = np.ones(nu_card) / nu_card

            cpuM = M.data.cpu().numpy()
            if self.ot_params.exact:
                T = ot.emd(mu, nu, cpuM)
            else:
                T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=self.ot_params.reg)

            T_var = torch.from_numpy(T).float().to(w_a.device)

            if self.ot_params.debug:
                logging.info("The trace of T is {}".format(T_var.trace()))

            if self.ot_params.correction:
                if not self.ot_params.proper_marginals:
                    # think of it as m x 1, scaling weights for m linear combinations of points in X
                    marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                    marginals.to(w_a.device)
                    marginals = torch.diag(1.0 / (marginals + eps)).to(
                        w_a.device
                    )  # take inverse
                    T_var = torch.matmul(T_var, marginals)
                else:
                    marginals_beta = T_var.t() @ torch.ones(
                        T_var.shape[0], dtype=T_var.dtype
                    )

                    marginals = 1 / (marginals_beta + eps)
                    print("shape of inverse marginals beta is ", marginals_beta.shape)
                    print("inverse marginals beta is ", marginals_beta)

                    T_var = T_var * marginals
                    # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                    # this should all be ones, and number equal to number of neurons in 2nd model
                    print(T_var.sum(dim=0))

            if self.ot_params.past_correction:
                matched_w_a = torch.matmul(
                    T_var.transpose(0, 1), aligned_wt.reshape(aligned_wt.shape[0], -1)
                )
            else:
                matched_w_a = torch.matmul(
                    T_var.transpose(0, 1), w_a.view(w_a.shape[0], -1)
                )

            w_tgt.copy_(matched_w_a)
