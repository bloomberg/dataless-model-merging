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
import re

import torch

from .misc import filter_params_to_merge


class ModelMergerBase:
    def __init__(
        self, config, merger_config, local_models, global_model, merger_ds=None
    ):
        self.local_models = local_models
        self.global_model = global_model
        self.merger_ds = merger_ds
        self.merger_config = merger_config
        self.config = config


class FedAvgMerger(ModelMergerBase):
    def __init__(
        self, config, merger_config, local_models, global_model, merger_ds=None
    ):
        super().__init__(config, merger_config, local_models, global_model, merger_ds)

    def merge_to_global(self, model_coeffs=None, fisher_weights=None, **kwargs):
        self.avg_merge(
            self.local_models, self.global_model, model_coeffs, fisher_weights, **kwargs
        )

    def avg_merge(
        self,
        local_models,
        global_model,
        model_coeffs=None,
        fisher_weights=None,
        all_grams=None,
        **kwargs,
    ):
        params = {}
        for local_model in local_models:
            n2p = {k: v for k, v in local_model.base.named_parameters()}
            merge_param_names = filter_params_to_merge(
                [n for n in n2p], self.merger_config.exclude_param_regex
            )
            for n in merge_param_names:
                if n not in params:
                    params[n] = []
                params[n].append(n2p[n])

        if all_grams:
            assert not fisher_weights
            avg_params = self.regmean_mean_params(
                params, all_grams, model_coeffs=model_coeffs
            )

        elif fisher_weights:  # fisher weighted averaging
            if not model_coeffs:
                model_coeffs = torch.ones(len(local_models)) / len(local_models)
            elif not torch.is_tensor(model_coeffs):
                model_coeffs = torch.FloatTensor(model_coeffs)
            avg_params = self.fisher_weighted_average(
                params, model_coeffs, fisher_weights
            )

        # average all the params for params t- merge
        else:  # simple average
            if not model_coeffs:
                avg_params = {k: torch.stack(v, 0).mean(0) for k, v in params.items()}
            else:
                avg_params = {}
                for k, v in params.items():
                    s = 0
                    for model_idx, local_model in enumerate(local_models):
                        coeff = model_coeffs[model_idx]
                        param = v[model_idx]
                        s += coeff * param
                    avg_params[k] = s

        # special treatments for multi label classification
        if (
            self.local_models[0].model_config.task_type == "multi_label"
            and not fisher_weights
            and self.merger_config.multi_label_head_special
        ):
            (
                head_param_name,
                head_param,
            ) = self.avg_merge_multilabel_classification_heads(params)
            avg_params[head_param_name] = head_param

        for n, p in global_model.base.named_parameters():
            if n in avg_params:
                p.data.copy_(avg_params[n])

    def to_diag(self, cov_mat):
        mask = torch.diag(torch.ones(cov_mat.size(0))).to(cov_mat.device)
        diag_cov_mat = mask * cov_mat
        return diag_cov_mat

    def reduce_non_diag(self, cov_mat, a):
        diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
        non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
        weight = diag_weight + non_diag_weight
        ret = cov_mat * weight
        return ret

    def regmean_mean_params(self, all_params, all_grams, model_coeffs=None):
        # here param is dict(list) but all_grams is list(dict) -- why did i write in this way
        avg_params = {}
        n_model = len(all_grams)
        # special treatments for linear weight params
        for name in all_params:
            h_avged = False
            valid_for_regmean = not any(
                [
                    re.match(patt, name)
                    for patt in self.merger_config.regmean_exclude_param_regex
                ]
            )
            if name.endswith(".weight") and valid_for_regmean:
                logging.info(f"Regmean: {name}")
                module_name = name[: -len(".weight")]
                if module_name in all_grams[0]:
                    gram_m_ws, grams = [], []

                    is_conv = "meta_info" in all_grams[0] and module_name in all_grams[
                        0
                    ]["meta_info"].get("conv1d", [])

                    for model_id, model_grams in enumerate(all_grams):
                        param_grams = model_grams[module_name]

                        if self.merger_config.regmean_diag:
                            param_grams = self.to_diag(param_grams)

                        if self.merger_config.regmean_reduce_nondiag >= 0:
                            param_grams = self.reduce_non_diag(
                                param_grams, a=self.merger_config.regmean_reduce_nondiag
                            )

                        param = all_params[name][model_id]
                        if model_coeffs is not None:
                            coeff = (
                                model_coeffs[model_id] * n_model
                            )  # according to formula
                            param_grams = param_grams * coeff

                        if is_conv:
                            gram_m_ws.append(torch.matmul(param_grams, param))
                        else:
                            gram_m_ws.append(
                                torch.matmul(param_grams, param.transpose(0, 1))
                            )

                        grams.append(param_grams)
                    sum_cov = sum(grams)
                    sum_gram_m_ws = sum(gram_m_ws)
                    sum_cov_inv = torch.inverse(sum_cov)
                    wt = torch.matmul(sum_cov_inv, sum_gram_m_ws)

                    if is_conv:
                        w = wt
                    else:
                        w = wt.transpose(0, 1)

                    avg_params[name] = w
                    h_avged = True
            if not h_avged:
                if model_coeffs is None:
                    avg_params[name] = torch.stack(all_params[name], 0).mean(0)
                else:
                    params = torch.stack(all_params[name], 0)
                    coeff = model_coeffs
                    if not torch.is_tensor(coeff):
                        coeff = torch.FloatTensor(coeff)
                    coeff = coeff.view(-1, *[1 for _ in range(params.dim() - 1)]).to(
                        params.device
                    )
                    avg_params[name] = (params * coeff).sum(0)
        return avg_params

    def deliver_to_local(self):
        n2p = {k: v for k, v in self.global_model.named_parameters()}
        merge_param_names = filter_params_to_merge(
            [n for n in n2p], self.merger_config.exclude_param_regex
        )
        for local_model in self.local_models:
            for n, p in local_model.named_parameters():
                if n in merge_param_names:
                    p.data.copy_(n2p[n].data)

    def fisher_weighted_average(self, all_params, model_coeffs, fisher_weights):
        avg_params = {}

        fisher_norms, concat_norm = None, None
        if self.merger_config.fisher_normalize is not None:
            fisher_norms, concat_norm = self.get_fisher_norm_coeff(
                all_params, fisher_weights
            )

        for n, params in all_params.items():
            params = torch.stack(params)  # [N, *]
            fisher = (
                torch.stack([x[n] for x in fisher_weights])
                + self.merger_config.fisher_smooth
            )  # [N, *]

            coeff = model_coeffs.view(-1, *[1 for _ in range(params.dim() - 1)]).to(
                params.device
            )

            if self.merger_config.fisher_normalize:
                if self.merger_config.fisher_normalize == "param":
                    fisher_norm = fisher_norms[n]
                elif self.merger_config.fisher_normalize == "model":
                    fisher_norm = concat_norm
                else:
                    raise NotImplementedError
                fisher_norm_coeff = 1.0 / (
                    fisher_norm + self.merger_config.fisher_smooth
                )  # [N]
                fisher_norm_coeff = fisher_norm_coeff / fisher_norm_coeff.sum()
                fisher_norm_coeff = fisher_norm_coeff.view(
                    -1, *[1 for _ in range(params.dim() - 1)]
                )
                coeff = coeff * fisher_norm_coeff

            sum_p = params * fisher * coeff
            sum_p = sum_p.sum(0)

            denom = (fisher * coeff).sum(0)

            avg_p = sum_p / denom
            avg_params[n] = avg_p
        return avg_params

    def get_fisher_norm_coeff(self, all_params, fisher_weights):
        norms = {}
        for n, _ in all_params.items():
            fisher = torch.stack([x[n] for x in fisher_weights])
            dims = [_ for _ in range(1, fisher.dim())]
            fisher_norm = torch.norm(fisher, dim=dims)  # [N]
            norms[n] = fisher_norm

        concat_norm = torch.stack([v for v in norms.values()], 1)  # [N, Pc]
        concat_norm = torch.norm(concat_norm, dim=1)
        return norms, concat_norm

    def avg_merge_multilabel_classification_heads(self, params):
        head_params = None
        head_param_name = None
        for n, ps in params.items():
            if any(
                [
                    re.match(patt, n)
                    for patt in self.merger_config.multilabel_head_params
                ]
            ):
                if head_params is not None:
                    raise ValueError
                head_params = ps
                head_param_name = n

        merged_head = torch.zeros_like(head_params[0])
        n_valid = torch.zeros(merged_head.size(0)).to(merged_head.device)
        for model_id, param in enumerate(head_params):
            freq_dist = self.local_models[model_id].model_config.freq_dist
            for label_id, _ in freq_dist.items():
                merged_head[label_id] += param[label_id]
                n_valid[label_id] += 1

        for i in range(n_valid.size(0)):
            if n_valid[i].item() < 1e-10:
                n_valid[i] = 1e-10

        merged_head = merged_head / n_valid.view(-1, 1)
        return head_param_name, merged_head
