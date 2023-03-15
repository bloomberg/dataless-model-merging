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
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from transformers import Seq2SeqTrainingArguments, TrainingArguments

from ..remote_io import zoo
from ..utils.config import merge_config, post_process_hf_training_args
from .avg_merger import FedAvgMerger
from .ensembler import Ensembler
from .local_trainer import MyModelTrainer, MySeq2SeqModelTrainer
from .net import create_model
from .ot_merger import OptimalTransportMerger


def fmt(name, met_dict):
    return "{}\t{}".format(name, json.dumps(met_dict))


class LocalModel(nn.Module):
    def __init__(self, config, name, model_config, dm) -> None:
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.name = name
        self.base, _ = create_model(self.config, model_config)
        if torch.has_cuda:
            self.base = self.base.to(model_config.device)
        self.dm = dm
        self._prev_load_dir = None
        self._remote_zoo_dir, self._local_zoo_dir = None, None
        self.mtl = config.mtl

        self.dataset_names, self.train_dataset, self.eval_dataset = None, None, None
        if self.mtl:
            (
                self.dataset_names,
                self.train_dataset,
                self.eval_dataset,
                self.test_dataset,
            ) = self.dm.load_mtl_dataset(mtl_config=model_config)
        else:
            (
                self.train_dataset,
                self.eval_dataset,
                self.test_dataset,
            ) = self.dm.load_dataset(
                model_config=model_config,
            )

        self.output_dir = config.local_models.output_dir_format.format(
            main_output_dir=config.main_output_dir, name=name
        )
        self.load_dir = (
            None
            if config.load_dir is None
            else config.local_models.output_dir_format.format(
                main_output_dir=config.load_dir, name=name
            )
        )
        training_args = (
            TrainingArguments(output_dir=self.output_dir)
            if not self.config.seq2seq
            else Seq2SeqTrainingArguments(output_dir=self.output_dir)
        )
        merge_config(training_args, self.model_config)
        # print('Training args', training_args)
        post_process_hf_training_args(training_args)
        self.training_args = training_args
        collator_cls = dm.get_collator_cls()

        if self.config.seq2seq:
            self.trainer = MySeq2SeqModelTrainer(
                self.base,
                training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.dm.get_metrics_func(model_config=model_config),
                data_collator=collator_cls(dm.tokenizer),
                tokenizer=dm.tokenizer,
                is_mtl=self.mtl,
            )
        else:
            self.trainer = MyModelTrainer(
                self.base,
                training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.dm.get_metrics_func(model_config=model_config),
                data_collator=collator_cls(dm.tokenizer),
                tokenizer=dm.tokenizer,
                is_mtl=self.mtl,
            )

    def train_model(self):
        logging.info("Started training model {}".format(self.name))
        self.trainer.train()

        self.trainer.save_model()
        self.model_config.save(self.output_dir, "model_config.json")

        if self.config.push_to_local_zoo:
            logging.info("Pushing trained {} to the local zoo".format(self.name))
            self._local_zoo_dir = zoo.save_to_zoo(
                self.config, self.model_config, self.output_dir
            )
        if self.config.push_to_remote_zoo:
            logging.info("Pushing trained {} to the remote zoo".format(self.name))
            self._remote_zoo_dir = zoo.save_to_remote_zoo(
                self.config, self.model_config, self.output_dir
            )

    def train_if_needed(self):
        if self.config.load_from_checkpoint:
            # load checkpoints from the output dir
            logging.info("Loading checkpoint from {}".format(self.load_dir))
            self.trainer.load_from_checkpoint(self.load_dir)
        elif self._prev_load_dir is not None:
            # directly load from previous dir
            self.trainer.load_from_checkpoint(self._prev_load_dir)
        elif self.config.load_from_zoo in ["maybe", "yes"]:
            if self.config.load_from_zoo_use_remote:
                ckpt_dir, zoo_dir = zoo.fetch_from_remote_zoo(
                    self.config, self.model_config
                )
                self._remote_zoo_dir = zoo_dir
            else:
                ckpt_dir = zoo.fetch_from_zoo(self.config, self.model_config)
                self._local_zoo_dir = ckpt_dir
            if ckpt_dir:
                self._prev_load_dir = ckpt_dir
                self.trainer.load_from_checkpoint(ckpt_dir)
            elif self.config.load_from_zoo in ["maybe"]:
                self.train_model()
                self._prev_load_dir = self.output_dir
            else:
                raise FileNotFoundError("Cannot find model from the zoo")
        else:
            self.train_model()
            self._prev_load_dir = self.output_dir

    def post_merge_train(self):
        # deprecated
        logging.info("Post merge training of {}".format(self.name))
        # train some components (e.g. classification head after merge)

        post_merge_training_args = copy.deepcopy(self.training_args)
        merge_config(post_merge_training_args, self.config.post_merge_model_args)

        collator_cls = self.dm.get_collator_cls()
        post_merge_trainer = MyModelTrainer(
            self.base,
            post_merge_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.dm.get_metrics_func(model_config=self.model_config),
            data_collator=collator_cls(self.dm.tokenizer),
            tokenizer=self.dm.tokenizer,
        )

        if self.config.post_merge_reinit:
            post_merge_trainer.reinit_trainable_params()

        post_merge_trainer.train()

    def prepare_fisher(self):
        return self.prepare_extra_data(
            extra_data_name="fisher",
            version=self.config.merger.fisher_version,
            compute_func=self.trainer.compute_fisher,
        )

    def prepare_gram(self):
        return self.prepare_extra_data(
            extra_data_name="gram",
            version=self.config.merger.gram_version,
            compute_func=self.trainer.compute_grams,
        )

    def prepare_extra_data(self, extra_data_name, version, compute_func):
        # must be called after "train_if_needed"
        assert self._prev_load_dir is not None
        name = "{}_v{}.pkl".format(extra_data_name, version)
        logging.info(os.path.join(self._prev_load_dir, name))
        logging.info(os.path.exists(os.path.join(self._prev_load_dir, name)))
        if os.path.exists(os.path.join(self._prev_load_dir, name)):
            with open(os.path.join(self._prev_load_dir, name), "rb") as f:
                extra_data = torch.load(f)
        elif os.path.exists(os.path.join(self.output_dir, name)):
            with open(os.path.join(self.output_dir, name), "rb") as f:
                extra_data = torch.load(f)
        else:
            extra_data = compute_func(self.config, self.model_config)
            with open(os.path.join(self.output_dir, name), "wb") as wf:
                torch.save(extra_data, wf)

            if self.config.push_to_local_zoo:
                logging.info(
                    "Pushing {} {} to the local zoo".format(extra_data_name, self.name)
                )
                zoo.save_extra_info_to_zoo(name, self.output_dir, self._local_zoo_dir)
            if self.config.push_to_remote_zoo:
                logging.info(
                    "Pushing {} {} to the remote zoo".format(extra_data_name, self.name)
                )
                zoo.save_extra_info_to_remote_zoo(
                    name, self.output_dir, self._remote_zoo_dir
                )
        return extra_data

    def evaluate(self, dataset, **kwargs):
        met = self.trainer.evaluate(eval_dataset=dataset, **kwargs)
        return met

    def load_previous_checkpoint(self):
        return self.trainer.load_from_checkpoint(self._prev_load_dir)


class GlobalModel(nn.Module):
    def __init__(self, config, dm) -> None:
        super().__init__()
        self.config = config
        # use first local model config as the global model config
        model_config = next(iter(vars(config.local_models.models).values()))
        self.base, _ = create_model(self.config, model_config)
        if torch.has_cuda:
            self.base = self.base.to(config.global_device)
        self.dm = dm

        self.output_dir = config.global_model.output_dir_format.format(
            main_output_dir=config.main_output_dir
        )
        training_args = (
            TrainingArguments(output_dir=self.output_dir)
            if not self.config.seq2seq
            else Seq2SeqTrainingArguments(output_dir=self.output_dir)
        )
        post_process_hf_training_args(training_args)
        collator_cls = dm.get_collator_cls()
        if self.config.seq2seq:
            self.trainer = MySeq2SeqModelTrainer(
                self.base,
                training_args,
                train_dataset=None,
                eval_dataset=None,
                compute_metrics=self.dm.get_metrics_func(model_config=model_config),
                data_collator=collator_cls(dm.tokenizer),
                is_mtl=self.config.mtl,
            )
        else:
            self.trainer = MyModelTrainer(
                self.base,
                training_args,
                train_dataset=None,
                eval_dataset=None,
                compute_metrics=self.dm.get_metrics_func(model_config=model_config),
                data_collator=collator_cls(dm.tokenizer),
                is_mtl=self.config.mtl,
            )

    def evaluate(self, dataset, **kwargs):
        met = self.trainer.evaluate(eval_dataset=dataset, **kwargs)
        return met


class ModelMergeExp:
    def __init__(self, config, dm):
        """
        dm: data manager, given client id provides corresponding dataset
        """
        self.config = config
        self.dm = dm

        self.local_models, self.local_model_names = self.create_multiple_local_models()
        self.global_model = GlobalModel(config, dm)
        self.met_logger = logging.getLogger("metrics")
        self.merger = self.create_merger()

        # for ood evaluation
        if (
            self.config.evaluate_locals_ood_after_merge
            or self.config.evaluate_locals_ood_before_merge
            or self.config.evaluate_ensemble_ood
        ):
            self.ood_datasets = (
                self.dm.load_all_ood_eval_datasets()
            )  # dic, ds_name:dataset

    def create_multiple_local_models(self):
        local_model_configs = self.config.local_models
        local_models = nn.ModuleList()
        local_model_names = []

        for model_name, model_config in vars(local_model_configs.models).items():
            local_model = LocalModel(self.config, model_name, model_config, self.dm)
            local_models.append(local_model)
            local_model_names.append(model_name)

        return local_models, local_model_names

    def create_merger(self):
        if self.config.merger.algo == "fedavg":
            merger = FedAvgMerger(
                self.config,
                self.config.merger,
                self.local_models,
                self.global_model,
                merger_ds=None,
            )
        elif self.config.merger.algo == "ot":
            merger = OptimalTransportMerger(
                self.config,
                self.config.merger,
                self.local_models,
                self.global_model,
                merger_ds=None,
            )
        else:
            raise NotImplementedError
        return merger

    def train_local_models_if_needed(self):
        for local_model in self.local_models:
            local_model.train_if_needed()

    def single_round(self, save_results=True, merger_options=None):
        met = {}
        merger_options = {} if merger_options is None else merger_options
        self.train_local_models_if_needed()

        # if self.config.evaluate_locals_before:
        before_met = self.evaluate_local_models(when="before_merge_locals", merged="no")
        if before_met:
            met["before_merge_locals"] = before_met

        if hasattr(self.config, "ensembler") and self.config.ensembler.enabled:
            ensembler = Ensembler(self.local_models)
            ensemble_met = self.evaluate_ensemble_model(
                ensembler, when="before_merge_locals"
            )
            met["before_merge_ensemble"] = ensemble_met

        if self.config.merger.fisher_weighted:
            merger_options["fisher_weights"] = []
            for local_model in self.local_models:
                fisher = local_model.prepare_fisher()
                merger_options["fisher_weights"].append(fisher)

        if self.config.merger.regmean_mean:
            merger_options["grams"] = []
            for local_model in self.local_models:
                gram = local_model.prepare_gram()
                merger_options["grams"].append(gram)

        if self.config.merger.enabled:
            self.merger.merge_to_global(**merger_options)
            self.merger.deliver_to_local()

            if self.config.post_merge_train:
                for local_model in self.local_models:
                    local_model.post_merge_train()

            # if self.config.evaluate_locals_after:
            after_met = self.evaluate_local_models(
                when="after_merge_locals", merged="yes"
            )
            if after_met:
                met["after_merge_locals"] = after_met

            if self.config.evaluate_global_model:
                global_met = self.evaluate_global_model()
                met["after_merge_global"] = global_met

        if save_results:
            with open(
                os.path.join(self.config.main_output_dir, "metrics.json"), "w"
            ) as wf:
                json.dump(met, wf)
        return met

    def search_coeffs(self):
        n_models = len(self.local_models)

        if self.config.merger.coeff_search_method == "random":
            dist = torch.distributions.dirichlet.Dirichlet(torch.ones(n_models))
        else:
            dist = None

        n_trials = self.config.merger.n_trials

        best_coeffs = None
        best_score, best_key_scores, best_met = -1, None, None

        for trial_id in range(n_trials):
            if self.config.merger.coeff_search_method == "random":
                coeffs = dist.sample().cpu().tolist()
            else:
                assert len(self.local_models) == 2
                coeffs = [
                    1.0 * trial_id / (n_trials - 1),
                    1.0 - 1.0 * trial_id / (n_trials - 1),
                ]
            met = self.single_round(
                save_results=False, merger_options={"model_coeffs": coeffs}
            )
            key_met_scores = self.dm.extract_main_metrics(
                met["after_merge_locals"],
                self.local_model_names,
                [x.model_config for x in self.local_models],
                prefix="eval_",
            )
            avg_score = np.mean(key_met_scores)
            self.met_logger.info(
                "Trial\t{}\t{}\t{}\t{}".format(
                    trial_id, json.dumps(coeffs), json.dumps(key_met_scores), avg_score
                )
            )

            if avg_score > best_score:
                best_score = avg_score
                best_coeffs, best_key_scores, best_met = coeffs, key_met_scores, met

        self.met_logger.info(
            "Best Trial\t{}\t{}\t{}\t{}\t{}".format(
                trial_id,
                json.dumps(best_coeffs),
                json.dumps(best_key_scores),
                best_score,
                json.dumps(best_met),
            )
        )

        return best_coeffs

    def evaluate_global_model(self, when=""):
        met = {}
        all_local_eval_sets = []
        for model_name, model in zip(self.local_model_names, self.local_models):
            local_eval_set = (
                model.eval_dataset
                if not self.config.eval_on_test
                else model.test_dataset
            )
            logging.info(f"Evaluating global model over local dataset of {model_name}")
            metrics = self.global_model.evaluate(local_eval_set)
            self.met_logger.info(
                fmt(f"{when}/GlobalModel/LocalEvalSet::{model_name}", metrics)
            )
            met[model_name] = metrics
            all_local_eval_sets.append(local_eval_set)

        if self.config.evaluate_global_joint:
            all_local_eval_sets = torch.utils.data.ConcatDataset(all_local_eval_sets)
            logging.info("Evaluating global model over joint dataset")
            metrics = self.global_model.evaluate(all_local_eval_sets)
            self.met_logger.info(fmt(f"{when}/GlobalModel/JointEvalSet", metrics))
            met["joint"] = metrics
        return met

    def evaluate_ensemble_model(self, ensembler, when=""):
        met = {}
        all_local_eval_sets = []

        if self.config.evaluate_ensemble_locals:
            for model_name, model in zip(self.local_model_names, self.local_models):
                local_eval_set = (
                    model.eval_dataset
                    if not self.config.eval_on_test
                    else model.test_dataset
                )
                logging.info(
                    f"Evaluating ensemble model over local dataset of {model_name}"
                )
                metrics = ensembler.evaluate_ensemble(local_eval_set)
                self.met_logger.info(
                    fmt(f"{when}/EnsembleModel/LocalEvalSet::{model_name}", metrics)
                )
                met[model_name] = metrics
                all_local_eval_sets.append(local_eval_set)

        if self.config.evaluate_ensemble_ood:
            met["ensemble_ood"] = {}
            for dataset_name, ood_eval_ds in self.ood_datasets.items():
                logging.info(
                    f"Evaluating ensemble model over OOD dataset {dataset_name}"
                )
                metrics = ensembler.evaluate_ensemble(ood_eval_ds)
                self.met_logger.info(
                    fmt(f"{when}/EnsembleModel/OODEvalSet::{dataset_name}", metrics)
                )
                met["ensemble_ood"][dataset_name] = metrics

        return met

    def evaluate_local_models(self, when="", merged="unknown"):
        met = defaultdict(dict)
        self.met_logger.info(
            "MergingModels::{}".format("+".join(self.local_model_names))
        )
        if (self.config.evaluate_locals_before and merged == "no") or (
            self.config.evaluate_locals_after and merged == "yes"
        ):
            all_local_eval_sets = []
            for model_name, model in zip(self.local_model_names, self.local_models):
                logging.info(
                    f"Evaluating local model {model_name} over local dataset of {model_name}"
                )
                local_eval_set = (
                    model.eval_dataset
                    if not self.config.eval_on_test
                    else model.test_dataset
                )
                metrics = model.trainer.evaluate(local_eval_set)
                self.met_logger.info(
                    fmt(
                        f"{when}/LocalModel::{model_name}/LocalEvalSet::{model_name}",
                        metrics,
                    )
                )
                met[model_name] = metrics

                all_local_eval_sets.append(local_eval_set)
            all_local_eval_sets = self.dm.join_datasets(all_local_eval_sets)

        if self.config.evaluate_locals_other_tasks and merged == "no":
            for model_name, model in zip(self.local_model_names, self.local_models):
                met[model_name]["other"] = {}
                for dataset_name, model2 in zip(
                    self.local_model_names, self.local_models
                ):
                    if model_name != dataset_name:
                        other_eval_set = (
                            model2.eval_dataset
                            if not self.config.eval_on_test
                            else model2.test_dataset
                        )
                        logging.info(
                            f"Evaluating local model {model_name} over local dataset of {dataset_name}"
                        )
                        metrics = model.trainer.evaluate(other_eval_set)
                        self.met_logger.info(
                            fmt(
                                f"{when}/LocalModel::{model_name}/OtherEvalSet::{dataset_name}",
                                metrics,
                            )
                        )

                        met[model_name]["other"][dataset_name] = metrics

        if (self.config.evaluate_locals_ood_after_merge and merged == "yes") or (
            self.config.evaluate_locals_ood_before_merge and merged == "no"
        ):
            # print(self.local_models, self.local_model_names, self.ood_datasets)
            for model_name, model in zip(self.local_model_names, self.local_models):
                met[model_name]["ood"] = {}
                for dataset_name, ood_eval_ds in self.ood_datasets.items():
                    logging.info(
                        f"Evaluating local model {model_name} over OOD dataset {dataset_name}"
                    )
                    metrics = model.trainer.evaluate(ood_eval_ds)
                    self.met_logger.info(
                        fmt(
                            f"{when}/LocalModel::{model_name}/OODEvalSet::{dataset_name}",
                            metrics,
                        )
                    )
                    met[model_name]["ood"][dataset_name] = metrics
                if self.config.evaluate_locals_ood_after_merge and merged == "yes":
                    break  # assuming merged local models are the same
        return met

    # for finding the best coeffient over a subset of validation examples
