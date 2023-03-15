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
import math
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import Seq2SeqTrainer, Trainer
from transformers.pytorch_utils import Conv1D
from transformers.trainer import (
    DataLoader,
    EvalLoopOutput,
    EvalPrediction,
    IterableDatasetShard,
    datasets,
    deepspeed_init,
    denumpify_detensorize,
    find_batch_size,
    has_length,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    logger,
    nested_concat,
    nested_numpify,
    nested_truncate,
    speed_metrics,
)
from transformers.trainer_pt_utils import get_parameter_names
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from ..data_manager.data_utils import (
    DataCollatorWithPaddingExcludeStr,
    DatasetWithDsName,
    MTLDataloader,
)
from .misc import filter_modules_by_regex


class MyModelTrainer(Trainer):
    def __init__(
        self,
        model,
        args,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        is_mtl=False,
    ):
        if data_collator is None:
            data_collator = DataCollatorWithPaddingExcludeStr(tokenizer)
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.mtl_compute_metrics = compute_metrics
        self.is_mtl = is_mtl

    def _prepare_inputs(self, inputs):
        """
        Remove unused keys
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        new_inputs = {}
        for k in ["input_ids", "attention_mask", "labels", "dataset"]:
            if k in inputs:
                new_inputs[k] = inputs[k]

        return new_inputs

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        return_output=False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Any]]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if hasattr(eval_dataset, "is_mtl") and eval_dataset.is_mtl:
            ds_names = eval_dataset.get_ds_names()
            metrics = []
            for ds_name in ds_names:
                ds = eval_dataset.get_ds(ds_name)
                self.compute_metrics = self.mtl_compute_metrics[ds_name]
                output = self._eval_single(ds, ignore_keys, metric_key_prefix)
                metrics.append(output.metrics)
            output_metrics = self.merge_metrics(ds_names, metrics)
        else:
            if (
                self.is_mtl and type(self.mtl_compute_metrics) is dict
            ):  # dirty fix because on ner emotion actually all metrics_func are same
                self.compute_metrics = next(iter(self.mtl_compute_metrics.values()))

            output = self._eval_single(eval_dataset, ignore_keys, metric_key_prefix)
            output_metrics = output.metrics

        logging.info("Metrics: {}".format(output_metrics))
        if return_output:
            return (output_metrics, output)
        else:
            return output_metrics

    def merge_metrics(self, ds_names, metrics):
        output_metrics = {}
        for ds_name, metric in zip(ds_names, metrics):
            for k, v in metric.items():
                output_metrics["{}_{}".format(ds_name, k)] = v
        key_scores, all_has_keyscore = [], True
        for metric in metrics:
            if "eval_key_score" in metric:
                key_scores.append(metric["eval_key_score"])
            else:
                all_has_keyscore = False
        if all_has_keyscore:
            output_metrics["eval_key_score"] = np.mean(key_scores)
        return output_metrics

    def _eval_single(self, eval_dataset, ignore_keys, metric_key_prefix):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def get_train_dataloader(self) -> DataLoader:
        if self.is_mtl:
            return self.get_mtl_train_dataloader()
        else:
            return Trainer.get_train_dataloader(self)

    def get_mtl_train_dataloader(self):
        task_keys = self.train_dataset.get_ds_names()

        train_dataset = self.train_dataset  # MTLDataset instance
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )
        data_loaders = [
            DataLoader(
                DatasetWithDsName(ds, ds_name),
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            for ds, ds_name in zip(
                train_dataset.get_dss(), train_dataset.get_ds_names()
            )
        ]
        mtl_dataloader = MTLDataloader(
            train_dataset,
            data_loaders=data_loaders,
            mtl_bal_sampling=False,
            task_keys=task_keys,
            task_num=len(task_keys),
            pad_token_id=self.tokenizer.pad_token_id,
            sqrt=False,
        )
        return mtl_dataloader

    def load_from_checkpoint(self, checkpoint_dir):
        return self._load_from_checkpoint(checkpoint_dir)

    def reinit_trainable_params(self):
        logger.info("Reinitialized trainable params")
        optim_param_regex = self.args.optim_param_regex
        assert optim_param_regex
        params = []
        for n, p in self.model.named_parameters():
            valid = any([re.match(patt, n) for patt in optim_param_regex])
            if valid:
                params.append((n, p))
        for name, param in params:
            if param.dim() == 2:
                param.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            elif param.dim() == 1:
                param.data.zero_()

    def compute_grams(self, config, model_config):
        train_dataloader = self.get_train_dataloader()
        covs = {}
        xn = {}

        def get_grams(name):
            def hook(module, input, output):
                """
                Note: adhere to signature of hook functions
                """
                x = input[0].detach()  # $[b,t,h]
                x = x.view(-1, x.size(-1))
                xtx = torch.matmul(x.transpose(0, 1), x)  # [h,h]
                if name not in covs:
                    covs[name] = xtx / x.size(0)
                    xn[name] = x.size(0)
                else:
                    covs[name] = (covs[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                    xn[name] += x.size(0)

            return hook

        model = self.model
        linear_modules = filter_modules_by_regex(
            model, None, [nn.Linear, nn.Conv1d, Conv1D]
        )
        print("Linear modules: {}".format(linear_modules))
        handles = []
        for name, module in linear_modules.items():
            handle = module.register_forward_hook(get_grams(name))
            handles.append(handle)

        # mark cov modules as special
        covs["meta_info"] = {
            "conv1d": [
                n
                for n, m in filter_modules_by_regex(
                    model, None, [nn.Conv1d, Conv1D]
                ).items()
            ]
        }

        n_step = config.merger.gram_n_example
        if n_step <= 0:
            n_step = model_config.max_steps

        total = n_step if n_step > 0 else len(train_dataloader)
        for step, inputs in tqdm(
            enumerate(train_dataloader), total=total, desc="Computing gram matrix"
        ):
            if n_step > 0 and step == n_step:
                break
            # print(inputs['labels'])
            inputs = self._prepare_inputs(inputs)
            _ = self.forward_model_pass(model, inputs)

        for handle in handles:
            handle.remove()

        return covs

    def forward_model_pass(self, model, inputs):
        outputs = model(**inputs)
        return outputs

    def compute_fisher(self, config, model_config):
        train_dataloader = self.get_train_dataloader()
        model = self.model
        fisher = {}
        n_b = 0

        n_step = config.merger.fisher_n_example
        if n_step <= 0:
            n_step = model_config.max_steps

        total = n_step if n_step > 0 else len(train_dataloader)

        for step, inputs in tqdm(
            enumerate(train_dataloader), total=total, desc="Computing fisher"
        ):
            if n_step > 0 and step == n_step:
                break
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            logits = outputs.logits
            n_b += 1
            # computer empirical fisher

            if logits.size(-1) == 1 or config.merger.emp_fisher:
                # is regression task. can only compute empiricial fisher
                # assume f(x; theta) is gaussian with fixed var. log likelihood proportional to || f(x) - y ||^2
                loss = outputs.loss
                model.zero_grad()
                loss.backward()
                b_n2fisher = self.collect_squared_gradients(model)
            else:
                if config.merger.fisher_variant == "hard":
                    log_probs = torch.log_softmax(logits, -1)
                    _, target_labels = logits.max(-1)
                    nll_loss = F.nll_loss(log_probs, target_labels)
                    model.zero_grad()
                    nll_loss.backward()
                    b_n2fisher = self.collect_squared_gradients(model)
                elif config.merger.fisher_variant == "soft":
                    probs = torch.softmax(logits, -1).detach()  # [b,c]
                    log_probs = torch.log_softmax(logits, -1)
                    num_labels = probs.size(-1)
                    nll_losses = []
                    for label in range(num_labels):
                        target = (
                            torch.full(probs.size()[:-1], label).long().to(probs.device)
                        )
                        nll_loss = F.nll_loss(log_probs, target, reduction="none")
                        nll_losses.append(nll_loss)
                    nll_losses = torch.stack(nll_losses, -1)  # [b,c]
                    weighted_nll_losses = probs * nll_losses
                    mean_nll_loss = weighted_nll_losses.sum(-1).mean()
                    model.zero_grad()
                    mean_nll_loss.backward()
                    b_n2fisher = self.collect_squared_gradients(model)

            for n, f in b_n2fisher.items():
                if n not in fisher:
                    fisher[n] = f
                else:
                    fisher[n] += f
        assert n_b
        for n, f in fisher.items():
            fisher[n] = f / n_b
        return fisher

    def collect_squared_gradients(self, model):
        n2fisher = {n: p.grad.detach() ** 2 for n, p in model.named_parameters()}
        return n2fisher

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = get_optim_param_groups(
                self.model, self.args, self.args.optim_param_regex
            )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        return self.optimizer


class MySeq2SeqModelTrainer(MyModelTrainer, Seq2SeqTrainer):
    def __init__(
        self,
        model,
        args,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        is_mtl=False,
    ):
        if data_collator is None:
            data_collator = DataCollatorWithPaddingExcludeStr(tokenizer)
        MyModelTrainer.__init__(
            self,
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            is_mtl,
        )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        self._max_length = self.args.generation_max_length
        self._num_beams = self.args.generation_num_beams
        return MyModelTrainer.evaluate(
            self,
            eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

    def forward_model_pass(self, model, inputs):
        self._max_length = self.args.generation_max_length
        self._num_beams = self.args.generation_num_beams
        outputs = self.prediction_step(model, inputs, prediction_loss_only=False)
        return outputs

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            # prune # added
            if self.args.prune_logits:
                logits = torch.stack(
                    [logits[..., idx] for idx in self.args.prune_idxs], -1
                )

            inputs_decode = (
                inputs["input_ids"] if args.include_inputs_for_metrics else None
            )

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(
                            all_inputs, inputs_decode, padding_index=-100
                        )
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(
            eval_dataset, "num_examples"
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds, label_ids=all_labels, inputs=all_inputs
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels)
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        if args.include_inputs_for_metrics:
            ret = EvalLoopOutputWithInput(
                predictions=all_preds,
                label_ids=all_labels,
                metrics=metrics,
                num_samples=num_samples,
                inputs=all_inputs,
            )
            return ret
        else:
            return EvalLoopOutput(
                predictions=all_preds,
                label_ids=all_labels,
                metrics=metrics,
                num_samples=num_samples,
            )


class EvalLoopOutputWithInput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    inputs: Union[np.ndarray, Tuple[np.ndarray]]


def get_optim_param_groups(model, training_args, optim_param_regex=None):
    if not optim_param_regex:
        params = model.named_parameters()
    else:
        params = []
        for n, p in model.named_parameters():
            valid = any([re.match(patt, n) for patt in optim_param_regex])
            if valid:
                params.append((n, p))

    assert params

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in params if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
