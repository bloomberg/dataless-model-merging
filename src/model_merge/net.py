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
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    SequenceClassifierOutputWithPast,
)

from ..utils.config import get_component_configs

IGNORE_INDEX = -100


def create_model(full_config, maybe_mtl_model_config):
    if full_config.mtl:
        if full_config.mtl_shared_label_space:
            if full_config.seq2seq:
                sample_model_configs = get_component_configs(
                    full_config, maybe_mtl_model_config.components
                )
                model, config = create_stl_model(full_config, sample_model_configs[0])
            else:
                model = MultiDomainModel(
                    full_config, maybe_mtl_model_config
                )  # legacy, should fix
                config = model.base_model_configs
        else:
            model = MTLModel(full_config, maybe_mtl_model_config)
            config = model.base_model_configs
    else:
        model, config = create_stl_model(full_config, maybe_mtl_model_config)
    return model, config


def get_reweight_factor(freq_dist, labels, schema):
    weight = torch.zeros_like(labels).float()
    for label_id in range(labels.size(1)):
        if (
            label_id in freq_dist
            and freq_dist[label_id][0] > 0
            and freq_dist[label_id][1] > 0
        ):
            if schema == "freq":
                n0, n1 = 1.0 / freq_dist[label_id][0], 1.0 / freq_dist[label_id][1]
            elif schema == "sqrt":
                n0, n1 = 1.0 / (freq_dist[label_id][0] ** 0.5), 1.0 / (
                    freq_dist[label_id][1] ** 0.5
                )
            else:
                raise ValueError(schema)

            zero_weight_new = (n0 + n1) / (2 * n0)
            one_weight_new = (n0 + n1) / (2 * n1)

            cls_label = labels[:, label_id]

            mask = torch.zeros_like(labels).bool()
            mask[:, label_id] = cls_label == 0
            weight.masked_fill_(mask, zero_weight_new)

            mask = torch.zeros_like(labels).bool()
            mask[:, label_id] = cls_label == 1
            weight.masked_fill_(mask, one_weight_new)
    return weight


def create_stl_model(full_config, model_config):
    model_name_short = model_config.model_name.split("/")[-1]

    if full_config.seq2seq:
        if model_name_short.startswith("t5"):
            model_cls = T5ForConditionalGeneration
        elif "gpt2" in model_name_short:
            model_cls = GPT2LMHeadModel
    elif model_config.task_type == "multi_label":
        if model_name_short in ["roberta-base", "roberta-large"]:
            model_cls = RoBERTaForMultiLabelClassification
        elif "distilbert" in model_name_short:
            model_cls = DistilBERTForMultiLabelClassification
        elif "gpt2" in model_name_short:
            model_cls = GPT2ForMultiLabelClassification
        elif "deberta-v3" in model_name_short:
            model_cls = DebertaV2ForMultiLabelClassification
        else:
            raise NotImplementedError(
                "Unknown model type {} for multi label classification".format(
                    model_config.model_name
                )
            )
    elif model_config.task_type == "token_classification":
        if model_name_short in ["roberta-base", "roberta-large"]:
            model_cls = RobertaForTokenClassification
        elif "distilbert" in model_name_short:
            model_cls = DistilBertForTokenClassification
        elif "deberta" in model_name_short:
            model_cls = DebertaV2ForTokenClassification
        else:
            raise NotImplementedError(
                "Unknown model type {} for token classification".format(
                    model_config.model_name
                )
            )
    else:
        MODEL_CLASSES = {
            "classification": AutoModelForSequenceClassification,
            "seq_tagging": AutoModelForTokenClassification,
            "span_extraction": AutoModelForQuestionAnswering,
            "seq2seq": AutoModelForSeq2SeqLM,
        }

        model_cls = MODEL_CLASSES[model_config.task_type]
    config = AutoConfig.from_pretrained(model_config.model_name)
    config.num_labels = model_config.num_labels
    if "gpt2" in model_name_short:
        config.pad_token_id = config.eos_token_id
    if model_config.task_type == "multi_label":
        config.freq_dist = model_config.freq_dist
        config.reweight_loss_schema = model_config.reweight_loss_schema
    model = model_cls.from_pretrained(model_config.model_name, config=config)

    return model, config


class MTLModel(nn.Module):
    def __init__(self, config, mtl_config):
        super().__init__()
        self.mtl_config = mtl_config
        self.model_configs = get_component_configs(config, mtl_config.components)

        base_models = {}
        self.base_model_configs = {}
        self.dataset_names = []
        for idx, model_config in enumerate(self.model_configs):
            model, tf_config = create_stl_model(config, model_config)
            self.dataset_names.append(model_config.dataset_name)
            base_models[model_config.dataset_name] = model
            self.base_model_configs[model_config.dataset_name] = tf_config
            if idx != 0:
                delattr(model, config.model_type)  # e.g., bert, distilbert etc.
                setattr(
                    model,
                    config.model_type,
                    getattr(base_models[self.dataset_names[0]], config.model_type),
                )
            base_models[model_config.dataset_name] = model
        self.config = self.base_model_configs[self.dataset_names[0]]
        self.base_models = nn.ModuleDict(base_models)

    def forward(self, input_ids, attention_mask, labels, dataset=None):
        if dataset is not None:
            ds_unique = list(set(dataset))
            rets = []
            for ds in ds_unique:
                idxs = np.array([x for x in range(len(input_ids)) if dataset[x] == ds])
                input_ids_ss, attention_mask_ss, labels_ss = (
                    input_ids[idxs],
                    attention_mask[idxs],
                    labels[idxs],
                )
                base_model = self.base_models[ds]
                ret = base_model(
                    input_ids=input_ids_ss,
                    attention_mask=attention_mask_ss,
                    labels=labels_ss,
                )
                rets.append(ret)
            ret = self.join_rets(rets)
            ret.ds = ds_unique
        else:
            # just pick first base model
            base_model = self.base_models[self.dataset_names[0]]
            ret = base_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        return ret

    def join_rets(self, rets):
        if len(rets) == 1:
            return rets[0]
        bss = [x.logits.size(0) for x in rets]
        logits = [x.logits for x in rets]
        loss = sum(bs * x.loss for bs, x in zip(bss, rets)) / sum(bss)
        rets[0].logits = logits
        rets[0].loss = loss
        return rets[0]


class MultiDomainModel(nn.Module):
    def __init__(self, config, mtl_config):
        super().__init__()
        self.mtl_config = mtl_config
        self.model_configs = get_component_configs(config, mtl_config.components)

        base_models = {}
        self.base_model_configs = {}
        self.dataset_names = []
        assert config.model_type in [
            "distilbert",
            "roberta",
            "t5",
            "deberta",
        ]  # only considered these cases in implementation
        for idx, model_config in enumerate(self.model_configs):
            model, tf_config = create_stl_model(config, model_config)
            self.dataset_names.append(model_config.dataset_name)
            base_models[model_config.dataset_name] = model
            self.base_model_configs[model_config.dataset_name] = tf_config
            if idx != 0:
                for module_name in [
                    config.model_type,
                    "classifier",
                    "pre_classifier",
                    "shared",
                    "encoder",
                    "decoder",
                    "lm_head",
                    "pooler",
                ]:
                    self.replace_module_if_exists(
                        model, module_name, base_models[self.dataset_names[0]]
                    )
            base_models[model_config.dataset_name] = model
        self.config = self.base_model_configs[self.dataset_names[0]]
        self.base_models = nn.ModuleDict(base_models)

    def replace_module_if_exists(self, model, module_name, other_model):
        if hasattr(model, module_name):
            logging.info("Replacing {} for multi domain model".format(module_name))
            delattr(model, module_name)
            setattr(model, module_name, getattr(other_model, module_name))

    def forward(self, input_ids, attention_mask, labels, dataset=None):
        if dataset is not None:
            ds_unique = list(set(dataset))
            rets = []
            for ds in ds_unique:
                idxs = np.array([x for x in range(len(input_ids)) if dataset[x] == ds])
                input_ids_ss, attention_mask_ss, labels_ss = (
                    input_ids[idxs],
                    attention_mask[idxs],
                    labels[idxs],
                )
                base_model = self.base_models[ds]
                ret = base_model(
                    input_ids=input_ids_ss,
                    attention_mask=attention_mask_ss,
                    labels=labels_ss,
                )
                rets.append(ret)
            ret = self.join_rets(rets)
            ret.ds = ds_unique
        else:
            # just pick first base model
            base_model = self.base_models[self.dataset_names[0]]
            ret = base_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        return ret

    def join_rets(self, rets):
        if len(rets) == 1:
            return rets[0]
        bss = [x.logits.size(0) for x in rets]
        logits = [x.logits for x in rets]
        loss = sum(bs * x.loss for bs, x in zip(bss, rets)) / sum(bss)
        rets[0].logits = logits
        rets[0].loss = loss
        return rets[0]


def compute_bce_loss(
    config,
    logits,
    labels,
):
    valid = labels != IGNORE_INDEX
    n_item = valid.sum()
    if n_item.item() == 0:
        loss = torch.zeros().to(labels.device)
    else:
        loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        if config.reweight_loss_schema != "no":
            reweight_factor = get_reweight_factor(
                config.freq_dist, labels, config.reweight_loss_schema
            ).to(logits.device)
            loss_raw = loss_fct(logits, labels.float())
            loss = loss_raw.masked_fill(~valid, 0.0)
            loss = reweight_factor * loss
        else:
            loss_raw = loss_fct(logits, labels.float())
            loss = loss_raw.masked_fill(~valid, 0.0)
        loss = loss.sum() / n_item
    return loss


class RoBERTaForMultiLabelClassification(RobertaForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = compute_bce_loss(self.config, logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DistilBERTForMultiLabelClassification(DistilBertForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss = compute_bce_loss(self.config, logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class GPT2ForMultiLabelClassification(GPT2ForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                )
            else:
                sequence_lengths = -1
                logging.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            loss = compute_bce_loss(self.config, pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class DebertaV2ForMultiLabelClassification(DebertaV2ForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = compute_bce_loss(self.config, logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
