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

import json
import logging
import os

import yaml
from transformers.trainer_utils import IntervalStrategy


class FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class Struct:
    def __init__(self) -> None:
        pass

    def __repr__(self, space=0) -> str:
        s = ""
        for k, v in self.__dict__.items():
            if type(v) is not Struct:
                s += "{}{}: {}\n".format(" " * space, k, v)
            else:
                s += "{}{}:\n{}".format(" " * space, k, v.__repr__(space=space + 2))
        return s

    def to_dict(self):
        dic = {}
        for k, v in vars(self).items():
            if type(v) is Struct:
                v = v.to_dict()
            dic[k] = v
        return dic

    def save(self, output_dir, filename):
        d = self.to_dict()
        with open(os.path.join(output_dir, filename), "w") as wf:
            json.dump(d, wf)


def get_component_configs(config, model_names):
    model_configs = [getattr(config.local_models._models, x) for x in model_names]
    return model_configs


def get_same_item_from_list(item, lst):
    assert item is not None
    if type(item) is dict:
        assert len(item) == 1
        for x in lst:
            if (
                type(x) is dict
                and len(x) == 1
                and next(iter(x.keys())) == next(iter(item.keys()))
            ):
                return x
        return None
    else:
        return item if item in lst else None


def merge_config_dict(parent, src):
    for k, v in src.items():
        if k in parent:
            if type(v) is dict:
                merge_config_dict(parent[k], v)
            elif type(v) is list:
                if v and type(v[0]) is dict:  # is list of dicts
                    for item in v:
                        same_parent_item = get_same_item_from_list(item, parent[k])
                        if same_parent_item is not None:
                            merge_config_dict(same_parent_item, item)
                        else:
                            parent[k].append(item)
                else:
                    parent[k] = v
            else:
                parent[k] = v
        else:
            parent[k] = v


def weak_merge_config_dict(parent, src):
    """
    Unlike merge config, only add entries that are not present in parent
    """
    for k, v in src.items():
        if type(v) is dict and k in parent:
            weak_merge_config_dict(parent[k], v)
        elif k not in parent:
            parent[k] = v


def merge_config(parent, src):
    for k, v in vars(src).items():
        if hasattr(parent, k):
            if type(v) is not Struct:
                try:
                    setattr(parent, k, v)
                except AttributeError:
                    logging.debug("Skip setting {} to {}".format(k, v))
            else:
                merge_config(parent[k], v)
        else:
            setattr(parent, k, v)


def weak_merge_config(parent, src):
    for k, v in src.__dict__.items():
        if k not in parent:
            setattr(parent, k, v)


def merge_config_discard_conflict(cfga, cfgb):
    cfg = Struct()
    for k, va in vars(cfga).items():
        if hasattr(cfgb, k):
            vb = getattr(cfgb, k)
            if type(va) is Struct and type(vb) is Struct:
                v = merge_config_discard_conflict(va, vb)
                setattr(cfg, k, v)
            else:
                if va == vb:
                    setattr(cfg, k, va)
    return cfg


def merge_config_discard_conflict_dict(cfga, cfgb):
    cfg = {}
    for k, va in cfga.items():
        if k in cfgb:
            vb = cfgb[k]
            if type(va) is dict and type(vb) is dict:
                v = merge_config_discard_conflict_dict(va, vb)
                cfg[k] = v
            else:
                if va == vb:
                    cfg[k] = va
    return cfg


def merge_many_configs_discard_conflict_dict(cfgs):
    base = cfgs[0]
    for cfg in cfgs[1:]:
        base = merge_config_discard_conflict_dict(base, cfg)
    return base


def dic_to_object(dic):
    obj = Struct()
    for k, v in dic.items():
        if type(v) is dict:
            sub_obj = dic_to_object(v)
            setattr(obj, k, sub_obj)
        else:
            setattr(obj, k, v)

    return obj


def resolve_template(dic, template):
    template_map = FormatDict()
    for k, v in template.items():
        template_map[k] = v
    for k in dic:
        if type(dic[k]) is str:
            dic[k] = dic[k].format_map(template_map)
            if type(dic[k]) is str and dic[k].isnumeric():
                dic[k] = int(dic[k])
            else:
                try:
                    dic[k] = float(dic[k])
                except ValueError:
                    pass
        elif type(dic[k]) is dict:
            resolve_template(dic[k], template)


def load_template_from_args(dic, templates):
    # preprocess templates
    if templates:
        for item in templates:
            k, v = item.split("=")
            if k in dic["templates"]:
                logging.warning(
                    "Overwritten {} as {} in template from args".format(k, v)
                )
            # else:
            dic["templates"][k] = v


def postprocess_config(dic, filter_model):
    dic["data_file_path"] = dic["data_file_path"].format(
        resource_dir=dic["resource_dir"], dataset=dic["dataset"]
    )
    dic["partition_file_path"] = dic["partition_file_path"].format(
        resource_dir=dic["resource_dir"], dataset=dic["dataset"]
    )
    dic["parition_num"] = len(dic["local_models"]["models"])
    dic["tokenizer"] = dic["tokenizer"].format(resource_dir=dic["resource_dir"])
    dic["hf_datasets_cache_dir"] = dic["hf_datasets_cache_dir"].format(
        resource_dir=dic["resource_dir"]
    )
    if dic["output_dir_keys"]:
        for key in dic["output_dir_keys"]:
            dic["main_output_dir"] = os.path.join(
                dic["main_output_dir"], "{}_{}".format(key, dic[key])
            )

    resolve_template(dic, dic["templates"])

    for local_model_config in dic["local_models"]["models"].values():
        weak_merge_config_dict(local_model_config, dic["default_model_args"])
        local_model_config["model_name"] = local_model_config["model_name"].format(
            resource_dir=dic["resource_dir"]
        )
        if "seed" not in local_model_config:
            local_model_config["seed"] = dic["seed"]
        if local_model_config["dataset_name"] == "stsb":
            local_model_config["is_regression"] = True

    filter_models_dict(dic, filter_model)

    # special treatments for mtl
    if dic["mtl"]:
        logging.info("MTL mode is enabled. Refactorizing the config...")
        dic["local_models"]["_models"] = dic["local_models"]["models"]
        dic["local_models"].pop("models")

        if dic["mtl_all_tasks"]:
            model_keys = [_ for _ in dic["local_models"]["_models"].keys()]
            dic["local_models"]["_mtl_models"] = {
                "mtl_{}".format("+".join(model_keys)): {"components": model_keys}
            }

        for mtl_model_name in dic["local_models"]["_mtl_models"]:
            configs = [
                dic["local_models"]["_models"][x]
                for x in dic["local_models"]["_mtl_models"][mtl_model_name][
                    "components"
                ]
            ]
            merged = merge_many_configs_discard_conflict_dict(configs)
            for k, v in merged.items():
                dic["local_models"]["_mtl_models"][mtl_model_name][k] = v

        dic["local_models"]["models"] = dic["local_models"].pop("_mtl_models")


def post_process_hf_training_args(args):
    args.evaluation_strategy = IntervalStrategy(args.evaluation_strategy)
    args.logging_strategy = IntervalStrategy(args.logging_strategy)
    args.save_strategy = IntervalStrategy(args.save_strategy)


def post_process_templates(dic):
    if "dseed_generator" in dic["templates"]:
        gen = int(dic["templates"]["dseed_generator"])
        for i in range(1, dic["dseed_n"] + 1):
            dic["templates"][f"dseed{i}"] = (gen + i - 2) % dic["dseed_n"] + 1


def load_configs(*config_files, **kwargs):
    assert len(config_files) >= 1
    configs = []
    for file in config_files:
        with open(file) as f:
            dic = yaml.safe_load(f)
            configs.append(dic)
    for config in configs[1:]:
        merge_config_dict(configs[0], config)
    load_template_from_args(configs[0], kwargs["templates"])
    post_process_templates(configs[0])
    postprocess_config(configs[0], kwargs.get("filter_model"))
    config_obj = dic_to_object(configs[0])
    logging.info("Merged config: {}".format(config_obj))
    return config_obj


def maybe_load_remote_configs(*config_paths, **kwargs):
    from ..remote_io import get_remote_config

    config_files = []
    for idx, path in enumerate(config_paths):
        if "s3://" in path:
            config_file = get_remote_config(path, idx)
        else:
            config_file = path
        config_files.append(config_file)
    config = load_configs(*config_files, **kwargs)
    return config


def filter_models(config, filt):
    if filt:
        rm_models = []
        for name in vars(config.local_models.models):
            if name not in filt:
                rm_models.append(name)
        for name in rm_models:
            delattr(config.local_models.models, name)
        logging.info("Removed {} entries from the config".format(rm_models))


def filter_models_dict(config, filt):
    if filt:
        rm_models = []
        for name in config["local_models"]["models"]:
            if name not in filt:
                rm_models.append(name)
        for name in rm_models:
            config["local_models"]["models"].pop(name)
        logging.info("Removed {} entries from the config".format(rm_models))


if __name__ == "__main__":
    parent = yaml.safe_load(
        """
    local_models:
        models:
            model0:
                model_name: distill-bert-uncased
                task_type: classification
                dataset_name: 20news
                partition: 0
                device: 0
                no_cache: False
                seeds: [1,2]
            model1:
                model_name: distill-bert-uncased
                task_type: classification
                dataset_name: 20news
                partition: 0
                device: 0
                no_cache: False
    """
    )

    another = yaml.safe_load(
        """
    local_models:
        models:
            model0:
                model_name: distill-bert-uncased
                task_type: classification
                dataset_name: 20news
                partition: 0
                device: 0
                no_cache: False
                seeds: [3,4,5]
            model2:
                model_name: distill-bert-uncased
                task_type: classification
                dataset_name: 20news
                partition: 0
                device: 0
                no_cache: False
    """
    )

    """
    should be:

    local_models:
        models:
            model0:
                xxx
                seeds: [3,4,5]
            model1: xxx
            model2: xxx
    """

    merge_config_dict(parent, another)

    print(parent)

    parent_obj = dic_to_object(parent)

    print(parent_obj)
