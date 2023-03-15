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
import random
import shutil
import string
from datetime import datetime

from .get_resources import ls, ropen, transfer


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def save_to_zoo(config, model_config, copy_src_dir):
    dataset_name = model_config.dataset_name
    model_arch = model_config.model_name.split("/")[-1]
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    save_path = os.path.join(config.local_zoo_dir, dataset_name, model_arch, str(ts))
    shutil.copytree(copy_src_dir, save_path)
    return save_path


def save_extra_info_to_zoo(name, copy_src_dir, save_path):
    shutil.copy(os.path.join(copy_src_dir, name), save_path)


def fetch_from_zoo(config, model_config):
    dataset_name = model_config.dataset_name
    model_arch = model_config.model_name.split("/")[-1]
    d = os.path.join(config.local_zoo_dir, dataset_name, model_arch)
    if not os.path.exists(d):
        return None
    model_dirs = os.listdir(d)

    md_ts = []
    for model_dir in model_dirs:
        model_dir_l = os.path.join(d, model_dir)
        should_use = filter_check(model_config, model_dir_l)
        if should_use:
            md_ts.append(model_dir)
    if not md_ts:
        logging.info("No matched model from the zoo")
        return None
    md_ts.sort(key=lambda x: float(x))
    sel = md_ts[model_config.zoo_idx]
    logging.info(
        "Matched {} models from the zoo. Using model with zoo id {} at {}".format(
            len(md_ts), model_config.zoo_idx, sel
        )
    )
    return os.path.join(d, sel)


def save_to_remote_zoo(config, model_config, copy_src_dir):
    dataset_name = model_config.dataset_name
    model_arch = model_config.model_name.split("/")[-1]
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    save_path = os.path.join(config.remote_zoo_dir, dataset_name, model_arch, str(ts))
    transfer(copy_src_dir, save_path)
    return save_path


def save_extra_info_to_remote_zoo(name, copy_src_dir, remote_dir):
    transfer(os.path.join(copy_src_dir, name), os.path.join(remote_dir, name))


def fetch_from_remote_zoo(config, model_config):
    dataset_name = model_config.dataset_name
    model_arch = model_config.model_name.split("/")[-1]
    d = os.path.join(config.remote_zoo_dir, dataset_name, model_arch)
    model_dirs = ls(d)

    md_ts = []
    for model_dir in model_dirs:
        logging.info("Checking {}".format(model_dir))
        should_use = filter_check_remote(model_config, model_dir)
        if should_use:
            md_ts.append(model_dir)
    md_ts.sort(key=lambda x: float(x.split("/")[-1]))

    if not md_ts:
        logging.info(
            "No matched model from the remote zoo with {}".format(
                vars(model_config.zoo_filter)
            )
        )
        return None, None

    remote_dir = md_ts[model_config.zoo_idx]
    logging.info(
        "Matched {} models from the zoo. Using model with zoo_id {} at {}".format(
            len(md_ts), model_config.zoo_idx, remote_dir
        )
    )

    tmp_dir = os.path.join("remote_zoo_cache/", remote_dir.split("/")[-1])
    if os.path.exists(tmp_dir) and len(os.listdir(tmp_dir)) == len(ls(remote_dir)):
        logging.info("Reusing cached remote zoo model at {}".format(tmp_dir))
    else:
        logging.info("Transfering {} to local tmp dir {}".format(remote_dir, tmp_dir))
        transfer(remote_dir, tmp_dir, overwrite=True)

    return tmp_dir, remote_dir


def filter_check(model_config, model_dir):
    zoo_filter = model_config.zoo_filter
    if not zoo_filter:
        return True
    else:
        with open(os.path.join(model_dir, "model_config.json")) as f:
            meta = json.load(f)
        for k, v in vars(zoo_filter).items():
            if k not in meta or meta[k] != v:
                return False
    return True


def filter_check_remote(model_config, model_dir):
    zoo_filter = model_config.zoo_filter
    if not zoo_filter:
        return True
    else:
        with ropen(os.path.join(model_dir, "model_config.json")) as f:
            meta = json.load(f)
        for k, v in vars(zoo_filter).items():
            if k not in meta or meta[k] != v:
                return False
    return True
