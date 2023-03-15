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
import os

from datasets.config import HF_CACHE_HOME


def transfer(dir1, dir2, **kwargs):
    raise NotImplementedError(
        f"""
        Please implement your own transfer function taking {dir1} {dir2} {kwargs} to push/pull remote models and download resources from a service provider.
        Otherwise, please disable "load_from_zoo_use_remote" and "push_to_remote" in config to disable the remote model zoo feature and use local model zoo instead.
        Please also make sure all the files/folders required by "required_resources" are already present under "resource_dir".
        """
    )


def ropen(remote_filename):
    raise NotImplementedError(
        f"""
        Please implement your own ropen function taking {remote_filename} to create IO handles for remote resources
        Otherwise, please disable "load_from_zoo_use_remote" and "push_to_remote" in config to disable the remote model zoo feature and use local model zoo instead.
        """
    )


def ls(remote_dir):
    raise NotImplementedError(
        f"""
        Please implement your own remote ls function taking {remote_dir} to list files in a remote directory
        Otherwise, please disable "load_from_zoo_use_remote" and "push_to_remote" in config to disable the remote model zoo feature and use local model zoo instead.
        """
    )


def check_and_get_remote_resources(config):
    if config.required_resources and config.download_remote_resources:
        for l_pth, r_pth in vars(config.required_resources).items():
            save_pth = os.path.join(config.resource_dir, l_pth)
            if not os.path.exists(save_pth):
                logging.info("Starting to transfer {} to {}".format(r_pth, save_pth))
                transfer(r_pth, save_pth)
                # some ad-hoc processing for hf datasets cache
                if l_pth == "huggingface":
                    print("Also copying hf datasets cache to {}".format(HF_CACHE_HOME))
                    os.makedirs(HF_CACHE_HOME, exist_ok=True)
                    os.system("cp -r {}/* {}".format(save_pth, HF_CACHE_HOME))


def get_remote_config(s3_path, idx):
    print("Getting config from {}".format(s3_path))
    config_path = "./tmp_{}.yaml".format(idx)
    transfer(s3_path, config_path, overwrite=True)
    return config_path


def upload_runs(config):
    output_dir, remote_dir = config.main_output_dir, config.s3_runs_dir
    dst_dir = os.path.join(remote_dir, output_dir)

    if dst_dir[-1] == "/":
        dst_dir = dst_dir[:-1]
    dst_dir = "/".join(dst_dir.split("/")[:-1])

    logging.info("Uploading files under {} to remote {}".format(output_dir, dst_dir))
    transfer(output_dir, dst_dir, overwrite=True)


def get_remote_args(s3_path, idx):
    print("Getting cmdline args from {}".format(s3_path))
    args_path = "./tmp_{}.args.txt".format(idx)
    transfer(s3_path, args_path, overwrite=True)
    return args_path
