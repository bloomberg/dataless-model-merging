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

import argparse
import logging
import os

from transformers import AutoTokenizer

from .data_manager import get_dm_class
from .model_merge.base import ModelMergeExp
from .remote_io import check_and_get_remote_resources, upload_runs
from .utils.config import maybe_load_remote_configs
from .utils.initializer import set_seed

os.environ["WANDB_DISABLED"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_files", "-c", nargs="+")
    parser.add_argument("--filter_model", nargs="*")
    parser.add_argument("--templates", nargs="*")
    parser.add_argument("--upload", action="store_true")
    return parser


def main(args):
    config = maybe_load_remote_configs(
        *args.config_files, filter_model=args.filter_model, templates=args.templates
    )
    # filter_models(config, args.filter_model)
    set_seed(config.seed)

    os.makedirs(config.main_output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(config.main_output_dir, "log.txt"))
    )
    logging.info(config)

    check_and_get_remote_resources(config)

    # metrics logger
    logger = logging.getLogger("metrics")
    logger.addHandler(
        logging.FileHandler(os.path.join(config.main_output_dir, "metrics_log.txt"))
    )
    logger.setLevel(logging.DEBUG)
    logger.info("\n")

    # data manager stuffs
    if "gpt2" in config.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer,
            add_prefix_space=config.tokenizer_add_prefix_space,
            pad_token="<|endoftext|>",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer, add_prefix_space=config.tokenizer_add_prefix_space
        )

    dm = get_dm_class(config.dataset, seq2seq=config.seq2seq)(
        config, tokenizer=tokenizer
    )

    local_models_configs = (
        config.local_models.models if not config.mtl else config.local_models._models
    )
    for _, model_config in vars(local_models_configs).items():
        dm.update_model_config(model_config)

    if os.environ.get("DRY_RUN") == "1":
        print("Dry run success")
        exit(0)

    exp = ModelMergeExp(config, dm)

    if config.merger.coeff_search_method is None:
        exp.single_round()
    elif config.merger.coeff_search_method in ["random", "grid"]:
        exp.search_coeffs()
    else:
        raise NotImplementedError(config.merger.coeff_search_method)

    if args.upload:
        upload_runs(config)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
