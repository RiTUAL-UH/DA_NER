import os
import re
import json
import torch
import random
import logging
import argparse
import datasets
import transformers

import numpy as np
import src.exp_lm.experiment as exp_lm
import src.commons.globals as glb

from accelerate import Accelerator


accelerator = Accelerator()
logger = logging.getLogger('DA_NER')


class Arguments(dict):
    def __init__(self, *args, **kwargs):
        super(Arguments, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        if not isinstance(data, dict):
              return data
        else: return Arguments({key: Arguments.from_nested_dict(data[key]) for key in data})


def load_args(default_config=None, verbose=False, logger=None):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--config', default=default_config, type=str, required=default_config is None, help='Provide the JSON config file with the experiment parameters')
    parser.add_argument('--mode', choices=['pretrain', 'train', 'eval', 'generate'], default='train')
    parser.add_argument('--replicable', action='store_true', help='If provided, a seed will be used to allow replicability')

    if default_config is None:
        arguments = parser.parse_args()
    else:
        arguments = parser.parse_args("")

    # Override the default values with the JSON arguments
    with open(os.path.join(glb.PROJ_DIR, arguments.config)) as f:
        params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])  # Remove comments from the JSON config
        args = Arguments.from_nested_dict(json.loads(params))


    args.replicable = arguments.replicable

    # Training Mode ['train', 'eval', 'generate']
    args.mode = arguments.mode

    # Data Args
    args.data.directory = os.path.join(glb.PROJ_DIR, args.data.directory)

    # Exp Args
    args.experiment.pretraining_dir = os.path.join(glb.PROJ_DIR, args.experiment.output_dir, args.model.pretrained_model_name_or_path, 'checkpoint') if args.mode != 'pretrain' else None
    args.experiment.output_dir = os.path.join(glb.PROJ_DIR, args.experiment.output_dir, args.experiment.id)
    args.experiment.checkpoint_dir = os.path.join(args.experiment.output_dir, 'checkpoint')
    args.experiment.pseudo_dir = os.path.join(glb.PROJ_DIR, 'data/linearized_pseudo')

    # Optim Args
    args.optim.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.optim.n_gpu = torch.cuda.device_count()

    if verbose:
        for main_field in ['experiment', 'data', 'config', 'tokenizer', 'model', 'optim', 'generation']:
            if hasattr(args, main_field):
                logger.info(f'  [{main_field.title()}]')
                for k,v in args[main_field].items():
                    if k == 'new_tokens':
                        logger.info(f'      num_new_tokens: {len(v)}')
                    else:
                        logger.info(f"      {k}: {v}")

        logger.info('\n')

    return args


def main():

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        logger.info(accelerator.state)
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    args = load_args(verbose=True, logger=logger)

    # Setup seed to replicate results
    if args.replicable:
        seed_num = args.experiment.seed
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

    exp_lm.main(args)


if __name__ == '__main__':
    main()
