import os
import re
import json
import torch
import random
import argparse
import transformers

import numpy as np
import src.commons.globals as glb
import src.exp_ner.experiment_ner as exp_ner


class Arguments(dict):
    def __init__(self, *args, **kwargs):
        super(Arguments, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        if not isinstance(data, dict):
              return data
        else: return Arguments({key: Arguments.from_nested_dict(data[key]) for key in data})


def load_args(default_config=None, verbose=False):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--config', default=default_config, type=str, required=default_config is None, help='Provide the JSON config file with the experiment parameters')
    parser.add_argument('--mode', choices=['train', 'eval', 'generate'], default='train')
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
    args.experiment.output_dir = os.path.join(glb.PROJ_DIR, args.experiment.output_dir, args.experiment.id)
    args.experiment.checkpoint_dir = os.path.join(args.experiment.output_dir, "checkpoint")

    # Optim Args
    args.optim.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.optim.n_gpu = torch.cuda.device_count()

    if verbose:
        print("[LOG] {}".format('=' * 40))
        for main_field in ['experiment', 'data', 'config', 'tokenizer', 'model', 'optim']:
            assert hasattr(args, main_field)
            print(f"[{main_field.title()}]:")
            for k,v in args[main_field].items():
                if k == 'label_scheme':
                    continue
                print(f"    {k}: {v}")
        print("[LOG] {}".format('=' * 40))
    return args


def main():
    args = load_args(verbose=True)

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
    
    transformers.utils.logging.set_verbosity_error()

    exp_ner.main(args)


if __name__ == '__main__':
    main()
