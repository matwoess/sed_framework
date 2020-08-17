import os

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def main(results_path: str, network_config: dict, eval_settings: dict, classes: list, learning_rate: int = 1e-3,
         weight_decay: float = 1e-4, n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    path_plots = os.path.join(results_path, 'plots')
    os.makedirs(path_plots, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))
    pass


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)
