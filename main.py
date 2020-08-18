# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from datasets import BaseDataset, TrainingDataset, SpectrogramDataset


def main(results_path: str, network_config: dict, eval_settings: dict, classes: list, learning_rate: int = 1e-3,
         weight_decay: float = 1e-4, n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))
    base_dataset = BaseDataset()
    spectrogram_dataset = SpectrogramDataset(dataset=base_dataset)
    training_dataset = TrainingDataset(dataset=spectrogram_dataset)
    base_dataset_eval = BaseDataset(data_path=os.path.join('data', 'eval'))
    spectrogram_dataset_eval = SpectrogramDataset(dataset=base_dataset_eval)
    training_dataset_eval = TrainingDataset(dataset=spectrogram_dataset_eval)
    print(training_dataset[0])
    print(training_dataset_eval[0])


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
