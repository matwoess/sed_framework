# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime
from typing import Tuple, Iterator

import torch
import torch.utils.data
import numpy as np
from torch import Tensor
from torch.nn import Parameter
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

import evaluation
from evaluation import Metrics
from architectures import SimpleCNN
from datasets import BaseDataset, ExcerptDataset
import utils

torch.random.manual_seed(0)  # Set a known random seed for reproducibility


def validate_model(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, classes: list, update: int,
                   device: torch.device, loss_fn=torch.nn.BCELoss()) -> Tuple[Tensor, Metrics, Metrics]:
    plots_path = os.path.join('results', 'itermediate', 'plots')
    metrics_path = os.path.join('results', 'itermediate', 'metrics')
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    loss = torch.tensor(0., device=device)
    with torch.no_grad():
        target_list = []
        prediction_list = []
        for data in tqdm.tqdm(dataloader, desc='scoring', position=0):
            inputs, targets, _, idx = data
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            predictions = net(inputs)
            loss += loss_fn(predictions, targets)
            # plot results
            target_array = targets.detach().cpu().numpy()
            prediction_array = predictions.detach().cpu().numpy()
            target_list.extend([*target_array])
            prediction_list.extend([*prediction_array])
        loss /= len(dataloader)
        # pick some excerpts and plot them
        num_plots = 3
        indices = random.choices(np.arange(len(target_list)), k=num_plots)
        targets = np.stack([t for i, t in enumerate(target_list) if i in indices])
        predictions = np.stack([t for i, t in enumerate(prediction_list) if i in indices])
        utils.plot(targets, predictions, classes, plots_path, update)
        # compute dcase metrics
        targets = np.stack(target_list)
        predictions = np.stack(prediction_list)
        metrics = evaluation.compute_dcase_metrics(targets, predictions, classes)
        metrics_pp = evaluation.compute_dcase_metrics(targets, predictions, classes, post_process=True)
        evaluation.write_dcase_metrics_to_file(metrics, metrics_path, f"{update:07d}.txt")
        evaluation.write_dcase_metrics_to_file(metrics_pp, metrics_path, f"{update:07d}_pp.txt")
    return loss, metrics, metrics_pp


def main(network_config: dict, eval_settings: dict, classes: list, scenes: list, feature_type: str,
         learning_rate: int = 1e-3, weight_decay: float = 1e-4, n_updates: int = int(1e5), excerpt_size: int = 384,
         batch_size: int = 16, device: torch.device = torch.device("cuda:0"),
         ):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    writer = SummaryWriter(log_dir=os.path.join('results', 'tensorboard', time_string))
    training_dataset = BaseDataset(scenes=scenes, features=feature_type)

    # Create Network
    network_config['out_features'] = len(classes)
    net = SimpleCNN(**network_config)
    # Save initial model as "best" model (will be overwritten later)
    model_path = os.path.join('results', f'best_{"_".join(scenes)}_{feature_type}_model.pt')
    if not os.path.exists(model_path):
        torch.save(net, model_path)
    else:  # if there already exists a model load parameters
        print(f'reusing pre-trained model: "{model_path}"')
        net = torch.load(model_path, map_location=torch.device('cpu'))
    net.to(device)
    # Get loss function
    loss_fn = torch.nn.BCELoss()
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_stats_at = eval_settings['train_stats_at']
    validate_at = eval_settings['validate_at']
    best_validation_loss = np.inf  # best validation loss so far
    progress_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    update = 0  # current update counter

    fold_idx = 0
    train_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes, fold_idx)[0])
    val_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes, fold_idx)[1])
    train_set = ExcerptDataset(train_set, classes, excerpt_size=excerpt_size)
    val_set = ExcerptDataset(val_set, classes, excerpt_size=excerpt_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    while update <= n_updates:
        for data in train_loader:
            inputs, targets, _, idx = data
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            if update % train_stats_at == 0 and update > 0:
                # log training loss
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            if update % validate_at == 0 and update > 0:
                # Evaluate model on validation set (after every complete training fold)
                val_loss, metrics, metrics_pp = validate_model(net, val_loader, classes, update, device)
                params = net.parameters()
                log_validation_params(writer, val_loss, params, metrics, metrics_pp, update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    print(f'{val_loss} < {best_validation_loss}... saving as new {os.path.split(model_path)[-1]}')
                    best_validation_loss = val_loss
                    torch.save(net, model_path)

            # update progress and update-counter
            progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            progress_bar.update()
            update += 1
            if update >= n_updates:
                break

    progress_bar.close()
    print('finished training.')

    print('starting evaluation...')
    evaluation.final_evaluation(classes, excerpt_size, feature_type, model_path, scenes, training_dataset, device)
    print('zipping "results" folder...')
    utils.zip_folder('results')


def log_validation_params(writer: SummaryWriter, val_loss: Tensor, params: Iterator[Parameter],
                          metrics: Metrics, metrics_pp: Metrics, update: int) -> None:
    writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
    # Add weights to tensorboard
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(), global_step=update)
    # Add gradients to tensorboard
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/gradients_{i}', values=param.grad.cpu(), global_step=update)
    write_categories = ['dcase/segment_based/overall', 'dcase_pp/segment_based/overall',
                        'dcase/segment_based/class_wise_average', 'dcase_pp/segment_based/class_wise_average',
                        'dcase/event_based/onset', 'dcase_pp/event_based/onset',
                        'dcase/event_based/onset-offset', 'dcase_pp/event_based/onset-offset'
                        ]
    write_scalars = ['F', 'ER']

    def write_metric_dict(m_dict: dict, identifier: str):
        for key in m_dict.keys():
            if type(m_dict[key]) == dict:
                write_metric_dict(m_dict[key], '/'.join([identifier, key]))
            elif key in write_scalars and identifier in write_categories:
                tag = '/'.join([identifier.replace('dcase', f'dcase_{key}'), key])
                writer.add_scalar(tag=tag, scalar_value=m_dict[key], global_step=update)

    def write_metrics(metrics: Metrics, identifier: str):
        write_metric_dict(metrics.segment_based, '/'.join([identifier, 'segment_based']))
        write_metric_dict(metrics.event_based, '/'.join([identifier, 'event_based']))
        write_metric_dict(metrics.class_based, '/'.join([identifier, 'class_based']))
        write_metric_dict(metrics.frame_based, '/'.join([identifier, 'frame_based']))

    write_metrics(metrics, 'dcase')
    write_metrics(metrics_pp, 'dcase_pp')


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
