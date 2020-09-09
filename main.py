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


def main(hyper_params: dict, network_config: dict, eval_settings: dict, classes: list, scenes: list, fft_params: dict,
         device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f' - {" - ".join(scenes)}'
    writer = SummaryWriter(log_dir=os.path.join('results', 'tensorboard', experiment_id))
    training_dataset = BaseDataset(scenes, classes, hyper_params, fft_params)

    # Create Network
    network_config['out_features'] = len(classes)
    net = SimpleCNN(**network_config)
    # Save initial model as "best" model (will be overwritten later)
    model_path = os.path.join('results', f'best_{"_".join(scenes)}_{hyper_params["feature_type"]}_model.pt')
    if not os.path.exists(model_path):
        torch.save(net, model_path)
    else:  # if there already exists a model load parameters
        print(f'reusing pre-trained model: "{model_path}"')
        net = torch.load(model_path, map_location=torch.device('cpu'))
    net.to(device)
    # Get loss function
    loss_fn = torch.nn.BCELoss()
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params['learning_rate'],
                                 weight_decay=hyper_params['weight_decay'])

    train_stats_at = eval_settings['train_stats_at']
    validate_at = eval_settings['validate_at']
    best_validation_loss = np.inf  # best validation loss so far
    best_f_score = 0
    progress_bar = tqdm.tqdm(total=hyper_params['n_updates'], desc=f"loss: {np.nan:7.5f}", position=0)
    update = 0  # current update counter

    fold_idx = 0
    rnd_augment = hyper_params['rnd_augment']
    train_subset = Subset(training_dataset, training_dataset.get_fold_indices(scenes, fold_idx)[0])
    val_subset = Subset(training_dataset, training_dataset.get_fold_indices(scenes, fold_idx)[1])
    train_set = ExcerptDataset(train_subset, hyper_params['feature_type'], classes, hyper_params['excerpt_size'],
                               fft_params, overlap=hyper_params['overlap_train_excerpts'], rnd_augment=rnd_augment)
    val_set = ExcerptDataset(val_subset, hyper_params['feature_type'], classes, hyper_params['excerpt_size'],
                             fft_params, overlap=False, rnd_augment=False)
    train_loader = DataLoader(train_set, batch_size=hyper_params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=0)

    n_updates = hyper_params['n_updates']
    while update <= n_updates:
        if update == 0:
            train_set.generate_excerpts()
            val_set.generate_excerpts()
        elif rnd_augment:
            train_set.generate_excerpts()
        for data in train_loader:
            inputs, targets, audio_file, idx = data
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
                print(f'val_loss: {val_loss}')
                params = net.parameters()
                f_score, err_rate = log_validation_params(writer, val_loss, params, metrics, metrics_pp, update)
                print(f'f_score: {f_score}')
                print(f'err_rate: {err_rate}')
                # Save best model for early stopping
                if f_score > best_f_score:
                    print(f'{f_score} > {best_f_score}... saving as new {os.path.split(model_path)[-1]}')
                    best_f_score = f_score
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
    evaluation.final_evaluation(classes, hyper_params, fft_params, model_path, scenes, training_dataset, device)
    print('zipping "results" folder...')
    utils.zip_folder('results', f'results_{"_".join(scenes)}')


def log_validation_params(writer: SummaryWriter, val_loss: Tensor, params: Iterator[Parameter],
                          metrics: Metrics, metrics_pp: Metrics, update: int) -> Tuple[float, float]:
    writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
    # Add weights to tensorboard
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(), global_step=update)
    # Add gradients to tensorboard
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/gradients_{i}', values=param.grad.cpu(), global_step=update)
    write_categories = ['dcase/segment_based/overall',
                        'dcase/segment_based/class_wise_average',
                        'dcase_pp/segment_based/overall',
                        'dcase_pp/segment_based/class_wise_average',
                        'dcase/event_based/onset',
                        'dcase/event_based/onset-offset',
                        'dcase_pp/event_based/onset',
                        'dcase_pp/event_based/onset-offset'
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
    return metrics.segment_based['overall']['F'], metrics.segment_based['overall']['ER']


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
