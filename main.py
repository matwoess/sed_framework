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
import metric
from architecture import SimpleCNN
from dataset import BaseDataset, ExcerptDataset
import util
from plot import Plotter

torch.random.manual_seed(0)  # Set a known random seed for reproducibility


def validate_model(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, classes: list, update: int,
                   device: torch.device, plotter: Plotter, loss_fn=torch.nn.BCELoss()) -> Tuple[Tensor, dict, dict]:
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
        plotter.plot(targets, predictions, plots_path, update)
        # compute dcase metrics
        targets = np.stack(target_list)
        predictions = np.stack(prediction_list)
        metrics = metric.compute_dcase_metrics(targets, predictions, classes)
        metrics_pp = metric.compute_dcase_metrics(targets, predictions, classes, post_process=True)
        metric.write_dcase_metrics_to_file(metrics, metrics_path, f"{update:07d}.txt")
        metric.write_dcase_metrics_to_file(metrics_pp, metrics_path, f"{update:07d}_pp.txt")
    return loss, metrics, metrics_pp


def main(eval_mode: bool, feature_type: str, scene: str, hyper_params: dict, network_config: dict, eval_settings: dict,
         fft_params: dict, device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    os.makedirs('results', exist_ok=True)
    experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f' - {feature_type} - {scene}'
    writer = SummaryWriter(log_dir=os.path.join('tensorboard', experiment_id))
    training_dataset = BaseDataset(feature_type, scene, hyper_params, fft_params)
    # create network
    classes = util.get_scene_classes(scene)
    plotter = Plotter(classes, hop_size=fft_params['hop_size'], sampling_rate=22050)
    network_config['out_features'] = len(classes)
    if feature_type == 'spec':
        network_config['n_features'] = fft_params['hop_size'] + 1
    elif feature_type == 'mfcc':
        network_config['n_features'] = fft_params['n_mfcc']
    elif feature_type == 'mels':
        network_config['n_features'] = fft_params['n_mels']
    net = SimpleCNN(**network_config)
    # Save initial model as "best" model (will be overwritten later)
    model_path = os.path.join('results', f'best_{feature_type}_{scene}_model.pt')
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
    best_loss = np.inf  # best validation loss so far
    progress_bar = tqdm.tqdm(total=hyper_params['n_updates'], desc=f"loss: {np.nan:7.5f}", position=0)
    update = 0  # current update counter

    fold_idx = 0
    rnd_augment = hyper_params['rnd_augment']
    if eval_mode:
        train_subset = training_dataset
        val_loader = None
    else:
        train_subset = Subset(training_dataset, training_dataset.get_fold_indices(fold_idx)[0])
        val_subset = Subset(training_dataset, training_dataset.get_fold_indices(fold_idx)[1])
        val_set = ExcerptDataset(val_subset, feature_type, classes, hyper_params['excerpt_size'],
                                 fft_params, overlap_factor=1, rnd_augment=False)
        val_loader = DataLoader(val_set, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=0)

    train_set = ExcerptDataset(train_subset, feature_type, classes, hyper_params['excerpt_size'],
                               fft_params, overlap_factor=hyper_params['train_overlap_factor'], rnd_augment=rnd_augment)
    train_loader = DataLoader(train_set, batch_size=hyper_params['batch_size'], shuffle=True, num_workers=0)

    n_updates = hyper_params['n_updates']
    while update <= n_updates:
        if rnd_augment and update > 0:
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

            if not eval_mode and update % validate_at == 0 and update > 0:
                # Evaluate model on validation set (after every complete training fold)
                val_loss, metrics, metrics_pp = validate_model(net, val_loader, classes, update, device, plotter)
                print(f'val_loss: {val_loss}')
                f_score = metrics['segment_based']['overall']['F']
                err_rate = metrics['segment_based']['overall']['ER']
                f_score_pp = metrics_pp['segment_based']['overall']['F']
                err_rate_pp = metrics_pp['segment_based']['overall']['ER']
                print(f'f_score: {f_score}')
                print(f'err_rate: {err_rate}')
                print(f'f_score_pp: {f_score_pp}')
                print(f'err_rate_pp: {err_rate_pp}')
                params = net.parameters()
                log_validation_params(writer, val_loss, params, metrics, metrics_pp, update)
                # Save best model for early stopping
                if val_loss < best_loss:
                    print(f'{val_loss} < {best_loss}... saving as new {os.path.split(model_path)[-1]}')
                    best_loss = val_loss
                    torch.save(net, model_path)

            if eval_mode:
                train_loss = loss.cpu()
                if train_loss < best_loss:
                    print(f'{train_loss} < {best_loss}... saving as new {os.path.split(model_path)[-1]}')
                    best_loss = train_loss
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
    evaluation.final_evaluation(feature_type, scene, hyper_params, network_config, fft_params, model_path, device,
                                writer, plotter)
    print('zipping "results" folder...')
    util.zip_folder('results', f'results_{feature_type}_{scene}')
    print('deleting "results" folder')
    import shutil
    shutil.rmtree('./results')


def log_validation_params(writer: SummaryWriter, val_loss: Tensor, params: Iterator[Parameter],
                          metrics: dict, metrics_pp: dict, update: int) -> None:
    writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
    # Add weights to tensorboard
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(), global_step=update)
    # Add gradients to tensorboard
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/gradients_{i}', values=param.grad.cpu(), global_step=update)

    def write_metrics(m_dict: dict, identifier: str):
        flat_metrics = util.flatten_dict(m_dict, identifier)
        filtered_metrics = metric.filter_metrics_dict(flat_metrics)
        for key in filtered_metrics:
            writer.add_scalar(tag=key, scalar_value=filtered_metrics[key], global_step=update)

    write_metrics(metrics, 'metric')
    write_metrics(metrics_pp, 'metric_pp')


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('feature_type', help='"spec", "mels" or "mfcc"', type=str)
    parser.add_argument('scene', help='"indoor", "outdoor" or "all"', type=str)
    parser.add_argument('config_file', help='path to config file', type=str)
    parser.add_argument('-eval', help='train on whole development set for maximum eval set score', required=False,
                        dest='eval_mode', action='store_true', default=False)
    args = parser.parse_args()
    feature_type_arg = args.feature_type
    scene_arg = args.scene
    config_file_arg = args.config_file
    eval_mode_arg = args.eval_mode

    with open(config_file_arg, 'r') as fh:
        config_args = json.load(fh)
    main(eval_mode_arg, feature_type_arg, scene_arg, **config_args)
