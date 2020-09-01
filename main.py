# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Tuple

import torch
import torch.utils.data
import numpy as np
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

import evaluation
from architectures import DorferCNN, SimpleCNN
from datasets import BaseDataset, ExcerptDataset
import utils

torch.random.manual_seed(0)  # Set a known random seed for reproducibility


def validate_model(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device,
                   loss_fn=torch.nn.BCELoss()) -> torch.Tensor:
    loss = torch.tensor(0., device=device)
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, desc='scoring', position=0):
            inputs, targets, _, idx = data
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            predictions = net(inputs)
            loss += loss_fn(predictions, targets)
            # loss += (torch.stack([loss_fn(pred, target) for pred, target in zip(predictions, targets)]).sum()
            #          / len(dataloader.dataset))
    return loss


def main(network_config: dict, eval_settings: dict, classes: list, scenes: list, feature_type: str,
         learning_rate: int = 1e-3, weight_decay: float = 1e-4, n_updates: int = int(1e5), excerpt_size: int = 384,
         batch_size: int = 16, device: torch.device = torch.device("cuda:0"),
         ):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    plots_path = os.path.join('results', 'itermediate', 'plots')
    metrics_path = os.path.join('results', 'itermediate', 'metrics')
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
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
        net = torch.load(model_path)
    net.to(device)
    # Get loss function
    loss_fn = torch.nn.BCELoss()
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    plot_at = eval_settings['plot_at']
    stats_at = eval_settings['stats_at']
    validate_at = eval_settings['validate_at']
    metrics_at = eval_settings['metrics_at']
    best_validation_loss = np.inf  # best validation loss so far
    progress_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    update = 0  # current update counter

    fold_idx = 0
    train_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes, fold_idx)[0])
    val_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes, fold_idx)[1])
    train_set = ExcerptDataset(train_set, classes, excerpt_size=excerpt_size)
    val_set = ExcerptDataset(val_set, classes, excerpt_size=excerpt_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    while update <= n_updates:
        for data in train_loader:
            inputs, targets, _, idx = data
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_fn(predictions, targets)
            # loss = torch.stack([loss_fn(pred, target) for pred, target in zip(predictions, targets)]).sum()
            # loss = loss.mean()
            loss.backward()
            optimizer.step()

            if update % stats_at == 0 and update > 0:
                # log training loss
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            if update % validate_at == 0 and update > 0:
                # Evaluate model on validation set (after every complete training fold)
                val_loss = validate_model(net, dataloader=val_loader, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(), global_step=update)
                # Add gradients to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}', values=param.grad.cpu(), global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    print(f'{val_loss} < {best_validation_loss}... saving as new best_model.pt')
                    best_validation_loss = val_loss
                    torch.save(net, model_path)

            # plot output
            if update % plot_at == 0 and update > 0:
                plot_targets = targets.detach().cpu().numpy()
                plot_predictions = predictions.detach().cpu().numpy()
                utils.plot(plot_targets, plot_predictions, classes, plots_path, update)
            # compute metrics
            if update % metrics_at == 0 and update > 0:
                metric_targets = targets.detach().cpu().numpy()
                metric_predictions = predictions.detach().cpu().numpy()
                utils.compute_metrics(metric_targets, metric_predictions, metrics_path, update)

            # update progress and update-counter
            progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            progress_bar.update()
            update += 1
            if update >= n_updates:
                break

    progress_bar.close()
    print('finished training')

    # Save final net as best (as by now validation loss keeps going up) TODO: remove
    print('saving final model as best model, TODO:remove')
    torch.save(net, model_path)

    evaluation.final_evaluation(classes, excerpt_size, feature_type, model_path, scenes, training_dataset, device)
    utils.zip_folder('results')


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
