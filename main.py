# -*- coding: utf-8 -*-
import os

import torch
import torch.utils.data
import numpy as np
import typing
from typing import Dict
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

from architectures import DorferCNN, SimpleCNN
from datasets import BaseDataset, FeatureDataset
import utils

torch.random.manual_seed(0)  # Set a known random seed for reproducibility


def evaluate_model(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device,
                   loss_fn=torch.nn.BCELoss()) -> torch.Tensor:
    loss = torch.tensor(0., device=device)
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, desc='scoring', position=0):
            inputs, targets, idx = data
            inputs.unsqueeze_(1)
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            predictions = net(inputs)
            predictions = torch.reshape(predictions, (predictions.shape[0], -1))
            # loss += loss_fn(predictions, targets)
            loss += (torch.stack([loss_fn(pred, target) for pred, target in zip(predictions, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def main(results_path: str, network_config: dict, eval_settings: dict, classes: list, learning_rate: int = 1e-3,
         weight_decay: float = 1e-4, n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0"),
         scenes=None):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    if scenes is None:
        scenes = ['residential_area']
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))
    training_dataset = BaseDataset(scenes=scenes, features='mels')

    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)
    # Get loss function
    loss_fn = torch.nn.BCELoss()
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    tb_stats_at = eval_settings['tb_stats_at']  # print stats to tensorboard every x updates
    plot_at = eval_settings['plot_at']  # plot every x updates
    validate_at = eval_settings['validate_at']  # test on validation set and check for new best model every x updates
    best_validation_loss = np.inf  # best validation loss so far
    progress_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    update = 0  # current update counter
    fold_idx = -1
    while update <= n_updates:
        fold_idx = (fold_idx + 1) % 4
        train_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[0])
        val_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[1])
        train_set = FeatureDataset(train_set, classes, excerpt_size=384)
        val_set = FeatureDataset(val_set, classes, excerpt_size=384)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
        for data in train_loader:
            inputs, targets, idx = data
            inputs.unsqueeze_(1)
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_fn(predictions, targets)
            # loss = torch.stack([loss_fn(pred, target) for pred, target in zip(predictions, targets)]).sum()
            # loss = loss.mean()
            loss.backward()
            optimizer.step()

            # Print current status and score
            if update % tb_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)
            # Plot output
            if update % plot_at == 0 and update > 0:
                pass
                utils.plot(inputs.detach().numpy(), targets.detach().numpy(), predictions.detach().numpy(), update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=val_loader, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(), global_step=update)
                # Add gradients to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}', values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    print(f'{val_loss} < {best_validation_loss}... saving as new best_model.pt')
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))
            progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            progress_bar.update()
            update += 1
            if update >= n_updates:
                break

    progress_bar.close()
    print('finished training')

    # final evaluation on best model
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    fold_idx = 0

    train_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[0])
    train_set = FeatureDataset(train_set, classes, excerpt_size=384)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=0)

    val_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[1])
    val_set = FeatureDataset(val_set, classes, excerpt_size=384)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    test_set = BaseDataset(scenes=scenes, features='mels', data_path=os.path.join('data', 'eval'))
    test_set = FeatureDataset(test_set, classes, excerpt_size=384)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

    test_loss = evaluate_model(net, dataloader=test_loader, device=device)
    val_loss = evaluate_model(net, dataloader=val_loader, device=device)
    train_loss = evaluate_model(net, dataloader=train_loader, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to separate file
    with open(os.path.join(results_path, 'results.txt'), 'w') as f:
        print(f"Scores:", file=f)
        print(f"test loss: {test_loss}", file=f)
        print(f"validation loss: {val_loss}", file=f)
        print(f"training loss: {train_loss}", file=f)


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
