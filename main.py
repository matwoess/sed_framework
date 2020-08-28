# -*- coding: utf-8 -*-
import os

import torch
import torch.utils.data
import numpy as np
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
            # loss += loss_fn(predictions, targets)
            loss += (torch.stack([loss_fn(pred, target) for pred, target in zip(predictions, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def main(results_path: str, network_config: dict, eval_settings: dict, classes: list, scenes: list, feature_type: str,
         learning_rate: int = 1e-3, weight_decay: float = 1e-4, n_updates: int = int(1e5), excerpt_size: int = 384,
         batch_size: int = 16, device: torch.device = torch.device("cuda:0"),
         ):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))
    training_dataset = BaseDataset(scenes=scenes, features=feature_type)

    # Create Network
    network_config['out_features'] = len(classes)
    net = SimpleCNN(**network_config)
    net.to(device)
    # Get loss function
    loss_fn = torch.nn.BCELoss()
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    plot_at = eval_settings['plot_at']
    best_validation_loss = np.inf  # best validation loss so far
    progress_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    update = 0  # current update counter
    fold_iteration = 0
    fold_idx = -1

    loss = torch.empty(0)
    inputs = torch.empty(0)
    targets = torch.empty(0)
    predictions = torch.empty(0)

    while update <= n_updates:
        fold_idx = (fold_idx + 1) % 4
        train_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[0])
        val_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[1])
        train_set = FeatureDataset(train_set, classes, excerpt_size=excerpt_size)
        val_set = FeatureDataset(val_set, classes, excerpt_size=excerpt_size)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

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

            # Plot output
            if update % plot_at == 0 and update > 0:
                plot_targets = targets.detach().numpy()
                plot_predictions = predictions.detach().numpy()
                utils.plot(plot_targets, plot_predictions, classes, excerpt_size, plots_path, update)

            # update progress and update-counter
            progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            progress_bar.update()
            update += 1
            if update >= n_updates:
                break

        # log training loss
        writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)
        # Evaluate model on validation set (after every complete training fold)
        val_loss = evaluate_model(net, dataloader=val_loader, device=device)
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
            torch.save(net, os.path.join(results_path, 'best_model.pt'))

        fold_iteration += 1

    progress_bar.close()
    print('finished training')

    # final evaluation on best model
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    fold_idx = 0

    train_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[0])
    train_set = FeatureDataset(train_set, classes, excerpt_size=excerpt_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

    val_set = Subset(training_dataset, training_dataset.get_fold_indices(scenes[0], fold_idx)[1])
    val_set = FeatureDataset(val_set, classes, excerpt_size=excerpt_size)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    test_set = BaseDataset(scenes=scenes, features=feature_type, data_path=os.path.join('data', 'eval'))
    test_set = FeatureDataset(test_set, classes, excerpt_size=excerpt_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

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
