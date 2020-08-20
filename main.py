# -*- coding: utf-8 -*-
import os

import torch
import torch.utils.data
import numpy as np
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

from architectures import DorferCNN
from datasets import TrainingDataset
import utils


def evaluate_model(net, dataloader, device) -> torch.Tensor:
    return torch.empty()


def main(results_path: str, network_config: dict, eval_settings: dict, classes: list, learning_rate: int = 1e-3,
         weight_decay: float = 1e-4, n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters, creates the architecture, trains the model and evaluates it"""
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))
    training_dataset = TrainingDataset()

    # Create Network
    net = DorferCNN(**network_config)
    net.to(device)
    # Get loss function
    loss_fn = torch.nn.MSELoss()
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    tb_stats_at = eval_settings['tb_stats_at']  # print stats to tensorboard every x updates
    plot_at = eval_settings['plot_at']  # plot every x updates
    validate_at = eval_settings['validate_at']  # test on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    progress_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))

    for fold_idx in range(4):
        train_set = Subset(training_dataset, training_dataset.fold_indices[fold_idx][0])
        val_set = Subset(training_dataset, training_dataset.fold_indices[fold_idx][0])
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
        for data in train_loader:
            file, spec, mfccs, mels, ann, idx = data
            inputs = spec  # TODO
            targets = ann  # TODO
            inputs.to(device)
            optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_fn(predictions - targets)  # TODO
            loss = loss.mean()
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

    progress_bar.close()
    print('finished training')

    # final evaluation on best model
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    fold_idx = 0

    train_set = Subset(training_dataset, training_dataset.fold_indices[fold_idx][0])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)

    val_set = Subset(training_dataset, training_dataset.fold_indices[fold_idx][0])
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    test_dataset = TrainingDataset(data_path=os.path.join('data', 'eval'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

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
