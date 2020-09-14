# -*- coding: utf-8 -*-
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import util
from dataset import ExcerptDataset, BaseDataset
import metric
from plot import Plotter


def final_evaluation(feature_type: str, scene: str, hyper_params: dict, network_params: dict, fft_params: dict,
                     model_path: str, device: torch.device, writer: SummaryWriter, plotter: Plotter) -> None:
    # final evaluation on best model
    net = torch.load(model_path)
    classes = util.get_scene_classes(scene)
    dev_dataset = BaseDataset(feature_type, scene, hyper_params, fft_params)
    eval_dataset = BaseDataset(feature_type, scene, hyper_params, fft_params, data_path=os.path.join('data', 'eval'))
    dev_set = ExcerptDataset(dev_dataset, feature_type, classes, hyper_params['excerpt_size'], fft_params,
                             excerpts_per_file=-1, rnd_augment=False)
    eval_set = ExcerptDataset(eval_dataset, feature_type, classes, hyper_params['excerpt_size'], fft_params,
                              excerpts_per_file=-1, rnd_augment=False)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

    eval_loss, metrics_eval, metrics_pp_eval = evaluate_model_on_files(net, eval_loader, classes, device, plotter)
    dev_loss, metrics_dev, metrics_pp_dev = evaluate_model_on_files(net, dev_loader, classes, device, plotter)
    # Write result to separate file
    with open(os.path.join('results', 'final_losses.txt'), 'w') as f:
        print(f"Scores:", file=f)
        print(f"evaluation set loss: {eval_loss}", file=f)
        print(f"development set loss: {dev_loss}", file=f)

    metric.write_dcase_metrics_to_file(metrics_eval, os.path.join('results', 'final', 'metrics'), 'eval_average')
    metric.write_dcase_metrics_to_file(metrics_dev, os.path.join('results', 'final', 'metrics'), 'dev_average')
    metric.write_dcase_metrics_to_file(metrics_pp_eval, os.path.join('results', 'final_pp', 'metrics'), 'eval_average')
    metric.write_dcase_metrics_to_file(metrics_pp_dev, os.path.join('results', 'final_pp', 'metrics'), 'dev_average')

    # log hyper parameters and metrics to tensorboard
    all_params = {}
    for key in hyper_params.keys():
        all_params[key] = hyper_params[key]
    for key in network_params.keys():
        all_params[key] = network_params[key]
    for key in fft_params.keys():
        all_params[key] = fft_params[key]
    flat_results = util.flatten_dict(metrics_eval, 'final_metric')
    flat_results_pp = util.flatten_dict(metrics_pp_eval, 'final_metric_pp')
    filtered_results = metric.filter_metrics_dict(flat_results)
    filtered_results_pp = metric.filter_metrics_dict(flat_results_pp)
    for key in filtered_results_pp.keys():
        filtered_results[key] = filtered_results_pp[key]
    writer.add_hparams(all_params, filtered_results)
    print(f'final eval ER: {filtered_results["final_metric/segment_based/overall/ER"]}')
    print(f'final eval F: {filtered_results["final_metric/segment_based/overall/F"]}')
    print(f'final eval ER (post-processed): {filtered_results["final_metric_pp/segment_based/overall/ER"]}')
    print(f'final eval F (post-processed): {filtered_results["final_metric_pp/segment_based/overall/F"]}')


def evaluate_model_on_files(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, classes: list,
                            device: torch.device, plotter: Plotter, loss_fn=torch.nn.BCELoss()) \
        -> Tuple[torch.Tensor, dict, dict]:
    plot_path = os.path.join('results', 'final', 'plots')
    metrics_path = os.path.join('results', 'final', 'metrics')
    plot_pp_path = os.path.join('results', 'final_pp', 'plots')
    metrics_pp_path = os.path.join('results', 'final_pp', 'metrics')
    loss = torch.tensor(0., device=device)
    all_metrics = []
    all_pp_metrics = []
    file_targets = []
    file_predictions = []
    all_targets = []
    all_predictions = []
    curr_file = None
    with torch.no_grad():
        for data in tqdm(dataloader, desc='evaluating', position=0):
            inputs, targets, audio_file, idx = data
            audio_file = audio_file[0]
            if curr_file is None:
                curr_file = audio_file
            elif audio_file != curr_file:
                # combine all targets and predictions from current file
                concat_targets = np.concatenate(file_targets, axis=2)
                concat_predictions = np.concatenate(file_predictions, axis=2)
                all_targets.append(concat_targets)
                all_predictions.append(concat_predictions)
                # plot and compute metrics
                filename = os.path.split(curr_file)[-1]
                plotter.plot(concat_targets, concat_predictions, plot_path, filename, False, to_seconds=True)
                plotter.plot(concat_targets, concat_predictions, plot_pp_path, filename, True, to_seconds=True)
                metrics = metric.compute_dcase_metrics([concat_targets], [concat_predictions], classes, False)
                metric.write_dcase_metrics_to_file(metrics, metrics_path, filename)
                metrics_pp = metric.compute_dcase_metrics([concat_targets], [concat_predictions], classes, True)
                metric.write_dcase_metrics_to_file(metrics_pp, metrics_pp_path, filename)
                all_metrics.append(metrics)
                all_pp_metrics.append(metrics_pp)
                # set up variables for next file
                curr_file = audio_file
                file_targets.clear()
                file_predictions.clear()

            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            predictions = net(inputs)
            loss += loss_fn(predictions, targets)
            file_targets.append(targets.detach().cpu().numpy())
            file_predictions.append(predictions.detach().cpu().numpy())
        loss /= len(dataloader)
        metrics = metric.compute_dcase_metrics(all_targets, all_predictions, classes, False)
        metrics_pp = metric.compute_dcase_metrics(all_targets, all_predictions, classes, True)
    return loss, metrics, metrics_pp


if __name__ == '__main__':
    np.random.seed(1)
    test_excerpt_length = 32
    test_targets = np.where(np.random.rand(16, 3, test_excerpt_length) >= 0.8, 1, 0)
    test_predictions = np.where(np.random.rand(16, 3, test_excerpt_length) >= 0.9, 1, 0)
    test_classes = ["bird singing", "children shouting", "wind blowing"]
    test_path = 'results/metrics'
    test_update = 1
    test_metrics = metric.compute_dcase_metrics([test_targets], [test_predictions], test_classes)
    metric.write_dcase_metrics_to_file(test_metrics, os.path.join('results', 'metrics'), 'test')
    test_predictions = np.where(np.random.rand(16, 3, test_excerpt_length) >= 0.99, 1, 0)
    test_metrics2 = metric.compute_dcase_metrics([test_targets], [test_predictions], test_classes)
    test_metrics_list = [test_metrics, test_metrics, test_metrics2]
    test_metrics = metric.calc_avg_metrics(test_metrics_list)
    metric.write_dcase_metrics_to_file(test_metrics, os.path.join('results', 'metrics'), 'test_avg')
