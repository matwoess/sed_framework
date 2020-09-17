# -*- coding: utf-8 -*-
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import postproc
import util
from dataset import ExcerptDataset, BaseDataset
import metric
from plot import Plotter


class Evaluator:
    def __init__(self, feature_type: str, scene: str, hyper_params: dict, network_params: dict, fft_params: dict,
                 model_path: str, device: torch.device, writer: SummaryWriter, plotter: Plotter):
        self.feature_type = feature_type
        self.scene = scene
        self.classes = util.get_scene_classes(scene)
        self.hyper_params = hyper_params
        self.network_params = network_params
        self.fft_params = fft_params
        self.device = device
        self.writer = writer
        self.plotter = plotter
        self.loss_fn = torch.nn.BCELoss()
        self.net = torch.load(model_path, map_location='cpu').to(device)

    def evaluate(self) -> None:
        dev_dataset = BaseDataset(self.feature_type, self.scene, self.hyper_params, self.fft_params)
        eval_dataset = BaseDataset(self.feature_type, self.scene, self.hyper_params, self.fft_params,
                                   data_path=os.path.join('data', 'eval'))
        dev_set = ExcerptDataset(dev_dataset, self.feature_type, self.classes, self.hyper_params['excerpt_size'],
                                 self.fft_params, overlap_factor=1, rnd_augment=False)
        eval_set = ExcerptDataset(eval_dataset, self.feature_type, self.classes, self.hyper_params['excerpt_size'],
                                  self.fft_params, overlap_factor=1, rnd_augment=False)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)
        eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

        eval_loss, metrics_eval, metrics_pp_eval = self.evaluate_model_on_files(eval_loader)
        dev_loss, metrics_dev, metrics_pp_dev = self.evaluate_model_on_files(dev_loader)

        # write results to files and log parameters
        self.write_losses(dev_loss, eval_loss)
        self.write_metrics(metrics_dev, metrics_eval, metrics_pp_dev, metrics_pp_eval)
        filtered_results = self.log_params(metrics_eval, metrics_pp_eval)
        print(f'final eval ER: {filtered_results["final_metric/segment_based/overall/ER"]}')
        print(f'final eval F: {filtered_results["final_metric/segment_based/overall/F"]}')
        print(f'final eval ER (post-processed): {filtered_results["final_metric_pp/segment_based/overall/ER"]}')
        print(f'final eval F (post-processed): {filtered_results["final_metric_pp/segment_based/overall/F"]}')

    @staticmethod
    def write_losses(dev_loss, eval_loss):
        with open(os.path.join('results', 'final_losses.txt'), 'w') as f:
            print(f"Scores:", file=f)
            print(f"evaluation set loss: {eval_loss}", file=f)
            print(f"development set loss: {dev_loss}", file=f)

    @staticmethod
    def write_metrics(metrics_dev, metrics_eval, metrics_pp_dev, metrics_pp_eval):
        metrics_path = os.path.join('results', 'final', 'metrics')
        metrics_pp_path = os.path.join('results', 'final_pp', 'metrics')
        metric.write_dcase_metrics_to_file(metrics_eval, metrics_path, 'eval_average')
        metric.write_dcase_metrics_to_file(metrics_dev, metrics_path, 'dev_average')
        metric.write_dcase_metrics_to_file(metrics_pp_eval, metrics_pp_path, 'eval_average')
        metric.write_dcase_metrics_to_file(metrics_pp_dev, metrics_pp_path, 'dev_average')

    def log_params(self, metrics_eval, metrics_pp_eval):
        all_params = {}
        for key in self.hyper_params.keys():
            all_params[key] = self.hyper_params[key]
        for key in self.network_params.keys():
            all_params[key] = self.network_params[key]
        for key in self.fft_params.keys():
            all_params[key] = self.fft_params[key]
        flat_results = util.flatten_dict(metrics_eval, 'final_metric')
        flat_results_pp = util.flatten_dict(metrics_pp_eval, 'final_metric_pp')
        filtered_results = metric.filter_metrics_dict(flat_results)
        filtered_results_pp = metric.filter_metrics_dict(flat_results_pp)
        # log hyper parameters and metrics to tensorboard
        for key in filtered_results_pp.keys():
            filtered_results[key] = filtered_results_pp[key]
        self.writer.add_hparams(all_params, filtered_results)
        return filtered_results

    def evaluate_model_on_files(self, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, dict, dict]:
        plot_per_file = True
        plot_path = os.path.join('results', 'final', 'plots')
        metrics_path = os.path.join('results', 'final', 'metrics')
        plot_pp_path = os.path.join('results', 'final_pp', 'plots')
        metrics_pp_path = os.path.join('results', 'final_pp', 'metrics')
        loss = torch.tensor(0., device=self.device)
        file_targets, file_predictions, file_pp_predictions = [], [], []
        all_targets, all_predictions, all_pp_predictions = [], [], []
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
                    concat_pp_predictions = np.concatenate(file_pp_predictions, axis=2)
                    all_targets.append(concat_targets)
                    all_predictions.append(concat_predictions)
                    all_pp_predictions.append(concat_pp_predictions)
                    # plot and compute metrics
                    filename = os.path.split(curr_file)[-1]
                    if plot_per_file:
                        self.plotter.plot(concat_targets, concat_predictions, plot_path, filename, True)
                        self.plotter.plot(concat_targets, concat_pp_predictions, plot_pp_path, filename, True)
                    metrics = metric.compute_dcase_metrics([concat_targets], [concat_predictions], self.classes)
                    metric.write_dcase_metrics_to_file(metrics, metrics_path, filename)
                    metrics_pp = metric.compute_dcase_metrics([concat_targets], [concat_pp_predictions], self.classes)
                    metric.write_dcase_metrics_to_file(metrics_pp, metrics_pp_path, filename)
                    # set up variables for next file
                    curr_file = audio_file
                    file_targets.clear()
                    file_predictions.clear()

                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)
                predictions = self.net(inputs)
                loss += self.loss_fn(predictions, targets)
                file_targets.append(targets.detach().cpu().numpy())
                detach_predictions = predictions.detach().cpu().numpy()
                file_predictions.append(detach_predictions)
                file_pp_predictions.append(postproc.post_process_predictions(detach_predictions))
            loss /= len(dataloader)

            metrics = metric.compute_dcase_metrics(all_targets, all_predictions, self.classes)
            metrics_pp = metric.compute_dcase_metrics(all_targets, [postproc.post_process_predictions(pred) for pred in
                                                                    all_predictions], self.classes)
        return loss, metrics, metrics_pp
