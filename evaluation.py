import os
from typing import Tuple, NamedTuple, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import utils
from datasets import ExcerptDataset, BaseDataset
from dcase16_evaluation import DCASE2013_EventDetection_Metrics, DCASE2016_EventDetection_SegmentBasedMetrics


class Metrics(NamedTuple):
    frame_based: dict
    event_based: dict
    class_based: dict
    segment_based: dict


def final_evaluation(classes: list, excerpt_size: int, feature_type: str, model_path: str, scenes: list,
                     training_dataset: BaseDataset, device: torch.device) -> None:
    # final evaluation on best model
    net = torch.load(model_path)
    dev_set = ExcerptDataset(training_dataset, classes, excerpt_size=excerpt_size)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)
    eval_set = BaseDataset(scenes=scenes, features=feature_type, data_path=os.path.join('data', 'eval'))
    eval_set = ExcerptDataset(eval_set, classes, excerpt_size=excerpt_size)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

    eval_loss, eval_metrics, eval_pp_metrics = evaluate_model_on_files(net, eval_loader, classes, device=device)
    dev_loss, dev_metrics, dev_pp_metrics = evaluate_model_on_files(net, dev_loader, classes, device=device)

    print(f"Scores:")
    print(f"evaluation set loss: {eval_loss}")
    print(f"development set loss: {dev_loss}")
    # Write result to separate file
    with open(os.path.join('results', 'losses.txt'), 'w') as f:
        print(f"Scores:", file=f)
        print(f"evaluation set loss: {eval_loss}", file=f)
        print(f"development set loss: {dev_loss}", file=f)

    avg_eval_metrics = calc_avg_metrics(eval_metrics)
    avg_dev_metrics = calc_avg_metrics(dev_metrics)
    avg_eval_pp_metrics = calc_avg_metrics(eval_pp_metrics)
    avg_dev_pp_metrics = calc_avg_metrics(dev_pp_metrics)
    write_dcase_metrics_to_file(avg_eval_metrics, os.path.join('results', 'final', 'metrics'), 'eval_average')
    write_dcase_metrics_to_file(avg_dev_metrics, os.path.join('results', 'final', 'metrics'), 'dev_average')
    write_dcase_metrics_to_file(avg_eval_pp_metrics, os.path.join('results', 'final_pp', 'metrics'), 'eval_average')
    write_dcase_metrics_to_file(avg_dev_pp_metrics, os.path.join('results', 'final_pp', 'metrics'), 'dev_average')


def evaluate_model_on_files(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, classes: list,
                            device: torch.device, loss_fn=torch.nn.BCELoss()) -> Tuple[torch.Tensor, list, list]:
    plot_path = os.path.join('results', 'final', 'plots')
    metrics_path = os.path.join('results', 'final', 'metrics')
    plot_pp_path = os.path.join('results', 'final_pp', 'plots')
    metrics_pp_path = os.path.join('results', 'final_pp', 'metrics')
    loss = torch.tensor(0., device=device)
    all_metrics = []
    all_pp_metrics = []
    file_targets = []
    file_predictions = []
    curr_file = None
    with torch.no_grad():
        for data in tqdm(dataloader, desc='evaluating', position=0):
            inputs, targets, audio_file, idx = data
            audio_file = audio_file[0]
            if curr_file is None:
                curr_file = audio_file
            elif audio_file != curr_file:
                # combine all targets and predictions from current file
                all_targets = np.concatenate(file_targets, axis=2)
                all_predictions = np.concatenate(file_predictions, axis=2)
                # plot and compute metrics
                filename = os.path.split(curr_file)[-1]
                utils.plot(all_targets, all_predictions, classes, plot_path, filename, False, to_seconds=True)
                utils.plot(all_targets, all_predictions, classes, plot_pp_path, filename, True, to_seconds=True)
                metrics = compute_dcase_metrics(all_targets, all_predictions, classes, False)
                write_dcase_metrics_to_file(metrics, metrics_path, filename)
                metrics_pp = compute_dcase_metrics(all_targets, all_predictions, classes, True)
                write_dcase_metrics_to_file(metrics_pp, metrics_pp_path, filename)
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
    return loss, all_metrics, all_pp_metrics


def calc_avg_metrics(metrics_list: List[Metrics]) -> Metrics:
    class_based = [m.class_based for m in metrics_list]
    frame_based = [m.frame_based for m in metrics_list]
    event_based = [m.event_based for m in metrics_list]
    segment_based = [m.segment_based for m in metrics_list]

    def calc_avg_dict(dict_list: list) -> dict:
        result = {}
        for dictionary in dict_list:
            for key in dictionary.keys():
                if type(dictionary[key]) == dict:
                    result[key] = calc_avg_dict([d[key] for d in dict_list])
                else:
                    result[key] = np.sum([d[key] for d in dict_list]) / len(dict_list)
        return result

    avg = Metrics(calc_avg_dict(frame_based),
                  calc_avg_dict(event_based),
                  calc_avg_dict(class_based),
                  calc_avg_dict(segment_based))
    return avg


def write_dcase_metrics_to_file(metrics: Metrics, folder, file) -> None:
    os.makedirs(folder, exist_ok=True)

    def get_dict_string(d: dict, depth=0) -> str:
        result = ''
        ident = depth * "  "
        for key in d:
            if type(d[key]) == dict:
                result += f'{ident}{key}:\n{get_dict_string(d[key], depth + 1)}'
            else:
                result += f'{ident}{key}: {d[key]}'
            result += '\n'
        return result

    with open(os.path.join(folder, f'{file}.txt'), 'w') as f:
        print(f"=== DCASE2016 - segment based ===\n", file=f)
        print(get_dict_string(metrics.segment_based), file=f)
        print(f"=== DCASE2013 - event based ===\n", file=f)
        print(get_dict_string(metrics.event_based), file=f)
        print(f"=== DCASE2013 - class based ===\n", file=f)
        print(get_dict_string(metrics.class_based), file=f)
        print(f"=== DCASE2013 - frame based ===\n", file=f)
        print(get_dict_string(metrics.frame_based), file=f)


def get_event_list(batch: np.ndarray, classes: list, time_factor=512 / 22050) -> List:
    event_list = []
    for cls_idx, cls in enumerate(batch):
        event = False
        onset = -1.0
        # offset = -1.0
        for i, val in enumerate(cls):
            if val == 1 and not event:
                event = True
                onset = i * time_factor
            if val == 0 and event:
                event = False
                offset = i * time_factor
                event_list.append({
                    'event_label': classes[cls_idx],
                    'event_onset': onset,
                    'event_offset': offset
                })
    return event_list


def compute_dcase_metrics(targets: np.ndarray, predictions: np.ndarray, classes: list, post_process=False) -> Metrics:
    threshold = 0.5
    predictions = np.where(predictions >= threshold, 1, 0)
    if post_process:
        predictions = utils.median_filter_predictions(predictions, frame_size=10)
    # create metric classes and get lists
    dcase2013metric = DCASE2013_EventDetection_Metrics(class_list=classes)
    dcase2016_metric = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=classes)
    system_output = get_event_list(predictions[0, ...], classes)
    annotated_groundtruth = get_event_list(targets[0, ...], classes)
    # get dcase metrics
    frame_based_metrics = dcase2013metric.frame_based(annotated_groundtruth, system_output)
    even_based_metrics = dcase2013metric.event_based(annotated_groundtruth, system_output)
    class_based_metrics = dcase2013metric.class_based(annotated_groundtruth, system_output)
    segment_based_metrics = dcase2016_metric.evaluate(annotated_groundtruth, system_output).results()
    return Metrics(frame_based_metrics, even_based_metrics, class_based_metrics, segment_based_metrics)


if __name__ == '__main__':
    np.random.seed(1)
    test_excerpt_length = 32
    test_targets = np.where(np.random.rand(16, 3, test_excerpt_length) >= 0.8, 1, 0)
    test_predictions = np.where(np.random.rand(16, 3, test_excerpt_length) >= 0.9, 1, 0)
    test_classes = ["bird singing", "children shouting", "wind blowing"]
    test_path = 'results/metrics'
    test_update = 1
    test_metrics = compute_dcase_metrics(test_targets, test_predictions, test_classes)
    write_dcase_metrics_to_file(test_metrics, os.path.join('results', 'metrics'), 'test')
    test_predictions = np.where(np.random.rand(16, 3, test_excerpt_length) >= 0.99, 1, 0)
    test_metrics2 = compute_dcase_metrics(test_targets, test_predictions, test_classes)
    test_metrics_list = [test_metrics, test_metrics, test_metrics2]
    test_metrics = calc_avg_metrics(test_metrics_list)
    write_dcase_metrics_to_file(test_metrics, os.path.join('results', 'metrics'), 'test_avg')
