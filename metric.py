# -*- coding: utf-8 -*-
import os
from typing import List

import numpy as np

import util
from dcase16_evaluation import DCASE2016_EventDetection_SegmentBasedMetrics, DCASE2016_EventDetection_EventBasedMetrics


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


def compute_dcase_metrics(targets: np.ndarray, predictions: np.ndarray, classes: list, post_process=False) -> dict:
    threshold = 0.5
    predictions = np.where(predictions >= threshold, 1, 0)
    if post_process:
        predictions = util.median_filter_predictions(predictions, frame_size=10)
    # create metric classes and get lists
    dcase2016_segment_based = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=classes)
    dcase2016_event_based = DCASE2016_EventDetection_EventBasedMetrics(class_list=classes)
    if len(targets.shape) == 3:  # concatenate batches
        predictions = np.concatenate([*predictions], axis=1)
        targets = np.concatenate([*targets], axis=1)
    system_output = get_event_list(predictions, classes)
    annotated_groundtruth = get_event_list(targets, classes)
    # get dcase metrics
    segment_based_metrics = dcase2016_segment_based.evaluate(annotated_groundtruth, system_output).results()
    dcase2016_event_based.evaluate(annotated_groundtruth, system_output)
    even_based_metrics = dcase2016_event_based.results()
    return {'segment_based': segment_based_metrics, 'event_based': even_based_metrics}


def calc_avg_metrics(metrics_list: List[dict]) -> dict:
    dictionaries = [m for m in metrics_list]

    def calc_avg_dict(dict_list: list) -> dict:
        result = {}
        for dictionary in dict_list:
            for key in dictionary.keys():
                if type(dictionary[key]) == dict:
                    result[key] = calc_avg_dict([d[key] for d in dict_list])
                else:
                    result[key] = np.sum([d[key] for d in dict_list]) / len(dict_list)
        return result

    avg = calc_avg_dict(dictionaries)
    return avg


def filter_metrics_dict(dictionary: dict) -> dict:
    whitelist = [
        'segment_based/overall/ER', 'segment_based/overall/F',
        'event_based/overall/ER', 'event_based/overall/F',
    ]
    result_dict = {}
    for key in dictionary.keys():
        if str(key).endswith(tuple(whitelist)):
            result_dict[key] = dictionary[key]
    return result_dict


def write_dcase_metrics_to_file(metrics: dict, folder, file) -> None:
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
        print(get_dict_string(metrics['segment_based']), file=f)
        print(f"=== DCASE2016 - event based ===\n", file=f)
        print(get_dict_string(metrics['event_based']), file=f)
