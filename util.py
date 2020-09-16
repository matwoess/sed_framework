# -*- coding: utf-8 -*-
import os
import zipfile
from typing import List

import numpy as np
from scipy.ndimage import median_filter


def get_scene_classes(scene: str) -> List[str]:
    classes = []
    if scene in ['indoor', 'all']:
        classes.extend([
            "(object) rustling",
            "(object) snapping",
            "cupboard",
            "cutlery",
            "dishes",
            "drawer",
            "glass jingling",
            "object impact",
            "people walking",
            "washing dishes",
            "water tap running"])
    if scene in ['outdoor', 'all']:
        classes.extend([
            "(object) banging",
            "bird singing",
            "car passing by",
            "children shouting",
            "people speaking",
            "people walking",
            "wind blowing"])
    # add a class for "no event"
    classes.append('background')
    return classes


def median_filter_predictions(array: np.ndarray, frame_size: int = 10) -> np.ndarray:
    n_dimensions = len(array.shape)
    filter_shape = (*np.ones(n_dimensions - 1, dtype=np.int), frame_size)
    result = median_filter(array, size=filter_shape)
    return result


def remove_events_if_background(array: np.ndarray) -> np.ndarray:
    background_idx = array.shape[1] - 1
    for b, batch in enumerate(array):
        for c, class_values in enumerate(batch):
            if c == background_idx:
                continue
            for i in range(array.shape[-1]):
                if array[b, background_idx, i] == 1:
                    array[b, c, i] = 0
    return array


def post_process_predictions(array: np.ndarray, min_event_len=0.1, min_gap=0.1, hop_size=512, sr=22050) -> np.ndarray:
    threshold = 0.5
    min_length_indices = int(min_event_len * sr / hop_size)
    min_gap_indices = int(min_gap * sr / hop_size)

    def remove_short_segments(bool_array, min_seg_length, replacement):
        # Find the changes in the bool_array
        change_indices = np.diff(bool_array).nonzero()[0]
        # Shift change_index with one, focus on frame after the change
        change_indices += 1
        if bool_array[0]:
            # If the first element of bool_array is True add 0 at the beginning
            change_indices = np.r_[0, change_indices]
        if bool_array[-1]:
            # If the last element of bool_array is True, add the length of the array
            change_indices = np.r_[change_indices, bool_array.size]
        # Reshape the result into two columns
        segments = change_indices.reshape((-1, 2))

        # remove short segments
        for seg in segments:
            if seg[1] - seg[0] < min_seg_length:
                array[b, c, seg[0]:seg[1] + 1] = replacement

    for b, batch in enumerate(array):
        for c, class_values in enumerate(batch):
            # no changes if no events happened
            if np.all(class_values == 0):
                continue
            event_segments = class_values >= threshold
            remove_short_segments(event_segments, min_length_indices, 0)
            gap_segments = class_values < threshold
            remove_short_segments(gap_segments, min_gap_indices, 1)

    return array


def flatten_dict(dictionary: dict, root: str = '', separator: str = '/') -> dict:
    def flatten(obj, string=''):
        if type(obj) == dict:
            string = string + separator if string else string
            for k in obj.keys():
                yield from flatten(obj[k], string + str(k))
        else:
            yield string, obj

    flat_dict = {k: v for k, v in flatten(dictionary, root)}
    return flat_dict


def zip_folder(folder: str, zipfile_name: str) -> None:
    archive = zipfile.ZipFile(f'{zipfile_name}.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder):
        for file in files:
            archive.write(os.path.join(root, file))
    archive.close()


if __name__ == '__main__':
    zip_folder('results', 'results')
