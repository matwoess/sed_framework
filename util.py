# -*- coding: utf-8 -*-
import os
import zipfile
from typing import List

import numpy as np
import requests
import tqdm
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
    return classes


def median_filter_predictions(array: np.ndarray, frame_size: int = 10) -> np.ndarray:
    n_dimensions = len(array.shape)
    filter_shape = (*np.ones(n_dimensions - 1, dtype=np.int), frame_size)
    result = median_filter(array, size=filter_shape)
    return result


def flatten_dict(dictionary: dict, root: str = '', separator: str = '/'):
    def flatten(obj, string=''):
        if type(obj) == dict:
            string = string + separator if string else string
            for k in obj.keys():
                yield from flatten(obj[k], string + str(k))
        else:
            yield string, obj

    flat_dict = {k: v for k, v in flatten(dictionary, root)}
    return flat_dict


def download_url(url, save_path, description, chunk_size=4096):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size), desc=description,
                               total=total_length // chunk_size):
            fd.write(chunk)


def download_dataset(data_path: str = 'data'):
    print('downloading dataset files...')
    os.makedirs(data_path, exist_ok=True)
    url_dev_data = 'https://zenodo.org/record/45759/files/TUT-sound-events-2016-development.audio.zip?download=1'
    url_dev_meta = 'https://zenodo.org/record/45759/files/TUT-sound-events-2016-development.meta.zip?download=1'
    url_eval_data = 'https://zenodo.org/record/996424/files/TUT-sound-events-2016-evaluation.audio.zip?download=1'
    url_eval_meta = 'https://zenodo.org/record/996424/files/TUT-sound-events-2016-evaluation.meta.zip?download=1'
    path_extracted_dev = os.path.join(data_path, 'TUT-sound-events-2016-development')
    path_target_dev = os.path.join(data_path, 'dev')
    path_extracted_eval = os.path.join(data_path, 'TUT-sound-events-2016-evaluation')
    path_target_eval = os.path.join(data_path, 'eval')

    path_temp_file = os.path.join(data_path, 'temp.zip')
    to_download = [
        (url_dev_data, 'development data'),
        (url_dev_meta, 'development meta')
    ]
    # development set
    for url, name in to_download:
        download_url(url, path_temp_file, name)
        archive = zipfile.ZipFile(path_temp_file)
        archive.extractall(data_path)
    os.rename(path_extracted_dev, path_target_dev)
    # evaluation set
    to_download = [
        (url_eval_data, 'evaluation audio files'),
        (url_eval_meta, 'evaluation meta files')
    ]
    for url, name in to_download:
        download_url(url, path_temp_file, name)
        archive = zipfile.ZipFile(path_temp_file)
        archive.extractall(data_path)
    os.rename(path_extracted_eval, path_target_eval)
    # cleanup
    os.remove(path_temp_file)


def zip_folder(folder: str, zipfile_name: str):
    archive = zipfile.ZipFile(f'{zipfile_name}.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder):
        for file in files:
            archive.write(os.path.join(root, file))
    archive.close()


if __name__ == '__main__':
    download_dataset()
    zip_folder('results')
