# -*- coding: utf-8 -*-
import os
import zipfile
from typing import Dict

import numpy as np
import requests
import tqdm
from matplotlib import pyplot as plt


def plot(targets: np.ndarray, predictions: np.ndarray, classes: list, path: str, iteration: int) -> None:
    print('plotting results...')
    os.makedirs(path, exist_ok=True)
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    label_size = 6 if len(classes) < 12 else 4
    # compute errors
    threshold = 0.5
    thresh_predictions = np.where(predictions >= threshold, 1, 0)
    errors = thresh_predictions != targets
    # create subplots for targets, predictions and errors
    to_plot = [targets, thresh_predictions, errors]
    plot_titles = ['target', 'prediction', 'error']
    batches = targets.shape[0]
    length = targets.shape[-1]
    for b in range(batches):
        plt.rc('ytick', labelsize=label_size)
        fig, ax = plt.subplots(3, 1)
        plt.setp(ax, yticklabels=classes, yticks=np.arange(len(classes)),
                 xlim=(-10, length + 5), ylim=(-0.5, len(classes) - 0.5))
        for a, array in enumerate(to_plot):
            for c, cls in enumerate(classes):
                mask = [x == 1 for x in array[b, c, :]]
                x_data = [i for i, m in enumerate(mask) if m]
                y_data = [c for m in mask if m]
                ax[a].set_title(plot_titles[a])
                color = cmap[c % len(cmap)]
                ax[a].plot(x_data, y_data, marker='.', color=color, linestyle='None', markersize=1)
        # fig.show()
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"{iteration:07d}_{b:02d}.png"), dpi=500)
        fig.clf()
    del fig


def compute_metrics(targets: np.ndarray, predictions: np.ndarray, save_path: str, iteration: int) -> None:
    print('computing metrics...')
    os.makedirs(save_path, exist_ok=True)
    targets = targets == 1
    targets = targets.reshape(-1)
    n_samples = len(targets)
    threshold = 0.5
    predictions = predictions.reshape(-1)
    predictions = [b for b in predictions >= threshold]
    if not np.any(targets):
        print('no positive targets, aborting...')
        return
    if not np.any(predictions):
        print('no positive predictions, aborting...')
        return
    tp = sum([1 for p, t in zip(predictions, targets) if t and p])
    tn = sum([1 for p, t in zip(predictions, targets) if not t and not p])
    fp = sum([1 for p, t in zip(predictions, targets) if not t and p])
    fn = sum([1 for p, t in zip(predictions, targets) if t and not p])
    assert tp + tn + fp + fn == n_samples
    P = tp + fn
    N = tn + fp
    tpr = 1.0 if P == 0 else tp / P
    tnr = 1.0 if N == 0 else tn / N
    fpr = 1 - tnr
    fnr = 1 - tpr
    ppr = tp / (tp + fp)
    acc = (tp + tn) / n_samples
    bacc = (tpr + tnr) / 2
    f1score = 0.0 if tpr + ppr == 0 else 2 * (tpr * ppr) / (tpr + ppr)
    metrics = {'F1-score': f1score, 'error rate / false negative rate': fnr,
               'accuracy': acc, 'balanced accuracy': bacc,
               'recall / true positive rate': tpr, 'precision / positive predictive rate': ppr,
               'fall-out / false positive rate': fpr, 'true negative rate': tnr}
    with open(os.path.join(save_path, f'{iteration:07d}.txt'), 'w') as f:
        print(f"Metrics:", file=f)
        for key in metrics.keys():
            print(f'{key}: {metrics[key]}', file=f)


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


def zip_folder(folder):
    archive = zipfile.ZipFile(f'{folder}.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder):
        for file in files:
            archive.write(os.path.join(root, file))
    archive.close()


if __name__ == '__main__':
    # download_dataset()
    np.random.seed(0)
    test_excerpt_length = 32
    test_targets = np.random.randint(low=0, high=1 + 1, size=(16, 3, test_excerpt_length))
    test_predictions = np.random.rand(16, 3, test_excerpt_length)
    test_classes = ["bird singing", "children shouting", "wind blowing"]
    test_path = 'results/plots'
    test_update = 1
    # plot(test_targets, test_predictions, test_classes, test_path, test_update)
    # compute_metrics(test_targets, test_predictions, 'results/metrics', test_update)
    # zip_folder('results')
