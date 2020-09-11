# -*- coding: utf-8 -*-
import os
import threading
from queue import Queue

import numpy as np

from util import median_filter_predictions
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, classes: list, hop_size: int = 512, sampling_rate: int = 22050):
        self.classes = classes
        self.threshold = 0.5
        self.label_size = 6 if len(classes) < 12 else 4
        self.cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm']
        self.time_factor = hop_size / sampling_rate

        self.semaphore = threading.Semaphore(value=1)

    def plot(self, targets: np.ndarray, predictions: np.ndarray, path: str, identifier,
             post_process=False, to_seconds=False) -> None:
        print('plotting results...')
        threading.Thread(target=self.plot_thread,
                         args=(targets, predictions, path, identifier, post_process, to_seconds)).start()

    def plot_thread(self, targets: np.ndarray, predictions: np.ndarray, path: str, identifier,
                    post_process: bool, to_seconds: bool):
        self.semaphore.acquire(blocking=True)
        # compute errors
        os.makedirs(path, exist_ok=True)
        thresh_predictions = np.where(predictions >= self.threshold, 1, 0)
        if post_process:
            thresh_predictions = median_filter_predictions(thresh_predictions, frame_size=10)
        errors = thresh_predictions != targets
        # create subplots for targets, predictions and errors
        to_plot = [targets, thresh_predictions, errors]
        plot_titles = ['target', 'prediction', 'error']
        batches = targets.shape[0]
        length = targets.shape[-1]
        x_factor = 1
        if to_seconds:
            x_factor = self.time_factor
            length = int(length * x_factor)

        for b in range(batches):
            plt.rc('ytick', labelsize=self.label_size)
            fig, ax = plt.subplots(3, 1)
            plt.setp(ax, yticklabels=self.classes, yticks=np.arange(len(self.classes)),
                     xlim=(-5, length + 5), ylim=(-0.5, len(self.classes) - 0.5))
            for a, array in enumerate(to_plot):
                for c, cls in enumerate(self.classes):
                    mask = [x == 1 for x in array[b, c, :]]
                    x_data = [i * x_factor for i, m in enumerate(mask) if m]
                    y_data = [c for m in mask if m]
                    ax[a].set_title(plot_titles[a])
                    color = self.cmap[c % len(self.cmap)]
                    ax[a].plot(x_data, y_data, marker='.', color=color, linestyle='None', markersize=1)
            # fig.show()
            fig.tight_layout()
            if type(identifier) == int:
                save_path = os.path.join(path, f"{identifier:07d}_{b:02d}.png")
            else:
                save_path = os.path.join(path, f'{identifier}.png')
            fig.savefig(save_path, dpi=500)
            plt.close(fig)

        self.semaphore.release()