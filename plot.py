# -*- coding: utf-8 -*-
import os
import threading

import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    """
    Plotter class, plotting targets predictions and differences/error.
    plotting happens in a background thread for not stalling the main thread.
    Only one thread at a time gets access to pyplot to avoid state-machine errors.
    """
    def __init__(self, classes: list, hop_size: int = 512, sampling_rate: int = 22050):
        self.classes = classes
        self.threshold = 0.5
        self.label_size = 6 if len(classes) < 12 else 4
        self.cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm']
        self.time_factor = hop_size / sampling_rate
        # access pyplot only by one thread at a time
        self.semaphore = threading.Semaphore(value=1)

    def plot(self, targets: np.ndarray, predictions: np.ndarray, path: str, identifier, to_seconds=False) -> None:
        print('plotting results...')
        # start plotting in background
        threading.Thread(target=self.plot_thread,
                         args=(targets, predictions, path, identifier, to_seconds)).start()

    def plot_thread(self, targets: np.ndarray, predictions: np.ndarray, path: str, identifier,
                    to_seconds: bool) -> None:
        # acquire access to pyplot
        self.semaphore.acquire(blocking=True)
        os.makedirs(path, exist_ok=True)
        # compute errors
        thresh_predictions = np.where(predictions >= self.threshold, 1, 0)
        errors = thresh_predictions != targets
        # create subplots for targets, predictions and errors
        to_plot = [targets, thresh_predictions, errors]
        plot_titles = ['target', 'prediction', 'error']
        batches = targets.shape[0]
        length = targets.shape[-1]
        x_factor = 1
        # covert x-axis from sequence position to seconds
        if to_seconds:
            x_factor = self.time_factor
            length = int(length * x_factor)
        # create separate plot for each sample in batch
        for b in range(batches):
            plt.rc('ytick', labelsize=self.label_size)
            fig, ax = plt.subplots(3, 1)
            # set up axis descriptions and dimensions
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
        # hand over access to pyplot
        self.semaphore.release()


if __name__ == '__main__':
    np.random.seed(0)
    test_batches = 16
    test_excerpt_length = 32
    test_classes = ["bird singing", "children shouting", "wind blowing"]
    test_targets = np.random.randint(low=0, high=1 + 1, size=(test_batches, len(test_classes), test_excerpt_length))
    test_predictions = np.random.rand(test_batches, len(test_classes), test_excerpt_length)
    test_path = 'results/plots'
    test_update = 1
    plotter = Plotter(test_classes, 512, 22050)
    plotter.plot(test_targets, test_predictions, test_path, test_update)
    import postproc
    plotter.plot(test_targets, postproc.post_process_predictions(test_predictions), test_path, test_update + 1)
