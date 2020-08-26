# -*- coding: utf-8 -*-
import os

import numpy as np
from matplotlib import pyplot as plt


def plot(targets: np.ndarray, predictions: np.ndarray, classes: list, length: int, path: str, iteration: int) -> None:
    os.makedirs(path, exist_ok=True)
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    # compute errors
    threshold = 0.5
    thresh_predictions = np.where(predictions >= threshold, 1, 0)
    errors = thresh_predictions != targets
    # create subplots for targets, predictions and errors
    to_plot = [targets, thresh_predictions, errors]
    plot_titles = ['target', 'prediction', 'error']
    batches = targets.shape[0]
    for b in range(batches):
        plt.rc('ytick', labelsize=6)
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
        fig.savefig(os.path.join(path, f"{iteration:07d}_{b:02d}.png"), dpi=750)
        del fig


if __name__ == '__main__':
    np.random.seed(0)
    test_excerpt_length = 32
    test_targets = np.random.randint(low=0, high=1+1, size=(16, 3, test_excerpt_length))
    test_predictions = np.random.rand(16, 3, test_excerpt_length)
    test_classes = ["bird singing", "children shouting", "wind blowing"]
    test_path = 'results/plots'
    test_update = 1
    plot(test_targets, test_predictions, test_classes, test_excerpt_length, test_path, test_update)
