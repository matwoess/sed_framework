# -*- coding: utf-8 -*-

import glob
import os

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_path: str = os.path.join('data', 'dev')):
        if not os.path.exists(data_path):
            raise ValueError(f'dataset path "{data_path}" does not exist')
        audio_path = os.path.join(data_path, 'audio')
        self.audio_files = sorted(glob.glob(os.path.join(audio_path, '**/*.wav'), recursive=True))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx], idx


class TrainingDataset(Dataset):
    def __init__(self, dataset: Dataset, data_path: str = os.path.join('data', 'dev')):
        self.dataset = dataset
        if not os.path.exists(data_path):
            raise ValueError(f'dataset path "{data_path}" does not exist')
        meta_path = os.path.join(data_path, 'meta')
        self.annotations = sorted(glob.glob(os.path.join(meta_path, '**/*.ann'), recursive=True))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_file, _ = self.dataset.__getitem__(idx)
        return audio_file, self.annotations[idx], idx
