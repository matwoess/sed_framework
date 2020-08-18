# -*- coding: utf-8 -*-
import csv
import glob
import os

import librosa
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


class SpectrogramDataset(Dataset):
    def __init__(self, dataset: Dataset, data_path: str = os.path.join('data', 'dev')):
        self.dataset = dataset
        if not os.path.exists(data_path):
            raise ValueError(f'dataset path "{data_path}" does not exist')

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        audio_file, _ = self.dataset.__getitem__(idx)
        audio, sr = librosa.load(audio_file)
        spec = np.abs(librosa.stft(audio, n_fft=2048))
        mels = librosa.mel_frequencies(n_mels=64)
        mfccs = librosa.feature.mfcc(audio, sr=sr)
        return spec, mfccs, mels, sr, idx


def read_annotations(annotation_files):
    annotations = []
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as af:
            content = list(csv.reader(af, delimiter='\t'))
        data = {
            'onsets': [on for on, _, _ in content],
            'offsets': [off for _, off, _ in content],
            'events': [evt for _, _, evt in content]
        }
        annotations.append(data)
    return annotations


class TrainingDataset(Dataset):
    def __init__(self, dataset: Dataset, data_path: str = os.path.join('data', 'dev')):
        self.dataset = dataset
        if not os.path.exists(data_path):
            raise ValueError(f'dataset path "{data_path}" does not exist')
        meta_path = os.path.join(data_path, 'meta')
        annotation_files = sorted(glob.glob(os.path.join(meta_path, '**/*.ann'), recursive=True))
        self.annotations = read_annotations(annotation_files)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        spec, mfccs, mels, sr, _ = self.dataset.__getitem__(idx)
        return spec, mfccs, mels, sr, self.annotations[idx], idx
