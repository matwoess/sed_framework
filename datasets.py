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
        self.data_path = data_path
        audio_path = os.path.join(data_path, 'audio')
        self.audio_files = sorted(glob.glob(os.path.join(audio_path, '**/*.wav'), recursive=True))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx], idx


class SpectrogramDataset(BaseDataset):
    def __getitem__(self, idx):
        audio_file, _ = super(SpectrogramDataset, self).__getitem__(idx)
        audio, sr = librosa.load(audio_file)
        spec = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))  # magnitudes only
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512)
        mfccs = librosa.feature.mfcc(audio, sr=sr)
        return spec.T, mfccs.T, mels.T, sr, audio_file, idx


def read_annotations(meta_path):
    annotations = []
    annotation_files = sorted(glob.glob(os.path.join(meta_path, '**/*.ann'), recursive=True))
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


def get_folds(data_path, eval_setup_folder, num_folds=4):
    folds = []
    for i in range(1, num_folds + 1):
        train_fold = sorted(glob.glob(os.path.join(data_path, eval_setup_folder, f'**/*fold{i}_train.txt'), recursive=True))
        test_fold = sorted(glob.glob(os.path.join(data_path, eval_setup_folder, f'**/*fold{i}_test.txt'), recursive=True))
        # eval_fold = sorted(glob.glob(os.path.join(eval_setup_path, f'**/*fold{i}_evaluate.txt'), recursive=True))
        train_files, test_files = [], []
        for file in train_fold:
            with open(file, 'r') as f:
                content = list(csv.reader(f, delimiter='\t'))
            train_files.extend(set([row[0] for row in content]))
        for file in test_fold:
            with open(file, 'r') as f:
                content = list(csv.reader(f, delimiter='\t'))
            test_files.extend(set([row[0] for row in content]))
        # for file in eval_fold:
        #     with open(file, 'r') as f:
        #         content = list(csv.reader(f, delimiter='\t'))
        #     eval_files.extend(set([row[0] for row in content]))
        train_files = [os.path.join(data_path, f) for f in train_files]
        test_files = [os.path.join(data_path, f) for f in test_files]
        folds.append({'fold': i, 'train_files': train_files, 'test_files': test_files})
    return folds


def get_fold_indices(audio_files, folds):
    fold_indices = []
    for fold in folds:
        train_indices, test_indices = [], []
        for file in fold['train_files']:
            train_indices.append(audio_files.index(file))
        for file in fold['test_files']:
            test_indices.append(audio_files.index(file))
        fold_indices.append((train_indices, test_indices))
    return fold_indices


class FoldsDataset(SpectrogramDataset):
    def __init__(self, data_path: str = os.path.join('data', 'dev')):
        super().__init__(data_path)
        meta_path = os.path.join(self.data_path, 'meta')
        self.annotations = read_annotations(meta_path)
        self.folds = get_folds(data_path, 'evaluation_setup', num_folds=4)
        self.fold_indices = get_fold_indices(self.audio_files, self.folds)

    def __getitem__(self, idx):
        spec, mfccs, mels, sr, audio_file, _ = super(FoldsDataset, self).__getitem__(idx)
        return spec, mfccs, mels, self.annotations[idx], sr, audio_file, idx
