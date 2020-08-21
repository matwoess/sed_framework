# -*- coding: utf-8 -*-
import csv
import glob
import os
from typing import List, Tuple, Dict

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def read_annotations(meta_path):
    annotations = []
    annotation_files = sorted(glob.glob(os.path.join(meta_path, '**/*.ann'), recursive=True))
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as af:
            content = list(csv.reader(af, delimiter='\t'))
        file_annotations = []
        for row in content:
            file_annotations.append({'onset': row[0], 'offset': row[1], 'event': row[2]})
        annotations.append(file_annotations)
    return annotations


def get_folds(data_path: str, scene: str, num_folds: int = 4):
    folds = []
    for i in range(1, num_folds + 1):
        train_fold = os.path.join(data_path, 'evaluation_setup', f'{scene}_fold{i}_train.txt')
        test_fold = os.path.join(data_path, 'evaluation_setup', f'{scene}_fold{i}_test.txt')
        with open(train_fold, 'r') as f:
            content = list(csv.reader(f, delimiter='\t'))
        train_files = list(set([row[0] for row in content]))
        with open(test_fold, 'r') as f:
            content = list(csv.reader(f, delimiter='\t'))
        val_files = list(set([row[0] for row in content]))
        train_files = [os.path.join(data_path, f) for f in train_files]
        val_files = [os.path.join(data_path, f) for f in val_files]
        folds.append({'fold': i, 'train_files': train_files, 'val_files': val_files})
    return folds


def get_fold_indices(audio_files, folds):
    fold_indices = []
    for fold in folds:
        train_indices, val_indices = [], []
        for file in fold['train_files']:
            train_indices.append(audio_files.index(file))
        for file in fold['val_files']:
            val_indices.append(audio_files.index(file))
        fold_indices.append((train_indices, val_indices))
    return fold_indices


class SceneDataset(Dataset):
    def __init__(self, scene: str, features: str, data_path: str = os.path.join('data', 'dev')):
        self.scene = scene
        scene_audio_path = os.path.join(data_path, 'audio', scene)
        self.audio_files = sorted(glob.glob(os.path.join(scene_audio_path, '**/*.wav'), recursive=True))
        meta_path = os.path.join(data_path, 'meta', scene)
        self.annotations = read_annotations(meta_path)
        self.folds = get_folds(data_path, scene, num_folds=4)
        self.fold_indices = get_fold_indices(self.audio_files, self.folds)
        self.features = features

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio, sr = librosa.load(audio_file)
        if self.features == 'spec':
            feature = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))  # magnitudes only
        elif self.features == 'mels':
            feature = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512)
        elif self.features == 'mfccs':
            feature = librosa.feature.mfcc(audio, sr=sr)
        else:
            raise ValueError(f'specified feature extraction "{self.features}" is not supported.')
        ann = self.annotations[idx]
        return feature.T, ann, sr, audio_file, idx


class BaseDataset(Dataset):
    def __init__(self, scenes: List[str], features: str, data_path: str = os.path.join('data', 'dev')):
        if not os.path.exists(data_path):
            raise ValueError(f'dataset path "{data_path}" does not exist')  # TODO: auto-download
        self.data_path = data_path
        self.home_dataset = SceneDataset('home', features, data_path) if 'home' in scenes else None
        self.residential_dataset = SceneDataset(
            'residential_area', features, data_path) if 'residential_area' in scenes else None

    def get_fold_indices(self, scene: str, fold_idx) -> Tuple[list, list]:
        if scene == 'home':
            return self.home_dataset.fold_indices[fold_idx]
        elif scene == 'residential_area':
            return self.residential_dataset.fold_indices[fold_idx]
        else:
            return [], []

    def __len__(self):
        return len(self.home_dataset) + len(self.residential_dataset)

    def __getitem__(self, idx):
        i = idx
        if self.home_dataset is None:
            from_set = self.residential_dataset
        elif self.residential_dataset is None:
            from_set = self.home_dataset
        # in case we use both scenes
        elif idx > len(self.home_dataset):
            from_set = self.residential_dataset
            i = idx - len(self.home_dataset)
        else:
            from_set = self.home_dataset
        feature, ann, sr, audio_file, _ = from_set[i]
        return feature, ann, sr, audio_file, idx


def get_target_array(length: int, annotations: list, classes: list, sr: int) -> np.ndarray:
    target_array = np.zeros(shape=(len(classes), length))
    for i, cls in enumerate(classes):
        class_events = [(item['onset'], item['offset']) for item in annotations if item['event'] == cls]
        for onset, offset in class_events:
            onset_idx = int(onset * sr)
            offset_idx = int(offset * sr)
            target_array[i, onset_idx:offset_idx] = 1
    return target_array


class FeatureDataset(Dataset):
    def __init__(self, dataset: Dataset, classes: list):
        self.dataset = dataset
        total_len = 0
        features = []
        targets = np.empty(shape=(len(classes), 0))
        for feature, ann, sr, audio_file, idx in dataset:
            feature_vector = [row for row in feature]
            length = len(feature_vector)
            target_array = get_target_array(length, ann, classes, sr)
            features.extend(feature_vector)
            targets = np.hstack((targets, target_array))
            total_len += length
        self.features = features
        self.targets = targets
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return self.features[idx], self.targets[:, idx], idx
