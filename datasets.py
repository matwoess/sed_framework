# -*- coding: utf-8 -*-
import csv
import glob
import os
from typing import List, Tuple, Dict

import librosa
import numpy as np
import dill as pickle
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


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


def create_dataset(data_path, dataset_file, scene, features_type, n_folds: int = 4):
    scene_audio_path = os.path.join(data_path, 'audio', scene)
    audio_files = sorted(glob.glob(os.path.join(scene_audio_path, '**/*.wav'), recursive=True))
    meta_path = os.path.join(data_path, 'meta', scene)
    annotations = read_annotations(meta_path)
    if n_folds > 0:
        folds = get_folds(data_path, scene, num_folds=n_folds)
        fold_indices = get_fold_indices(audio_files, folds)
    else:
        fold_indices = []
    feature_type = features_type

    features = []
    sampling_rates = []
    desc = f'creating {scene} dataset with {features_type}'
    for i, file in tqdm(enumerate(audio_files), desc=desc, total=len(audio_files)):
        audio, sr = librosa.load(file)
        if feature_type == 'spec':
            feature = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))  # magnitudes only
        elif feature_type == 'mels':
            feature = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512)
        elif feature_type == 'mfccs':
            feature = librosa.feature.mfcc(audio, sr=sr)
        else:
            raise ValueError(f'specified feature extraction "{feature_type}" is not supported.')
        features.append(feature.T)
        sampling_rates.append(sr)

    data = {'features': features, 'annotations': annotations, 'sampling_rates': sampling_rates,
            'audio_files': audio_files, 'fold_indices': fold_indices}
    with open(dataset_file, 'wb') as f:
        pickle.dump(data, f)
    return data


class SceneDataset(Dataset):
    def __init__(self, scene: str, features_type: str, data_path: str = os.path.join('data', 'dev')):
        self.scene = scene
        self.feature_type = features_type
        n_folds = 4 if 'dev' in data_path else 0
        dataset_file = os.path.join(data_path, f'{scene}_{features_type}.pkl')
        if not os.path.exists(dataset_file):
            data = create_dataset(data_path, dataset_file, scene, features_type, n_folds)
        else:
            with open(dataset_file, 'rb') as f:
                data = pickle.load(f)
            print(f'loaded dataset from {dataset_file}')
        self.features = data['features']
        self.annotations = data['annotations']
        self.sampling_rates = data['sampling_rates']
        self.audio_files = data['audio_files']
        self.fold_indices = data['fold_indices']

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        feature = self.features[idx]
        ann = self.annotations[idx]
        sr = self.sampling_rates[idx]
        audio_file = self.audio_files[idx]
        return feature, ann, sr, audio_file, idx


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


def get_target_array(curr_seq_idx: int, annotations: list, classes: list, sr: int) -> np.ndarray:
    target_array = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        class_events = [(item['onset'], item['offset']) for item in annotations if item['event'] == cls]
        for onset, offset in class_events:
            onset_idx = int(float(onset) * sr)
            offset_idx = int(float(offset) * sr)
            if onset_idx <= curr_seq_idx <= offset_idx:
                target_array[i] = 1
                break
    return target_array


class FeatureDataset(Dataset):
    def __init__(self, dataset: Dataset, classes: list, excerpt_size: int = 384):
        self.dataset = dataset
        total_len = 0
        features = []
        targets = []
        for feature, ann, sr, audio_file, idx in dataset:
            feature_vector = [row for row in feature]
            length = len(feature_vector)
            n_excerpts = int(np.ceil(length / excerpt_size))
            for i in range(n_excerpts):
                excerpt = np.zeros(shape=(excerpt_size, len(feature_vector[0])))
                begin_idx = i * excerpt_size
                end_idx = min((i + 1) * excerpt_size, feature.shape[0])
                excerpt[0:end_idx - begin_idx, :] = feature[begin_idx:end_idx]
                features.append(excerpt)
                curr_seq_idx = i * excerpt_size + excerpt_size // 2
                target_array = get_target_array(curr_seq_idx, ann, classes, sr)
                targets.append(target_array)
            total_len += n_excerpts
        self.features = features
        self.targets = targets
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], idx
