# -*- coding: utf-8 -*-
import csv
import glob
import os
from typing import List, Tuple

import librosa
import numpy as np
import dill as pickle
from torch.utils.data import Dataset
from tqdm import tqdm

import augment
import utils


class SceneDataset(Dataset):
    def __init__(self, scene: str, feature_type: str, data_path: os.path.join('data', 'dev'), rnd_augment: bool = False,
                 n_fft: int = 1024, hop_length: int = 512, n_mfcc: int = 64):
        self.scene = scene
        self.feature_type = feature_type
        self.data_path = data_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_folds = 4 if 'dev' in data_path else 0
        self.rnd_augment = rnd_augment

        dataset_file = os.path.join(data_path, f'{scene}_spectrograms.pkl')
        if not os.path.exists(dataset_file):
            data = self.create_dataset(dataset_file)
        else:
            with open(dataset_file, 'rb') as f:
                data = pickle.load(f)
            print(f'loaded dataset from {dataset_file}')

        self.features = data['features']
        self.annotations = data['annotations']
        self.sampling_rates = data['sampling_rates']
        self.audio_files = data['audio_files']
        self.fold_indices = data['fold_indices']

    def create_dataset(self, dataset_file):
        scene_audio_path = os.path.join(self.data_path, 'audio', self.scene)
        audio_files = sorted(glob.glob(os.path.join(scene_audio_path, '**/*.wav'), recursive=True))
        meta_path = os.path.join(self.data_path, 'meta', self.scene)
        annotations = self.read_annotations(meta_path)
        if self.n_folds > 0:
            folds = self.get_folds(self.data_path, self.scene, num_folds=self.n_folds)
            fold_indices = self.get_fold_indices(audio_files, folds)
        else:
            fold_indices = []

        features = []
        sampling_rates = []
        desc = f'pre-computing spectrograms for {self.scene} dataset'
        for i, file in tqdm(enumerate(audio_files), desc=desc, total=len(audio_files)):
            audio, sr = librosa.load(file)
            feature = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))  # TODO:**2?
            features.append(feature)
            sampling_rates.append(sr)

        data = {'features': features, 'annotations': annotations, 'sampling_rates': sampling_rates,
                'audio_files': audio_files, 'fold_indices': fold_indices}
        with open(dataset_file, 'wb') as f:
            pickle.dump(data, f)
        return data

    @staticmethod
    def get_fold_indices(audio_files, folds):
        fold_indices = []
        for fold in folds:
            train_indices, val_indices = [], []
            for file in fold['train_files']:
                train_indices.append(audio_files.index(file))
            for file in fold['val_files']:
                val_indices.append(audio_files.index(file))
            train_indices.sort()
            val_indices.sort()
            fold_indices.append((train_indices, val_indices))
        return fold_indices

    @staticmethod
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

    @staticmethod
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

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        feature = self.features[idx]
        ann = self.annotations[idx]
        sr = self.sampling_rates[idx]
        audio_file = self.audio_files[idx]
        if self.feature_type in ['mels', 'mfccs']:
            mel_basis = librosa.filters.mel(sr=sr, n_fft=self.n_fft)
            feature = np.dot(mel_basis, feature)
        feature = librosa.amplitude_to_db(feature, ref=np.max)
        if self.feature_type == 'mfccs':
            feature = librosa.feature.mfcc(S=feature, n_mfcc=self.n_mfcc)

        return feature, ann, sr, audio_file, idx


class BaseDataset(Dataset):
    def __init__(self, scenes: List[str], feature_type: str, data_path: str = os.path.join('data', 'dev'),
                 n_fft: int = 1024, hop_length: int = 512, rnd_augment=False):
        if not os.path.exists(data_path):
            utils.download_dataset()
        self.data_path = data_path
        self.home_dataset = None
        self.residential_dataset = None
        if 'home' in scenes:
            self.home_dataset = SceneDataset('home', feature_type, data_path, rnd_augment, n_fft, hop_length)
        if 'residential_area' in scenes:
            self.residential_dataset = SceneDataset('residential_area', feature_type, data_path, rnd_augment, n_fft,
                                                    hop_length)

    def get_fold_indices(self, scenes: list, fold_idx) -> Tuple[list, list]:
        train = []
        val = []
        if 'home' in scenes:
            train_indices, val_indices = self.home_dataset.fold_indices[fold_idx]
            train.extend(train_indices)
            val.extend(val_indices)
        if 'residential_area' in scenes:
            train_indices, val_indices = self.residential_dataset.fold_indices[fold_idx]
            train.extend(train_indices)
            val.extend(val_indices)
        return train, val

    def __len__(self):
        total_len = 0
        if self.home_dataset is None:
            total_len += len(self.residential_dataset)
        if self.residential_dataset is None:
            total_len += len(self.home_dataset)
        return total_len

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


def get_target_array(from_idx: int, to_idx: int, annotations: list, classes: list, iteration: int,
                     excerpt_size: int, sr: int, hop_size: int) -> np.ndarray:
    target_array = np.zeros(shape=(len(classes), excerpt_size))
    from_seconds = (from_idx * hop_size / sr)
    to_seconds = (to_idx * hop_size / sr)
    for i, cls in enumerate(classes):
        # get all events for current class
        class_events = [(float(item['onset'].replace(',', '.')), float(item['offset'].replace(',', '.')))
                        for item in annotations if item['event'] == cls]
        # filter for those events actually occurring in current excerpt time
        class_events = [(onset, offset) for onset, offset in class_events
                        if from_seconds <= onset <= to_seconds or from_seconds <= offset <= to_seconds]
        for onset, offset in class_events:
            onset_idx = int(onset * sr / hop_size)
            offset_idx = int(offset * sr / hop_size)
            start = max(onset_idx, from_idx)
            end = min(offset_idx, to_idx)
            start = start - iteration * excerpt_size
            end = end - iteration * excerpt_size
            target_array[i, start:end] = 1
    return target_array


class ExcerptDataset(Dataset):
    def __init__(self, dataset: Dataset, classes: list, excerpt_size: int = 384, hop_size=512, rnd_augment=False):
        self.dataset = dataset
        self.rnd_augment = rnd_augment
        total_len = 0
        features = []
        targets = []
        file_names = []
        for feature, ann, sr, audio_file, _ in dataset:
            feature_count = feature.shape[0]
            sequence_positions = feature.shape[1]
            n_excerpts = int(np.ceil(sequence_positions / excerpt_size))
            for i in range(n_excerpts):  # TODO: might wanna overlap excerpts with half excerpt_size?
                excerpt = np.zeros(shape=(feature_count, excerpt_size))
                begin_idx = i * excerpt_size
                end_idx = min((i + 1) * excerpt_size, sequence_positions)
                excerpt[:, 0:end_idx - begin_idx] = feature[:, begin_idx:end_idx]
                target_array = get_target_array(begin_idx, end_idx, ann, classes, i, excerpt_size, sr, hop_size)
                features.append(excerpt)
                targets.append(target_array)
                file_names.append(audio_file)
            total_len += n_excerpts
        self.features = features
        self.targets = targets
        self.file_names = file_names
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # nomalize inputs
        feature = self.features[idx]
        if self.rnd_augment:
            feature = augment.apply_random_augmentations(feature)
        feature = augment.apply_normalization(feature)
        feature = np.expand_dims(feature, axis=0)
        return feature, self.targets[idx], self.file_names[idx], idx
