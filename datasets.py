# -*- coding: utf-8 -*-
import csv
import glob
import os
import threading
from queue import Queue
from typing import List, Tuple

import librosa
import numpy as np
import dill as pickle
from torch.utils.data import Dataset
from tqdm import tqdm

import augment
import utils


class SceneDataset(Dataset):
    def __init__(self, feature_type: str, scene: str, hyper_params: dict, fft_params: dict, data_path: str):
        self.scene = scene
        self.classes = utils.get_scene_classes(scene)
        self.feature_type = feature_type
        self.excerpt_size = hyper_params['excerpt_size']
        self.data_path = data_path
        self.n_fft = fft_params['n_fft']
        self.hop_size = fft_params['hop_size']
        self.n_folds = 4 if 'dev' in data_path else 0

        dataset_file = os.path.join(data_path, f'{scene}_data.pkl')
        if not os.path.exists(dataset_file):
            data = self.create_data_file(dataset_file)
        else:
            with open(dataset_file, 'rb') as f:
                data = pickle.load(f)
            print(f'loaded dataset from {dataset_file}')

        self.features = data['features']
        self.targets = data['targets']
        self.annotations = data['annotations']
        self.sampling_rates = data['sampling_rates']
        self.audio_files = data['audio_files']
        self.fold_indices = data['fold_indices']

    def create_data_file(self, dataset_file):
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
        target_arrays = []
        desc = f'pre-computing data for {self.scene} dataset'
        for i, file in tqdm(enumerate(audio_files), desc=desc, total=len(audio_files)):
            audio, sr = librosa.load(file)
            feature = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_size))  # TODO:**2?
            target = self.get_target_array(feature.shape[1], self.classes, annotations[i], sr, self.hop_size)
            features.append(feature)
            target_arrays.append(target)
            sampling_rates.append(sr)

        data = {'features': features, 'targets': target_arrays, 'annotations': annotations,
                'sampling_rates': sampling_rates, 'audio_files': audio_files, 'fold_indices': fold_indices}
        with open(dataset_file, 'wb') as f:
            pickle.dump(data, f)
        return data

    @staticmethod
    def get_target_array(seq_length: int, classes: list, annotations: list, sr: int, hop_size: int) -> np.ndarray:
        target_array = np.zeros(shape=(len(classes), seq_length))
        for i, cls in enumerate(classes):
            # get all events for current class
            class_events = [(float(item['onset'].replace(',', '.')), float(item['offset'].replace(',', '.')))
                            for item in annotations if item['event'] == cls]
            for onset, offset in class_events:
                onset_idx = int(onset * sr / hop_size)
                offset_idx = int(offset * sr / hop_size)
                target_array[i, onset_idx:offset_idx] = 1
        return target_array

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
        target = self.targets[idx]
        sr = self.sampling_rates[idx]
        audio_file = self.audio_files[idx]
        return feature, target, sr, audio_file, idx


class BaseDataset(Dataset):
    def __init__(self, feature_type: str, scene: str, hyper_params: dict, fft_params: dict,
                 data_path: str = os.path.join('data', 'dev')):
        if not os.path.exists(data_path):
            utils.download_dataset()
        self.data_path = data_path
        self.home_dataset = None
        self.residential_dataset = None
        self.classes = utils.get_scene_classes(scene)
        self.datasets = []
        if scene in ['indoor', 'all']:
            self.datasets.append(SceneDataset(feature_type, 'home', hyper_params, fft_params, data_path))
        if scene in ['outdoor', 'all']:
            self.datasets.append(SceneDataset(feature_type, 'residential_area', hyper_params, fft_params, data_path))

    def get_fold_indices(self, fold_idx) -> Tuple[list, list]:
        train = []
        val = []
        for dataset_idx, dataset in enumerate(self.datasets):
            train_indices, val_indices = dataset.fold_indices[fold_idx]
            # add offset to indices according to previous dataset lengths
            for prev_dataset_idx in range(0, dataset_idx):
                prev_len = len(self.datasets[prev_dataset_idx])
                train_indices = [orig_idx + prev_len for orig_idx in train_indices]
                val_indices = [orig_idx + prev_len for orig_idx in val_indices]
            train.extend(train_indices)
            val.extend(val_indices)
        return train, val

    def __len__(self):
        return np.sum([len(d) for d in self.datasets], dtype=int)

    def __getitem__(self, idx):
        dataset_idx = idx
        i = 0
        from_set = self.datasets[i]
        while dataset_idx >= len(from_set):
            i += 1
            dataset_idx -= len(from_set)
            from_set = self.datasets[i]
        feature, target, sr, audio_file, _ = from_set[dataset_idx]
        # 0-pad target array if more than 1 dataset
        if len(self.datasets) > 1:
            all_targets = np.zeros(shape=(len(self.classes), target.shape[1]))
            datasets_before = self.datasets[:i]
            pad_indices = np.sum([len(d.classes) for d in datasets_before], dtype=int)
            all_targets[pad_indices:(pad_indices + target.shape[0])] = target
            target = all_targets
        return feature, target, sr, audio_file, dataset_idx


class ExcerptDataset(Dataset):
    def __init__(self, dataset: Dataset, feature_type: str, classes: list, excerpt_size: int, fft_params: dict,
                 overlap_factor: int = 1, rnd_augment: bool = False):
        self.dataset = dataset
        self.feature_type = feature_type
        self.excerpt_size = excerpt_size
        self.classes = classes
        self.fft_params = fft_params
        self.overlap_factor = overlap_factor
        self.rnd_augment = rnd_augment

        self.feature_excerpts = None
        self.excerpt_targets = None
        self.audio_files = None
        self.excerpt_count = None

        self.excerpt_queue = Queue(maxsize=1)
        threading.Thread(target=self.excerpt_producer_thread).start()
        self.generate_excerpts()

    def generate_excerpts(self):
        feature_excerpts, excerpt_targets, audio_files, excerpt_count = self.excerpt_queue.get(block=True)
        self.feature_excerpts = feature_excerpts
        self.excerpt_targets = excerpt_targets
        self.audio_files = audio_files
        self.excerpt_count = excerpt_count
        threading.Thread(target=self.excerpt_producer_thread).start()

    def excerpt_producer_thread(self):
        excerpt_tuple = self.get_excerpts()
        self.excerpt_queue.put(excerpt_tuple, block=True)

    def get_excerpts(self):
        excerpt_count = 0
        feature_excerpts = []
        excerpt_targets = []
        audio_files = []
        hop = self.excerpt_size // self.overlap_factor
        for idx, data in enumerate(self.dataset):
            feature, targets, sr, audio_file, _ = data
            if self.rnd_augment:
                feature = augment.apply_random_stretching(feature)
            if self.feature_type in ['mels', 'mfccs']:
                mel_basis = librosa.filters.mel(sr=sr, n_fft=self.fft_params['n_fft'])
                feature = np.dot(mel_basis, feature)
            feature = librosa.amplitude_to_db(feature, ref=np.max)
            if self.feature_type == 'mfccs':
                feature = librosa.feature.mfcc(S=feature, n_mfcc=self.fft_params['n_mfcc'])
            feature_count = feature.shape[0]
            sequence_positions = feature.shape[1]
            n_excerpts = int(np.ceil(sequence_positions / hop))
            for i in range(n_excerpts):
                begin_idx = i * hop
                end_idx = min(begin_idx + self.excerpt_size, sequence_positions)
                feature_excerpt = np.zeros(shape=(feature_count, self.excerpt_size))
                target_excerpt = np.zeros(shape=(len(self.classes), self.excerpt_size))
                feature_excerpt[:, 0:end_idx - begin_idx] = feature[:, begin_idx:end_idx]
                target_excerpt[:, 0:end_idx - begin_idx] = targets[:, begin_idx:end_idx]
                feature_excerpts.append(feature_excerpt)
                excerpt_targets.append(target_excerpt)
                audio_files.append(audio_file)
            excerpt_count += n_excerpts
        return feature_excerpts, excerpt_targets, audio_files, excerpt_count

    def __len__(self):
        return self.excerpt_count

    def __getitem__(self, idx):
        excerpt = self.feature_excerpts[idx]
        if self.rnd_augment and self.feature_type != 'mfccs':
            excerpt = augment.apply_random_noise(excerpt)
            excerpt = augment.apply_random_db_alteration(excerpt)
            excerpt = augment.apply_random_excerpt_filter(excerpt)
        excerpt = augment.apply_normalization(excerpt)
        excerpt = np.expand_dims(excerpt, axis=0)
        return excerpt, self.excerpt_targets[idx], self.audio_files[idx], idx
