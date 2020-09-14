# -*- coding: utf-8 -*-
import random

import numpy as np
import scipy.ndimage

random.seed(0)


def apply_random_stretching(spectrogram: np.ndarray, factor: float = 0.3) -> np.ndarray:
    stretch = 1 + random.random() * factor
    factor *= 1 if random.random() >= 0.5 else -1  # invert on half the cases
    shift = 1  # + random.random() * factor
    spectrogram = scipy.ndimage.affine_transform(spectrogram, np.array((1 / stretch, 1 / shift)))
    return spectrogram


def apply_random_noise(spectrogram: np.ndarray, mean: float = 0.0, var: float = 1.0) -> np.ndarray:
    noise = np.random.normal(mean, var, spectrogram.shape)
    spectrogram += noise
    return spectrogram


def apply_random_db_alteration(spectrogram: np.ndarray, max_decibel: float = 30) -> np.ndarray:
    constant_factor = random.random() * max_decibel
    constant_factor *= 1 if random.random() >= 0.5 else -1  # invert on half the cases
    spectrogram += constant_factor
    return spectrogram


def apply_random_excerpt_filter(spectrogram: np.ndarray, max_decibel: int = 10) -> np.ndarray:
    constant_factor = - random.random() * max_decibel
    excerpt_len = spectrogram.shape[0] // 6
    lower_bound = excerpt_len + int(random.random() * (spectrogram.shape[0] - 2 * excerpt_len))
    upper_bound = lower_bound + excerpt_len
    spectrogram[lower_bound:upper_bound, ...] += constant_factor
    return spectrogram


def apply_normalization(feature: np.ndarray) -> np.ndarray:
    mean = feature.mean()
    std = feature.std()
    if std == 0:
        std = 1  # in case std equals zero, do not divide by zero
    feature -= mean
    feature /= std
    return feature


def apply_random_augmentations(spectrogram: np.ndarray) -> np.ndarray:
    spectrogram = apply_random_stretching(spectrogram)
    spectrogram = apply_random_noise(spectrogram)
    spectrogram = apply_random_db_alteration(spectrogram)
    spectrogram = apply_random_excerpt_filter(spectrogram)
    return spectrogram


if __name__ == '__main__':
    import librosa
    import librosa.display
    from matplotlib import pyplot as plt


    def show_spectrogram(s, augmentation=""):
        plt.figure()
        librosa.display.specshow(s, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'spectrogram {" - " + augmentation if augmentation else ""}')
        plt.show()


    file = 'data/dev/audio/residential_area/a001.wav'
    audio, sr = librosa.load(file)
    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=1024)
    S = np.dot(mel_basis, S)
    S = librosa.amplitude_to_db(S)
    show_spectrogram(S, 'original')
    show_spectrogram(apply_random_noise(S.copy()), 'noise1')
    show_spectrogram(apply_random_noise(S.copy()), 'noise2')
    show_spectrogram(apply_random_stretching(S.copy()), 'streched1')
    show_spectrogram(apply_random_stretching(S.copy()), 'streched2')
    show_spectrogram(apply_random_db_alteration(S.copy()), 'decibel1')
    show_spectrogram(apply_random_db_alteration(S.copy()), 'decibel2')
    show_spectrogram(apply_random_excerpt_filter(S.copy()), 'filter1')
    show_spectrogram(apply_random_excerpt_filter(S.copy()), 'filter2')
    show_spectrogram(apply_random_augmentations(S.copy()), 'rand_augs1')
    show_spectrogram(apply_random_augmentations(S.copy()), 'rand_augs2')
    show_spectrogram(apply_random_augmentations(S.copy()), 'rand_augs3')
    show_spectrogram(apply_random_augmentations(S.copy()), 'rand_augs4')
