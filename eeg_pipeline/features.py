"""
Feature extraction utilities for the EEG word imagery project.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.signal import welch
import mne
from sklearn.preprocessing import LabelEncoder

from .config import BandDefinition, PipelineConfig


def build_feature_matrix(
    epochs: mne.Epochs, config: PipelineConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute band-power features for each epoch and channel.

    Returns:
        features: shape (n_epochs, n_channels * n_bands * 2)
        feature_names: labels for each feature column.
    """

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    n_epochs, n_channels, _ = data.shape

    band_definitions = list(config.band_definitions)
    if not band_definitions:
        raise ValueError("Band definitions are required for feature extraction.")

    psd = _compute_welch_psd(
        data,
        sfreq,
        n_fft=config.psd_n_fft,
        n_overlap=config.psd_n_overlap,
    )  # shape: (n_epochs, n_channels, n_freqs)
    freqs = psd["freqs"]
    power = psd["psd"]

    absolute, names = _bandpowers(power, freqs, epochs.ch_names, band_definitions)
    relative = _relative_bandpowers(absolute, len(band_definitions), n_channels)

    features = np.concatenate([absolute, relative], axis=1)
    feature_names = names + [f"{name}_rel" for name in names]
    return features, feature_names


def encode_labels(
    epochs: mne.Epochs, config: PipelineConfig
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Transform target word labels into integers and return a scikit-learn encoder.
    """

    if epochs.metadata is not None and config.label_key in epochs.metadata:
        labels = epochs.metadata[config.label_key].to_numpy()
    else:
        id_to_name = {code: name for name, code in epochs.event_id.items()}
        labels = np.array(
            [id_to_name.get(code, str(code)) for code in epochs.events[:, 2]]
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y, encoder


def _compute_welch_psd(
    data: np.ndarray,
    sfreq: float,
    n_fft: int | None = None,
    n_overlap: int | None = None,
) -> dict:
    """Compute Welch PSD for each epoch and channel."""

    n_epochs, n_channels, n_times = data.shape

    if n_fft is None:
        n_fft = min(256, n_times)
    if n_overlap is None:
        n_overlap = n_fft // 2

    psd_list = []
    for epoch in data:
        epoch_psd = []
        for channel in epoch:
            freq, pxx = welch(
                channel,
                fs=sfreq,
                nperseg=n_fft,
                noverlap=n_overlap,
                scaling="density",
            )
            epoch_psd.append(pxx)
        psd_list.append(epoch_psd)

    psd_array = np.array(psd_list)  # shape (n_epochs, n_channels, n_freqs)
    return {"freqs": freq, "psd": psd_array}


def _bandpowers(
    psd: np.ndarray,
    freqs: np.ndarray,
    ch_names: Sequence[str],
    band_definitions: Sequence[BandDefinition],
) -> Tuple[np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    feature_names: List[str] = []

    for band in band_definitions:
        idx = np.logical_and(freqs >= band.fmin, freqs < band.fmax)
        if not np.any(idx):
            raise ValueError(f"No frequency bins fall inside band {band.name}")
        power = psd[:, :, idx].mean(axis=-1)
        features.append(power)
        feature_names.extend(f"{ch}_{band.name}" for ch in ch_names)

    concatenated = np.concatenate(features, axis=1)
    return concatenated, feature_names


def _relative_bandpowers(
    absolute: np.ndarray, n_bands: int, n_channels: int
) -> np.ndarray:
    if absolute.size == 0:
        return absolute

    reshaped = absolute.reshape(absolute.shape[0], n_bands, n_channels)
    total_power = reshaped.sum(axis=1, keepdims=True)
    total_power[total_power == 0.0] = 1.0
    relative = reshaped / total_power
    return relative.reshape(absolute.shape)
