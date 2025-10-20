"""
Signal conditioning helpers for EEG epochs.
"""

from __future__ import annotations

from typing import Optional, Sequence

from mne import Epochs
from mne.filter import filter_data, notch_filter

from .config import PipelineConfig


def preprocess_epochs(epochs: Epochs, config: PipelineConfig) -> Epochs:
    """
    Apply filtering, notch removal, baseline correction, and resampling.
    """

    epochs = epochs.copy()

    data = epochs.get_data(copy=True)
    sfreq = epochs.info["sfreq"]

    if config.notch_freqs:
        data = notch_filter(
            data,
            Fs=sfreq,
            freqs=list(config.notch_freqs),
            method="iir",
            iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )

    if config.l_freq is not None or config.h_freq is not None:
        data = filter_data(
            data,
            sfreq=sfreq,
            l_freq=config.l_freq,
            h_freq=config.h_freq,
            method="iir",
            iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )

    epochs._data[:] = data

    baseline = _validate_baseline(epochs, config.baseline)
    if baseline:
        epochs.apply_baseline(baseline)

    if config.resample_sfreq:
        epochs.resample(config.resample_sfreq)

    return epochs


def _validate_baseline(
    epochs: Epochs, baseline: Optional[Sequence[Optional[float]]]
):
    if not baseline:
        return None
    if len(epochs.times) < 2:
        return None

    tmin, tmax = baseline
    if tmin is None and tmax is None:
        return None

    sfreq = epochs.info["sfreq"]
    eps = 1.0 / sfreq

    if tmin is None:
        tmin = epochs.times[0]
    if tmax is None:
        tmax = epochs.times[-1]

    if tmin == tmax:
        tmax = tmin + eps

    mask = (epochs.times >= tmin) & (epochs.times <= tmax)
    if mask.sum() < 2:
        return None

    return (tmin, tmax)
