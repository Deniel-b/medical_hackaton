"""
Utilities for loading EEG epochs from EDF or FIF sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mne

from .config import PipelineConfig
from .edf_loader import load_epochs_from_edf


def load_epochs(config: PipelineConfig) -> mne.Epochs:
    """
    Load epochs from the configured FIF file and apply basic channel selection.
    """

    channel_selection: Optional[Sequence[str]] = config.eeg_channels
    edf_path = Path(config.edf_path) if config.edf_path else None

    if edf_path and edf_path.exists():
        epochs = load_epochs_from_edf(config)
    else:
        fif_path = Path(config.fif_path)
        if not fif_path.exists():
            if edf_path:
                raise FileNotFoundError(
                    "Neither EDF nor FIF file could be found. "
                    f"Checked EDF: {edf_path.resolve() if edf_path else 'None'}, "
                    f"FIF: {fif_path.resolve()}."
                )
            raise FileNotFoundError(f"Cannot locate FIF file: {fif_path.resolve()}")

        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

    if config.max_epochs is not None and config.max_epochs < len(epochs):
        epochs = epochs[: config.max_epochs]

    epochs = _pick_channels(epochs, channel_selection)
    epochs = _set_reference(epochs, config.reference)
    epochs = _crop_times(epochs, config.crop_time)
    return epochs


def _pick_channels(epochs: mne.Epochs, channels: Optional[Sequence[str]]) -> mne.Epochs:
    if not channels:
        return epochs
    available = set(epochs.ch_names)
    requested = list(channels)
    included = [ch for ch in requested if ch in available]
    missing = sorted(set(requested) - available)

    if missing:
        missing_str = ", ".join(missing)
        print(
            f"Warning: dropping {len(missing)} requested channels absent from the "
            f"recording: {missing_str}"
        )

    if not included:
        raise ValueError(
            "None of the requested EEG channels were found in the recording."
        )

    picks = mne.pick_channels(epochs.ch_names, include=included)
    return epochs.copy().pick(picks)


def _set_reference(epochs: mne.Epochs, reference) -> mne.Epochs:
    if reference is None or reference == "keep":
        return epochs
    return epochs.copy().set_eeg_reference(reference)


def _crop_times(
    epochs: mne.Epochs, crop_time: Optional[Sequence[Optional[float]]]
) -> mne.Epochs:
    if not crop_time:
        return epochs
    tmin, tmax = crop_time
    return epochs.copy().crop(tmin=tmin, tmax=tmax)
