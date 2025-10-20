"""
Helpers for converting EDF recordings with annotations into MNE epochs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import mne
import pandas as pd

from .config import PipelineConfig


def load_epochs_from_edf(config: PipelineConfig) -> mne.Epochs:
    """
    Read an EDF recording and convert annotated segments into epochs.
    """

    edf_path = Path(config.edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"Cannot locate EDF file: {edf_path.resolve()}")

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False,
    )

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=config.edf_event_id_map,
        verbose=False,
    )
    if events.size == 0:
        raise ValueError(
            "No events extracted from EDF annotations. "
            "Provide `edf_event_id_map` or verify annotations."
        )

    inverse_map: Dict[int, str] = {code: desc for desc, code in event_id.items()}
    labels = [inverse_map[event[2]] for event in events]
    metadata = pd.DataFrame({config.label_key: labels})

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
        baseline=None,
        preload=True,
        metadata=metadata,
        verbose=False,
    )
    return epochs
