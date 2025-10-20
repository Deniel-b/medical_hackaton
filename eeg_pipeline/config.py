"""
Configuration objects used across the EEG word imagery classification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class BandDefinition:
    """Describes a frequency band."""

    name: str
    fmin: float
    fmax: float


@dataclass
class PipelineConfig:
    """Holds filesystem paths and processing hyperparameters."""

    fif_path: Union[str, Path] = "sub-1-epo.fif"
    edf_path: Union[str, Path] = "sub-1-ru.edf"
    eeg_channels: Optional[Sequence[str]] = None
    reference: Union[str, Sequence[str], None] = "average"
    l_freq: Optional[float] = 1.0
    h_freq: Optional[float] = 40.0
    notch_freqs: Optional[Iterable[float]] = field(
        default_factory=lambda: (50.0,)
    )
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = (None, 0.0)
    resample_sfreq: Optional[float] = None
    crop_time: Optional[Tuple[Optional[float], Optional[float]]] = None
    band_definitions: Sequence[BandDefinition] = field(
        default_factory=lambda: (
            BandDefinition("delta", 1.0, 4.0),
            BandDefinition("theta", 4.0, 8.0),
            BandDefinition("alpha", 8.0, 13.0),
            BandDefinition("beta", 13.0, 30.0),
            BandDefinition("gamma", 30.0, 45.0),
        )
    )
    psd_n_fft: Optional[int] = None
    psd_n_overlap: Optional[int] = None
    test_size: float = 0.2
    random_state: int = 42
    max_epochs: Optional[int] = None
    n_jobs: int = 1
    label_key: Optional[str] = "word"  # Metadata column storing the target word
    export_model_path: Optional[Union[str, Path]] = "trained_word_classifier.joblib"
    use_feature_scaling: bool = False
    tree_max_depth: Optional[int] = None
    tree_min_samples_split: int = 2
    tree_min_samples_leaf: int = 1
    tree_class_weight: Optional[Dict[str, float]] = None
    epoch_tmin: float = 0.0
    epoch_tmax: float = 1.0
    edf_event_id_map: Optional[Dict[str, int]] = None


DEFAULT_CONFIG = PipelineConfig(
    eeg_channels=[
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "Fc3",
        "Fc4",
        "F7",
        "F8",
        "Ft7",
        "Ft8",
        "T3",
        "T4",
        "Tp7",
        "Tp8",
        "T5",
        "T6",
        "C3",
        "C4",
        "Cp3",
        "Cp4",
        "P3",
        "P4",
        "Po3",
        "Po4",
        "P5",
        "P6",
        "Po7",
        "Po8",
        "O1",
        "O2",
        "Fpz",
        "Fz",
        "Fcz",
        "Cz",
        "Cpz",
        "Pz",
        "Poz",
        "Oz",
        "A1",
        "A2",
    ],
    notch_freqs=(50.0,),
)
