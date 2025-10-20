"""High-level helpers for the EEG word imagery classification pipeline."""

from .config import PipelineConfig, BandDefinition, DEFAULT_CONFIG
from .data import load_epochs
from .preprocessing import preprocess_epochs
from .features import build_feature_matrix, encode_labels
from .model import build_classifier, evaluate_classifier

__all__ = [
    "PipelineConfig",
    "BandDefinition",
    "DEFAULT_CONFIG",
    "load_epochs",
    "preprocess_epochs",
    "build_feature_matrix",
    "encode_labels",
    "build_classifier",
    "evaluate_classifier",
]
