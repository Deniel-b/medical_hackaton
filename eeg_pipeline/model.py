"""
Machine-learning utilities for classifying EEG word imagery epochs.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import PipelineConfig


def build_classifier(config: PipelineConfig) -> Pipeline:
    """
    Construct a scikit-learn pipeline with scaling and multinomial logistic regression.
    """

    classifier = DecisionTreeClassifier(
        random_state=config.random_state,
        max_depth=config.tree_max_depth,
        min_samples_split=config.tree_min_samples_split,
        min_samples_leaf=config.tree_min_samples_leaf,
        class_weight=config.tree_class_weight,
    )

    steps = []
    # Decision Trees do not require scaling, but we keep the pipeline flexible.
    if config.use_feature_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", classifier))
    return Pipeline(steps=steps)


def evaluate_classifier(
    model: Pipeline,
    X: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[int],
    label_encoder,
) -> Dict[str, np.ndarray]:
    """
    Produce evaluation metrics including a confusion matrix and classification report.
    """

    y_pred = model.predict(X)
    label_names = label_encoder.inverse_transform(labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        digits=3,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "classification_report": report,
        "confusion_matrix": matrix,
        "f1_macro": f1_macro,
    }
