"""
End-to-end training script for EEG word imagery classification.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for training. Install it with 'pip install scikit-learn'."
    ) from exc

from eeg_pipeline import (
    DEFAULT_CONFIG,
    PipelineConfig,
    build_classifier,
    build_feature_matrix,
    encode_labels,
    evaluate_classifier,
    load_epochs,
    preprocess_epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classifier for imagined word EEG patterns."
    )
    parser.add_argument(
        "--fif-path",
        type=Path,
        default=None,
        help="Path to the epochs FIF file (defaults to config).",
    )
    parser.add_argument(
        "--export-model",
        type=Path,
        default=None,
        help="Where to store the trained model (joblib).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the trained model to disk.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for model assessment.",
    )
    parser.add_argument(
        "--edf-path",
        type=Path,
        default=None,
        help="Path to an EDF recording (with annotations) to build epochs from.",
    )
    parser.add_argument(
        "--epoch-tmin",
        type=float,
        default=None,
        help="Start time (s) for each epoch relative to annotation onset.",
    )
    parser.add_argument(
        "--epoch-tmax",
        type=float,
        default=None,
        help="End time (s) for each epoch relative to annotation onset.",
    )
    parser.add_argument(
        "--event-map",
        type=str,
        default=None,
        help=(
            "JSON mapping from annotation labels to event IDs "
            "(e.g. '{\"GZ\": 1, \"UY\": 2}')."
        ),
    )
    parser.add_argument(
        "--use-scaling",
        action="store_true",
        help="Apply feature scaling before classification (optional for trees).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _derive_config(args)

    print("Loading epochs...")
    epochs = load_epochs(config)
    print(f"Loaded epochs: {len(epochs)}")

    print("Preprocessing epochs (filters, baseline, resampling)...")
    epochs = preprocess_epochs(epochs, config)

    print("Extracting features...")
    X, feature_names = build_feature_matrix(epochs, config)
    y, encoder = encode_labels(epochs, config)
    print(f"Feature matrix shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    valid_labels = unique[counts >= 2]
    if len(valid_labels) < len(unique):
        dropped = unique[counts < 2]
        dropped_names = encoder.inverse_transform(dropped)
        print(
            "Removing classes with fewer than 2 samples: "
            + ", ".join(dropped_names)
        )
        mask = np.isin(y, valid_labels)
        X = X[mask]
        y = y[mask]
        if len(y) == 0:
            raise SystemExit("No data left after removing sparse classes.")
    unique, counts = np.unique(y, return_counts=True)
    valid_labels = unique
    min_class_count = counts.min()

    if len(valid_labels) < 2:
        raise SystemExit(
            f"At least two classes are required for classification; found {len(valid_labels)}."
        )

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    print("Training classifier...")
    model = build_classifier(config)
    model.fit(X_train, y_train)
    print("Training complete.")

    print("Evaluating on hold-out set...")
    active_labels = np.unique(np.concatenate([y_train, y_test]))
    metrics = evaluate_classifier(model, X_test, y_test, active_labels, encoder)
    print("Classification report:\n", metrics["classification_report"])
    print("Confusion matrix:\n", metrics["confusion_matrix"])
    print(f"Macro F1 score: {metrics['f1_macro']:.3f}")

    if args.cv_folds and args.cv_folds > 1:
        cv_folds = min(args.cv_folds, int(min_class_count))
        if cv_folds < 2:
            print(
                "Skipping cross-validation because the minimum class count "
                f"({min_class_count}) is less than 2."
            )
        else:
            if cv_folds != args.cv_folds:
                print(
                    f"Reducing CV folds from {args.cv_folds} to {cv_folds} to "
                    "respect the smallest class size."
                )
            cv_model = build_classifier(config)
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=config.random_state,
            )
            scores = cross_val_score(cv_model, X, y, cv=cv, n_jobs=config.n_jobs)
            print(
                f"{cv_folds}-fold CV accuracy: "
                f"{scores.mean():.3f} +/- {scores.std():.3f}"
            )

    print("Pipeline complete.")
    if not args.no_save and config.export_model_path and joblib is not None:
        export_path = Path(config.export_model_path).resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "label_encoder": encoder,
                "feature_names": feature_names,
                "config": config,
            },
            export_path,
        )
        print(f"Model saved to {export_path}")
    elif not args.no_save and config.export_model_path:
        print(
            "Joblib is not installed; skipping model export. "
            "Install joblib to enable saving."
        )


def _derive_config(args: argparse.Namespace) -> PipelineConfig:
    config = replace(DEFAULT_CONFIG)
    if args.fif_path is not None:
        config.fif_path = args.fif_path
    if args.edf_path is not None:
        config.edf_path = args.edf_path
    if args.epoch_tmin is not None:
        config.epoch_tmin = args.epoch_tmin
    if args.epoch_tmax is not None:
        config.epoch_tmax = args.epoch_tmax
    if args.event_map is not None:
        try:
            config.edf_event_id_map = {
                str(k): int(v) for k, v in json.loads(args.event_map).items()
            }
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "Failed to parse --event-map JSON mapping. "
                "Use a format like '{\"label\": 1}'."
            ) from exc
    if args.use_scaling:
        config.use_feature_scaling = True
    if args.export_model is not None:
        config.export_model_path = args.export_model
    return config


if __name__ == "__main__":
    main()
