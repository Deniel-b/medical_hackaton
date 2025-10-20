"""
Visualise EEG epochs stored in an Excel spreadsheet as stacked traces.

Each channel is plotted with a vertical offset, and epoch boundaries are
marked with vertical lines so that each column corresponds to one epoch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SAMPLE_RATE = 500.0  # Hz, adjust via CLI if needed
DEFAULT_IGNORE_COLUMNS = {"Unnamed: 0", "condition", "epoch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked EEG traces from an Excel file."
    )
    parser.add_argument(
        "excel_path",
        type=Path,
        help="Path to the Excel file containing epochs.",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Sampling frequency in Hz (default {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=tuple(DEFAULT_IGNORE_COLUMNS),
        help="Column names to ignore (non-signal columns).",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Limit the number of channels plotted (for debugging).",
    )
    parser.add_argument(
        "--epoch-order",
        choices=["auto", "sorted"],
        default="auto",
        help="How to order epochs: 'auto' keeps file order, 'sorted' sorts by epoch id.",
    )
    return parser.parse_args()


def load_excel(
    path: Path, ignore_columns: Iterable[str], order: str
) -> tuple[pd.DataFrame, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path.resolve()}")

    df = pd.read_excel(path)
    ignore = set(ignore_columns)
    required = {"epoch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel file is missing required columns: {missing}")

    if order == "sorted":
        df = df.sort_values(["epoch"] + [col for col in df.columns if col not in ignore])
    else:
        df = df.sort_values(["epoch", df.columns[0]])

    channel_columns = [col for col in df.columns if col not in ignore]
    if not channel_columns:
        raise ValueError("No channel columns detected after ignoring metadata.")

    return df.reset_index(drop=True), channel_columns


def prepare_data(
    df: pd.DataFrame,
    channel_columns: Sequence[str],
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    epochs = df["epoch"].to_numpy()
    unique_epochs, counts = np.unique(epochs, return_counts=True)
    if not np.all(counts == counts[0]):
        raise ValueError("Epochs have varying numbers of samples; cannot align them.")

    samples_per_epoch = counts[0]
    epoch_duration = samples_per_epoch / sfreq
    total_samples = len(df)

    time_axis = np.arange(total_samples) / sfreq
    data = df[channel_columns].to_numpy(dtype=float)

    return time_axis, data, unique_epochs, epoch_duration


def plot_epochs(
    time_axis: np.ndarray,
    data: np.ndarray,
    channel_names: Sequence[str],
    epoch_ids: np.ndarray,
    epoch_duration: float,
    limit: int | None = None,
):
    if limit is not None:
        data = data[:, :limit]
        channel_names = channel_names[:limit]

    n_channels = data.shape[1]
    offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
    # Normalise each channel to similar amplitude
    stds = data.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    scaled = (data / stds) * 20  # scale factor for readability

    fig, ax = plt.subplots(figsize=(16, max(6, n_channels * 0.3)))
    for idx, (channel, offset) in enumerate(zip(channel_names, offsets)):
        trace = scaled[:, idx] + offset
        ax.plot(time_axis, trace, linewidth=0.6, color="black")

    for epoch_idx in range(1, len(epoch_ids)):
        boundary = epoch_idx * epoch_duration
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.set_yticks(offsets)
    ax.set_yticklabels(list(channel_names))
    ax.set_title("EEG epochs stacked by channel (each column = epoch)")
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    df, channels = load_excel(args.excel_path, args.ignore, args.epoch_order)
    time_axis, data, epoch_ids, epoch_duration = prepare_data(df, channels, args.sfreq)
    plot_epochs(time_axis, data, channels, epoch_ids, epoch_duration, args.max_channels)


if __name__ == "__main__":
    main()

