"""
Plot stacked EEG traces for epochs extracted from an EDF file using pyedflib.

Each annotation in the EDF is treated as an epoch. Каналы формируются по
классическому «double banana» монтажу, затем для каждого condition
(описание аннотации) строится отдельная фигура: эпизоды одного condition
склеиваются вдоль времени, а между ними отображаются вертикальные границы.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from pyedflib import EdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Визуализация эпох из EDF без использования mne/json."
    )
    parser.add_argument(
        "edf_path",
        type=Path,
        nargs="?",
        default=None,
        help="Путь к EDF файлу. Если не указан, появится диалог выбора.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Ограничить количество отображаемых эпох.",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Ограничить количество каналов на графике.",
    )
    return parser.parse_args()

DOUBLE_BANANA_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("Fp1", "Fpz"),
    ("Fpz", "Fp2"),
    ("F7", "F3"),
    ("F3", "Cz"),
    ("Cz", "Pz"),
    ("Pz", "O1"),
    ("F8", "F4"),
    ("F4", "Cz"),
    ("Cz", "Pz"),
    ("Pz", "O2"),
    ("T3", "C3"),
    ("C3", "Cz"),
    ("Cz", "C4"),
    ("C4", "T4"),
    ("P3", "Pz"),
    ("Pz", "P4"),
)


def read_epochs(path: Path, max_epochs: int | None) -> tuple[np.ndarray, List[str], List[str], float]:
    if not path.exists():
        raise FileNotFoundError(f"EDF файл не найден: {path.resolve()}")

    reader = EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        freqs = reader.getSampleFrequencies()
        nsamples = reader.getNSamples()

        if len(set(freqs)) != 1:
            raise ValueError("Частоты дискретизации каналов различаются, нужен ресемплинг.")
        sfreq = float(freqs[0])
        max_length = min(nsamples)

        onsets, durations, descriptions = reader.readAnnotations()
        descriptions = [
            desc.decode("utf-8") if isinstance(desc, bytes) else desc for desc in descriptions
        ]

        epochs: List[np.ndarray] = []
        epoch_labels: List[str] = []

        for onset, duration, desc in zip(onsets, durations, descriptions):
            start = max(0, int(round(onset * sfreq)))
            if duration <= 0:
                duration = 1.0
            length = int(round(duration * sfreq))
            stop = start + length
            if start >= max_length or stop > max_length:
                continue

            segments = [
                reader.readSignal(ch_index, start=start, n=length)
                for ch_index in range(reader.signals_in_file)
            ]
            epochs.append(np.stack(segments, axis=0))
            epoch_labels.append(desc)
            if max_epochs and len(epochs) >= max_epochs:
                break

        if not epochs:
            # Если аннотаций нет — берём весь сигнал одной эпохой.
            segments = [
                reader.readSignal(ch_index) for ch_index in range(reader.signals_in_file)
            ]
            epochs = [np.stack(segments, axis=0)]
            epoch_labels = ["full"]

        return np.stack(epochs, axis=0), labels, epoch_labels, sfreq
    finally:
        reader.close()

def apply_bipolar_montage(
    epochs: np.ndarray,
    channel_names: Sequence[str],
    montage: Sequence[Tuple[str, str]],
) -> Tuple[np.ndarray, List[str]]:
    index: Dict[str, int] = {name: idx for idx, name in enumerate(channel_names)}
    bipolar_data = []
    bipolar_names: List[str] = []
    missing_pairs = []

    for left, right in montage:
        if left not in index or right not in index:
            missing_pairs.append(f"{left}-{right}")
            continue
        diff = epochs[:, index[left], :] - epochs[:, index[right], :]
        bipolar_data.append(diff)
        bipolar_names.append(f"{left}-{right}")

    if not bipolar_data:
        raise ValueError("Не удалось создать ни одного биполярного соединения.")

    if missing_pairs:
        print(
            "Предупреждение: отсутствуют каналы для соединений: "
            + ", ".join(missing_pairs)
        )

    stacked = np.stack(bipolar_data, axis=1)  # (epochs, bipolar_channels, samples)
    return stacked, bipolar_names


def flatten_epochs(epochs: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    n_epochs, _, samples = epochs.shape
    epoch_duration = samples / sfreq
    base_time = np.arange(samples) / sfreq

    times = []
    stacked = []
    for idx in range(n_epochs):
        times.append(base_time + idx * epoch_duration)
        stacked.append(epochs[idx].T)
    return np.concatenate(times), np.concatenate(stacked, axis=0)


def plot_stacked(
    times: np.ndarray,
    data: np.ndarray,
    channel_names: Sequence[str],
    epoch_labels: Sequence[str],
    sfreq: float,
    samples_per_epoch: int,
    max_channels: int | None,
    title: str,
) -> None:
    n_samples, n_channels = data.shape
    if max_channels is not None:
        channel_names = channel_names[:max_channels]
        data = data[:, :max_channels]
        n_channels = len(channel_names)

    offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
    stds = data.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    scaled = (data / stds) * 20

    fig, ax = plt.subplots(figsize=(16, max(6, n_channels * 0.3)))
    for idx, (name, offset) in enumerate(zip(channel_names, offsets)):
        ax.plot(times, scaled[:, idx] + offset, color="black", linewidth=0.6)

    epoch_duration = samples_per_epoch / sfreq
    for idx, label in enumerate(epoch_labels):
        center = (idx + 0.5) * epoch_duration
        ax.text(
            center,
            offsets[0] + 20,
            label,
            fontsize=8,
            ha="center",
            va="bottom",
            rotation=90,
        )
        if idx < len(epoch_labels) - 1:
            boundary = (idx + 1) * epoch_duration
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Каналы")
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names)
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()


def main() -> None:
    args = parse_args()
    edf_path = args.edf_path
    if edf_path is None:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Выберите EDF файл",
            filetypes=(("EDF files", "*.edf"), ("All files", "*.*")),
        )
        root.destroy()
        if not file_path:
            print("Файл не выбран, завершение работы.")
            return
        edf_path = Path(file_path)

    epochs, channels, labels, sfreq = read_epochs(edf_path, args.max_epochs)
    epochs, bipolar_names = apply_bipolar_montage(epochs, channels, DOUBLE_BANANA_PAIRS)
    unique_conditions = sorted(set(labels))
    for condition in unique_conditions:
        indices = [idx for idx, label in enumerate(labels) if label == condition]
        if not indices:
            continue
        print(f"Condition {condition}: {len(indices)} эпох(и)")
        subset_epochs = epochs[indices]
        subset_labels = [f"{condition} #{i + 1}" for i in range(len(indices))]
        times, data = flatten_epochs(subset_epochs, sfreq)
        plot_stacked(
            times,
            data,
            bipolar_names,
            subset_labels,
            sfreq,
            subset_epochs.shape[2],
            max_channels=args.max_channels,
            title=f"Condition: {condition}",
        )
    plt.show()


if __name__ == "__main__":
    main()
