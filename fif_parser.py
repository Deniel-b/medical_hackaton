from __future__ import annotations

import sys
from pathlib import Path

import mne

DEFAULT_FIF_PATH = Path("russianDataSet/sub-1/sub-1-epo-ru.fif")


def main(path: str | None = None) -> None:
    fif_path = Path(path) if path else DEFAULT_FIF_PATH
    if not fif_path.exists():
        raise FileNotFoundError(f"FIF file not found: {fif_path}")

    epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    print(f"FIF path: {fif_path}")
    print(f"Epochs shape: {data.shape}")
    print("Channel names:", epochs.ch_names)

    if epochs.metadata is not None:
        print("Metadata:")
        print(epochs.metadata)

    for idx, epoch in enumerate(data):
        print(f"\nEpoch {idx}:")
        print(epoch.tolist())


if __name__ == "__main__":
    user_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(user_path)
