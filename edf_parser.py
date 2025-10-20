from __future__ import annotations

import sys
from pathlib import Path

from pyedflib import EdfReader

DEFAULT_EDF_PATH = Path("russianDataSet/sub-1/sub-1-ru.edf")


def main(path: str | None = None) -> None:
    edf_path = Path(path) if path else DEFAULT_EDF_PATH
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    reader = EdfReader(str(edf_path))
    try:
        signal_count = reader.signals_in_file
        labels = reader.getSignalLabels()
        sample_rates = reader.getSampleFrequencies()

        print(f"EDF path: {edf_path}")
        print(f"Signals in file: {signal_count}")
        print("Signal labels:", labels)
        print("Sample rates (Hz):", sample_rates)

        for idx in range(signal_count):
            signal = reader.readSignal(idx)
            print(f"\nSignal {idx} - {labels[idx]} (samples: {len(signal)}):")
            print(signal.tolist())
    finally:
        reader.close()


if __name__ == "__main__":
    user_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(user_path)
