"""
Utility script to inspect basic metadata of an EDF recording.
"""

from __future__ import annotations

from pathlib import Path

import pyedflib


def inspect_edf(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    reader = pyedflib.EdfReader(str(path))
    try:
        n_signals = reader.signals_in_file
        print(f"Signals in EDF: {n_signals}")
        print(f"Recording duration (s): {reader.file_duration}")
        print(f"Signal labels: {reader.getSignalLabels()}")
        print(f"Sample frequencies: {reader.getSampleFrequencies()}")
        print(f"Start date/time: {reader.getStartdatetime()}")
    finally:
        reader.close()


if __name__ == "__main__":
    inspect_edf(Path("sub-1-ru.edf"))
