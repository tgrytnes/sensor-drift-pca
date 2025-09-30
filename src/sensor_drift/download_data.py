#!/usr/bin/env python3
"""Setup for Gas Sensor Array Drift Dataset"""

from pathlib import Path


def setup_data_dir():
    """Create data directory and check for existing files"""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = list(data_dir.glob('batch*.dat'))
    if existing:
        print(f"Found {len(existing)} batch files")
    else:
        print(f"Download batch1-10.dat.zip from UCI ML Repository to: {data_dir}")
        print("URL: https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset")

    return data_dir, existing


if __name__ == "__main__":
    setup_data_dir()