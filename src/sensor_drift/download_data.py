#!/usr/bin/env python3
"""Download Gas Sensor Array Drift Dataset from UCI Repository"""

import os
import urllib.request
import zipfile
from pathlib import Path
import requests


def download_dataset():
    """Download and extract the Gas Sensor Array Drift Dataset"""

    print("=" * 70)
    print("GAS SENSOR ARRAY DRIFT DATASET - DOWNLOAD INSTRUCTIONS")
    print("=" * 70)

    # Create data directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 DATASET INFORMATION:")
    print("-" * 40)
    print("Name: Gas Sensor Array Drift Dataset")
    print("Source: UCI Machine Learning Repository")
    print("Size: ~30 MB compressed")
    print("Samples: 13,910 measurements")
    print("Features: 128 (16 sensors × 8 features each)")
    print("Classes: 6 gas types")
    print("Time span: 36 months (10 batches)")

    print("\n📥 DOWNLOAD STEPS:")
    print("-" * 40)
    print("1. Visit the UCI repository page:")
    print("   https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset")
    print("")
    print("2. Click on 'Data Folder' link")
    print("")
    print("3. Download these files:")
    print("   • batch1.dat.zip")
    print("   • batch2.dat.zip")
    print("   • batch3.dat.zip")
    print("   • batch4.dat.zip")
    print("   • batch5.dat.zip")
    print("   • batch6.dat.zip")
    print("   • batch7.dat.zip")
    print("   • batch8.dat.zip")
    print("   • batch9.dat.zip")
    print("   • batch10.dat.zip")
    print("")
    print("4. Extract all .dat files to:")
    print(f"   {data_dir}")

    print("\n📝 DATA FORMAT:")
    print("-" * 40)
    print("Each .dat file contains semicolon-separated values with:")
    print("• Column 1: Gas type (1=Ethanol, 2=Ethylene, etc.)")
    print("• Column 2: Concentration (ppm)")
    print("• Columns 3-130: 128 sensor features")

    print("\n🔬 GAS TYPES:")
    print("-" * 40)
    print("1 = Ethanol")
    print("2 = Ethylene")
    print("3 = Ammonia")
    print("4 = Acetaldehyde")
    print("5 = Acetone")
    print("6 = Toluene")

    print("\n⚠️  IMPORTANT NOTES:")
    print("-" * 40)
    print("• The dataset cannot be automatically downloaded due to UCI's terms")
    print("• Manual download ensures you accept the dataset's citation requirements")
    print("• Please cite the original paper if you use this dataset in publications")

    print("\n📚 CITATION:")
    print("-" * 40)
    print("Vergara, A., et al. (2012). 'Chemical gas sensor drift compensation")
    print("using classifier ensembles.' Sensors and Actuators B: Chemical, 166, 320-329.")

    print("\n✅ AFTER DOWNLOADING:")
    print("-" * 40)
    print("Run the data preprocessing script to load and format the data:")
    print("  python3 src/sensor_drift/preprocess_uci_data.py")
    print("")
    print(f"Current data directory: {data_dir}")
    print(f"Files currently in raw data: {list(data_dir.glob('*.dat'))}")

    # Check if any data files exist
    existing_files = list(data_dir.glob('batch*.dat'))
    if existing_files:
        print(f"\n✅ Found {len(existing_files)} batch files already downloaded!")
        print("Files:", [f.name for f in existing_files])
    else:
        print("\n❌ No batch files found yet. Please download them from UCI.")


def load_uci_batch(file_path: Path) -> tuple:
    """Load a single batch file from UCI dataset

    Args:
        file_path: Path to batch*.dat file

    Returns:
        Tuple of (features, labels, batch_number)
    """
    import numpy as np

    # Read the data file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Each line: gas_type;concentration;sensor1;sensor2;...;sensor128
            parts = line.strip().split(';')
            if len(parts) >= 130:  # gas_type + concentration + 128 sensors
                gas_type = int(parts[0])
                concentration = float(parts[1])
                sensors = [float(x) for x in parts[2:130]]
                data.append([gas_type] + [concentration] + sensors)

    if not data:
        raise ValueError(f"No valid data found in {file_path}")

    # Convert to numpy array
    data = np.array(data)

    # Extract components
    labels = data[:, 0].astype(int)  # Gas type
    concentrations = data[:, 1]      # Concentration (might be useful later)
    features = data[:, 2:]           # 128 sensor features

    # Extract batch number from filename (e.g., "batch1.dat" -> 1)
    batch_num = int(file_path.stem.replace('batch', ''))

    print(f"Loaded {file_path.name}: {len(features)} samples, {features.shape[1]} features")

    return features, labels, batch_num


def load_all_uci_data():
    """Load all UCI batch files and combine into a single dataset

    Returns:
        Combined pandas DataFrame with all batches
    """
    import pandas as pd
    import numpy as np

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"

    # Find all batch files
    batch_files = sorted(data_dir.glob('batch*.dat'))

    if not batch_files:
        raise FileNotFoundError(
            f"No batch files found in {data_dir}\n"
            f"Please download the UCI dataset first by running:\n"
            f"  python3 src/sensor_drift/download_data.py"
        )

    all_data = []

    for batch_file in batch_files:
        features, labels, batch_num = load_uci_batch(batch_file)

        # Create dataframe for this batch
        df_batch = pd.DataFrame(
            features,
            columns=[f'sensor_{i:03d}' for i in range(features.shape[1])]
        )
        df_batch['chemical'] = labels
        df_batch['batch'] = batch_num

        all_data.append(df_batch)

    # Combine all batches
    df = pd.concat(all_data, ignore_index=True)

    # Map chemical IDs to names
    chemical_map = {
        1: 'Ethanol',
        2: 'Ethylene',
        3: 'Ammonia',
        4: 'Acetaldehyde',
        5: 'Acetone',
        6: 'Toluene'
    }
    df['chemical_name'] = df['chemical'].map(chemical_map)

    # Add approximate timestamps (batches are ~3-4 months apart)
    from datetime import datetime, timedelta
    batch_dates = {
        1: datetime(2009, 1, 1),    # Month 1
        2: datetime(2009, 2, 1),    # Month 2
        3: datetime(2009, 3, 1),    # Month 3
        4: datetime(2009, 8, 1),    # Month 8
        5: datetime(2009, 9, 1),    # Month 9
        6: datetime(2009, 10, 1),   # Month 10
        7: datetime(2010, 1, 1),    # Month 13
        8: datetime(2010, 2, 1),    # Month 14
        9: datetime(2010, 5, 1),    # Month 17
        10: datetime(2011, 9, 1),   # Month 21
    }

    df['timestamp'] = df['batch'].map(batch_dates)

    print("\n" + "=" * 50)
    print("UCI DATASET LOADED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Batches: {sorted(df['batch'].unique())}")
    print(f"Chemicals: {sorted(df['chemical_name'].unique())}")
    print(f"Shape: {df.shape}")

    # Save to processed folder
    output_path = project_root / 'data' / 'processed' / 'uci_sensor_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--load':
        # If called with --load, try to load and process the UCI data
        try:
            df = load_all_uci_data()
            print("\nFirst few rows:")
            print(df.head())
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            download_dataset()
    else:
        # Default: show download instructions
        download_dataset()