#!/usr/bin/env python3
"""
Convert raw .dat files to CSV with proper column naming
NO DATA MODIFICATIONS - just conversion from LibSVM format

Dataset: Gas Sensor Array Drift Dataset (Original version WITHOUT concentration)
Format: <class_label> 1:<feature_1> 2:<feature_2> ... 128:<feature_128>

- 16 chemical sensors
- 8 features per sensor = 128 total features
- Feature 1 is ΔR_1 (absolute resistance change for sensor 1), NOT concentration
- Large values (15000+) are normal for ΔR (resistance in Ohms)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Gas type mapping
GAS_TYPES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene"
}

# Feature types for each of the 8 features (per UCI documentation)
FEATURE_TYPES = [
    "DR",           # ΔR - Absolute resistance change (Ohms)
    "DR_norm",      # |ΔR| - Normalized resistance change (dimensionless)
    "EMAi_001",     # EMAi0.001 - Rising EMA α=0.001
    "EMAi_01",      # EMAi0.01 - Rising EMA α=0.01
    "EMAi_1",       # EMAi0.1 - Rising EMA α=0.1
    "EMAd_001",     # EMAd0.001 - Decay EMA α=0.001
    "EMAd_01",      # EMAd0.01 - Decay EMA α=0.01
    "EMAd_1"        # EMAd0.1 - Decay EMA α=0.1
]

def process_dat_files():
    """Convert .dat files to CSV - no data modifications"""

    base_path = Path("/Users/thomasfey-grytnes/Documents/Artificial Intelligence - Studying/sensor-drift-pca")
    raw_path = base_path / "data" / "raw"
    processed_path = base_path / "data" / "processed"

    print("="*70)
    print(" CONVERTING .DAT FILES TO CSV (NO MODIFICATIONS)")
    print("="*70)

    # Process all batch files
    all_data = []

    for batch_file in sorted(raw_path.glob("batch*.dat")):
        batch_num = int(re.search(r'batch(\d+)', batch_file.name).group(1))
        print(f"\nProcessing {batch_file.name} (batch {batch_num})...")

        with open(batch_file, 'r') as f:
            for line in f:
                parts = line.strip().split()

                # First element is gas type
                gas_type = int(parts[0])

                # Parse feature:value pairs
                features = {}
                for part in parts[1:]:
                    if ':' in part:
                        idx, val = part.split(':')
                        features[int(idx)] = float(val)

                # Create row with metadata
                row = {
                    'gas_type': gas_type,
                    'gas_name': GAS_TYPES.get(gas_type, f"Unknown_{gas_type}"),
                    'batch': batch_num
                }

                # Add all 128 sensor features
                # Feature 1 is ΔR_1 (NOT concentration!)
                for i in range(1, 129):
                    # Calculate sensor and feature numbers
                    sensor_num = (i - 1) // 8 + 1
                    feature_num = (i - 1) % 8 + 1
                    feature_type = FEATURE_TYPES[feature_num - 1]

                    # Create column name
                    col_name = f"S{sensor_num:02d}_F{feature_num}_{feature_type}"
                    row[col_name] = features.get(i, np.nan)

                all_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    print(f"\nTotal samples loaded: {len(df)}")

    # Report info only - NO MODIFICATIONS
    print("\nDataset Info:")

    sensor_cols = [col for col in df.columns if col.startswith('S')]
    print(f"  Total samples: {len(df)}")
    print(f"  Sensor columns: {len(sensor_cols)}")
    print(f"  Batches: {sorted(df['batch'].unique())}")
    print(f"  Gas types: {sorted(df['gas_name'].unique())}")

    # Report if S01_F1_DR has large values (expected for ΔR)
    if 'S01_F1_DR' in df.columns:
        dr_values = df['S01_F1_DR']
        print(f"\n  S01_F1_DR (ΔR_1) range: [{dr_values.min():.1f}, {dr_values.max():.1f}] Ohms")
        print(f"  Note: Large values are normal for absolute resistance change")

    # Reorder columns
    metadata_cols = ['gas_type', 'gas_name', 'batch']
    df = df[metadata_cols + sensor_cols]

    # Save to CSV
    output_path = processed_path / "sensor_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   NO DATA MODIFICATIONS APPLIED")
    print(f"   Columns: 3 metadata (gas_type, gas_name, batch) + 128 sensor features")
    print(f"   Column naming: S{{01-16}}_F{{1-8}}_{{type}}")
    print(f"   NO concentration column (not in original dataset)")

    return df

if __name__ == "__main__":
    df = process_dat_files()