#!/usr/bin/env python3
"""
Preprocess HDF5 files to add mask columns permanently.
This script reads all HDF5 files in a directory, adds mask columns for NaN values,
performs NaN imputation and outlier capping, then saves the processed data
back to new HDF5 files. This eliminates the need to process data every time
it's loaded.

USAGE:
python preprocess_h5.py --input /path/to/folder/with/h5/files
python preprocess_h5.py --input /path/to/folder/with/h5/files --output /path/to/output/folder

ARGUMENTS:
--input, -i    : Directory containing HDF5 files to process (required)
--output, -o   : Output directory for processed files (optional, defaults to input directory)

WHAT IT DOES:
- Processes ALL .h5 files in the specified directory
- Adds mask columns for NaN values in 'pe' and 'turnover_rate' columns
- Imputes NaN values with 0.0
- Caps outliers using 1st and 99th percentiles
- Saves processed data to new HDF5 files with '_processed' suffix
"""

import h5py
import numpy as np
import os
import argparse
from pathlib import Path

def preprocess_h5_file(input_path, output_path=None):
    """
    Preprocess HDF5 file by adding mask columns and performing data cleaning.
    
    Args:
        input_path: Path to the input HDF5 file
        output_path: Path to the output HDF5 file (if None, will append '_processed')
    """
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_processed{input_file.suffix}")
    
    print(f"Processing {input_path} -> {output_path}")
    
    with h5py.File(input_path, 'r') as h5_in, h5py.File(output_path, 'w') as h5_out:
        # Get all dataset names
        dataset_names = list(h5_in.keys())
        print(f"Found {len(dataset_names)} datasets to process")
        
        for name in dataset_names:
            print(f"Processing dataset: {name}")
            
            # Load the data
            arr = np.array(h5_in[name])
            if not hasattr(arr, 'shape'):
                arr = np.array(arr)
            
            print(f"  Original shape: {arr.shape}")
            
            # Check if data already has mask columns
            if arr.shape[1] == 13:
                print(f"  Dataset {name} already has mask columns, skipping...")
                # Copy as-is
                h5_out.create_dataset(name, data=arr)
                continue
            elif arr.shape[1] != 11:
                print(f"  Warning: Unexpected shape {arr.shape} for {name}")
                continue
            
            # Add mask columns for turnover_rate and pe
            # Based on the preprocessing structure from preprocess.py:
            # Column 5: pe, Column 6: turnover_rate
            pe_idx = 5
            turnover_idx = 6
            
            print(f"  Creating masks for pe (col {pe_idx}) and turnover_rate (col {turnover_idx})")
            
            # Create masks (1 if NaN, 0 if present) - BEFORE imputation
            pe_nan_mask = np.isnan(arr[:, pe_idx]).astype(np.float32)
            turnover_nan_mask = np.isnan(arr[:, turnover_idx]).astype(np.float32)
            
            # Debug: Show how many NaNs we found
            pe_nan_count = np.sum(pe_nan_mask)
            turnover_nan_count = np.sum(turnover_nan_mask)
            total_rows = arr.shape[0]
            print(f"    PE NaNs: {pe_nan_count}/{total_rows} ({pe_nan_count/total_rows*100:.1f}%)")
            print(f"    Turnover NaNs: {turnover_nan_count}/{total_rows} ({turnover_nan_count/total_rows*100:.1f}%)")
            
            # Impute NaNs in turnover_rate and pe with 0.0
            arr[:, pe_idx] = np.nan_to_num(arr[:, pe_idx], nan=0.0)
            arr[:, turnover_idx] = np.nan_to_num(arr[:, turnover_idx], nan=0.0)
            
            # Add masks as new features (columns 11 and 12)
            # Column 11: turnover_rate_mask, Column 12: pe_nan_mask
            arr = np.concatenate([arr, turnover_nan_mask[:, None], pe_nan_mask[:, None]], axis=1)
            print(f"  After adding masks: {arr.shape}")
            print(f"    Column 11 (turnover_rate_mask): {np.sum(arr[:, 11])}/{total_rows} NaN indicators")
            print(f"    Column 12 (pe_nan_mask): {np.sum(arr[:, 12])}/{total_rows} NaN indicators")
            
            # Remove any remaining NaNs
            if np.isnan(arr).any():
                nan_count = np.isnan(arr).sum()
                print(f"  Removing {nan_count} remaining NaNs")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Cap outliers (1st and 99th percentiles)
            for col_idx in range(arr.shape[1]):
                col_data = arr[:, col_idx]
                q1, q99 = np.percentile(col_data, [1, 99])
                arr[:, col_idx] = np.clip(col_data, q1, q99)
            
            # Save processed data
            h5_out.create_dataset(name, data=arr)
            print(f"  Saved processed data for {name}")
    
    print(f"Preprocessing complete! Output saved to: {output_path}")
    return output_path

def process_directory(input_dir, output_dir=None):
    """
    Process all HDF5 files in a directory.
    
    Args:
        input_dir: Directory containing HDF5 files to process
        output_dir: Directory to save processed files (if None, saves in input_dir with '_processed' suffix)
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .h5 files in the input directory
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(h5_files)} HDF5 files to process")
    
    processed_files = []
    failed_files = []
    
    for i, filename in enumerate(h5_files, 1):
        print(f"\n[{i}/{len(h5_files)}] Processing {filename}...")
        
        input_path = os.path.join(input_dir, filename)
        
        # Create output filename
        if filename.endswith('_processed.h5'):
            print(f"  Skipping {filename} (already processed)")
            continue
            
        base_name = filename.replace('.h5', '')
        output_filename = f"{base_name}_processed.h5"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            preprocess_h5_file(input_path, output_path)
            processed_files.append(output_filename)
            print(f"  ✅ Successfully processed: {output_filename}")
        except Exception as e:
            print(f"  ❌ Failed to process {filename}: {e}")
            failed_files.append(filename)
    
    print(f"\n📊 Processing Summary:")
    print(f"  ✅ Successfully processed: {len(processed_files)} files")
    print(f"  ❌ Failed: {len(failed_files)} files")
    
    if processed_files:
        print(f"\n📁 Processed files saved in: {output_dir}")
        for f in processed_files[:5]:  # Show first 5
            print(f"  • {f}")
        if len(processed_files) > 5:
            print(f"  • ... and {len(processed_files) - 5} more")
    
    if failed_files:
        print(f"\n⚠️  Failed files:")
        for f in failed_files:
            print(f"  • {f}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess HDF5 files to add mask columns')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing HDF5 files to process')
    parser.add_argument('--output', '-o', help='Output directory for processed files (optional, defaults to input directory)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return
    
    if not os.path.isdir(args.input):
        print(f"Error: Input path '{args.input}' is not a directory")
        print("Please provide a directory containing HDF5 files")
        return
    
    try:
        # Process directory - this is now the default behavior
        print(f"Processing directory: {args.input}")
        process_directory(args.input, args.output)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

if __name__ == "__main__":
    main() 