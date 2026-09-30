#!/usr/bin/env python3
"""
HDF5 File Combiner Script

This script combines multiple HDF5 files back into a single HDF5 file.
It can handle files created by the split_hdf5.py script or any collection of HDF5 files.

Usage:
    python combine_hdf5.py --input split_files/ --output combined_data.h5
    python combine_hdf5.py --input "part_*.h5" --output combined_data.h5
    python combine_hdf5.py --input file1.h5 file2.h5 file3.h5 --output combined_data.h5
"""

import h5py
import numpy as np
import argparse
import os
import glob
from datetime import datetime
import time

def analyze_hdf5_file(file_path):
    """Analyze the structure of an HDF5 file"""
    print(f"🔍 Analyzing {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} does not exist!")
        return None
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.1f} MB")
    
    with h5py.File(file_path, 'r') as f:
        # List all datasets
        datasets = list(f.keys())
        print(f"📊 Found {len(datasets)} datasets")
        
        if datasets:
            # Analyze first dataset to understand structure
            first_dataset = datasets[0]
            data = f[first_dataset]
            print(f"📈 Sample dataset '{first_dataset}':")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Size: {data.size * data.dtype.itemsize / (1024 * 1024):.1f} MB")
            
            # Show first few dataset names
            print(f"📋 First 5 datasets: {datasets[:5]}")
            if len(datasets) > 5:
                print(f"   ... and {len(datasets) - 5} more")
        
        return {
            'datasets': datasets,
            'file_size_mb': file_size,
            'sample_shape': data.shape if datasets else None,
            'sample_dtype': data.dtype if datasets else None
        }

def get_input_files(input_paths):
    """Get list of all input files to combine"""
    all_files = []
    
    for input_path in input_paths:
        if os.path.isdir(input_path):
            # Directory: find all .h5 files
            pattern = os.path.join(input_path, "*.h5")
            files = glob.glob(pattern)
            all_files.extend(files)
        elif '*' in input_path:
            # Glob pattern
            files = glob.glob(input_path)
            all_files.extend(files)
        else:
            # Single file
            if os.path.exists(input_path):
                all_files.append(input_path)
            else:
                print(f"⚠️  File not found: {input_path}")
    
    # Sort files to ensure consistent ordering
    all_files.sort()
    
    print(f"📁 Found {len(all_files)} files to combine:")
    for i, file in enumerate(all_files[:10]):  # Show first 10
        print(f"   {i+1:2d}. {file}")
    if len(all_files) > 10:
        print(f"   ... and {len(all_files) - 10} more files")
    
    return all_files

def combine_hdf5_files(input_files, output_file):
    """Combine multiple HDF5 files into one"""
    print(f"🔗 Combining {len(input_files)} files into {output_file}...")
    
    if not input_files:
        print("❌ No input files found!")
        return False
    
    # Analyze first file to understand structure
    first_analysis = analyze_hdf5_file(input_files[0])
    if not first_analysis:
        return False
    
    total_datasets = 0
    total_size_mb = 0
    all_dataset_names = set()  # To check for duplicates
    
    # First pass: analyze all files
    print(f"📊 Analyzing all input files...")
    for i, input_file in enumerate(input_files):
        analysis = analyze_hdf5_file(input_file)
        if analysis:
            total_datasets += len(analysis['datasets'])
            total_size_mb += analysis['file_size_mb']
            
            # Check for duplicate dataset names
            for dataset_name in analysis['datasets']:
                if dataset_name in all_dataset_names:
                    print(f"⚠️  WARNING: Duplicate dataset name '{dataset_name}' found!")
                all_dataset_names.add(dataset_name)
    
    print(f"📊 Total datasets to combine: {total_datasets}")
    print(f"📁 Total size: {total_size_mb:.1f} MB")
    
    # Second pass: combine all files
    print(f"🔗 Combining files...")
    with h5py.File(output_file, 'w') as dst:
        for i, input_file in enumerate(input_files):
            print(f"   Processing {input_file} ({i+1}/{len(input_files)})...")
            
            with h5py.File(input_file, 'r') as src:
                for dataset_name in src.keys():
                    print(f"     Copying {dataset_name}...")
                    src.copy(dataset_name, dst)
    
    # Verify the combined file
    print(f"✅ Verifying combined file...")
    final_analysis = analyze_hdf5_file(output_file)
    
    if final_analysis:
        print(f"✅ Successfully combined {len(input_files)} files into {output_file}")
        print(f"📊 Final file contains {len(final_analysis['datasets'])} datasets")
        print(f"📁 Final file size: {final_analysis['file_size_mb']:.1f} MB")
        
        # Verify we got all datasets
        if len(final_analysis['datasets']) == total_datasets:
            print(f"✅ All {total_datasets} datasets successfully combined!")
        else:
            print(f"⚠️  WARNING: Expected {total_datasets} datasets, got {len(final_analysis['datasets'])}")
        
        return True
    else:
        print(f"❌ Failed to create combined file!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Combine multiple HDF5 files into one')
    parser.add_argument('--input', nargs='+', required=True, 
                       help='Input files/directories/patterns (e.g., "split_files/" or "part_*.h5" or file1.h5 file2.h5)')
    parser.add_argument('--output', required=True, help='Output HDF5 file path')
    
    args = parser.parse_args()
    
    print("🚀 HDF5 File Combiner")
    print("=" * 50)
    
    start_time = time.time()
    
    # Get all input files
    input_files = get_input_files(args.input)
    
    if not input_files:
        print("❌ No input files found!")
        return
    
    # Combine the files
    success = combine_hdf5_files(input_files, args.output)
    
    end_time = time.time()
    
    if success:
        print(f"\n✅ Combine completed successfully in {end_time - start_time:.1f} seconds")
        print(f"📁 Output: {args.output}")
    else:
        print(f"\n❌ Combine failed!")

if __name__ == "__main__":
    main() 