#!/usr/bin/env python3
"""
HDF5 File Splitter Script

This script can split HDF5 files in several ways:
1. Split into N equal files (test mode) - distributes all data across N files
2. Split by time periods (e.g., split each day into separate files)
3. Split by stocks (e.g., create separate files for each stock)
4. Split into smaller chunks (e.g., create files with fewer days)

Usage:
    python split_hdf5.py --input h5_rty_data.h5 --output split_files/ --mode test --limit 20
    python split_hdf5.py --input h5_rty_data.h5 --output h5_rty_data_split.h5 --mode time --days 10
    python split_hdf5.py --input h5_rty_data.h5 --output h5_rty_data_split.h5 --mode stocks --stocks 5
"""

import h5py
import numpy as np
import argparse
import os
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

def split_by_test_mode(input_file, output_file, limit=5):
    """Create multiple files that together contain all data"""
    print(f"🧪 Creating {limit} files that together contain all data...")
    
    analysis = analyze_hdf5_file(input_file)
    if not analysis:
        return False
    
    datasets = analysis['datasets']
    total_datasets = len(datasets)
    
    if total_datasets == 0:
        print(f"❌ No datasets found in input file")
        return False
    
    # Calculate how many datasets per file
    datasets_per_file = (total_datasets + limit - 1) // limit  # Ceiling division
    
    print(f"📊 Total datasets: {total_datasets}")
    print(f"📁 Creating {limit} files with ~{datasets_per_file} datasets each")
    
    # Create output directory if output_file is a directory
    output_dir = output_file
    if not output_file.endswith('/'):
        output_dir = output_file + '/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Split datasets across files
    with h5py.File(input_file, 'r') as src:
        for file_idx in range(limit):
            # Calculate which datasets go in this file
            start_idx = file_idx * datasets_per_file
            end_idx = min((file_idx + 1) * datasets_per_file, total_datasets)
            
            if start_idx >= total_datasets:
                break
                
            file_datasets = datasets[start_idx:end_idx]
            output_filename = os.path.join(output_dir, f"part_{file_idx+1:02d}.h5")
            
            print(f"   Creating {output_filename} with {len(file_datasets)} datasets...")
            
            with h5py.File(output_filename, 'w') as dst:
                for dataset_name in file_datasets:
                    src.copy(dataset_name, dst)
            
            # Verify the file
            file_analysis = analyze_hdf5_file(output_filename)
            print(f"   ✅ Created {output_filename}: {len(file_datasets)} datasets")
    
    print(f"✅ Created {limit} files in {output_dir}")
    print(f"📊 Total datasets distributed: {total_datasets}")
    return True

def split_by_time_periods(input_file, output_file, days=10):
    """Split by selecting first N days of data"""
    print(f"⏰ Creating time-split version with {days} days...")
    
    analysis = analyze_hdf5_file(input_file)
    if not analysis:
        return False
    
    datasets = analysis['datasets']
    
    # Group datasets by date (assuming format like "STOCK-YYYY-MM-DD")
    date_groups = {}
    for dataset_name in datasets:
        # Extract date from dataset name
        parts = dataset_name.split('-')
        if len(parts) >= 4:
            date_key = '-'.join(parts[-3:])  # YYYY-MM-DD
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(dataset_name)
    
    print(f"📅 Found {len(date_groups)} unique dates")
    
    # Sort dates and take first N days
    sorted_dates = sorted(date_groups.keys())
    selected_dates = sorted_dates[:days]
    
    selected_datasets = []
    for date in selected_dates:
        selected_datasets.extend(date_groups[date])
    
    print(f"📊 Selected {len(selected_datasets)} datasets from {len(selected_dates)} dates")
    
    with h5py.File(input_file, 'r') as src:
        with h5py.File(output_file, 'w') as dst:
            for dataset_name in selected_datasets:
                print(f"   Copying {dataset_name}...")
                src.copy(dataset_name, dst)
    
    # Verify the new file
    new_analysis = analyze_hdf5_file(output_file)
    print(f"✅ Time-split file created: {len(selected_datasets)} datasets from {len(selected_dates)} days")
    return True

def split_by_stocks(input_file, output_file, num_stocks=5):
    """Split by selecting first N stocks"""
    print(f"📈 Creating stock-split version with {num_stocks} stocks...")
    
    analysis = analyze_hdf5_file(input_file)
    if not analysis:
        return False
    
    datasets = analysis['datasets']
    
    # Group datasets by stock (assuming format like "STOCK-YYYY-MM-DD")
    stock_groups = {}
    for dataset_name in datasets:
        # Extract stock name from dataset name
        parts = dataset_name.split('-')
        if len(parts) >= 4:
            stock_name = parts[0]  # First part is stock name
            if stock_name not in stock_groups:
                stock_groups[stock_name] = []
            stock_groups[stock_name].append(dataset_name)
    
    print(f"🏢 Found {len(stock_groups)} unique stocks")
    
    # Sort stocks and take first N stocks
    sorted_stocks = sorted(stock_groups.keys())
    selected_stocks = sorted_stocks[:num_stocks]
    
    selected_datasets = []
    for stock in selected_stocks:
        selected_datasets.extend(stock_groups[stock])
    
    print(f"📊 Selected {len(selected_datasets)} datasets from {len(selected_stocks)} stocks")
    
    with h5py.File(input_file, 'r') as src:
        with h5py.File(output_file, 'w') as dst:
            for dataset_name in selected_datasets:
                print(f"   Copying {dataset_name}...")
                src.copy(dataset_name, dst)
    
    # Verify the new file
    new_analysis = analyze_hdf5_file(output_file)
    print(f"✅ Stock-split file created: {len(selected_datasets)} datasets from {len(selected_stocks)} stocks")
    return True

def create_small_chunks(input_file, output_dir, chunk_size=50):
    """Split into multiple smaller files"""
    print(f"📦 Creating chunks of {chunk_size} datasets each...")
    
    analysis = analyze_hdf5_file(input_file)
    if not analysis:
        return False
    
    datasets = analysis['datasets']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into chunks
    for i in range(0, len(datasets), chunk_size):
        chunk_datasets = datasets[i:i+chunk_size]
        chunk_file = os.path.join(output_dir, f"chunk_{i//chunk_size+1:03d}.h5")
        
        print(f"   Creating {chunk_file} with {len(chunk_datasets)} datasets...")
        
        with h5py.File(input_file, 'r') as src:
            with h5py.File(chunk_file, 'w') as dst:
                for dataset_name in chunk_datasets:
                    src.copy(dataset_name, dst)
        
        # Verify chunk file
        chunk_analysis = analyze_hdf5_file(chunk_file)
        print(f"   ✅ Created {chunk_file}: {len(chunk_datasets)} datasets")
    
    print(f"✅ Created {(len(datasets) + chunk_size - 1) // chunk_size} chunk files in {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Split HDF5 files in various ways')
    parser.add_argument('--input', required=True, help='Input HDF5 file path')
    parser.add_argument('--output', required=True, help='Output file or directory path')
    parser.add_argument('--mode', required=True, choices=['test', 'time', 'stocks', 'chunks'], 
                       help='Split mode: test (split into N files), time (by days), stocks (by stocks), chunks (multiple files)')
    parser.add_argument('--limit', type=int, default=5, help='Number of files to create for test mode')
    parser.add_argument('--days', type=int, default=10, help='Number of days for time mode')
    parser.add_argument('--stocks', type=int, default=5, help='Number of stocks for stocks mode')
    parser.add_argument('--chunk-size', type=int, default=50, help='Chunk size for chunks mode')
    
    args = parser.parse_args()
    
    print("🚀 HDF5 File Splitter")
    print("=" * 50)
    
    start_time = time.time()
    
    # Analyze input file
    analysis = analyze_hdf5_file(args.input)
    if not analysis:
        return
    
    # Perform the split based on mode
    success = False
    if args.mode == 'test':
        success = split_by_test_mode(args.input, args.output, args.limit)
    elif args.mode == 'time':
        success = split_by_time_periods(args.input, args.output, args.days)
    elif args.mode == 'stocks':
        success = split_by_stocks(args.input, args.output, args.stocks)
    elif args.mode == 'chunks':
        success = create_small_chunks(args.input, args.output, args.chunk_size)
    
    end_time = time.time()
    
    if success:
        print(f"\n✅ Split completed successfully in {end_time - start_time:.1f} seconds")
        print(f"📁 Output: {args.output}")
    else:
        print(f"\n❌ Split failed!")

if __name__ == "__main__":
    main() 