#!/usr/bin/env python3
"""
Quick UBM Lance Data Reader
Simple utility to quickly read and display UBM data from Lance datasets.
"""

import lance
import pandas as pd
import base64
import sys
import argparse

def quick_read(dataset_path: str = "ubm_lance_zstd_1", limit: int = 10):
    """
    Quick read of UBM Lance data with basic display.
    
    Args:
        dataset_path: Path to Lance dataset
        limit: Number of records to display
    """
    try:
        # Open dataset
        dataset = lance.dataset(dataset_path)
        print(f"📊 Dataset: {dataset_path}")
        print(f"📈 Total records: {dataset.count_rows()}")
        
        # Read data
        table = dataset.to_table()
        df = table.to_pandas()
        
        print(f"📋 Columns: {list(df.columns)}")
        print(f"💾 Original size: {df['total_size'].sum() / (1024*1024):.2f} MB")
        print()
        
        # Display sample records
        print("🔍 Sample Records:")
        print("-" * 80)
        
        for i in range(min(limit, len(df))):
            row = df.iloc[i]
            
            # Decode key
            try:
                key_binary = base64.b64decode(row['key'])
                key_text = key_binary.decode('utf-8', errors='ignore')
            except:
                key_text = "Binary data"
            
            print(f"Record {i}:")
            print(f"  ID: {row['id']}")
            print(f"  Key: {key_text}")
            print(f"  Key size: {row['key_size']} bytes")
            print(f"  Value size: {row['value_size']} bytes")
            print(f"  Total size: {row['total_size']} bytes")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def search_by_key(dataset_path: str, search_key: str):
    """
    Search for records by key pattern.
    
    Args:
        dataset_path: Path to Lance dataset
        search_key: Key pattern to search for
    """
    try:
        dataset = lance.dataset(dataset_path)
        table = dataset.to_table()
        df = table.to_pandas()
        
        print(f"🔍 Searching for key pattern: '{search_key}'")
        print("-" * 60)
        
        found = False
        for i, row in df.iterrows():
            try:
                key_binary = base64.b64decode(row['key'])
                key_text = key_binary.decode('utf-8', errors='ignore')
                
                if search_key.lower() in key_text.lower():
                    print(f"Match found in record {i}:")
                    print(f"  Key: {key_text}")
                    print(f"  Value size: {row['value_size']} bytes")
                    print()
                    found = True
                    
            except Exception as e:
                continue
        
        if not found:
            print("No matches found.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def get_stats(dataset_path: str):
    """
    Get basic statistics about the dataset.
    
    Args:
        dataset_path: Path to Lance dataset
    """
    try:
        dataset = lance.dataset(dataset_path)
        table = dataset.to_table()
        df = table.to_pandas()
        
        print(f"📊 Statistics for {dataset_path}")
        print("=" * 50)
        print(f"Total records: {len(df)}")
        print(f"Total size: {df['total_size'].sum() / (1024*1024):.2f} MB")
        print()
        
        print("Key Statistics:")
        print(f"  Min size: {df['key_size'].min()} bytes")
        print(f"  Max size: {df['key_size'].max()} bytes")
        print(f"  Mean size: {df['key_size'].mean():.1f} bytes")
        print()
        
        print("Value Statistics:")
        print(f"  Min size: {df['value_size'].min()} bytes")
        print(f"  Max size: {df['value_size'].max()} bytes")
        print(f"  Mean size: {df['value_size'].mean():.1f} bytes")
        print()
        
        print("Size Distribution:")
        size_ranges = [
            (0, 1000, "Small (<1KB)"),
            (1000, 10000, "Medium (1-10KB)"),
            (10000, 100000, "Large (10-100KB)"),
            (100000, float('inf'), "Very Large (>100KB)")
        ]
        
        for min_size, max_size, label in size_ranges:
            count = len(df[(df['value_size'] >= min_size) & (df['value_size'] < max_size)])
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count} records ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quick UBM Lance Data Reader")
    parser.add_argument("--dataset", "-d", default="ubm_lance_zstd_1", 
                       help="Path to Lance dataset (default: ubm_lance_zstd_1)")
    parser.add_argument("--limit", "-l", type=int, default=5,
                       help="Number of records to display (default: 5)")
    parser.add_argument("--search", "-s", 
                       help="Search for records by key pattern")
    parser.add_argument("--stats", action="store_true",
                       help="Show dataset statistics")
    
    args = parser.parse_args()
    
    if args.search:
        search_by_key(args.dataset, args.search)
    elif args.stats:
        get_stats(args.dataset)
    else:
        quick_read(args.dataset, args.limit)

if __name__ == "__main__":
    main() 