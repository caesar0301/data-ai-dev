import lance
import pandas as pd
import numpy as np
import time
import os
import json
import gzip
import pickle
import lmdb
import base64
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import shutil

def load_ubm_dataset(db_path: str = "ubm_data", sample_size: int = None) -> pd.DataFrame:
    """
    Load UBM dataset from LMDB and convert to DataFrame.
    UBM data contains binary key-value pairs stored in LMDB format.
    """
    print(f"Loading UBM dataset from {db_path}...")
    
    # Open the LMDB environment in read-only mode
    env = lmdb.open(db_path, readonly=True, lock=False)
    
    data = []
    total_records = 0
    
    # Start a read-only transaction
    with env.begin() as txn:
        # Create a cursor to iterate over the database
        cursor = txn.cursor()
        
        # First pass: count total records
        for key, value in cursor:
            total_records += 1
        
        print(f"Total records in UBM dataset: {total_records}")
        
        # Determine sample size if not specified
        if sample_size is None:
            sample_size = min(total_records, 100000)  # Default to 100k or all if less
        
        # Reset cursor
        cursor = txn.cursor()
        
        # Second pass: collect data
        for i, (key, value) in enumerate(cursor):
            if i >= sample_size:
                break
                
            # Convert binary data to base64 for storage
            key_b64 = base64.b64encode(key).decode('utf-8')
            value_b64 = base64.b64encode(value).decode('utf-8')
            
            data.append({
                'id': i,
                'key': key_b64,
                'value': value_b64,
                'key_size': len(key),
                'value_size': len(value),
                'total_size': len(key) + len(value),
                'key_type': 'binary',
                'value_type': 'binary'
            })
    
    env.close()
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records from UBM dataset")
    print(f"Key size statistics: min={df['key_size'].min()}, max={df['key_size'].max()}, mean={df['key_size'].mean():.1f}")
    print(f"Value size statistics: min={df['value_size'].min()}, max={df['value_size'].max()}, mean={df['value_size'].mean():.1f}")
    print(f"Total size statistics: min={df['total_size'].min()}, max={df['total_size'].max()}, mean={df['total_size'].mean():.1f}")
    
    return df

def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)

def get_dir_size(dir_path: str) -> int:
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, filenames in os.walk(dir_path)
               for f in filenames)

def compress_with_lance(df: pd.DataFrame, uri: str, compression: str, compression_level: str = "3") -> Dict[str, Any]:
    start_time = time.time()
    
    # Create PyArrow schema with compression metadata
    fields = []
    for col_name, col_type in df.dtypes.items():
        if col_name in ['key', 'value']:
            pa_type = pa.string()  # Base64 encoded binary data as strings
        elif col_type == 'int64':
            pa_type = pa.int64()
        elif col_type == 'float64':
            pa_type = pa.float64()
        else:
            pa_type = pa.string()
        
        # Add compression metadata for all columns
        metadata = {
            "lance-encoding:compression": compression,
            "lance-encoding:compression-level": compression_level,
        }
        
        fields.append(pa.field(col_name, pa_type, metadata=metadata))
    
    schema = pa.schema(fields)
    table = pa.Table.from_pandas(df, schema=schema)

    if os.path.exists(uri):
        shutil.rmtree(uri)

    # Write dataset with schema that includes compression metadata
    lance.write_dataset(
        table,
        uri,
        mode='overwrite'
    )
    
    write_time = time.time() - start_time
    
    # Calculate sizes
    lance_size = get_dir_size(uri)
    original_size = df['total_size'].sum()
    
    return {
        'compression': compression,
        'compression_level': compression_level,
        'write_time': float(write_time),
        'lance_size': int(lance_size),
        'original_size': int(original_size),
        'compression_ratio': float(original_size / lance_size if lance_size > 0 else 0),
        'space_saved': float((original_size - lance_size) / original_size * 100 if original_size > 0 else 0)
    }

def compress_with_other_formats(df: pd.DataFrame, base_path: str) -> Dict[str, Any]:
    """Compress data using other formats for comparison"""
    results = {}
    
    # Convert DataFrame to binary format for fair comparison
    binary_data = []
    for _, row in df.iterrows():
        # Decode base64 back to binary for compression
        key_bin = base64.b64decode(row['key'])
        value_bin = base64.b64decode(row['value'])
        binary_data.append((key_bin, value_bin))
    
    # Pickle compression
    pickle_path = f"{base_path}_pickle.pkl"
    start_time = time.time()
    with open(pickle_path, 'wb') as f:
        pickle.dump(binary_data, f)
    pickle_time = time.time() - start_time
    pickle_size = get_file_size(pickle_path)
    
    # Gzip compression
    gzip_path = f"{base_path}_gzip.gz"
    start_time = time.time()
    with gzip.open(gzip_path, 'wb') as f:
        pickle.dump(binary_data, f)
    gzip_time = time.time() - start_time
    gzip_size = get_file_size(gzip_path)
    
    # JSON compression (base64 encoded)
    json_data = []
    for _, row in df.iterrows():
        json_data.append({
            'key': row['key'],
            'value': row['value'],
            'key_size': row['key_size'],
            'value_size': row['value_size']
        })
    
    json_path = f"{base_path}_json.json"
    start_time = time.time()
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    json_time = time.time() - start_time
    json_size = get_file_size(json_path)
    
    # Gzipped JSON
    json_gzip_path = f"{base_path}_json_gzip.json.gz"
    start_time = time.time()
    with gzip.open(json_gzip_path, 'wt') as f:
        json.dump(json_data, f)
    json_gzip_time = time.time() - start_time
    json_gzip_size = get_file_size(json_gzip_path)
    
    original_size = df['total_size'].sum()
    
    results = {
        'pickle': {
            'format': 'pickle',
            'write_time': float(pickle_time),
            'size': int(pickle_size),
            'original_size': int(original_size),
            'compression_ratio': float(original_size / pickle_size if pickle_size > 0 else 0),
            'space_saved': float((original_size - pickle_size) / original_size * 100 if original_size > 0 else 0)
        },
        'gzip': {
            'format': 'gzip',
            'write_time': float(gzip_time),
            'size': int(gzip_size),
            'original_size': int(original_size),
            'compression_ratio': float(original_size / gzip_size if gzip_size > 0 else 0),
            'space_saved': float((original_size - gzip_size) / original_size * 100 if original_size > 0 else 0)
        },
        'json': {
            'format': 'json',
            'write_time': float(json_time),
            'size': int(json_size),
            'original_size': int(original_size),
            'compression_ratio': float(original_size / json_size if json_size > 0 else 0),
            'space_saved': float((original_size - json_size) / original_size * 100 if original_size > 0 else 0)
        },
        'json_gzip': {
            'format': 'json_gzip',
            'write_time': float(json_gzip_time),
            'size': int(json_gzip_size),
            'original_size': int(original_size),
            'compression_ratio': float(original_size / json_gzip_size if json_gzip_size > 0 else 0),
            'space_saved': float((original_size - json_gzip_size) / original_size * 100 if original_size > 0 else 0)
        }
    }
    
    return results

def benchmark_read_performance(base_path: str, lance_compressions: list) -> Dict[str, float]:
    """Benchmark read performance for different compression methods"""
    read_results = {}
    
    # Test Lance read performance
    for compression in lance_compressions:
        lance_path = f"{base_path}_lance_{compression}"
        if os.path.exists(lance_path):
            # Read entire dataset
            start_time = time.time()
            dataset = lance.dataset(lance_path)
            table = dataset.to_table()
            read_time = time.time() - start_time
            
            # Read with filters
            start_time = time.time()
            filtered_table = dataset.to_table(filter=f"key_size > {table['key_size'].mean()}")
            filter_time = time.time() - start_time
            
            read_results[f"lance_{compression}_full"] = read_time
            read_results[f"lance_{compression}_filtered"] = filter_time
    
    # Test other formats
    formats = ['pickle', 'gzip', 'json', 'json_gzip']
    for fmt in formats:
        file_path = f"{base_path}_{fmt}"
        if fmt == 'pickle':
            file_path += '.pkl'
        elif fmt == 'gzip':
            file_path += '.gz'
        elif fmt == 'json':
            file_path += '.json'
        elif fmt == 'json_gzip':
            file_path += '.json.gz'
        
        if os.path.exists(file_path):
            start_time = time.time()
            if fmt == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            elif fmt == 'gzip':
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            elif fmt == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif fmt == 'json_gzip':
                with gzip.open(file_path, 'rt') as f:
                    data = json.load(f)
            
            read_time = time.time() - start_time
            read_results[f"{fmt}_read"] = read_time
    
    return read_results

def plot_results(all_results: Dict[str, Any], read_results: Dict[str, float], output_dir: str = "ubm_results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    lance_results = []
    other_results = []
    
    for result in all_results.values():
        if isinstance(result, dict) and 'compression' in result:
            lance_results.append(result)
        elif isinstance(result, dict) and 'format' in result:
            other_results.append(result)
    
    # Create comparison DataFrame
    all_formats = []
    for result in lance_results:
        all_formats.append({
            'format': f"lance_{result['compression']}",
            'write_time': result['write_time'],
            'size_mb': result['lance_size'] / (1024 * 1024),
            'compression_ratio': result['compression_ratio'],
            'space_saved': result['space_saved']
        })
    
    for result in other_results:
        all_formats.append({
            'format': result['format'],
            'write_time': result['write_time'],
            'size_mb': result['size'] / (1024 * 1024),
            'compression_ratio': result['compression_ratio'],
            'space_saved': result['space_saved']
        })
    
    df_plot = pd.DataFrame(all_formats)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('UBM Data Compression Performance Comparison', fontsize=16)
    
    # 1. File sizes
    sns.barplot(data=df_plot, x='format', y='size_mb', ax=axes[0, 0])
    axes[0, 0].set_title('File Sizes (MB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylabel('Size (MB)')
    
    # 2. Compression ratios
    sns.barplot(data=df_plot, x='format', y='compression_ratio', ax=axes[0, 1])
    axes[0, 1].set_title('Compression Ratios')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylabel('Compression Ratio')
    
    # 3. Space saved percentage
    sns.barplot(data=df_plot, x='format', y='space_saved', ax=axes[1, 0])
    axes[1, 0].set_title('Space Saved (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylabel('Space Saved (%)')
    
    # 4. Write times
    sns.barplot(data=df_plot, x='format', y='write_time', ax=axes[1, 1])
    axes[1, 1].set_title('Write Times (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Read performance plot
    if read_results:
        read_df = pd.DataFrame([
            {'format': k, 'read_time': v} for k, v in read_results.items()
        ])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=read_df, x='format', y='read_time')
        plt.title('Read Performance Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Read Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'read_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save detailed results
    results_summary = {
        'compression_results': all_results,
        'read_results': read_results,
        'summary_stats': {
            'total_records': len(df_plot) if len(df_plot) > 0 else 0,
            'original_size_mb': df_plot['size_mb'].max() if len(df_plot) > 0 else 0,
            'best_compression': df_plot.loc[df_plot['compression_ratio'].idxmax(), 'format'] if len(df_plot) > 0 else 'none',
            'fastest_write': df_plot.loc[df_plot['write_time'].idxmin(), 'format'] if len(df_plot) > 0 else 'none',
            'smallest_size': df_plot.loc[df_plot['size_mb'].idxmin(), 'format'] if len(df_plot) > 0 else 'none'
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("UBM DATA COMPRESSION RESULTS SUMMARY")
    print("="*60)
    print(f"Total records processed: {results_summary['summary_stats']['total_records']}")
    print(f"Original data size: {results_summary['summary_stats']['original_size_mb']:.2f} MB")
    print(f"Best compression: {results_summary['summary_stats']['best_compression']}")
    print(f"Fastest write: {results_summary['summary_stats']['fastest_write']}")
    print(f"Smallest file size: {results_summary['summary_stats']['smallest_size']}")
    print("="*60)

def main():
    print("UBM Data Compression Performance Profiling")
    print("="*50)
    
    # Load UBM dataset
    df = load_ubm_dataset()
    
    # Define compression methods to test
    lance_compressions = ['lz4', 'zstd', 'gzip', 'brotli']
    compression_levels = ['1', '3', '6', '9']
    
    all_results = {}
    
    # Test Lance compressions
    print("\nTesting Lance compression methods...")
    for compression in lance_compressions:
        for level in compression_levels:
            print(f"Testing Lance {compression} level {level}...")
            result = compress_with_lance(df, f"ubm_lance_{compression}_{level}", compression, level)
            all_results[f"lance_{compression}_{level}"] = result
    
    # Test other compression formats
    print("\nTesting other compression formats...")
    other_results = compress_with_other_formats(df, "ubm_data")
    all_results.update(other_results)
    
    # Benchmark read performance
    print("\nBenchmarking read performance...")
    read_results = benchmark_read_performance("ubm_data", lance_compressions)
    
    # Plot and save results
    print("\nGenerating plots and saving results...")
    plot_results(all_results, read_results)
    
    print("\nProfiling complete! Results saved to ubm_results/ directory.")

if __name__ == "__main__":
    main()
