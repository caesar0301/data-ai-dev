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
        if compression == 'zstd':
            # Test different zstd compression levels
            for level in ['1', '3', '6', '9']:
                lance_path = f"{base_path}_{compression}_level{level}.lance"
                if os.path.exists(lance_path):
                    # Read entire dataset
                    start_time = time.time()
                    dataset = lance.dataset(lance_path)
                    table = dataset.to_table()
                    read_time = time.time() - start_time
                    
                    read_results[f"lance-{compression}-level{level}"] = read_time
        else:
            lance_path = f"{base_path}_{compression}.lance"
            if os.path.exists(lance_path):
                # Read entire dataset
                start_time = time.time()
                dataset = lance.dataset(lance_path)
                table = dataset.to_table()
                read_time = time.time() - start_time
                
                read_results[f"lance-{compression}"] = read_time
    
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
            read_results[fmt] = read_time
    
    return read_results

def plot_results(all_results: Dict[str, Any], read_results: Dict[str, float], output_dir: str = "ubm_results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    formats = list(all_results.keys())
    ratios = [res.get('compression_ratio', 0) for res in all_results.values()]
    sizes = [res.get('lance_size', res.get('size', 0)) / (1024*1024) for res in all_results.values()]
    write_times = [res.get('write_time', 0) for res in all_results.values()]
    read_times = [read_results.get(f, 0) for f in formats]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UBM Data Compression Performance Comparison', fontsize=16)

    ax1.bar(formats, ratios, color='skyblue')
    ax1.set_title('Compression Ratio (Higher is Better)', fontsize=12)
    ax1.set_ylabel('Ratio')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2.bar(formats, sizes, color='lightcoral')
    ax2.set_title('File Size (Lower is Better)', fontsize=12)
    ax2.set_ylabel('Size (MB)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    ax3.bar(formats, write_times, color='lightgreen')
    ax3.set_title('Write Time (Lower is Better)', fontsize=12)
    ax3.set_ylabel('Time (s)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    ax4.bar(formats, read_times, color='gold')
    ax4.set_title('Read Time (Lower is Better)', fontsize=12)
    ax4.set_ylabel('Time (s)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'compression_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Lance UBM Data Compression Benchmark ===")
    
    # Load UBM dataset
    df = load_ubm_dataset()
    base_path = "ubm_data"
    output_dir = "ubm_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    lance_compressions = ['zstd', 'lz4', 'gzip']

    print("\n=== Lance Write Tests ===")
    for compression in lance_compressions:
        if compression == 'zstd':
            # Test different zstd compression levels
            for level in ['1', '3', '6', '9']:
                key = f"lance-{compression}-level{level}"
                print(f"Testing {key}...")
                result = compress_with_lance(df, f"{base_path}_{compression}_level{level}.lance", compression, level)
                all_results[key] = result
        else:
            key = f"lance-{compression}"
            print(f"Testing {key}...")
            result = compress_with_lance(df, f"{base_path}_{compression}.lance", compression)
            all_results[key] = result

    print("\n=== Other Formats Write Tests ===")
    other_results = compress_with_other_formats(df, base_path)
    all_results.update(other_results)

    print("\n=== Read Performance Tests ===")
    read_results = benchmark_read_performance(base_path, lance_compressions)

    print("\n" + "="*90)
    print("                           UBM DATA COMPRESSION BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Format':<25} {'Size (MB)':<12} {'Ratio':<10} {'Write Time (s)':<15} {'Read Time (s)':<15}")
    print("-" * 90)
    for fmt, res in all_results.items():
        rtime = read_results.get(fmt, float('nan'))
        print(f"{fmt:<25} {res.get('lance_size', res.get('size', 0)) / (1024*1024):<12.2f} {res.get('compression_ratio', 0):<10.2f} "
              f"{res.get('write_time', 0):<15.3f} {rtime:<15.3f}")
    print("="*90 + "\n")

    # Save detailed results to output directory
    results_file = os.path.join(output_dir, 'ubm_compression_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': all_results, 
            'read_results': read_results,
            'dataset_info': {
                'total_records': len(df),
                'original_size_mb': df['total_size'].sum() / (1024*1024),
                'key_size_stats': {
                    'min': int(df['key_size'].min()),
                    'max': int(df['key_size'].max()),
                    'mean': float(df['key_size'].mean())
                },
                'value_size_stats': {
                    'min': int(df['value_size'].min()),
                    'max': int(df['value_size'].max()),
                    'mean': float(df['value_size'].mean())
                }
            }
        }, f, indent=2, default=str)

    plot_results(all_results, read_results, output_dir)
    print(f"Results saved to {results_file}")
    print(f"Visualization saved to {os.path.join(output_dir, 'compression_comparison.png')}")

    # Cleanup
    cleanup_files = [
        f"{base_path}_pickle.pkl",
        f"{base_path}_gzip.gz", 
        f"{base_path}_json.json",
        f"{base_path}_json_gzip.json.gz"
    ]
    for file in cleanup_files:
        if os.path.exists(file): 
            os.remove(file)
            
    for comp in lance_compressions:
        if comp == 'zstd':
            # Clean up zstd files with different levels
            for level in ['1', '3', '6', '9']:
                dir_to_remove = f"{base_path}_{comp}_level{level}.lance"
                if os.path.exists(dir_to_remove): 
                    shutil.rmtree(dir_to_remove)
        else:
            dir_to_remove = f"{base_path}_{comp}.lance"
            if os.path.exists(dir_to_remove): 
                shutil.rmtree(dir_to_remove)

    print("Cleanup completed.")

if __name__ == "__main__":
    main()
