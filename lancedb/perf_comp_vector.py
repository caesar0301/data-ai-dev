import lance
import pandas as pd
import numpy as np
import time
import os
import json
import gzip
import pickle
import struct
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import shutil

def load_sift_dataset(sample_size: int = 100000) -> pd.DataFrame:
    """
    Load SIFT dataset from .fvecs files and convert to DataFrame.
    SIFT (Scale-Invariant Feature Transform) vectors are 128-dimensional float vectors.

    wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    tar -xzf sift.tar.gz
    """
    print(f"Loading SIFT dataset (sampling {sample_size} vectors)...")
    
    # Load SIFT base vectors
    sift_file = "sift/sift_base.fvecs"
    if not os.path.exists(sift_file):
        raise FileNotFoundError(f"SIFT dataset not found at {sift_file}")
    
    vectors = []
    with open(sift_file, 'rb') as f:
        # Read first vector to determine dimension
        dim = struct.unpack('<I', f.read(4))[0]
        print(f"Vector dimensions: {dim}")
        
        # Read the first vector data
        vector_data = f.read(dim * 4)
        first_vector = struct.unpack(f'<{dim}f', vector_data)
        vectors.append(first_vector)
        
        # Reset file pointer
        f.seek(0)
        
        # Count total vectors by reading through the file
        total_vectors = 0
        while True:
            try:
                dim_header = f.read(4)
                if len(dim_header) < 4:
                    break
                f.read(dim * 4)  # Skip vector data
                total_vectors += 1
            except:
                break
        
        print(f"Total vectors in file: {total_vectors}")
        
        # Reset file pointer again
        f.seek(0)
        
        # Sample vectors if needed
        if sample_size < total_vectors:
            # Use random sampling
            np.random.seed(42)
            indices = np.random.choice(total_vectors, sample_size, replace=False)
            indices.sort()  # Keep original order for consistency
            
            current_idx = 0
            vectors = []  # Reset vectors list
            
            for i, target_idx in enumerate(indices):
                # Skip to target index
                while current_idx < target_idx:
                    # Skip dimension + vector data
                    f.read(4)  # dimension
                    f.read(dim * 4)  # vector data
                    current_idx += 1
                
                # Read target vector
                dim_read = struct.unpack('<I', f.read(4))[0]
                if dim_read != dim:
                    raise ValueError(f"Dimension mismatch: expected {dim}, got {dim_read}")
                
                vector_data = f.read(dim * 4)
                vector = struct.unpack(f'<{dim}f', vector_data)
                vectors.append(vector)
                current_idx += 1
        else:
            # Read all vectors
            vectors = []
            for i in range(min(sample_size, total_vectors)):
                dim_read = struct.unpack('<I', f.read(4))[0]
                if dim_read != dim:
                    raise ValueError(f"Dimension mismatch: expected {dim}, got {dim_read}")
                
                vector_data = f.read(dim * 4)
                vector = struct.unpack(f'<{dim}f', vector_data)
                vectors.append(vector)
    
    # Convert to DataFrame
    df = pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(dim)])
    
    # Add metadata columns
    df['vector_id'] = range(len(df))
    df['magnitude'] = np.linalg.norm(vectors, axis=1)
    df['timestamp'] = pd.Timestamp.now()
    
    print(f"Loaded {len(df)} vectors with {dim} dimensions")
    print(f"Vector statistics:")
    print(f"  Magnitude - min: {df['magnitude'].min():.4f}, max: {df['magnitude'].max():.4f}, mean: {df['magnitude'].mean():.4f}")
    print(f"  Values - min: {df.iloc[:, :dim].min().min():.4f}, max: {df.iloc[:, :dim].max().max():.4f}")
    
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
        if col_type == 'object':  # String columns
            pa_type = pa.string()
        elif col_type == 'int64':
            pa_type = pa.int64()
        elif col_type == 'float64':
            pa_type = pa.float64()
        elif 'datetime64' in str(col_type):
            pa_type = pa.timestamp('ns')
        else:
            pa_type = pa.float64()  # Default to float64 for vector dimensions
        
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
    end_time = time.time()

    file_size = get_dir_size(uri)
    # Calculate original size as raw float32 bytes
    vector_cols = [col for col in df.columns if col.startswith('dim_')]
    original_size = len(vector_cols) * len(df) * 4  # 4 bytes per float32
    
    return {
        'compression_algo': compression,
        'compression_level': compression_level,
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'write_time': end_time - start_time,
        'compression_ratio': original_size / file_size if file_size > 0 else float('inf'),
        'original_size_bytes': original_size,
        'original_size_mb': original_size / (1024 * 1024)
    }

def compress_with_other_formats(df: pd.DataFrame, base_path: str) -> Dict[str, Any]:
    results = {}
    
    # Calculate original size as raw float32 bytes
    vector_cols = [col for col in df.columns if col.startswith('dim_')]
    original_size = len(vector_cols) * len(df) * 4  # 4 bytes per float32
    
    # CSV (baseline)
    csv_path = f"{base_path}.csv"
    start_time = time.time()
    df.to_csv(csv_path, index=False)
    csv_time = time.time() - start_time
    csv_size = get_file_size(csv_path)
    
    results['CSV'] = {
        'format': 'CSV', 
        'file_size_bytes': csv_size, 
        'file_size_mb': csv_size / (1024*1024),
        'write_time': csv_time, 
        'compression_ratio': original_size / csv_size if csv_size > 0 else float('inf'),
        'original_size_bytes': original_size,
        'original_size_mb': original_size / (1024 * 1024)
    }

    # GZIP CSV
    gzip_path = f"{base_path}.csv.gz"
    start_time = time.time()
    with gzip.open(gzip_path, 'wt', encoding='utf-8') as f:
        df.to_csv(f, index=False)
    gzip_time = time.time() - start_time
    gzip_size = get_file_size(gzip_path)
    
    results['GZIP CSV'] = {
        'format': 'GZIP CSV', 
        'file_size_bytes': gzip_size, 
        'file_size_mb': gzip_size / (1024*1024),
        'write_time': gzip_time,
        'compression_ratio': original_size / gzip_size if gzip_size > 0 else float('inf'),
        'original_size_bytes': original_size,
        'original_size_mb': original_size / (1024 * 1024)
    }

    # Parquet (Snappy)
    parquet_path = f"{base_path}.parquet"
    start_time = time.time()
    df.to_parquet(parquet_path, compression='snappy')
    parquet_time = time.time() - start_time
    parquet_size = get_file_size(parquet_path)
    
    results['Parquet (Snappy)'] = {
        'format': 'Parquet (Snappy)', 
        'file_size_bytes': parquet_size, 
        'file_size_mb': parquet_size / (1024*1024),
        'write_time': parquet_time,
        'compression_ratio': original_size / parquet_size if parquet_size > 0 else float('inf'),
        'original_size_bytes': original_size,
        'original_size_mb': original_size / (1024 * 1024)
    }
    
    # Numpy NPZ (compressed)
    npz_path = f"{base_path}.npz"
    start_time = time.time()
    # Save only vector data as numpy array
    vector_data = df[vector_cols].values.astype(np.float32)
    np.savez_compressed(npz_path, vectors=vector_data)
    npz_time = time.time() - start_time
    npz_size = get_file_size(npz_path)
    
    results['NPZ (Compressed)'] = {
        'format': 'NPZ (Compressed)', 
        'file_size_bytes': npz_size, 
        'file_size_mb': npz_size / (1024*1024),
        'write_time': npz_time,
        'compression_ratio': original_size / npz_size if npz_size > 0 else float('inf'),
        'original_size_bytes': original_size,
        'original_size_mb': original_size / (1024 * 1024)
    }
    
    return results

def benchmark_read_performance(base_path: str, lance_compressions: list) -> Dict[str, float]:
    results = {}
    
    # Lance read
    for comp in lance_compressions:
        if comp == 'zstd':
            # Test different zstd compression levels
            for level in ['1', '3', '6', '9']:
                key = f"lance-{comp}-level{level}"
                uri = f"{base_path}_{comp}_level{level}.lance"
                if os.path.exists(uri):
                    start_time = time.time()
                    dataset = lance.dataset(uri)
                    _ = dataset.to_table().to_pandas()
                    results[key] = time.time() - start_time
        else:
            key = f"lance-{comp}"
            uri = f"{base_path}_{comp}.lance"
            if os.path.exists(uri):
                start_time = time.time()
                dataset = lance.dataset(uri)
                _ = dataset.to_table().to_pandas()
                results[key] = time.time() - start_time

    # Other formats
    if os.path.exists(f"{base_path}.csv"):
        start_time = time.time()
        _ = pd.read_csv(f"{base_path}.csv")
        results['CSV'] = time.time() - start_time
        
    if os.path.exists(f"{base_path}.csv.gz"):
        start_time = time.time()
        _ = pd.read_csv(f"{base_path}.csv.gz")
        results['GZIP CSV'] = time.time() - start_time
        
    if os.path.exists(f"{base_path}.parquet"):
        start_time = time.time()
        _ = pd.read_parquet(f"{base_path}.parquet")
        results['Parquet (Snappy)'] = time.time() - start_time
        
    if os.path.exists(f"{base_path}.npz"):
        start_time = time.time()
        _ = np.load(f"{base_path}.npz")
        results['NPZ (Compressed)'] = time.time() - start_time
            
    return results

def plot_results(all_results: Dict[str, Any], read_results: Dict[str, float], output_dir: str = "vector_results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    ax1, ax2, ax3, ax4 = axes.flatten()

    formats = list(all_results.keys())
    ratios = [res.get('compression_ratio', 0) for res in all_results.values()]
    sizes = [res.get('file_size_mb', 0) for res in all_results.values()]
    write_times = [res.get('write_time', 0) for res in all_results.values()]
    read_times = [read_results.get(f, 0) for f in formats]

    # Compression ratio plot
    ax1.bar(formats, ratios, color='skyblue')
    ax1.set_title('Compression Ratio (vs Raw Float32, Higher is Better)', fontsize=12)
    ax1.set_ylabel('Ratio')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # File size plot
    ax2.bar(formats, sizes, color='lightcoral')
    ax2.set_title('File Size (Lower is Better)', fontsize=12)
    ax2.set_ylabel('Size (MB)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Write time plot
    ax3.bar(formats, write_times, color='lightgreen')
    ax3.set_title('Write Time (Lower is Better)', fontsize=12)
    ax3.set_ylabel('Time (s)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # Read time plot
    ax4.bar(formats, read_times, color='gold')
    ax4.set_title('Read Time (Lower is Better)', fontsize=12)
    ax4.set_ylabel('Time (s)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'vector_compression_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Lance Vector Compression Benchmark (SIFT Dataset) ===")
    
    # Load SIFT dataset
    df = load_sift_dataset(sample_size=100000)  # Sample 100k vectors for faster testing
    base_path = "sift_vectors"
    output_dir = "vector_results"
    
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
    print("                           VECTOR COMPRESSION BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Format':<25} {'Size (MB)':<12} {'Ratio':<10} {'Write Time (s)':<15} {'Read Time (s)':<15}")
    print("-" * 90)
    for fmt, res in all_results.items():
        rtime = read_results.get(fmt, float('nan'))
        print(f"{fmt:<25} {res.get('file_size_mb', 0):<12.2f} {res.get('compression_ratio', 0):<10.2f} "
              f"{res.get('write_time', 0):<15.3f} {rtime:<15.3f}")
    print("="*90 + "\n")

    # Save detailed results to output directory
    results_file = os.path.join(output_dir, 'vector_compression_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': all_results, 
            'read_results': read_results,
            'dataset_info': {
                'vectors': len(df),
                'dimensions': len([col for col in df.columns if col.startswith('dim_')]),
                'total_elements': len(df) * len([col for col in df.columns if col.startswith('dim_')]),
                'original_size_mb': all_results[list(all_results.keys())[0]]['original_size_mb']
            }
        }, f, indent=2, default=str)

    plot_results(all_results, read_results, output_dir)
    print(f"Results saved to {results_file}")
    print(f"Visualization saved to {os.path.join(output_dir, 'vector_compression_comparison.png')}")

    # Cleanup
    cleanup_files = [
        f"{base_path}.csv", 
        f"{base_path}.csv.gz", 
        f"{base_path}.parquet",
        f"{base_path}.npz"
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

if __name__ == "__main__":
    main() 