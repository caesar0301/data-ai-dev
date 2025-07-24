import lance
import pandas as pd
import numpy as np
import time
import os
import json
import gzip
import pickle
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import shutil

def load_text_dataset() -> pd.DataFrame:
    print("Creating synthetic text dataset...")
    np.random.seed(42)

    categories = {
        'short_text': ['Hello world', 'Good morning', 'How are you?', 'Nice day!', 'See you later'],
        'medium_text': [
            'This is a medium length text that contains multiple sentences and some repetition.',
            'The quick brown fox jumps over the lazy dog in a beautiful forest setting.',
            'Machine learning algorithms are becoming increasingly important in modern applications.',
            'Data compression techniques help reduce storage requirements and improve performance.',
            'Natural language processing enables computers to understand human language.'
        ],
        'long_text': [
            'This is a much longer text that contains many more words and sentences. It includes various punctuation marks, numbers like 123 and 456, and repeated phrases that might benefit from compression. The quick brown fox jumps over the lazy dog multiple times in this extended narrative. Machine learning and artificial intelligence are transforming the way we process and analyze data in the modern world.',
            'Another long text passage that demonstrates different compression characteristics. This text contains many common words like "the", "and", "is", "to", "in", "of", "a", "that", "it", "with", "as", "for", "his", "they", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time", "two", "more", "go", "no", "way", "could", "my", "than", "first", "been", "call", "who", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part".',
            'A comprehensive analysis of data compression algorithms reveals that different techniques work better for different types of data. Text compression algorithms like LZ77, LZ78, and their variants (LZMA, LZ4, Snappy) are particularly effective for natural language text. These algorithms exploit redundancy in text data, including repeated words, phrases, and patterns. The compression ratio achieved depends on the nature of the text, with highly repetitive text achieving better compression ratios than random or encrypted data.'
        ]
    }

    data = []
    for i in range(10000):
        category = np.random.choice(list(categories.keys()), p=[0.4, 0.4, 0.2])
        text = np.random.choice(categories[category])
        if np.random.random() < 0.3:
            text += f" ID:{i} Timestamp:{time.time()}"

        data.append({
            'id': i,
            'text': text,
            'category': category,
            'length': len(text),
            'word_count': len(text.split()),
            'timestamp': pd.Timestamp.now() - pd.Timedelta(seconds=np.random.randint(0, 86400))
        })

    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} records")
    print(f"Text length statistics: min={df['length'].min()}, max={df['length'].max()}, mean={df['length'].mean():.1f}")
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
        elif col_type == 'datetime64[ns]':
            pa_type = pa.timestamp('ns')
        else:
            pa_type = pa.string()  # Default to string
        
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
    csv_size = len(df.to_csv(index=False).encode('utf-8'))
    return {
        'compression_algo': compression,
        'compression_level': compression_level,
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'write_time': end_time - start_time,
        'compression_ratio': csv_size / file_size if file_size > 0 else float('inf')
    }

def compress_with_other_formats(df: pd.DataFrame, base_path: str) -> Dict[str, Any]:
    results = {}
    csv_path = f"{base_path}.csv"
    df.to_csv(csv_path, index=False)
    csv_size = get_file_size(csv_path)

    # Uncompressed CSV (baseline)
    results['CSV'] = {
        'format': 'CSV', 'file_size_bytes': csv_size, 'file_size_mb': csv_size / (1024*1024),
        'write_time': 0, 'compression_ratio': 1.0
    }

    # GZIP CSV
    gzip_path = f"{base_path}.csv.gz"
    start_time = time.time()
    with gzip.open(gzip_path, 'wt', encoding='utf-8') as f:
        df.to_csv(f, index=False)
    gzip_size = get_file_size(gzip_path)
    results['GZIP CSV'] = {
        'format': 'GZIP CSV', 'file_size_bytes': gzip_size, 'file_size_mb': gzip_size / (1024*1024),
        'write_time': time.time() - start_time,
        'compression_ratio': csv_size / gzip_size if gzip_size > 0 else float('inf')
    }

    # Parquet (Snappy)
    parquet_path = f"{base_path}.parquet"
    start_time = time.time()
    df.to_parquet(parquet_path, compression='snappy')
    parquet_size = get_file_size(parquet_path)
    results['Parquet (Snappy)'] = {
        'format': 'Parquet (Snappy)', 'file_size_bytes': parquet_size, 'file_size_mb': parquet_size / (1024*1024),
        'write_time': time.time() - start_time,
        'compression_ratio': csv_size / parquet_size if parquet_size > 0 else float('inf')
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
        start_time = time.time(); _ = pd.read_csv(f"{base_path}.csv"); results['CSV'] = time.time() - start_time
    if os.path.exists(f"{base_path}.csv.gz"):
        start_time = time.time(); _ = pd.read_csv(f"{base_path}.csv.gz"); results['GZIP CSV'] = time.time() - start_time
    if os.path.exists(f"{base_path}.parquet"):
        start_time = time.time(); _ = pd.read_parquet(f"{base_path}.parquet"); results['Parquet (Snappy)'] = time.time() - start_time
            
    return results

def plot_results(all_results: Dict[str, Any], read_results: Dict[str, float]):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax1, ax2, ax3, ax4 = axes.flatten()

    formats = list(all_results.keys())
    ratios = [res.get('compression_ratio', 0) for res in all_results.values()]
    sizes = [res.get('file_size_mb', 0) for res in all_results.values()]
    write_times = [res.get('write_time', 0) for res in all_results.values()]
    read_times = [read_results.get(f, 0) for f in formats]

    ax1.bar(formats, ratios, color='skyblue')
    ax1.set_title('Compression Ratio (vs CSV, Higher is Better)')
    ax1.set_ylabel('Ratio')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(formats, sizes, color='lightcoral')
    ax2.set_title('File Size (Lower is Better)')
    ax2.set_ylabel('Size (MB)')
    ax2.tick_params(axis='x', rotation=45)

    ax3.bar(formats, write_times, color='lightgreen')
    ax3.set_title('Write Time (Lower is Better)')
    ax3.set_ylabel('Time (s)')
    ax3.tick_params(axis='x', rotation=45)

    ax4.bar(formats, read_times, color='gold')
    ax4.set_title('Read Time (Lower is Better)')
    ax4.set_ylabel('Time (s)')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout(pad=2.0)
    plt.savefig('compression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== LanceDB Compression Benchmark ===")
    df = load_text_dataset()
    base_path = "text_data"
    
    all_results = {}
    lance_compressions = ['zstd', 'lz4', 'gzip']

    print("\n=== LanceDB Write Tests ===")
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

    print("\n" + "="*80)
    print("                           BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Format':<20} {'Size (MB)':<12} {'Ratio':<10} {'Write Time (s)':<15} {'Read Time (s)':<15}")
    print("-" * 80)
    for fmt, res in all_results.items():
        rtime = read_results.get(fmt, float('nan'))
        print(f"{fmt:<20} {res.get('file_size_mb', 0):<12.2f} {res.get('compression_ratio', 0):<10.2f} "
              f"{res.get('write_time', 0):<15.3f} {rtime:<15.3f}")
    print("="*80 + "\n")

    with open('compression_results.json', 'w') as f:
        json.dump({'all_results': all_results, 'read_results': read_results}, f, indent=2, default=str)

    plot_results(all_results, read_results)
    print("Results saved to compression_results.json")
    print("Visualization saved to compression_comparison.png")

    # Cleanup
    for file in [f"{base_path}.csv", f"{base_path}.csv.gz", f"{base_path}.parquet"]:
        if os.path.exists(file): os.remove(file)
    for comp in lance_compressions:
        if comp == 'zstd':
            # Clean up zstd files with different levels
            for level in ['1', '3', '6', '9']:
                dir_to_remove = f"{base_path}_{comp}_level{level}.lance"
                if os.path.exists(dir_to_remove): shutil.rmtree(dir_to_remove)
        else:
            dir_to_remove = f"{base_path}_{comp}.lance"
            if os.path.exists(dir_to_remove): shutil.rmtree(dir_to_remove)

if __name__ == "__main__":
    main()