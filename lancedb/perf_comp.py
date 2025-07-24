import lance
import pandas as pd
import numpy as np
import time
import os
import json
from pathlib import Path
import gzip
import pickle
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa


def load_text_dataset(dataset_path: str = "text_dataset.csv") -> pd.DataFrame:
    print("Creating synthetic text dataset...")
    np.random.seed(42)

    categories = {
        "short_text": [
            "Hello world",
            "Good morning",
            "How are you?",
            "Nice day!",
            "See you later",
        ],
        "medium_text": [
            "This is a medium length text that contains multiple sentences and some repetition.",
            "The quick brown fox jumps over the lazy dog in a beautiful forest setting.",
            "Machine learning algorithms are becoming increasingly important in modern applications.",
            "Data compression techniques help reduce storage requirements and improve performance.",
            "Natural language processing enables computers to understand human language.",
        ],
        "long_text": [
            "This is a much longer text that contains many more words and sentences. It includes various punctuation marks, numbers like 123 and 456, and repeated phrases that might benefit from compression. The quick brown fox jumps over the lazy dog multiple times in this extended narrative. Machine learning and artificial intelligence are transforming the way we process and analyze data in the modern world.",
            'Another long text passage that demonstrates different compression characteristics. This text contains many common words like "the", "and", "is", "to", "in", "of", "a", "that", "it", "with", "as", "for", "his", "they", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time", "two", "more", "go", "no", "way", "could", "my", "than", "first", "been", "call", "who", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part".',
            "A comprehensive analysis of data compression algorithms reveals that different techniques work better for different types of data. Text compression algorithms like LZ77, LZ78, and their variants (LZMA, LZ4, Snappy) are particularly effective for natural language text. These algorithms exploit redundancy in text data, including repeated words, phrases, and patterns. The compression ratio achieved depends on the nature of the text, with highly repetitive text achieving better compression ratios than random or encrypted data.",
        ],
    }

    data = []
    for i in range(10000):
        category = np.random.choice(list(categories.keys()), p=[0.4, 0.4, 0.2])
        text = np.random.choice(categories[category])
        if np.random.random() < 0.3:
            text += f" ID:{i} Timestamp:{time.time()}"

        data.append(
            {
                "id": i,
                "text": text,
                "category": category,
                "length": len(text),
                "word_count": len(text.split()),
                "timestamp": pd.Timestamp.now()
                - pd.Timedelta(seconds=np.random.randint(0, 86400)),
            }
        )

    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} records")
    print(
        f"Text length statistics: min={df['length'].min()}, max={df['length'].max()}, mean={df['length'].mean():.1f}"
    )
    return df


def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)


def compress_with_lance(
    df: pd.DataFrame, uri: str, compression: str = "zstd"
) -> Dict[str, Any]:
    start_time = time.time()
    table = pa.Table.from_pandas(df)

    if os.path.exists(uri):
        import shutil

        shutil.rmtree(uri)

    # Corrected section: Use write_params to specify compression
    write_params = {"compression": compression}

    lance.write_dataset(
        table,
        uri,
        mode="overwrite",
        max_rows_per_group=8192,
        max_rows_per_file=1024 * 1024,
        write_params=write_params,  # Pass the params here
    )
    end_time = time.time()

    file_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(uri)
        for f in filenames
    )

    csv_size = len(df.to_csv(index=False).encode("utf-8"))
    return {
        "compression": compression,
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "compression_time": end_time - start_time,
        "original_size": csv_size,
        "compression_ratio": csv_size / file_size if file_size else float("inf"),
    }


def compress_with_other_formats(df: pd.DataFrame, base_path: str) -> Dict[str, Any]:
    results = {}

    csv_path = f"{base_path}.csv"
    start_time = time.time()
    df.to_csv(csv_path, index=False)
    csv_time = time.time() - start_time
    csv_size = get_file_size(csv_path)
    results["csv"] = {
        "format": "CSV",
        "file_size_bytes": csv_size,
        "file_size_mb": csv_size / (1024 * 1024),
        "compression_time": csv_time,
        "compression_ratio": 1.0,
    }

    gzip_path = f"{base_path}.csv.gz"
    start_time = time.time()
    with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    gzip_time = time.time() - start_time
    gzip_size = get_file_size(gzip_path)
    results["gzip"] = {
        "format": "GZIP CSV",
        "file_size_bytes": gzip_size,
        "file_size_mb": gzip_size / (1024 * 1024),
        "compression_time": gzip_time,
        "compression_ratio": csv_size / gzip_size,
    }

    pickle_path = f"{base_path}.pkl"
    start_time = time.time()
    with open(pickle_path, "wb") as f:
        pickle.dump(df, f)
    pickle_time = time.time() - start_time
    pickle_size = get_file_size(pickle_path)
    results["pickle"] = {
        "format": "Pickle",
        "file_size_bytes": pickle_size,
        "file_size_mb": pickle_size / (1024 * 1024),
        "compression_time": pickle_time,
        "compression_ratio": csv_size / pickle_size,
    }

    parquet_path = f"{base_path}.parquet"
    start_time = time.time()
    df.to_parquet(parquet_path, compression="snappy")
    parquet_time = time.time() - start_time
    parquet_size = get_file_size(parquet_path)
    results["parquet"] = {
        "format": "Parquet (Snappy)",
        "file_size_bytes": parquet_size,
        "file_size_mb": parquet_size / (1024 * 1024),
        "compression_time": parquet_time,
        "compression_ratio": csv_size / parquet_size,
    }

    return results


def benchmark_read_performance(df: pd.DataFrame, base_path: str) -> Dict[str, float]:
    results = {}

    # Lance read
    lance_path_zstd = f"{base_path}_zstd.lance"
    start_time = time.time()
    dataset_zstd = lance.dataset(lance_path_zstd)
    df_lance_zstd = dataset_zstd.to_table().to_pandas()
    results["zstd"] = time.time() - start_time

    lance_path_lz4 = f"{base_path}_lz4.lance"
    start_time = time.time()
    dataset_lz4 = lance.dataset(lance_path_lz4)
    df_lance_lz4 = dataset_lz4.to_table().to_pandas()
    results["lz4"] = time.time() - start_time

    lance_path_gzip = f"{base_path}_gzip.lance"
    start_time = time.time()
    dataset_gzip = lance.dataset(lance_path_gzip)
    df_lance_gzip = dataset_gzip.to_table().to_pandas()
    results["gzip_lance"] = time.time() - start_time

    # CSV read
    csv_path = f"{base_path}.csv"
    start_time = time.time()
    df_csv = pd.read_csv(csv_path)
    results["csv"] = time.time() - start_time

    # Gzip CSV read
    gzip_path = f"{base_path}.csv.gz"
    start_time = time.time()
    df_gzip = pd.read_csv(gzip_path, compression="gzip")
    results["gzip"] = time.time() - start_time

    # Parquet read
    parquet_path = f"{base_path}.parquet"
    start_time = time.time()
    df_parquet = pd.read_parquet(parquet_path)
    results["parquet"] = time.time() - start_time

    return results


def plot_results(compression_results: Dict[str, Any], read_results: Dict[str, float]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Use format from compression_results as the primary key
    formats = [
        res["format"] if "format" in res else res["compression"]
        for res in compression_results.values()
    ]
    ratios = [res["compression_ratio"] for res in compression_results.values()]
    sizes = [res["file_size_mb"] for res in compression_results.values()]
    times = [res["compression_time"] for res in compression_results.values()]

    # Remap read results to match the order and names of compression_results
    read_keys_map = {
        "zstd": "zstd",
        "lz4": "lz4",
        "gzip": "gzip_lance",  # Lance formats
        "csv": "csv",
        "gzip_csv": "gzip",
        "pickle": "pickle",
        "parquet": "parquet",  # Other formats
    }

    # Build read times in the same order as the other plots
    ordered_read_times = []
    read_formats = []

    # Map from compression_results keys to read_results keys
    key_map = {
        "zstd": "zstd",
        "lz4": "lz4",
        "gzip": "gzip_lance",
        "csv": "csv",
        "GZIP CSV": "gzip",
        "Parquet (Snappy)": "parquet",
        "Pickle": "pickle",
    }

    all_formats = [
        res.get("format", res.get("compression"))
        for res in compression_results.values()
    ]

    ordered_read_times = [
        read_results.get(key_map.get(f), float("nan")) for f in all_formats
    ]

    ax1.bar(all_formats, ratios, color="skyblue")
    ax1.set_title("Compression Ratio Comparison (Higher is Better)")
    ax1.set_ylabel("Compression Ratio (Original CSV Size / Compressed Size)")
    ax1.tick_params(axis="x", rotation=45, ha="right")

    ax2.bar(all_formats, sizes, color="lightcoral")
    ax2.set_title("File Size Comparison (Lower is Better)")
    ax2.set_ylabel("Size (MB)")
    ax2.tick_params(axis="x", rotation=45, ha="right")

    ax3.bar(all_formats, times, color="lightgreen")
    ax3.set_title("Write Time Comparison (Lower is Better)")
    ax3.set_ylabel("Time (s)")
    ax3.tick_params(axis="x", rotation=45, ha="right")

    ax4.bar(all_formats, ordered_read_times, color="gold")
    ax4.set_title("Read Time Comparison (Lower is Better)")
    ax4.set_ylabel("Time (s)")
    ax4.tick_params(axis="x", rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("compression_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    print("=== Lance Compression Benchmark ===")
    df = load_text_dataset()
    base_path = "text_data"

    lance_results = {}
    # Renaming 'gzip' to 'gzip_lance' to avoid key collision with 'GZIP CSV'
    lance_compressions = ["zstd", "lz4", "gzip"]

    print("\n=== Lance Compression Tests ===")
    for compression in lance_compressions:
        print(f"Testing {compression}...")
        uri = f"{base_path}_{compression}.lance"
        result = compress_with_lance(df, uri, compression)
        # Use a distinct key for lance's gzip
        key = "gzip_lance" if compression == "gzip" else compression
        lance_results[key] = result
        print(
            f"  {compression}: {result['file_size_mb']:.2f} MB, "
            f"ratio: {result['compression_ratio']:.2f}x, "
            f"time: {result['compression_time']:.2f}s"
        )

    print("\n=== Other Formats ===")
    other_results = compress_with_other_formats(df, base_path)

    # Use the format name as the key for consistency
    other_results_by_format = {v["format"]: v for k, v in other_results.items()}

    all_results = {**lance_results, **other_results_by_format}
    read_results = benchmark_read_performance(df, base_path)

    print("\n=== SUMMARY ===")
    print(
        f"{'Format':<20} {'Size (MB)':<10} {'Ratio':<10} {'Write Time (s)':<15} {'Read Time (s)':<10}"
    )
    print("-" * 75)

    # Create a mapping from the result key to the read_results key
    read_key_map = {
        "zstd": "zstd",
        "lz4": "lz4",
        "gzip_lance": "gzip_lance",
        "CSV": "csv",
        "GZIP CSV": "gzip",
        "Parquet (Snappy)": "parquet",
        "Pickle": "pickle",
    }

    for fmt, res in all_results.items():
        read_key = read_key_map.get(fmt, None)
        rtime_val = read_results.get(read_key) if read_key else None
        rtime_str = f"{rtime_val:.3f}" if rtime_val is not None else "N/A"

        print(
            f"{fmt:<20} {res['file_size_mb']:<10.2f} {res['compression_ratio']:<10.2f} "
            f"{res['compression_time']:<15.2f} {rtime_str:<10}"
        )

    with open("compression_results.json", "w") as f:
        json.dump(
            {
                "compression_results": all_results,
                "read_results": read_results,
                "dataset_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "total_text_length": df["text"].str.len().sum(),
                },
            },
            f,
            indent=2,
            default=str,
        )

    plot_results(all_results, read_results)
    print("\nResults saved to compression_results.json")
    print("Visualization saved to compression_comparison.png")

    cleanup_files = [
        f"{base_path}.csv",
        f"{base_path}.csv.gz",
        f"{base_path}.pkl",
        f"{base_path}.parquet",
    ]
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
    # Also clean up lance directories
    for comp in lance_compressions:
        dir_to_remove = f"{base_path}_{comp}.lance"
        if os.path.exists(dir_to_remove):
            import shutil

            shutil.rmtree(dir_to_remove)


if __name__ == "__main__":
    main()
