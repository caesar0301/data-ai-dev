# Lance Compression Benchmark Suite

This project provides comprehensive benchmarks comparing Lance compression algorithms on both text and vector data, with detailed performance analysis and visualizations.

## Features

- **Dual Benchmark Scripts**: Separate tests for text and vector data compression
- **Multiple Compression Algorithms**: Tests Lance with zstd (levels 1,3,6,9), lz4, and gzip compression
- **Format Comparison**: Compares Lance against CSV, GZIP CSV, Parquet, and NPZ formats
- **Performance Metrics**: Measures compression ratios, file sizes, write time, and read time
- **Visualization**: Generates comprehensive charts showing comparison results
- **Organized Output**: Separate output directories for text and vector results
- **Real Datasets**: Uses synthetic text data and SIFT vector dataset for realistic testing

## Requirements

### Option 1: Using Poetry (Recommended)

Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install dependencies:
```bash
poetry install
```

Activate the virtual environment:
```bash
poetry shell
```

### Option 2: Using pip

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Text Compression Benchmark

Run the text compression comparison script:

```bash
# With Poetry
poetry run python perf_comp_text.py

# With pip
python perf_comp_text.py
```

### Vector Compression Benchmark

Run the vector compression comparison script (requires SIFT dataset):

```bash
# With Poetry
poetry run python perf_comp_vector.py

# With pip
python perf_comp_vector.py
```

**Note**: The vector benchmark requires the SIFT dataset. Download it with:
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

## Development

### Setting up development environment

```bash
# Install with development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Run linting
poetry run flake8 .
```

### Adding new dependencies

```bash
# Add production dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```

## Output

Both scripts generate organized output in separate directories:

### Text Compression Results (`text_results/`)

1. **Console Output**: Detailed comparison table showing:
   - File sizes in MB
   - Compression ratios (vs CSV baseline)
   - Write time
   - Read time

2. **JSON Results**: `text_results/compression_results.json` with detailed metrics

3. **Visualization**: `text_results/compression_comparison.png` with 4 charts:
   - Compression ratio comparison
   - File size comparison  
   - Write time comparison
   - Read time comparison

### Vector Compression Results (`vector_results/`)

1. **Console Output**: Detailed comparison table showing:
   - File sizes in MB
   - Compression ratios (vs raw float32 data)
   - Write time
   - Read time

2. **JSON Results**: `vector_results/vector_compression_results.json` with detailed metrics and dataset info

3. **Visualization**: `vector_results/vector_compression_comparison.png` with 4 charts:
   - Compression ratio comparison
   - File size comparison  
   - Write time comparison
   - Read time comparison

## Datasets

### Text Dataset (`perf_comp_text.py`)

The script creates a synthetic text dataset with:
- 10,000 records
- Mixed text types (short, medium, long)
- Various text characteristics including repetition
- Metadata columns (id, category, length, word_count, timestamp)

### Vector Dataset (`perf_comp_vector.py`)

Uses the SIFT (Scale-Invariant Feature Transform) dataset:
- 100,000 vectors (sampled from 1M total)
- 128-dimensional float vectors
- Standard benchmark dataset for vector compression
- Includes metadata (vector_id, magnitude, timestamp)

## Compression Algorithms Tested

### Lance Formats:
- **zstd (levels 1,3,6,9)**: High compression ratio with varying speed/compression trade-offs
- **lz4**: Fast compression/decompression
- **gzip**: Standard compression

### Other Formats:
- **CSV**: Uncompressed baseline
- **GZIP CSV**: Standard text compression
- **Parquet (Snappy)**: Columnar format with compression
- **NPZ (Compressed)**: NumPy compressed format (vector benchmark only)

## Example Output

### Text Compression Results
```
=== Lance Compression Benchmark ===
Format               Size (MB)    Ratio      Write Time (s)  Read Time (s)  
--------------------------------------------------------------------------------
lance-zstd-level1    0.44         4.80       0.012           0.006          
lance-zstd-level3    0.44         4.80       0.010           0.004          
lance-zstd-level6    0.44         4.88       0.013           0.003          
lance-zstd-level9    0.43         4.91       0.012           0.003          
lance-lz4            1.93         1.10       0.010           0.003          
lance-gzip           1.93         1.10       0.010           0.003          
CSV                  2.13         1.00       0.000           0.012          
GZIP CSV             0.15         13.95      0.051           0.013          
Parquet (Snappy)     0.26         8.32       0.049           0.003          
```

### Vector Compression Results
```
=== Lance Vector Compression Benchmark (SIFT Dataset) ===
Format                    Size (MB)    Ratio      Write Time (s)  Read Time (s)  
------------------------------------------------------------------------------------------
lance-zstd-level1         100.04       0.49       0.091           0.044          
lance-zstd-level3         100.04       0.49       0.053           0.044          
lance-zstd-level6         100.04       0.49       0.058           0.044          
lance-zstd-level9         100.04       0.49       0.060           0.043          
lance-lz4                 100.04       0.49       0.058           0.047          
lance-gzip                100.04       0.49       0.073           0.046          
CSV                       60.87        0.80       3.386           0.644          
GZIP CSV                  13.38        3.65       50.028          0.619          
Parquet (Snappy)          12.91        3.78       0.167           0.030          
NPZ (Compressed)          14.34        3.40       4.032           0.003          
```

## Notes

- Both scripts automatically clean up temporary files after completion
- Results are organized in separate output directories for easy management
- The synthetic text dataset mimics real-world text characteristics for meaningful comparisons
- The SIFT vector dataset provides a standard benchmark for vector compression analysis 