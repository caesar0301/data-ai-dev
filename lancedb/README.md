# Lance Text Compression Comparison

This project demonstrates and compares Lance compression algorithms on text data, providing comprehensive benchmarks and visualizations.

## Features

- **Multiple Compression Algorithms**: Tests Lance with zstd, lz4, and gzip compression
- **Format Comparison**: Compares Lance against CSV, GZIP CSV, Pickle, and Parquet formats
- **Performance Metrics**: Measures compression ratios, file sizes, compression time, and read time
- **Visualization**: Generates comprehensive charts showing comparison results
- **Synthetic Dataset**: Creates realistic text data with varying characteristics for testing

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

Run the compression comparison script:

```bash
# With Poetry
poetry run python comp.py

# With pip
python comp.py
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

The script generates:

1. **Console Output**: Detailed comparison table showing:
   - File sizes in MB
   - Compression ratios
   - Compression time
   - Read time

2. **JSON Results**: `compression_results.json` with detailed metrics

3. **Visualization**: `compression_comparison.png` with 4 charts:
   - Compression ratio comparison
   - File size comparison  
   - Compression time comparison
   - Read time comparison

## Dataset

The script creates a synthetic text dataset with:
- 10,000 records
- Mixed text types (short, medium, long)
- Various text characteristics including repetition
- Metadata columns (id, category, length, word_count, timestamp)

## Compression Algorithms Tested

### Lance Formats:
- **zstd**: High compression ratio, good speed
- **lz4**: Fast compression/decompression
- **gzip**: Standard compression

### Other Formats:
- **CSV**: Uncompressed baseline
- **GZIP CSV**: Standard text compression
- **Pickle**: Python serialization
- **Parquet (Snappy)**: Columnar format with compression

## Example Output

```
=== COMPRESSION SUMMARY ===
Format               Size (MB)    Ratio     Comp Time    Read Time
---------------------------------------------------------------
zstd                 2.45         8.32      1.23         0.045s
lz4                  3.12         6.54      0.89         0.038s
gzip                 2.78         7.34      1.45         0.052s
csv                  20.45        1.00      0.12         0.089s
gzip                 5.67         3.61      0.34         0.067s
pickle               15.23        1.34      0.23         0.078s
parquet              4.56         4.48      0.67         0.041s
```

## Customization

You can modify the script to:
- Use your own text dataset by replacing the `load_text_dataset()` function
- Test different compression parameters
- Add more comparison formats
- Adjust the dataset size and characteristics

## Notes

- The script automatically cleans up temporary files after completion
- Results are saved for further analysis
- The synthetic dataset mimics real-world text characteristics for meaningful comparisons 