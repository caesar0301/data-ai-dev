import lance
import pandas as pd
import base64
import time
from typing import Dict, Any

def read_ubm_lance_dataset(dataset_path: str = "ubm_lance_zstd_1") -> pd.DataFrame:
    """
    Read UBM data from Lance dataset and convert back to original format.
    
    Args:
        dataset_path: Path to the Lance dataset
        
    Returns:
        DataFrame with the UBM data
    """
    print(f"Reading Lance dataset from: {dataset_path}")
    
    # Open the Lance dataset
    dataset = lance.dataset(dataset_path)
    
    # Get dataset info
    print(f"Dataset schema: {dataset.schema}")
    print(f"Number of fragments: {len(dataset.get_fragments())}")
    print(f"Total rows: {dataset.count_rows()}")
    
    # Read the entire dataset
    start_time = time.time()
    table = dataset.to_table()
    read_time = time.time() - start_time
    
    print(f"Read time: {read_time:.4f} seconds")
    
    # Convert to pandas DataFrame
    df = table.to_pandas()
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def analyze_ubm_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the UBM data and provide statistics.
    
    Args:
        df: DataFrame containing UBM data
        
    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*60)
    print("UBM DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    total_records = len(df)
    total_original_size = df['total_size'].sum()
    total_compressed_size = df['key'].str.len().sum() + df['value'].str.len().sum()
    
    print(f"Total records: {total_records}")
    print(f"Original size: {total_original_size / (1024*1024):.2f} MB")
    print(f"Compressed size (base64): {total_compressed_size / (1024*1024):.2f} MB")
    
    # Key statistics
    print(f"\nKey Statistics:")
    print(f"  Min size: {df['key_size'].min()} bytes")
    print(f"  Max size: {df['key_size'].max()} bytes")
    print(f"  Mean size: {df['key_size'].mean():.1f} bytes")
    print(f"  Median size: {df['key_size'].median():.1f} bytes")
    
    # Value statistics
    print(f"\nValue Statistics:")
    print(f"  Min size: {df['value_size'].min()} bytes")
    print(f"  Max size: {df['value_size'].max()} bytes")
    print(f"  Mean size: {df['value_size'].mean():.1f} bytes")
    print(f"  Median size: {df['value_size'].median():.1f} bytes")
    
    # Total size statistics
    print(f"\nTotal Size Statistics:")
    print(f"  Min size: {df['total_size'].min()} bytes")
    print(f"  Max size: {df['total_size'].max()} bytes")
    print(f"  Mean size: {df['total_size'].mean():.1f} bytes")
    print(f"  Median size: {df['total_size'].median():.1f} bytes")
    
    # Sample some records
    print(f"\nSample Records:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"  Record {i}:")
        print(f"    ID: {row['id']}")
        print(f"    Key size: {row['key_size']} bytes")
        print(f"    Value size: {row['value_size']} bytes")
        print(f"    Total size: {row['total_size']} bytes")
        print(f"    Key (first 50 chars): {row['key'][:50]}...")
        print(f"    Value (first 50 chars): {row['value'][:50]}...")
        print()
    
    return {
        'total_records': total_records,
        'original_size_mb': total_original_size / (1024*1024),
        'compressed_size_mb': total_compressed_size / (1024*1024),
        'key_stats': {
            'min': int(df['key_size'].min()),
            'max': int(df['key_size'].max()),
            'mean': float(df['key_size'].mean()),
            'median': float(df['key_size'].median())
        },
        'value_stats': {
            'min': int(df['value_size'].min()),
            'max': int(df['value_size'].max()),
            'mean': float(df['value_size'].mean()),
            'median': float(df['value_size'].median())
        },
        'total_stats': {
            'min': int(df['total_size'].min()),
            'max': int(df['total_size'].max()),
            'mean': float(df['total_size'].mean()),
            'median': float(df['total_size'].median())
        }
    }

def decode_binary_data(df: pd.DataFrame, sample_size: int = 5) -> None:
    """
    Decode base64 data back to binary and show sample.
    
    Args:
        df: DataFrame with base64 encoded data
        sample_size: Number of samples to decode and display
    """
    print(f"\n" + "="*60)
    print("DECODING BINARY DATA SAMPLES")
    print("="*60)
    
    for i in range(min(sample_size, len(df))):
        row = df.iloc[i]
        
        try:
            # Decode key and value from base64
            key_binary = base64.b64decode(row['key'])
            value_binary = base64.b64decode(row['value'])
            
            print(f"Record {i}:")
            print(f"  ID: {row['id']}")
            print(f"  Key (binary, first 20 bytes): {key_binary[:20].hex()}")
            print(f"  Value (binary, first 20 bytes): {value_binary[:20].hex()}")
            print(f"  Key length: {len(key_binary)} bytes")
            print(f"  Value length: {len(value_binary)} bytes")
            
            # Try to interpret as text if possible
            try:
                key_text = key_binary.decode('utf-8', errors='ignore')
                print(f"  Key as text: {key_text}")
            except:
                print(f"  Key: Binary data (not UTF-8)")
            
            print()
            
        except Exception as e:
            print(f"Error decoding record {i}: {e}")
            print()

def query_lance_data(dataset_path: str = "ubm_lance_zstd_1") -> None:
    """
    Demonstrate querying capabilities of Lance dataset.
    
    Args:
        dataset_path: Path to the Lance dataset
    """
    print(f"\n" + "="*60)
    print("LANCE QUERYING CAPABILITIES")
    print("="*60)
    
    dataset = lance.dataset(dataset_path)
    
    # Get basic stats
    table = dataset.to_table()
    mean_key_size = int(table['key_size'].to_pandas().mean())
    mean_value_size = int(table['value_size'].to_pandas().mean())
    
    print(f"Mean key size: {mean_key_size} bytes")
    print(f"Mean value size: {mean_value_size} bytes")
    
    # Query with filters
    print(f"\nQuerying records with key_size > {mean_key_size}:")
    start_time = time.time()
    filtered_table = dataset.to_table(filter=f"key_size > {mean_key_size}")
    query_time = time.time() - start_time
    
    print(f"Query time: {query_time:.4f} seconds")
    print(f"Records found: {len(filtered_table)}")
    
    # Query with value size filter
    print(f"\nQuerying records with value_size > {mean_value_size}:")
    start_time = time.time()
    large_values = dataset.to_table(filter=f"value_size > {mean_value_size}")
    query_time = time.time() - start_time
    
    print(f"Query time: {query_time:.4f} seconds")
    print(f"Records found: {len(large_values)}")
    
    # Query with combined filter
    print(f"\nQuerying records with both key_size > {mean_key_size} AND value_size > {mean_value_size}:")
    start_time = time.time()
    combined_filter = dataset.to_table(
        filter=f"key_size > {mean_key_size} AND value_size > {mean_value_size}"
    )
    query_time = time.time() - start_time
    
    print(f"Query time: {query_time:.4f} seconds")
    print(f"Records found: {len(combined_filter)}")

def main():
    """
    Main function to read and analyze UBM Lance data.
    """
    print("UBM Lance Data Reader")
    print("="*50)
    
    try:
        # Read the dataset
        df = read_ubm_lance_dataset("ubm_lance_zstd_1")
        
        # Analyze the data
        analysis = analyze_ubm_data(df)
        
        # Decode sample binary data
        decode_binary_data(df)
        
        # Demonstrate querying capabilities
        query_lance_data("ubm_lance_zstd_1")
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Successfully read {analysis['total_records']} records")
        print(f"Original data size: {analysis['original_size_mb']:.2f} MB")
        print(f"Compressed data size: {analysis['compressed_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"Error reading Lance dataset: {e}")
        print("Make sure the dataset exists and is accessible.")

if __name__ == "__main__":
    main() 