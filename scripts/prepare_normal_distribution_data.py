#!/usr/bin/env python3
"""
Prepare normally distributed dataset from NYC TLC taxi trip data.

The trip_distance column in NYC taxi data follows an approximately normal distribution,
making it ideal for benchmarking OLAP operators on normally distributed data.

This script:
1. Reads the NYC taxi parquet file
2. Extracts the trip_distance column
3. Converts to int32 (multiplied by 100 to preserve precision)
4. Generates join tables for HashJoin benchmarks
5. Saves as a properly formatted parquet file

Usage:
    python prepare_normal_distribution_data.py [options]

Options:
    --input PATH        Input NYC taxi parquet file (default: nyc_taxi_data.parquet)
    --output PATH       Output file path (default: normal_data.parquet)
    --column NAME       Column to extract (default: trip_distance)
    --max-rows N        Maximum rows to process (default: all)
    --seed N            Random seed for join tables (default: 42)
"""

import argparse
import os
import sys


def analyze_distribution(data, column_name: str):
    """Analyze and print distribution statistics."""
    import numpy as np
    from scipy import stats

    print(f"\nDistribution Analysis for '{column_name}':")
    print("-" * 40)
    print(f"  Count: {len(data):,}")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std Dev: {np.std(data):.2f}")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Max: {np.max(data):.2f}")

    # Calculate skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    print(f"  Skewness: {skewness:.3f} (0 = symmetric)")
    print(f"  Kurtosis: {kurtosis:.3f} (0 = normal)")

    # Normality test (for sample)
    if len(data) > 5000:
        sample = np.random.choice(data, size=5000, replace=False)
    else:
        sample = data
    _, p_value = stats.normaltest(sample)
    print(f"  Normality p-value: {p_value:.4f}")

    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def main():
    parser = argparse.ArgumentParser(
        description='Prepare normally distributed dataset from NYC taxi data')
    parser.add_argument('--input', type=str, default='nyc_taxi_data.parquet',
                        help='Input NYC taxi parquet file')
    parser.add_argument('--output', type=str, default='normal_data.parquet',
                        help='Output file path')
    parser.add_argument('--column', type=str, default='trip_distance',
                        help='Column to extract (default: trip_distance)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum rows to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--scale', type=int, default=100,
                        help='Scale factor for float->int conversion (default: 100)')

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Run 'python scripts/fetch_nyc_tlc_data.py' first to download the data.")
        sys.exit(1)

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        import numpy as np
        import random
    except ImportError as e:
        print(f"Error: Required package not found: {e}")
        print("Install with: pip install pyarrow numpy scipy")
        sys.exit(1)

    print("=" * 60)
    print("Normal Distribution Data Preparation")
    print("=" * 60)
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Column: {args.column}")
    print(f"  Scale factor: {args.scale}")
    print()

    # Read input parquet
    print("Reading input parquet file...")
    parquet_file = pq.ParquetFile(args.input)
    table = parquet_file.read()

    print(f"  Total rows: {table.num_rows:,}")
    print(f"  Columns: {table.num_columns}")

    # Check if column exists
    column_names = [field.name for field in table.schema]
    print(f"  Available columns: {column_names[:10]}...")

    if args.column not in column_names:
        print(f"\nError: Column '{args.column}' not found.")
        print(f"Available numeric columns:")
        for field in table.schema:
            if pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
                print(f"  - {field.name}: {field.type}")
        sys.exit(1)

    # Extract column
    print(f"\nExtracting column '{args.column}'...")
    column_data = table.column(args.column).to_numpy()

    # Handle null values
    valid_mask = ~np.isnan(column_data) if np.issubdtype(column_data.dtype, np.floating) else np.ones(len(column_data), dtype=bool)
    column_data = column_data[valid_mask]

    # Limit rows if specified
    if args.max_rows and len(column_data) > args.max_rows:
        print(f"  Limiting to {args.max_rows:,} rows...")
        np.random.seed(args.seed)
        indices = np.random.choice(len(column_data), size=args.max_rows, replace=False)
        column_data = column_data[indices]

    print(f"  Valid rows: {len(column_data):,}")

    # Analyze distribution
    try:
        from scipy import stats
        stats_info = analyze_distribution(column_data, args.column)
    except ImportError:
        print("  (Install scipy for detailed distribution analysis)")
        stats_info = None

    # Convert to int32 (scale to preserve precision)
    print(f"\nConverting to int32 (scale factor: {args.scale})...")
    int_data = (column_data * args.scale).astype(np.int32)

    # Remove any values that would overflow or be negative (for trip_distance, shouldn't happen)
    int_data = int_data[int_data >= 0]
    print(f"  Final row count: {len(int_data):,}")
    print(f"  Int value range: [{int_data.min()}, {int_data.max()}]")

    # Create output table
    print("\nWriting output parquet file...")
    output_table = pa.table({
        'IntColumn': pa.array(int_data)
    })

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pq.write_table(output_table, args.output, compression='snappy')
    output_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Written: {args.output} ({output_size:.2f} MB)")

    # Generate join tables
    print("\nGenerating join tables for HashJoin benchmarks...")

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # Build table: unique keys
    unique_values = np.unique(int_data)
    num_build_rows = min(len(unique_values), 100000)

    build_keys = unique_values[:num_build_rows].astype(np.int32)
    build_values = np_rng.integers(0, 1000000, size=num_build_rows, dtype=np.int32)

    build_table = pa.table({
        'key': pa.array(build_keys),
        'value': pa.array(build_values)
    })

    # Probe table: sample from the data (maintaining distribution)
    num_probe_rows = min(1000000, len(int_data))
    probe_keys = np_rng.choice(int_data, size=num_probe_rows, replace=True).astype(np.int32)
    probe_data_values = np_rng.integers(0, 1000000, size=num_probe_rows, dtype=np.int32)

    probe_table = pa.table({
        'key': pa.array(probe_keys),
        'data': pa.array(probe_data_values)
    })

    # Write join tables
    base_path = args.output.rsplit('.', 1)[0]
    build_path = f"{base_path}_build.parquet"
    probe_path = f"{base_path}_probe.parquet"

    pq.write_table(build_table, build_path, compression='snappy')
    pq.write_table(probe_table, probe_path, compression='snappy')

    build_size = os.path.getsize(build_path) / (1024 * 1024)
    probe_size = os.path.getsize(probe_path) / (1024 * 1024)

    print(f"  Build table: {build_path} ({build_size:.2f} MB, {num_build_rows:,} rows)")
    print(f"  Probe table: {probe_path} ({probe_size:.2f} MB, {num_probe_rows:,} rows)")

    # Summary
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Main data: {args.output} ({len(int_data):,} rows)")
    print(f"  Build table: {build_path}")
    print(f"  Probe table: {probe_path}")
    print()
    print("This dataset contains normally distributed data extracted from")
    print(f"the '{args.column}' column of NYC taxi trip records.")
    print()
    print("To run the benchmark:")
    print(f"  ./build/benchmark_normal {args.output}")


if __name__ == '__main__':
    main()
