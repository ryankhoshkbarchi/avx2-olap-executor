#!/usr/bin/env python3
"""
Fetch NYC TLC (Taxi & Limousine Commission) Trip Record Data.

This dataset contains taxi trip records with uniform/normally distributed numeric data,
making it ideal for benchmarking OLAP operations on real-world data.

The data is already in Parquet format from the NYC TLC website, so we just download it.

Usage:
    python fetch_nyc_tlc_data.py [options]

Options:
    --year YEAR         Year of data to fetch (default: 2024)
    --month MONTH       Month of data to fetch (default: 1)
    --output PATH       Output file path (default: nyc_taxi_data.parquet)
    --taxi-type TYPE    Type of taxi: yellow, green, fhv, fhvhv (default: yellow)
"""

import argparse
import os
import urllib.request
import sys


def get_tlc_url(taxi_type: str, year: int, month: int) -> str:
    """
    Construct the download URL for NYC TLC trip data.

    Data is hosted on AWS CloudFront.
    URL pattern: https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{year}-{month:02d}.parquet
    """
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    filename = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
    return f"{base_url}/{filename}"


def download_with_progress(url: str, output_path: str) -> None:
    """Download a file with progress reporting."""

    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:5.1f}% ({mb_downloaded:.1f} / {mb_total:.1f} MB)")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloaded: {mb_downloaded:.1f} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print()  # Newline after progress
    except urllib.error.HTTPError as e:
        print(f"\nError: HTTP {e.code} - {e.reason}")
        print("The requested data may not be available. Try a different year/month combination.")
        raise


def validate_parquet(filepath: str) -> dict:
    """Validate the downloaded Parquet file and return metadata."""
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(filepath)
        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow

        return {
            'num_rows': metadata.num_rows,
            'num_columns': metadata.num_columns,
            'num_row_groups': metadata.num_row_groups,
            'schema': schema,
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
        }
    except ImportError:
        # PyArrow not available, just return file size
        return {
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
        }


def generate_join_tables(main_file: str, seed: int = 42):
    """Generate build and probe tables from the main NYC taxi data for HashJoin benchmarks."""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        import numpy as np
        import random

        print()
        print("Generating join tables for HashJoin benchmarks...")

        # Read the main parquet file to extract numeric data
        parquet_file = pq.ParquetFile(main_file)
        table = parquet_file.read()

        # Find integer columns suitable for join keys
        # For NYC taxi data, we'll use passenger_count or similar integer columns
        int_columns = []
        for field in table.schema:
            if pa.types.is_integer(field.type):
                int_columns.append(field.name)

        if not int_columns:
            print("  Warning: No integer columns found for join tables.")
            return

        print(f"  Found integer columns: {int_columns[:5]}...")  # Show first 5

        # Use the first suitable integer column
        key_column = int_columns[0]
        print(f"  Using column '{key_column}' for join keys")

        # Extract data
        key_data = table.column(key_column).to_numpy()

        # Get unique values for build table
        unique_keys = np.unique(key_data)
        num_build_rows = min(len(unique_keys), 100000)

        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        # Build table
        build_keys = unique_keys[:num_build_rows].astype(np.int32)
        build_values = np_rng.integers(0, 1000000, size=num_build_rows, dtype=np.int32)

        build_table = pa.table({
            'key': pa.array(build_keys),
            'value': pa.array(build_values)
        })

        # Probe table: sample from the original data
        num_probe_rows = min(1000000, len(key_data))
        probe_indices = np_rng.choice(len(key_data), size=num_probe_rows, replace=False)
        probe_keys = key_data[probe_indices].astype(np.int32)
        probe_data = np_rng.integers(0, 1000000, size=num_probe_rows, dtype=np.int32)

        probe_table = pa.table({
            'key': pa.array(probe_keys),
            'data': pa.array(probe_data)
        })

        # Write tables
        base_path = main_file.rsplit('.', 1)[0]
        build_path = f"{base_path}_build.parquet"
        probe_path = f"{base_path}_probe.parquet"

        pq.write_table(build_table, build_path, compression='snappy')
        pq.write_table(probe_table, probe_path, compression='snappy')

        build_size = os.path.getsize(build_path) / (1024 * 1024)
        probe_size = os.path.getsize(probe_path) / (1024 * 1024)

        print(f"  Build table: {build_path} ({build_size:.2f} MB, {num_build_rows:,} rows)")
        print(f"  Probe table: {probe_path} ({probe_size:.2f} MB, {num_probe_rows:,} rows)")

    except Exception as e:
        print(f"  Warning: Could not generate join tables: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch NYC TLC Trip Record Data (Parquet format)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year of data to fetch (default: 2024)')
    parser.add_argument('--month', type=int, default=1,
                        help='Month of data to fetch, 1-12 (default: 1)')
    parser.add_argument('--output', type=str, default='nyc_taxi_data.parquet',
                        help='Output file path (default: nyc_taxi_data.parquet)')
    parser.add_argument('--taxi-type', type=str, default='yellow',
                        choices=['yellow', 'green', 'fhv', 'fhvhv'],
                        help='Type of taxi data (default: yellow)')
    parser.add_argument('--generate-join-tables', action='store_true',
                        help='Also generate build/probe tables for HashJoin benchmarks')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for join table generation (default: 42)')

    args = parser.parse_args()

    # Validate month
    if args.month < 1 or args.month > 12:
        raise ValueError("Month must be between 1 and 12")

    # Validate year (TLC data available from 2009 onwards)
    if args.year < 2009:
        raise ValueError("TLC data is only available from 2009 onwards")

    print("=" * 60)
    print("NYC TLC Trip Record Data Fetcher")
    print("=" * 60)
    print(f"  Taxi type: {args.taxi_type}")
    print(f"  Period: {args.year}-{args.month:02d}")
    print(f"  Output: {args.output}")
    print()

    # Construct URL and download
    url = get_tlc_url(args.taxi_type, args.year, args.month)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    download_with_progress(url, args.output)

    # Validate and print summary
    print()
    print("Validating downloaded file...")
    info = validate_parquet(args.output)

    print()
    print("Download complete!")
    print("-" * 40)
    print(f"  File size: {info['file_size_mb']:.2f} MB")

    if 'num_rows' in info:
        print(f"  Rows: {info['num_rows']:,}")
        print(f"  Columns: {info['num_columns']}")
        print(f"  Row groups: {info['num_row_groups']}")
        print()
        print("Schema:")
        for field in info['schema']:
            print(f"    {field.name}: {field.type}")

    print()
    print("This data contains uniform/normally distributed numeric fields ideal for")
    print("benchmarking filter, aggregation, and join operations.")

    # Optionally generate join tables
    if args.generate_join_tables:
        generate_join_tables(args.output, args.seed)


if __name__ == '__main__':
    main()
