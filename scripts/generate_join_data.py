#!/usr/bin/env python3
"""
Generate test data for HashJoin operator benchmarking.

Creates two Parquet files:
1. build_table.parquet - Table with unique keys (the "dimension" table)
2. probe_table.parquet - Table with foreign keys (the "fact" table)

Usage:
    python generate_join_data.py [options]

Options:
    --build-rows N      Number of rows in build table (default: 100000)
    --probe-rows N      Number of rows in probe table (default: 1000000)
    --match-ratio F     Fraction of probe keys that match build keys (default: 0.8)
    --output-dir PATH   Output directory (default: current directory)
    --seed N            Random seed for reproducibility (default: 42)
"""

import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os


def generate_build_table(num_rows: int, seed: int) -> pa.Table:
    """
    Generate the build (dimension) table with unique keys.

    Schema: [key: int32, value: int32]
    - key: unique integers from 1 to num_rows
    - value: random payload data
    """
    rng = np.random.default_rng(seed)

    # Unique keys: 1 to num_rows
    keys = np.arange(1, num_rows + 1, dtype=np.int32)

    # Random payload values
    values = rng.integers(0, 1000000, size=num_rows, dtype=np.int32)

    table = pa.table({
        'key': pa.array(keys),
        'value': pa.array(values)
    })

    return table


def generate_probe_table(num_rows: int, build_keys: np.ndarray,
                         match_ratio: float, seed: int) -> pa.Table:
    """
    Generate the probe (fact) table with foreign keys.

    Schema: [key: int32, data: int32]
    - key: foreign key referencing build table (match_ratio fraction will match)
    - data: random payload data
    """
    rng = np.random.default_rng(seed + 1)  # Different seed for probe table

    num_matching = int(num_rows * match_ratio)
    num_non_matching = num_rows - num_matching

    # Generate matching keys: randomly sample from build keys
    matching_keys = rng.choice(build_keys, size=num_matching, replace=True)

    # Generate non-matching keys: use negative values (build keys are positive)
    non_matching_keys = -rng.integers(1, 1000000, size=num_non_matching, dtype=np.int32)

    # Combine and shuffle
    all_keys = np.concatenate([matching_keys, non_matching_keys]).astype(np.int32)
    rng.shuffle(all_keys)

    # Random payload data
    data = rng.integers(0, 1000000, size=num_rows, dtype=np.int32)

    table = pa.table({
        'key': pa.array(all_keys),
        'data': pa.array(data)
    })

    return table


def main():
    parser = argparse.ArgumentParser(
        description='Generate test data for HashJoin benchmarking')
    parser.add_argument('--build-rows', type=int, default=100000,
                        help='Number of rows in build table (default: 100000)')
    parser.add_argument('--probe-rows', type=int, default=1000000,
                        help='Number of rows in probe table (default: 1000000)')
    parser.add_argument('--match-ratio', type=float, default=0.8,
                        help='Fraction of probe keys that match (default: 0.8)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory (default: current directory)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Validate arguments
    if args.match_ratio < 0 or args.match_ratio > 1:
        raise ValueError("match-ratio must be between 0 and 1")

    print(f"Generating join test data...")
    print(f"  Build rows: {args.build_rows:,}")
    print(f"  Probe rows: {args.probe_rows:,}")
    print(f"  Match ratio: {args.match_ratio:.1%}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Seed: {args.seed}")
    print()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate build table
    print("Generating build table...")
    build_table = generate_build_table(args.build_rows, args.seed)
    build_path = os.path.join(args.output_dir, 'build_table.parquet')
    pq.write_table(build_table, build_path)
    print(f"  Written to: {build_path}")
    print(f"  Size: {os.path.getsize(build_path) / 1024 / 1024:.2f} MB")

    # Extract build keys for probe table generation
    build_keys = build_table.column('key').to_numpy()

    # Generate probe table
    print("Generating probe table...")
    probe_table = generate_probe_table(args.probe_rows, build_keys,
                                       args.match_ratio, args.seed)
    probe_path = os.path.join(args.output_dir, 'probe_table.parquet')
    pq.write_table(probe_table, probe_path)
    print(f"  Written to: {probe_path}")
    print(f"  Size: {os.path.getsize(probe_path) / 1024 / 1024:.2f} MB")

    # Print summary statistics
    print()
    print("Summary:")
    print(f"  Build table: {args.build_rows:,} rows, unique keys 1..{args.build_rows}")
    expected_matches = int(args.probe_rows * args.match_ratio)
    print(f"  Probe table: {args.probe_rows:,} rows, ~{expected_matches:,} matching keys")
    print()
    print("Done!")


if __name__ == '__main__':
    main()
