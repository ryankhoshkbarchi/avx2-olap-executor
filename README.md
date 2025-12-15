# OLAPExecutor

This is a OLAP SIMD based database query executor built in C++20. Used to benchmark
SIMD query execution on different types of analytical datasets (uniform, normal, skewed).

# Prerequisites

The following libraries need to be installed to use the OLAP executor

- The apache arrow library
- The apache parquet library

Python 3 is needed to run all the scripts.
To fetch the datasets and generate the plots the following python packages are needed

```
pip install pyarrow numpy matplotlib pandas
```

# Benchmarking Guide

## Overview

The benchmarking suite tests how data distribution affects SIMD performance for common OLAP operations:

| Distribution | Dataset Source | Characteristics |
|--------------|----------------|-----------------|
| **Uniform** | NYC Taxi (raw columns) | Evenly distributed values |
| **Normal** | NYC Taxi trip_distance | Gaussian/bell curve distribution |
| **Skewed** | Project Gutenberg | Zipfian distribution (few frequent values, many rare) |

## Quick Start

```bash
# 1. Build the project
cmake -S . -B build
cmake --build build -j8

# 2. Generate datasets
python scripts/fetch_nyc_tlc_data.py --generate-join-tables
python scripts/prepare_normal_distribution_data.py
python scripts/generate_gutenberg_skewed_data.py

# 3. Run benchmarks
./build/benchmark_uniform nyc_taxi_data.parquet
./build/benchmark_normal normal_data.parquet
./build/benchmark_skewed skewed_data.parquet

# 4. Plot results
python scripts/plot_benchmark_results.py benchmark_*_results.csv --output results.png
```

## Data Preparation Scripts

### `scripts/fetch_nyc_tlc_data.py`

Downloads NYC TLC (Taxi & Limousine Commission) trip record data, which is already in Parquet format.

```bash
python scripts/fetch_nyc_tlc_data.py [options]

Options:
  --year YEAR              Year of data (default: 2024)
  --month MONTH            Month 1-12 (default: 1)
  --taxi-type TYPE         yellow, green, fhv, fhvhv (default: yellow)
  --output PATH            Output file (default: nyc_taxi_data.parquet)
  --generate-join-tables   Also create build/probe tables for HashJoin
  --seed N                 Random seed (default: 42)
```

**Output files:**
- `nyc_taxi_data.parquet` - Main dataset
- `nyc_taxi_data_build.parquet` - HashJoin build table (if `--generate-join-tables`)
- `nyc_taxi_data_probe.parquet` - HashJoin probe table (if `--generate-join-tables`)

### `scripts/prepare_normal_distribution_data.py`

Extracts normally distributed data from NYC taxi trip_distance column.

```bash
python scripts/prepare_normal_distribution_data.py [options]

Options:
  --input PATH      Input NYC taxi parquet (default: nyc_taxi_data.parquet)
  --output PATH     Output file (default: normal_data.parquet)
  --column NAME     Column to extract (default: trip_distance)
  --max-rows N      Limit rows (default: all)
  --scale N         Float-to-int multiplier (default: 100)
  --seed N          Random seed (default: 42)
```

**Output files:**
- `normal_data.parquet` - Main dataset with IntColumn
- `normal_data_build.parquet` - HashJoin build table
- `normal_data_probe.parquet` - HashJoin probe table

### `scripts/generate_gutenberg_skewed_data.py`

Generates Zipfian-distributed data from Project Gutenberg text corpora based on word frequency ranks.

```bash
python scripts/generate_gutenberg_skewed_data.py [options]

Options:
  --books N        Number of books to download (default: 5)
  --rows N         Target row count (default: 10,000,000)
  --output PATH    Output file (default: skewed_data.parquet)
  --seed N         Random seed (default: 42)
```

**Output files:**
- `skewed_data.parquet` - Main dataset with Zipfian distribution
- `skewed_data_build.parquet` - HashJoin build table
- `skewed_data_probe.parquet` - HashJoin probe table

## Benchmark Executables

All executables share the same interface:

```bash
./build/benchmark_<type> <parquet_file> [csv_output] [num_runs]
```

### `benchmark_uniform`

Tests uniformly distributed data.

```bash
./build/benchmark_uniform nyc_taxi_data.parquet benchmark_uniform_results.csv 10
```

**Operations tested:**
- Filter (threshold > 5000)
- Sum aggregation
- Filter + Sum pipeline
- HashJoin (Linear Probing)
- HashJoin (Cuckoo Hashing)

### `benchmark_normal`

Tests normally distributed data with selectivity analysis.

```bash
./build/benchmark_normal normal_data.parquet benchmark_normal_results.csv 10
```

**Operations tested:**
- Filter at 3 selectivity levels:
  - Low selectivity (>100): keeps ~80% of data
  - Medium selectivity (>300): keeps ~50% of data
  - High selectivity (>1000): keeps ~10% of data
- Sum aggregation
- Filter + Sum pipeline
- HashJoin (Linear Probing)
- HashJoin (Cuckoo Hashing)

### `benchmark_skewed`

Tests Zipfian-distributed data with selectivity analysis.

```bash
./build/benchmark_skewed skewed_data.parquet benchmark_skewed_results.csv 10
```

**Operations tested:**
- Filter at 3 selectivity levels:
  - High selectivity (>10): filters common words
  - Medium selectivity (>100): filters mid-frequency
  - Low selectivity (>1000): filters rare words
- Sum aggregation
- Filter + Sum pipeline
- HashJoin (Linear Probing)
- HashJoin (Cuckoo Hashing)

## CSV Output Format

All benchmarks output CSV files with the following columns:

| Column | Description |
|--------|-------------|
| Dataset | Distribution type (Uniform, Normal, Skewed) |
| Operation | Operation name (Filter, Sum, HashJoin_Linear, etc.) |
| Mode | Execution mode (Scalar, AVX2) |
| TaskName | Full benchmark task name |
| MeanNs | Mean execution time in nanoseconds |
| MedianNs | Median execution time in nanoseconds |
| StdDevNs | Standard deviation in nanoseconds |
| MinNs | Minimum execution time |
| MaxNs | Maximum execution time |
| InputRows | Number of input rows |
| OutputRows | Number of output rows |
| ThroughputGBps | Throughput in GB/s |

## Plotting Results

### `scripts/plot_benchmark_results.py`

Generates comprehensive visualization of benchmark results.

```bash
python scripts/plot_benchmark_results.py [csv_files...] [options]

Options:
  --output PATH    Output image file (default: benchmark_results.png)
  --show           Display interactive plot window
  --dpi N          Image resolution (default: 150)
  --style STYLE    dark, light, paper (default: light)
  --summary        Print text summary to console
```

**Examples:**

```bash
# Plot all results
python scripts/plot_benchmark_results.py benchmark_*_results.csv

# High-res output for publication
python scripts/plot_benchmark_results.py *.csv --output fig.png --dpi 300 --style paper

# Interactive viewing with summary
python scripts/plot_benchmark_results.py *.csv --show --summary
```

### Generated Plots (8 subplots)

1. **Throughput Comparison** - GB/s for Scalar vs AVX2 (excludes HashJoin)
2. **Speedup Ratios** - How much faster AVX2 is vs Scalar
3. **Execution Time** - Median time in milliseconds
4. **Dataset Comparison** - Average speedup per distribution
5. **Operation Breakdown** - Speedup by operation category
6. **Variability** - Coefficient of variation (consistency)
7. **HashJoin Comparison** - Join speedup by dataset and algorithm
8. **Selectivity Comparison** - Filter speedup across distributions

### Requirements

```bash
pip install pandas matplotlib numpy
```

## Understanding the Results

### Why Test Different Distributions?

SIMD performance can vary significantly based on data distribution:

- **Uniform data**: Predictable branching, good cache utilization
- **Normal data**: Moderate branching near mean, good for typical workloads
- **Skewed data**: Unpredictable branching, potential cache misses on rare values

### Key Metrics

- **Speedup > 1.0**: AVX2 is faster than Scalar
- **Throughput (GB/s)**: Higher is better; measures data processing rate
- **Coefficient of Variation**: Lower indicates more consistent performance

### Plotting errors

Install required Python packages:
```bash
pip install pandas matplotlib numpy
```
