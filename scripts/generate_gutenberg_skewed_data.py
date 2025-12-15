#!/usr/bin/env python3
"""
Generate skewed dataset from Project Gutenberg text corpora.

Natural language text follows Zipf's law - word frequencies are highly skewed,
with a few words appearing very frequently and most words appearing rarely.
This makes it ideal for benchmarking OLAP operators on non-uniform data.

This script:
1. Downloads text from Project Gutenberg
2. Tokenizes and counts word frequencies
3. Generates a Parquet file with skewed integer data based on word frequency ranks

Usage:
    python generate_gutenberg_skewed_data.py [options]

Options:
    --books N           Number of books to download (default: 5)
    --output PATH       Output file path (default: skewed_data.parquet)
    --rows N            Target number of rows (default: 10000000)
    --seed N            Random seed (default: 42)
"""

import argparse
import os
import re
import urllib.request
import sys
from collections import Counter
import random


# Popular books from Project Gutenberg (public domain)
# Format: (book_id, title)
GUTENBERG_BOOKS = [
    (1342, "Pride and Prejudice"),
    (11, "Alice's Adventures in Wonderland"),
    (84, "Frankenstein"),
    (1661, "Sherlock Holmes"),
    (98, "A Tale of Two Cities"),
    (2701, "Moby Dick"),
    (1952, "The Yellow Wallpaper"),
    (74, "Tom Sawyer"),
    (76, "Huckleberry Finn"),
    (345, "Dracula"),
    (1232, "The Prince"),
    (46, "A Christmas Carol"),
    (174, "The Picture of Dorian Gray"),
    (1400, "Great Expectations"),
    (5200, "Metamorphosis"),
    (2542, "A Doll's House"),
    (2591, "Grimm's Fairy Tales"),
    (1080, "A Modest Proposal"),
    (219, "Heart of Darkness"),
    (844, "The Importance of Being Earnest"),
]


def get_gutenberg_url(book_id: int) -> str:
    """Get the plain text URL for a Gutenberg book."""
    return f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"


def download_book(book_id: int, title: str) -> str:
    """Download a book from Project Gutenberg."""
    url = get_gutenberg_url(book_id)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            # Try UTF-8 first, fall back to latin-1
            try:
                text = response.read().decode('utf-8')
            except UnicodeDecodeError:
                response.seek(0)
                text = response.read().decode('latin-1')
        return text
    except urllib.error.HTTPError:
        # Try alternative URL format
        alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            with urllib.request.urlopen(alt_url, timeout=30) as response:
                try:
                    text = response.read().decode('utf-8')
                except UnicodeDecodeError:
                    response.seek(0)
                    text = response.read().decode('latin-1')
            return text
        except Exception as e:
            print(f"  Warning: Could not download '{title}' (ID: {book_id}): {e}")
            return ""


def tokenize(text: str) -> list:
    """
    Tokenize text into words.
    Converts to lowercase and extracts alphabetic words.
    """
    # Remove Gutenberg header/footer markers
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF"]

    for marker in start_markers:
        if marker in text:
            idx = text.find(marker)
            newline_idx = text.find('\n', idx)
            if newline_idx != -1:
                text = text[newline_idx:]
            break

    for marker in end_markers:
        if marker in text:
            idx = text.find(marker)
            text = text[:idx]
            break

    # Extract words (alphabetic only, lowercase)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def create_skewed_distribution(word_counts: Counter, target_rows: int, seed: int) -> list:
    """
    Create skewed integer data based on word frequency distribution.

    Maps each unique word to an integer (its frequency rank).
    Generates data by sampling words according to their actual frequencies,
    producing a naturally skewed Zipfian distribution.
    """
    rng = random.Random(seed)

    # Sort words by frequency (most common first)
    sorted_words = word_counts.most_common()

    # Create word -> rank mapping (1 = most common)
    word_to_rank = {word: rank + 1 for rank, (word, _) in enumerate(sorted_words)}

    # Create sampling weights (use actual frequencies)
    total_count = sum(word_counts.values())
    words = list(word_counts.keys())
    weights = [word_counts[w] / total_count for w in words]

    print(f"  Unique words: {len(words):,}")
    print(f"  Total word occurrences: {total_count:,}")
    print(f"  Top 10 words: {[w for w, _ in sorted_words[:10]]}")

    # Generate data by sampling
    print(f"  Generating {target_rows:,} rows with Zipfian distribution...")

    # Use weighted random choice to generate data
    data = []
    batch_size = 100000

    for i in range(0, target_rows, batch_size):
        batch_count = min(batch_size, target_rows - i)
        sampled_words = rng.choices(words, weights=weights, k=batch_count)
        batch_data = [word_to_rank[w] for w in sampled_words]
        data.extend(batch_data)

        if (i + batch_count) % 1000000 == 0:
            print(f"    Generated {i + batch_count:,} / {target_rows:,} rows")

    return data


def analyze_skewness(data: list) -> dict:
    """Analyze the skewness of the generated data."""
    import statistics

    counts = Counter(data)
    values = list(data)

    # Calculate basic statistics
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0

    # Calculate skewness manually
    n = len(values)
    if stdev > 0:
        skewness = sum((x - mean) ** 3 for x in values) / (n * stdev ** 3)
    else:
        skewness = 0

    # Count distribution characteristics
    unique_values = len(counts)
    top_10_pct = sum(c for _, c in counts.most_common(10)) / len(data) * 100

    return {
        'mean': mean,
        'median': median,
        'stdev': stdev,
        'skewness': skewness,
        'unique_values': unique_values,
        'top_10_coverage_pct': top_10_pct,
        'min': min(values),
        'max': max(values)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate skewed dataset from Project Gutenberg')
    parser.add_argument('--books', type=int, default=5,
                        help='Number of books to download (default: 5)')
    parser.add_argument('--output', type=str, default='skewed_data.parquet',
                        help='Output file path (default: skewed_data.parquet)')
    parser.add_argument('--rows', type=int, default=10000000,
                        help='Target number of rows (default: 10000000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Limit books to available
    num_books = min(args.books, len(GUTENBERG_BOOKS))

    print("=" * 60)
    print("Project Gutenberg Skewed Dataset Generator")
    print("=" * 60)
    print(f"  Books to download: {num_books}")
    print(f"  Target rows: {args.rows:,}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print()

    # Download books and collect words
    print("Downloading books from Project Gutenberg...")
    all_words = []

    for i, (book_id, title) in enumerate(GUTENBERG_BOOKS[:num_books]):
        print(f"  [{i+1}/{num_books}] Downloading '{title}'...")
        text = download_book(book_id, title)
        if text:
            words = tokenize(text)
            all_words.extend(words)
            print(f"    Extracted {len(words):,} words")

    if not all_words:
        print("Error: No text could be downloaded. Check your internet connection.")
        sys.exit(1)

    print()
    print(f"Total words collected: {len(all_words):,}")
    print()

    # Count word frequencies
    print("Counting word frequencies...")
    word_counts = Counter(all_words)

    # Generate skewed data
    print()
    print("Generating skewed integer data...")
    data = create_skewed_distribution(word_counts, args.rows, args.seed)

    # Analyze skewness
    print()
    print("Analyzing distribution...")
    stats = analyze_skewness(data)

    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std Dev: {stats['stdev']:.2f}")
    print(f"  Skewness: {stats['skewness']:.2f}")
    print(f"  Unique values: {stats['unique_values']:,}")
    print(f"  Top 10 values cover: {stats['top_10_coverage_pct']:.1f}% of data")
    print(f"  Value range: [{stats['min']}, {stats['max']}]")

    # Write to Parquet
    print()
    print("Writing Parquet files...")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import numpy as np

        # Convert to numpy array for efficiency
        int_data = np.array(data, dtype=np.int32)

        # Create Arrow table with single column
        table = pa.table({
            'IntColumn': pa.array(int_data)
        })

        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write main data file with compression
        pq.write_table(table, args.output, compression='snappy')

        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)

        print(f"  Main file: {args.output} ({file_size_mb:.2f} MB)")

        # Generate join tables for HashJoin benchmarks
        print()
        print("Generating join tables for HashJoin benchmarks...")

        rng = random.Random(args.seed + 100)

        # Build table: unique keys from 1 to num_unique_values
        # Use the unique values from our skewed data as keys
        unique_values = list(set(data))
        num_build_rows = min(len(unique_values), 100000)  # Cap at 100K rows

        build_keys = np.array(sorted(unique_values)[:num_build_rows], dtype=np.int32)
        build_values = np.array([rng.randint(0, 1000000) for _ in range(num_build_rows)], dtype=np.int32)

        build_table = pa.table({
            'key': pa.array(build_keys),
            'value': pa.array(build_values)
        })

        # Determine output paths for join tables
        base_path = args.output.rsplit('.', 1)[0]
        build_path = f"{base_path}_build.parquet"
        probe_path = f"{base_path}_probe.parquet"

        pq.write_table(build_table, build_path, compression='snappy')
        build_size_mb = os.path.getsize(build_path) / (1024 * 1024)
        print(f"  Build table: {build_path} ({build_size_mb:.2f} MB, {num_build_rows:,} rows)")

        # Probe table: sample from the skewed data (so it has skewed key distribution)
        # Use a subset of the main data as probe keys
        num_probe_rows = min(1000000, args.rows)

        # Sample probe keys from the skewed data (maintaining skewed distribution)
        probe_keys = np.array(rng.choices(data, k=num_probe_rows), dtype=np.int32)
        probe_data = np.array([rng.randint(0, 1000000) for _ in range(num_probe_rows)], dtype=np.int32)

        probe_table = pa.table({
            'key': pa.array(probe_keys),
            'data': pa.array(probe_data)
        })

        pq.write_table(probe_table, probe_path, compression='snappy')
        probe_size_mb = os.path.getsize(probe_path) / (1024 * 1024)
        print(f"  Probe table: {probe_path} ({probe_size_mb:.2f} MB, {num_probe_rows:,} rows)")

        print()
        print("=" * 60)
        print("Done!")
        print("=" * 60)
        print(f"  Main data file: {args.output}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Rows: {args.rows:,}")
        print()
        print(f"  Join tables generated:")
        print(f"    Build: {build_path} ({num_build_rows:,} unique keys)")
        print(f"    Probe: {probe_path} ({num_probe_rows:,} rows, skewed key distribution)")
        print()
        print("This dataset has a Zipfian (highly skewed) distribution where")
        print("small values (frequent words) appear much more often than large")
        print("values (rare words). This is ideal for testing operator performance")
        print("on non-uniform data distributions.")

    except ImportError:
        print("Error: PyArrow is required. Install with: pip install pyarrow")
        sys.exit(1)


if __name__ == '__main__':
    main()
