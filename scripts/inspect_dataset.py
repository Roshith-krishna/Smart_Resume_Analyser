"""Inspect a job-description CSV before vectorizing it.

The real Kaggle dataset's exact column names must never be assumed — this
script exists so the user can confirm them before running
``scripts/vectorize_jobs.py`` (which accepts the column names as
arguments; see its ``--title-col`` / ``--description-col`` / etc. flags).

Usage (from the project root):

    python -m scripts.inspect_dataset --csv data/raw/your_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a job-description CSV: columns, row count, sample rows, null counts."
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to the job-description CSV file.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of sample rows to print (default: 5).",
    )
    return parser


def inspect_dataset(csv_path: Path, sample_rows: int = 5) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        dataframe = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        # Some Kaggle exports are not UTF-8; fall back to a permissive encoding.
        dataframe = pd.read_csv(csv_path, encoding="latin-1")
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {exc}") from exc

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"File: {csv_path}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns ({len(dataframe.columns)}):")
    for column in dataframe.columns:
        print(f"  - {column!r} (dtype: {dataframe[column].dtype})")

    print("\nNull counts per column:")
    null_counts = dataframe.isna().sum()
    for column, null_count in null_counts.items():
        print(f"  - {column!r}: {null_count} / {len(dataframe)}")

    # Add duplicate diagnostics
    # Attempt to use 'Job Id', 'Job Title', 'Job Description', 'skills', 'Role', 'Responsibilities'
    if "Job Id" in dataframe.columns:
        dup_ids = dataframe["Job Id"].duplicated().sum()
        print(f"\nDuplicate 'Job Id's: {dup_ids}")
        
    if "Job Title" in dataframe.columns:
        unique_titles = dataframe["Job Title"].nunique()
        print(f"Unique 'Job Title's: {unique_titles}")

    # Estimate duplicate semantic records based on Title + Description + skills
    sig_cols = [c for c in ["Job Title", "Job Description", "skills", "Role", "Responsibilities"] if c in dataframe.columns]
    if sig_cols:
        # Create a naive signature
        sigs = dataframe[sig_cols].fillna("").astype(str).agg("".join, axis=1)
        num_dup_sigs = sigs.duplicated().sum()
        pct_dup = (num_dup_sigs / len(dataframe)) * 100 if len(dataframe) > 0 else 0
        print(f"Duplicate semantic records (based on {sig_cols}): {num_dup_sigs} ({pct_dup:.1f}%)")

    print(f"\nSample rows (first {min(sample_rows, len(dataframe))}):")
    with pd.option_context("display.max_colwidth", 80, "display.width", 120):
        print(dataframe.head(sample_rows).to_string())

    print("\nNext step: pass the correct column names to scripts/vectorize_jobs.py, e.g.")
    print(
        "  python -m scripts.vectorize_jobs --csv "
        f'"{csv_path}" --title-col "Job Title" --description-col "Job Description" --skills-col "skills"'
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        inspect_dataset(args.csv, args.sample_rows)
    except Exception as exc:  # surface a clean, single-line error to the user
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
