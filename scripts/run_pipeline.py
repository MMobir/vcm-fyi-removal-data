#!/usr/bin/env python3
"""
Run the full Isometric data pipeline.

This script orchestrates:
1. Fetching raw CSV data from Isometric's registry
2. Processing the data into standardized format
3. Validating against schemas
4. Saving to Parquet files
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from removal_db_data.isometric import run_pipeline


def find_latest_csv(raw_dir: Path, prefix: str) -> Path | None:
    """
    Find the most recent CSV file with the given prefix.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw CSV files.
    prefix : str
        Prefix to match (e.g., 'isometric_issuances').

    Returns
    -------
    Path | None
        Path to the most recent matching file, or None if not found.
    """
    pattern = f"{prefix}*.csv"
    matching_files = list(raw_dir.glob(pattern))

    if not matching_files:
        return None

    # Sort by modification time, return newest
    return max(matching_files, key=lambda p: p.stat().st_mtime)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Isometric carbon removal data pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw",
        help="Directory containing raw CSV files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "output",
        help="Directory to save output Parquet files",
    )
    parser.add_argument(
        "--issuances",
        type=Path,
        help="Path to specific issuances CSV (overrides auto-detection)",
    )
    parser.add_argument(
        "--retirements",
        type=Path,
        help="Path to specific retirements CSV (overrides auto-detection)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip schema validation",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch fresh data before processing",
    )

    args = parser.parse_args()

    # Optionally fetch fresh data
    if args.fetch:
        print("Fetching fresh data from Isometric registry...")
        from fetch_isometric import fetch_all

        results = fetch_all(args.raw_dir, timestamp=False)
        if not results["issuances"]["success"]:
            print("ERROR: Failed to fetch issuances data")
            return 1

    # Find CSV files
    issuances_path = args.issuances
    retirements_path = args.retirements

    if issuances_path is None:
        issuances_path = find_latest_csv(args.raw_dir, "isometric_issuances")
        if issuances_path is None:
            print(f"ERROR: No issuances CSV found in {args.raw_dir}")
            print("Run with --fetch to download data first, or specify --issuances path")
            return 1
        print(f"Auto-detected issuances: {issuances_path}")

    if retirements_path is None:
        retirements_path = find_latest_csv(args.raw_dir, "isometric_retirements")
        if retirements_path:
            print(f"Auto-detected retirements: {retirements_path}")

    # Run the pipeline
    try:
        credits, projects = run_pipeline(
            issuances_path=issuances_path,
            retirements_path=retirements_path,
            output_dir=args.output_dir,
            validate_output=not args.skip_validation,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Output Summary")
        print("=" * 60)
        print(f"\nCredits ({len(credits):,} records):")
        print(f"  - Issuances: {(credits['transaction_type'] == 'issuance').sum():,}")
        print(f"  - Retirements: {(credits['transaction_type'] == 'retirement').sum():,}")
        print(f"  - Total quantity: {credits['quantity'].sum():,.0f} tonnes")

        print(f"\nProjects ({len(projects):,} records):")
        if "category" in projects.columns:
            print("  By category:")
            for cat, count in projects["category"].value_counts().items():
                print(f"    - {cat}: {count}")

        print(f"\nOutput files saved to: {args.output_dir.absolute()}")

        return 0

    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

