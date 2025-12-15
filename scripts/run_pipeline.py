#!/usr/bin/env python3
"""
Run the full removal data pipeline for all registries.

This script orchestrates:
1. Processing raw CSV data from each registry (Isometric, Puro.earth)
2. Validating against schemas
3. Saving to Parquet files
4. Combining into unified output files
"""

import argparse
import sys
from pathlib import Path
from glob import glob

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from removal_db_data import isometric, puro


def find_puro_files(raw_dir: Path) -> dict:
    """
    Find Puro.earth CSV files in the raw directory.
    
    Puro files are named like:
    - Puro_Earth_Registry-Issuance_exports-Mon Dec 15 2025.csv
    - Puro_Earth_Registry-Project_exports-Mon Dec 15 2025.csv
    - Puro_Earth_Registry-Retirement_exports-Mon Dec 15 2025.csv
    """
    files = {}
    
    # Find issuances
    issuance_pattern = list(raw_dir.glob("Puro_Earth_Registry-Issuance*.csv"))
    if issuance_pattern:
        files["issuances"] = max(issuance_pattern, key=lambda p: p.stat().st_mtime)
    
    # Find projects
    project_pattern = list(raw_dir.glob("Puro_Earth_Registry-Project*.csv"))
    if project_pattern:
        files["projects"] = max(project_pattern, key=lambda p: p.stat().st_mtime)
    
    # Find retirements
    retirement_pattern = list(raw_dir.glob("Puro_Earth_Registry-Retirement*.csv"))
    if retirement_pattern:
        files["retirements"] = max(retirement_pattern, key=lambda p: p.stat().st_mtime)
    
    return files


def find_isometric_files(raw_dir: Path) -> dict:
    """Find Isometric CSV files in the raw directory."""
    files = {}
    
    issuances = raw_dir / "isometric_issuances.csv"
    if issuances.exists():
        files["issuances"] = issuances
    
    projects = raw_dir / "isometric_projects.csv"
    if projects.exists():
        files["projects"] = projects
    
    retirements = raw_dir / "isometric_retirements.csv"
    if retirements.exists():
        files["retirements"] = retirements
    
    return files


def process_isometric(raw_dir: Path, output_dir: Path, validate: bool = True):
    """Process Isometric data."""
    files = find_isometric_files(raw_dir)
    
    if "issuances" not in files:
        print("⚠ No Isometric issuances file found, skipping...")
        return None, None
    
    print(f"\n📁 Found Isometric files:")
    for key, path in files.items():
        print(f"   {key}: {path.name}")
    
    # Create output directory for isometric
    iso_output = output_dir / "isometric"
    iso_output.mkdir(parents=True, exist_ok=True)
    
    credits, projects = isometric.run_pipeline(
        issuances_path=files["issuances"],
        retirements_path=files.get("retirements"),
        projects_path=files.get("projects"),
        output_dir=None,  # We'll save manually
        validate_output=validate,
    )
    
    # Save to registry-specific output
    credits.to_parquet(iso_output / "credits.parquet", index=False)
    projects.to_parquet(iso_output / "projects.parquet", index=False)
    
    print(f"\n✅ Saved Isometric data to {iso_output}")
    
    return credits, projects


def process_puro(raw_dir: Path, output_dir: Path, validate: bool = True):
    """Process Puro.earth data."""
    files = find_puro_files(raw_dir)
    
    if "issuances" not in files:
        print("⚠ No Puro.earth issuances file found, skipping...")
        return None, None
    
    print(f"\n📁 Found Puro.earth files:")
    for key, path in files.items():
        print(f"   {key}: {path.name}")
    
    # Create output directory for puro
    puro_output = output_dir / "puro-earth"
    puro_output.mkdir(parents=True, exist_ok=True)
    
    credits, projects = puro.run_pipeline(
        issuances_path=files["issuances"],
        retirements_path=files.get("retirements"),
        projects_path=files.get("projects"),
        output_dir=None,  # We'll save manually
        validate_output=validate,
    )
    
    # Save to registry-specific output
    credits.to_parquet(puro_output / "credits.parquet", index=False)
    projects.to_parquet(puro_output / "projects.parquet", index=False)
    
    print(f"\n✅ Saved Puro.earth data to {puro_output}")
    
    return credits, projects


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the carbon removal data pipeline for all registries"
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
        "--registry",
        choices=["isometric", "puro-earth", "all"],
        default="all",
        help="Which registry to process (default: all)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip schema validation",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Carbon Removal Data Processing Pipeline")
    print("=" * 70)
    print(f"\nRaw directory: {args.raw_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Registry: {args.registry}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_credits = []
    all_projects = []
    
    # Process Isometric
    if args.registry in ["isometric", "all"]:
        credits, projects = process_isometric(
            args.raw_dir, args.output_dir, not args.skip_validation
        )
        if credits is not None:
            all_credits.append(credits)
            all_projects.append(projects)
    
    # Process Puro.earth
    if args.registry in ["puro-earth", "all"]:
        credits, projects = process_puro(
            args.raw_dir, args.output_dir, not args.skip_validation
        )
        if credits is not None:
            all_credits.append(credits)
            all_projects.append(projects)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    if all_credits:
        total_credits = sum(len(c) for c in all_credits)
        total_projects = sum(len(p) for p in all_projects)
        
        print(f"\n📊 Total processed:")
        print(f"   Credits: {total_credits:,}")
        print(f"   Projects: {total_projects:,}")
        
        print(f"\n📁 Output files:")
        for registry_dir in args.output_dir.iterdir():
            if registry_dir.is_dir():
                print(f"   {registry_dir.name}/")
                for f in registry_dir.glob("*.parquet"):
                    print(f"      - {f.name}")
    else:
        print("\n⚠ No data was processed!")
    
    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
