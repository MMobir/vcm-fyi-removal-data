"""
Puro.earth registry data processor.

Transforms raw CSV data from Puro.earth's registry into the standardized
schema matching CarbonPlan's OffsetsDB format.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_flavor as pf

from .common import (
    add_category,
    add_first_issuance_and_retirement_dates,
    add_missing_columns,
    add_retired_and_issued_totals,
    convert_to_datetime,
    harmonize_country_names,
    load_type_category_mapping,
    map_project_type_to_display_name,
    set_registry,
    validate,
)
from .models import credit_without_id_schema, project_schema


# Registry identifier
REGISTRY_NAME = "puro-earth"

# Column mappings for Puro.earth CSV data
ISSUANCE_COLUMNS = {
    "transactionId": "transaction_id",
    "projectId": "project_id",
    "projectName": "name",
    "volume": "quantity",
    "issuanceDate": "transaction_date",
    "methodologyName": "protocol",
    "methodologyCode": "project_type",
    "owner": "proponent",
    "vintage": "vintage",
    "creditType": "credit_type",
}

RETIREMENT_COLUMNS = {
    "transactionId": "transaction_id",
    "projectId": "project_id",
    "projectName": "name",
    "volume": "quantity",
    "completedOn": "transaction_date",
    "methodologyName": "protocol",
    "methodologyCode": "project_type",
    "owner": "retirement_account",
    "beneficiary": "retirement_beneficiary",
    "usePurpose": "retirement_reason",
    "vintage": "vintage",
    "creditType": "credit_type",
}

PROJECT_COLUMNS = {
    "projectId": "project_id",
    "name": "name",
    "supplierName": "proponent",
    "methodologyCode": "project_type",
    "country": "country",
    "creditingPeriodStart": "crediting_start",
    "creditingPeriodEnd": "crediting_end",
}


@pf.register_dataframe_method
def set_puro_project_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure project_id is properly formatted for Puro.earth.
    
    Puro uses numeric project IDs, we prefix with 'PURO-' for uniqueness.
    """
    if "project_id" in df.columns:
        df["project_id"] = "PURO-" + df["project_id"].astype(str)
    return df


@pf.register_dataframe_method
def set_puro_transaction_type(
    df: pd.DataFrame, transaction_type: str
) -> pd.DataFrame:
    """Set the transaction type for Puro.earth records."""
    df["transaction_type"] = transaction_type
    return df


@pf.register_dataframe_method
def clean_puro_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert quantity column to integers."""
    if "quantity" not in df.columns:
        return df

    def parse_quantity(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            val = val.replace(",", "").strip()
        try:
            return int(round(float(val)))
        except (ValueError, TypeError):
            return 0

    df["quantity"] = df["quantity"].apply(parse_quantity)
    return df


@pf.register_dataframe_method
def map_puro_methodology_to_project_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Puro.earth methodology codes to standardized project types.
    
    Puro methodology codes:
    - C01000000: Carbonated Materials
    - C02000000: Geologically Stored Carbon
    - C03000000: Biochar
    - C04000000: Enhanced Rock Weathering
    - C05000000: Woody Biomass Burial
    """
    methodology_mapping = {
        "C01000000": "carbonated materials",
        "C02000000": "geologically stored carbon",
        "C03000000": "biochar",
        "C04000000": "enhanced weathering",
        "C05000000": "woody biomass burial",
    }
    
    if "project_type" in df.columns:
        # First try mapping the code
        df["project_type"] = (
            df["project_type"]
            .str.upper()
            .str.strip()
            .map(methodology_mapping)
            .fillna(df["project_type"].str.lower())
        )
    
    return df


def process_puro_credits(
    issuances_df: pd.DataFrame,
    retirements_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Process Puro.earth credit transactions (issuances and retirements).
    """
    print("Processing Puro.earth credits...")

    # Process issuances
    print("  Processing issuances...")
    issuances = (
        issuances_df.copy()
        .rename(columns=ISSUANCE_COLUMNS)
        .pipe(set_registry, REGISTRY_NAME)
        .pipe(set_puro_transaction_type, "issuance")
        .pipe(set_puro_project_id)
        .pipe(clean_puro_quantity)
        .pipe(convert_to_datetime, columns=["transaction_date"])
    )
    
    # Ensure vintage is numeric
    if "vintage" in issuances.columns:
        issuances["vintage"] = pd.to_numeric(issuances["vintage"], errors="coerce")

    # Process retirements if provided
    all_credits = [issuances]

    if retirements_df is not None and not retirements_df.empty:
        print("  Processing retirements...")
        retirements = (
            retirements_df.copy()
            .rename(columns=RETIREMENT_COLUMNS)
            .pipe(set_registry, REGISTRY_NAME)
            .pipe(set_puro_transaction_type, "retirement")
            .pipe(set_puro_project_id)
            .pipe(clean_puro_quantity)
            .pipe(convert_to_datetime, columns=["transaction_date"])
        )
        
        if "vintage" in retirements.columns:
            retirements["vintage"] = pd.to_numeric(retirements["vintage"], errors="coerce")
        
        all_credits.append(retirements)

    # Combine all credits
    credits = pd.concat(all_credits, ignore_index=True)

    # Add missing columns from schema with default values
    credits = credits.pipe(add_missing_columns, schema=credit_without_id_schema)

    # Ensure required columns exist
    for col in [
        "retirement_account",
        "retirement_reason",
        "retirement_note",
        "retirement_beneficiary_harmonized",
    ]:
        if col not in credits.columns:
            credits[col] = None

    # Copy retirement_beneficiary to harmonized version
    if "retirement_beneficiary" in credits.columns:
        credits["retirement_beneficiary_harmonized"] = credits["retirement_beneficiary"]

    print(f"  Total credit transactions: {len(credits):,}")
    print(f"    Issuances: {(credits['transaction_type'] == 'issuance').sum():,}")
    print(f"    Retirements: {(credits['transaction_type'] == 'retirement').sum():,}")

    return credits


def process_puro_projects(
    projects_df: pd.DataFrame,
    credits_df: pd.DataFrame,
    type_category_mapping: dict | None = None,
) -> pd.DataFrame:
    """
    Process Puro.earth projects from raw projects data.
    """
    print("Processing Puro.earth projects from raw data...")

    if type_category_mapping is None:
        type_category_mapping = load_type_category_mapping()

    # Start with raw projects
    projects = projects_df.copy()

    # Rename columns to match schema
    projects = projects.rename(columns=PROJECT_COLUMNS)

    # Add PURO- prefix to project_id
    projects = projects.pipe(set_puro_project_id)

    # Set registry
    projects = projects.pipe(set_registry, REGISTRY_NAME)

    # All Puro projects are registered (they've been validated to issue credits)
    projects["status"] = "registered"

    # Map methodology to standardized project type
    projects = projects.pipe(map_puro_methodology_to_project_type)

    # Add category based on project type
    projects = projects.pipe(add_category, type_category_mapping=type_category_mapping)

    # Map project type to display name
    projects = projects.pipe(
        map_project_type_to_display_name, type_category_mapping=type_category_mapping
    )

    # Set project type source
    projects["project_type_source"] = "puro-earth"

    # Calculate issued and retired from credits
    projects = projects.pipe(add_retired_and_issued_totals, credits=credits_df)

    # Add first issuance and retirement dates from credits
    projects = projects.pipe(
        add_first_issuance_and_retirement_dates, credits=credits_df
    )

    # Set additional fields
    projects["is_compliance"] = False  # Carbon removal credits are voluntary

    # Build project URL
    projects["project_url"] = None  # Puro doesn't have public project URLs

    # Protocol - use methodology name from credits
    projects["protocol"] = None
    projects["protocol_version"] = None

    # Set datetime columns
    projects["listed_at"] = pd.NaT
    projects["listed_at"] = pd.to_datetime(projects["listed_at"], utc=True)

    # Ensure first_issuance_at and first_retirement_at are proper datetime
    for col in ["first_issuance_at", "first_retirement_at"]:
        if col in projects.columns:
            projects[col] = pd.to_datetime(projects[col], utc=True)

    # Add missing columns from schema
    projects = projects.pipe(add_missing_columns, schema=project_schema)

    print(f"  Total projects: {len(projects):,}")
    print(f"    Status breakdown:")
    for status, count in projects["status"].value_counts().items():
        print(f"      {status}: {count}")

    return projects


def run_pipeline(
    issuances_path: Path,
    retirements_path: Path | None = None,
    projects_path: Path | None = None,
    output_dir: Path | None = None,
    validate_output: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full Puro.earth data processing pipeline.
    """
    print("=" * 60)
    print("Puro.earth Data Processing Pipeline")
    print("=" * 60)

    # Load raw issuances data
    print(f"\nLoading issuances from: {issuances_path}")
    issuances_df = pd.read_csv(issuances_path)
    print(f"  Loaded {len(issuances_df):,} issuance records")

    # Load raw retirements data
    retirements_df = None
    if retirements_path and retirements_path.exists():
        print(f"\nLoading retirements from: {retirements_path}")
        retirements_df = pd.read_csv(retirements_path)
        print(f"  Loaded {len(retirements_df):,} retirement records")

    # Load raw projects data
    raw_projects_df = None
    if projects_path and projects_path.exists():
        print(f"\nLoading raw projects from: {projects_path}")
        raw_projects_df = pd.read_csv(projects_path)
        print(f"  Loaded {len(raw_projects_df):,} raw projects")

    # Process credits
    print("\n" + "-" * 40)
    credits = process_puro_credits(issuances_df, retirements_df)

    # Process projects
    print("\n" + "-" * 40)
    if raw_projects_df is not None:
        projects = process_puro_projects(raw_projects_df, credits)
    else:
        # Fallback: derive from credits only
        print("Deriving projects from credits (no raw projects file)...")
        projects = _derive_projects_from_credits(credits)

    # Validate if requested
    if validate_output:
        print("\n" + "-" * 40)
        print("Validating output...")
        try:
            credits = credits.pipe(validate, schema=credit_without_id_schema)
            print("  ✓ Credits validated successfully")
        except Exception as e:
            print(f"  ✗ Credit validation failed: {e}")

        try:
            projects = projects.pipe(validate, schema=project_schema)
            print("  ✓ Projects validated successfully")
        except Exception as e:
            print(f"  ✗ Project validation failed: {e}")

    # Save output if directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        credits_output = output_dir / "credits-puro-earth.parquet"
        projects_output = output_dir / "projects-puro-earth.parquet"

        print("\n" + "-" * 40)
        print(f"Saving credits to: {credits_output}")
        credits.to_parquet(credits_output, index=False)

        print(f"Saving projects to: {projects_output}")
        projects.to_parquet(projects_output, index=False)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return credits, projects


def _derive_projects_from_credits(credits_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: derive projects from credits when no raw projects file available."""
    type_category_mapping = load_type_category_mapping()
    
    project_cols = ["project_id", "name", "proponent", "project_type", "country"]
    available_cols = [c for c in project_cols if c in credits_df.columns]
    
    projects = credits_df[available_cols].drop_duplicates(subset=["project_id"])
    projects = projects.pipe(set_registry, REGISTRY_NAME)
    projects = projects.pipe(map_puro_methodology_to_project_type)
    projects = projects.pipe(add_category, type_category_mapping=type_category_mapping)
    projects = projects.pipe(map_project_type_to_display_name, type_category_mapping=type_category_mapping)
    projects["project_type_source"] = "puro-earth"
    projects = projects.pipe(add_retired_and_issued_totals, credits=credits_df)
    projects = projects.pipe(add_first_issuance_and_retirement_dates, credits=credits_df)
    projects["status"] = "registered"
    if "country" not in projects.columns:
        projects["country"] = None
    projects["is_compliance"] = False
    projects["project_url"] = None
    projects["protocol"] = None
    projects["protocol_version"] = None
    projects["listed_at"] = pd.NaT
    projects["listed_at"] = pd.to_datetime(projects["listed_at"], utc=True)
    for col in ["first_issuance_at", "first_retirement_at"]:
        if col in projects.columns:
            projects[col] = pd.to_datetime(projects[col], utc=True)
    projects = projects.pipe(add_missing_columns, schema=project_schema)
    
    print(f"  Derived {len(projects):,} projects from credits")
    return projects

