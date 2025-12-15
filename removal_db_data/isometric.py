"""
Isometric registry data processor.

Transforms raw CSV data from Isometric's registry into the standardized
schema matching CarbonPlan's OffsetsDB format.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_flavor as pf

from .common import (
    CONFIG_DIR,
    add_category,
    add_first_issuance_and_retirement_dates,
    add_missing_columns,
    add_retired_and_issued_totals,
    aggregate_issuance_transactions,
    convert_to_datetime,
    harmonize_country_names,
    load_type_category_mapping,
    map_project_type_to_display_name,
    set_registry,
    validate,
)
from .models import credit_without_id_schema, project_schema


# Registry identifier
REGISTRY_NAME = "isometric"

# Column mappings for Isometric CSV data (from GraphQL API)
# These map from CSV column names to internal schema names
# project_id is already correct so not included

ISSUANCE_COLUMNS = {
    "supplier": "proponent",
    "project_name": "name",
    "pathway": "project_type",
    "pathway_full": "protocol",
    # transaction_date, quantity, project_id, country are already correctly named
}

RETIREMENT_COLUMNS = {
    "beneficiary": "retirement_beneficiary",
    "retired_by": "retirement_account",
    "supplier": "proponent",
    "pathway": "project_type",
    "pathway_full": "protocol",
    "project_name": "name",
    "notes": "retirement_note",
    "purposes": "retirement_reason",
    # transaction_date, quantity, project_id are already correctly named
}


@pf.register_dataframe_method
def set_isometric_project_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set or generate project IDs for Isometric projects.

    Uses the project_id from API if available, otherwise generates from name.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'project_id' or 'name' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'project_id' column set.
    """
    # If project_id already exists and is non-null, use it
    if "project_id" in df.columns and df["project_id"].notna().any():
        # For rows where project_id is null, generate from name
        mask = df["project_id"].isna()
        if mask.any() and "name" in df.columns:
            def sanitize_name(name: str) -> str:
                if pd.isna(name):
                    return "ISO-unknown"
                sanitized = re.sub(r"[^a-zA-Z0-9\s-]", "", str(name))
                sanitized = re.sub(r"\s+", "-", sanitized.strip())
                sanitized = sanitized.lower()[:50]
                return f"ISO-{sanitized}" if sanitized else "ISO-unknown"
            df.loc[mask, "project_id"] = df.loc[mask, "name"].apply(sanitize_name)
        return df
    
    # If no project_id, generate from name
    if "name" in df.columns:
        def sanitize_name(name: str) -> str:
            if pd.isna(name):
                return "ISO-unknown"
            sanitized = re.sub(r"[^a-zA-Z0-9\s-]", "", str(name))
            sanitized = re.sub(r"\s+", "-", sanitized.strip())
            sanitized = sanitized.lower()[:50]
            return f"ISO-{sanitized}" if sanitized else "ISO-unknown"
        df["project_id"] = df["name"].apply(sanitize_name)
    else:
        df["project_id"] = "ISO-unknown"
    
    return df


@pf.register_dataframe_method
def set_isometric_transaction_type(
    df: pd.DataFrame, transaction_type: str
) -> pd.DataFrame:
    """
    Set the transaction type for Isometric records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    transaction_type : str
        The transaction type ('issuance' or 'retirement').

    Returns
    -------
    pd.DataFrame
        DataFrame with 'transaction_type' column set.
    """
    df["transaction_type"] = transaction_type
    return df


@pf.register_dataframe_method
def clean_isometric_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert quantity column to integers.

    Handles comma-separated numbers and decimal values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'quantity' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned 'quantity' column as integers.
    """
    if "quantity" not in df.columns:
        return df

    def parse_quantity(val):
        if pd.isna(val):
            return 0
        # Handle string quantities with commas
        if isinstance(val, str):
            val = val.replace(",", "").strip()
        try:
            # Convert to float first (handles decimals), then round to int
            return int(round(float(val)))
        except (ValueError, TypeError):
            return 0

    df["quantity"] = df["quantity"].apply(parse_quantity)
    return df


@pf.register_dataframe_method
def extract_vintage_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract vintage year from transaction date if not already present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'transaction_date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'vintage' column populated.
    """
    if "vintage" not in df.columns:
        df["vintage"] = None

    # Fill missing vintages from transaction date
    if "transaction_date" in df.columns:
        mask = df["vintage"].isna()
        df.loc[mask, "vintage"] = pd.to_datetime(
            df.loc[mask, "transaction_date"], errors="coerce"
        ).dt.year

    return df


@pf.register_dataframe_method
def map_isometric_pathway_to_project_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Isometric pathway names to standardized project types.

    Pathways observed:
    - BiCRS (Bio-oil Carbon Removal and Storage)
    - Marine (Ocean Alkalinity Enhancement)
    - EW (Enhanced Weathering)
    - DAC (Direct Air Capture)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'project_type' column containing pathway names.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized project types.
    """
    pathway_mapping = {
        "bicrs": "bicrs",
        "bio-oil": "bicrs",
        "biocrs": "bicrs",
        "marine": "marine cdr",
        "oae": "oae",
        "ew": "enhanced weathering",
        "enhanced weathering": "enhanced weathering",
        "dac": "dac",
        "direct air capture": "dac",
        "biochar": "biochar",
    }

    if "project_type" in df.columns:
        df["project_type"] = (
            df["project_type"]
            .str.lower()
            .str.strip()
            .map(pathway_mapping)
            .fillna(df["project_type"].str.lower())
        )

    return df


def process_isometric_credits(
    issuances_df: pd.DataFrame,
    retirements_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Process Isometric credit transactions (issuances and retirements).

    Parameters
    ----------
    issuances_df : pd.DataFrame
        Raw issuances data from Isometric CSV.
    retirements_df : pd.DataFrame, optional
        Raw retirements data from Isometric CSV.

    Returns
    -------
    pd.DataFrame
        Processed credit transactions matching the credit schema.
    """
    print("Processing Isometric credits...")

    # Process issuances
    print("  Processing issuances...")
    issuances = (
        issuances_df.copy()
        .rename(columns=ISSUANCE_COLUMNS)
        .pipe(set_registry, REGISTRY_NAME)
        .pipe(set_isometric_transaction_type, "issuance")
        .pipe(set_isometric_project_id)
        .pipe(clean_isometric_quantity)
        .pipe(convert_to_datetime, columns=["transaction_date"])
        .pipe(extract_vintage_from_date)
    )

    # Process retirements if provided
    all_credits = [issuances]

    if retirements_df is not None and not retirements_df.empty:
        print("  Processing retirements...")
        retirements = (
            retirements_df.copy()
            .rename(columns=RETIREMENT_COLUMNS)
            .pipe(set_registry, REGISTRY_NAME)
            .pipe(set_isometric_transaction_type, "retirement")
            .pipe(set_isometric_project_id)
            .pipe(clean_isometric_quantity)
            .pipe(convert_to_datetime, columns=["transaction_date"])
            .pipe(extract_vintage_from_date)
        )
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

    # Copy retirement_beneficiary to harmonized version if not present
    if "retirement_beneficiary_harmonized" not in credits.columns or credits[
        "retirement_beneficiary_harmonized"
    ].isna().all():
        credits["retirement_beneficiary_harmonized"] = credits.get(
            "retirement_beneficiary", None
        )

    print(f"  Total credit transactions: {len(credits):,}")
    print(f"    Issuances: {(credits['transaction_type'] == 'issuance').sum():,}")
    print(f"    Retirements: {(credits['transaction_type'] == 'retirement').sum():,}")

    return credits


def process_isometric_projects(
    projects_df: pd.DataFrame,
    credits_df: pd.DataFrame,
    type_category_mapping: dict | None = None,
) -> pd.DataFrame:
    """
    Process Isometric projects from raw projects data.

    Uses ALL projects from the raw projects CSV (not just those with credits).
    This matches how Verra, Gold Standard, etc. include all projects.

    Parameters
    ----------
    projects_df : pd.DataFrame
        Raw projects data from Isometric API.
    credits_df : pd.DataFrame
        Processed credit transactions (for first issuance/retirement dates).
    type_category_mapping : dict, optional
        Mapping of project types to categories. If None, loads from config.

    Returns
    -------
    pd.DataFrame
        Projects DataFrame matching the project schema.
    """
    print("Processing Isometric projects from raw data...")

    if type_category_mapping is None:
        type_category_mapping = load_type_category_mapping()

    # Start with raw projects
    projects = projects_df.copy()

    # Rename columns to match schema
    column_mapping = {
        "project_id": "project_id",
        "name": "name",
        "supplier": "proponent",
        "pathway": "project_type",
        "country": "country",
        "status": "status",
        "issued": "issued",
        "retired": "retired",
        "validated_at": "validated_at",
    }
    projects = projects.rename(columns={k: v for k, v in column_mapping.items() if k in projects.columns})

    # Set registry
    projects = projects.pipe(set_registry, REGISTRY_NAME)

    # Map Isometric status to standard status values
    # Standard values: listed, registered, completed
    ISOMETRIC_STATUS_MAP = {
        "UNDER_VALIDATION": "listed",
        "VALIDATED": "registered",
        "VALIDATION_UNSUCCESSFUL": "completed",
    }
    projects["status"] = projects["status"].map(ISOMETRIC_STATUS_MAP).fillna("unknown")

    # Map pathway to standardized project type
    projects = projects.pipe(map_isometric_pathway_to_project_type)

    # Add category based on project type
    projects = projects.pipe(add_category, type_category_mapping=type_category_mapping)

    # Map project type to display name
    projects = projects.pipe(
        map_project_type_to_display_name, type_category_mapping=type_category_mapping
    )

    # Set project type source
    projects["project_type_source"] = "isometric"

    # Use issued/retired from raw data, convert to int
    projects["issued"] = pd.to_numeric(projects["issued"], errors="coerce").fillna(0).astype(int)
    projects["retired"] = pd.to_numeric(projects["retired"], errors="coerce").fillna(0).astype(int)

    # Add first issuance and retirement dates from credits
    projects = projects.pipe(
        add_first_issuance_and_retirement_dates, credits=credits_df
    )

    # Set additional fields
    projects["is_compliance"] = False  # Carbon removal credits are voluntary

    # Build project URL (Isometric uses format: https://registry.isometric.com/project/{short_code})
    if "short_code" in projects_df.columns:
        projects["project_url"] = "https://registry.isometric.com/project/" + projects_df["short_code"].astype(str)
    else:
        projects["project_url"] = None

    # Protocol - use pathway_full if available
    if "pathway_full" in projects_df.columns:
        projects["protocol"] = projects_df["pathway_full"].apply(lambda x: [x] if pd.notna(x) else None)
    else:
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
    Run the full Isometric data processing pipeline.

    Parameters
    ----------
    issuances_path : Path
        Path to the issuances CSV file.
    retirements_path : Path, optional
        Path to the retirements CSV file.
    projects_path : Path, optional
        Path to the raw projects CSV file. If not provided, will look for
        isometric_projects.csv in the same directory as issuances.
    output_dir : Path, optional
        Directory to save output Parquet files.
    validate_output : bool
        Whether to validate output against schemas.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (credits_df, projects_df).
    """
    print("=" * 60)
    print("Isometric Data Processing Pipeline")
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
    if projects_path is None:
        # Auto-detect in same directory
        projects_path = issuances_path.parent / "isometric_projects.csv"
    
    raw_projects_df = None
    if projects_path.exists():
        print(f"\nLoading raw projects from: {projects_path}")
        raw_projects_df = pd.read_csv(projects_path)
        print(f"  Loaded {len(raw_projects_df):,} raw projects")
    else:
        print(f"\n⚠ Warning: No raw projects file found at {projects_path}")
        print("  Projects will be derived from credits only.")

    # Process credits
    print("\n" + "-" * 40)
    credits = process_isometric_credits(issuances_df, retirements_df)

    # Process projects
    print("\n" + "-" * 40)
    if raw_projects_df is not None:
        projects = process_isometric_projects(raw_projects_df, credits)
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

        credits_output = output_dir / "credits-isometric.parquet"
        projects_output = output_dir / "projects-isometric.parquet"

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
    """
    Fallback: derive projects from credits when no raw projects file available.
    """
    type_category_mapping = load_type_category_mapping()
    
    project_cols = ["project_id", "name", "proponent", "project_type", "country"]
    available_cols = [c for c in project_cols if c in credits_df.columns]
    
    projects = credits_df[available_cols].drop_duplicates(subset=["project_id"])
    projects = projects.pipe(set_registry, REGISTRY_NAME)
    projects = projects.pipe(map_isometric_pathway_to_project_type)
    projects = projects.pipe(add_category, type_category_mapping=type_category_mapping)
    projects = projects.pipe(map_project_type_to_display_name, type_category_mapping=type_category_mapping)
    projects["project_type_source"] = "isometric"
    projects = projects.pipe(add_retired_and_issued_totals, credits=credits_df)
    projects = projects.pipe(add_first_issuance_and_retirement_dates, credits=credits_df)
    projects["status"] = "registered"  # Assume registered if has credits
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

