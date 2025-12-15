"""
Common utilities for processing carbon removal credit data.

Adapted from CarbonPlan's OffsetsDB common utilities.
"""

import json
import typing
from pathlib import Path

import country_converter as coco
import numpy as np
import pandas as pd
import pandas_flavor as pf
import pandera as pa


# Config file paths
CONFIG_DIR = Path(__file__).parent / 'configs'
TYPE_CATEGORY_MAPPING_PATH = CONFIG_DIR / 'type-category-mapping.json'
CREDITS_COLUMN_MAPPING_PATH = CONFIG_DIR / 'credits-column-mapping.json'
PROJECTS_COLUMN_MAPPING_PATH = CONFIG_DIR / 'projects-column-mapping.json'


def load_type_category_mapping() -> dict:
    """Load the type to category mapping from config."""
    return json.loads(TYPE_CATEGORY_MAPPING_PATH.read_text())


def load_column_mapping(*, registry_name: str, download_type: str, mapping_path: Path) -> dict:
    """Load column mapping for a specific registry and download type."""
    with open(mapping_path) as f:
        registry_column_mapping = json.load(f)
    return registry_column_mapping.get(registry_name, {}).get(download_type, {})


@pf.register_dataframe_method
def set_registry(df: pd.DataFrame, registry_name: str) -> pd.DataFrame:
    """
    Set the registry name for each record in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    registry_name : str
        Name of the registry to set.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'registry' column set to the specified registry name.
    """
    df['registry'] = registry_name
    return df


@pf.register_dataframe_method
def convert_to_datetime(
    df: pd.DataFrame, *, columns: list, utc: bool = True, **kwargs: typing.Any
) -> pd.DataFrame:
    """
    Convert specified columns in the DataFrame to datetime format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        List of column names to convert to datetime.
    utc : bool, optional
        Whether to convert to UTC (default is True).
    **kwargs : typing.Any
        Additional keyword arguments passed to pd.to_datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns converted to datetime format.
    """
    for column in columns:
        if column not in df.columns:
            continue  # Skip if column doesn't exist
        try:
            df[column] = pd.to_datetime(df[column], utc=utc, **kwargs).dt.normalize()
        except ValueError:
            df[column] = pd.to_datetime(df[column], utc=utc).dt.normalize()
    return df


@pf.register_dataframe_method
def add_missing_columns(df: pd.DataFrame, *, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """
    Add any missing columns to the DataFrame and initialize them with default values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    schema : pa.DataFrameSchema
        Pandera schema to validate against.

    Returns
    -------
    pd.DataFrame
        DataFrame with all specified columns, adding missing ones initialized to default values.
    """
    default_values = {
        np.dtype('int64'): 0,
        np.dtype('int32'): 0,
        np.dtype('float64'): 0.0,
        np.dtype('float32'): 0.0,
        np.dtype('O'): None,
        np.dtype('<U'): None,
        np.dtype('U'): None,
        np.dtype('bool'): False,
        np.dtype('<M8[ns]'): pd.NaT,  # datetime64[ns]
    }

    for column, value in schema.columns.items():
        dtype = value.dtype.type
        if column not in df.columns:
            default_value = default_values.get(dtype)
            df[column] = pd.Series([default_value] * len(df), index=df.index, dtype=dtype)
    return df


@pf.register_dataframe_method
def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """
    Validate the DataFrame against a given Pandera schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    schema : pa.DataFrameSchema
        Pandera schema to validate against.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns sorted according to the schema and validated against it.
    """
    results = schema.validate(df)
    keys = sorted(list(schema.columns.keys()))
    results = results[keys]
    return results


@pf.register_dataframe_method
def harmonize_country_names(df: pd.DataFrame, *, country_column: str = 'country') -> pd.DataFrame:
    """
    Harmonize country names in the DataFrame to standardized country names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with country data.
    country_column : str, optional
        The name of the column containing country names to be harmonized (default is 'country').

    Returns
    -------
    pd.DataFrame
        DataFrame with harmonized country names in the specified column.
    """
    if country_column not in df.columns:
        return df
    
    print('Harmonizing country names...')
    cc = coco.CountryConverter()
    df[country_column] = cc.pandas_convert(df[country_column], to='name')
    print('Done converting country names...')
    return df


@pf.register_dataframe_method
def add_category(df: pd.DataFrame, *, type_category_mapping: dict) -> pd.DataFrame:
    """
    Add a category to each record in the DataFrame based on its project type.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing project type data.
    type_category_mapping : dict
        Dictionary mapping types to categories.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'category' column, derived from the project type information.
    """
    print('Adding category based on project type...')
    df['category'] = (
        df['project_type']
        .str.lower()
        .map({key.lower(): value['category'] for key, value in type_category_mapping.items()})
        .fillna('carbon-removal')  # Default category for removal credits
    )
    return df


@pf.register_dataframe_method
def map_project_type_to_display_name(
    df: pd.DataFrame, *, type_category_mapping: dict
) -> pd.DataFrame:
    """
    Map project types in the DataFrame to display names based on a mapping dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing project data.
    type_category_mapping : dict
        Dictionary mapping project type strings to display names.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'project_type' column, containing mapped display names.
    """
    print('Mapping project types to display names...')
    df['project_type'] = (
        df['project_type']
        .str.lower()
        .map(
            {
                key.lower(): value['project-type-display-name']
                for key, value in type_category_mapping.items()
            }
        )
        .fillna(df['project_type'])  # Keep original if no mapping
    )
    return df


@pf.register_dataframe_method
def add_first_issuance_and_retirement_dates(
    projects: pd.DataFrame, *, credits: pd.DataFrame
) -> pd.DataFrame:
    """
    Add the first issuance date and first retirement date to each project.

    Parameters
    ----------
    projects : pd.DataFrame
        A pandas DataFrame containing project data with a 'project_id' column.
    credits : pd.DataFrame
        A pandas DataFrame containing credit transaction data.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'first_issuance_at' and 'first_retirement_at' columns added.
    """
    first_issuance = (
        credits[credits['transaction_type'] == 'issuance']
        .groupby('project_id')['transaction_date']
        .min()
        .reset_index()
    )
    first_retirement = (
        credits[credits['transaction_type'].str.contains('retirement', na=False)]
        .groupby('project_id')['transaction_date']
        .min()
        .reset_index()
    )

    # Merge the projects DataFrame with the first issuance and retirement dates
    projects_with_dates = pd.merge(projects, first_issuance, on='project_id', how='left')
    projects_with_dates = pd.merge(
        projects_with_dates, first_retirement, on='project_id', how='left'
    )

    # Rename the merged columns for clarity
    projects_with_dates = projects_with_dates.rename(
        columns={
            'transaction_date_x': 'first_issuance_at',
            'transaction_date_y': 'first_retirement_at',
        }
    )

    return projects_with_dates


@pf.register_dataframe_method
def add_retired_and_issued_totals(projects: pd.DataFrame, *, credits: pd.DataFrame) -> pd.DataFrame:
    """
    Add total quantities of issued and retired credits to each project.

    Parameters
    ----------
    projects : pd.DataFrame
        DataFrame containing project data.
    credits : pd.DataFrame
        DataFrame containing credit transaction data.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'issued' and 'retired' columns representing total quantities.
    """
    # Drop conflicting columns if they exist
    projects = projects.drop(columns=['issued', 'retired'], errors='ignore')

    # Group and sum
    credit_totals = (
        credits.groupby(['project_id', 'transaction_type'])['quantity'].sum().reset_index()
    )
    
    # Pivot the table
    credit_totals_pivot = credit_totals.pivot(
        index='project_id', columns='transaction_type', values='quantity'
    ).reset_index()

    # Ensure columns exist
    for col in ['issuance', 'retirement']:
        if col not in credit_totals_pivot.columns:
            credit_totals_pivot[col] = 0

    # Merge with projects
    projects_combined = pd.merge(
        projects,
        credit_totals_pivot[['project_id', 'issuance', 'retirement']],
        left_on='project_id',
        right_on='project_id',
        how='left',
    )

    # Rename columns for clarity
    projects_combined = projects_combined.rename(
        columns={'issuance': 'issued', 'retirement': 'retired'}
    )

    # Replace NaNs with 0 if any
    projects_combined[['issued', 'retired']] = projects_combined[['issued', 'retired']].fillna(0)

    return projects_combined


@pf.register_dataframe_method  
def aggregate_issuance_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate issuance transactions by summing the quantity for each combination 
    of project ID, transaction date, and vintage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing issuance transaction data.

    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated issuance transactions.
    """
    if 'transaction_type' not in df.columns:
        raise KeyError("The column 'transaction_type' is missing.")

    df_issuance_agg = pd.DataFrame()
    df_issuance = df[df['transaction_type'] == 'issuance']

    if not df_issuance.empty:
        df_issuance_agg = (
            df_issuance.groupby(['project_id', 'transaction_date', 'vintage'])
            .agg(
                {
                    'quantity': 'sum',
                    'registry': 'first',
                    'transaction_type': 'first',
                }
            )
            .reset_index()
        )
        df_issuance_agg = df_issuance_agg[df_issuance_agg['quantity'] > 0]
    return df_issuance_agg

