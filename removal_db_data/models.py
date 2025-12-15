"""
Pandera schemas for validating carbon removal credit data.

These schemas EXACTLY match CarbonPlan's OffsetsDB schemas to ensure compatibility.
See: offsets-db-data/offsets_db_data/models.py
"""

import typing

import pandas as pd
import pandera as pa


RegistryType = typing.Literal[
    'isometric',
    'puro',
    'none',
]


# Exact copy of CarbonPlan's project_schema
project_schema = pa.DataFrameSchema(
    {
        'protocol': pa.Column(pa.Object, nullable=True),  # Array of strings
        'protocol_version': pa.Column(pa.Object, nullable=True),  # Array of strings (parallel to protocol)
        'category': pa.Column(pa.String, nullable=True),
        'project_type': pa.Column(pa.String, nullable=False),
        'project_type_source': pa.Column(pa.String, nullable=False),
        'retired': pa.Column(
            pa.Int, pa.Check.greater_than_or_equal_to(0), nullable=True, coerce=True
        ),
        'issued': pa.Column(
            pa.Int, pa.Check.greater_than_or_equal_to(0), nullable=True, coerce=True
        ),
        'project_id': pa.Column(pa.String, nullable=False),
        'name': pa.Column(pa.String, nullable=True),
        'registry': pa.Column(pa.String, nullable=False),
        'proponent': pa.Column(pa.String, nullable=True),
        'status': pa.Column(pa.String, nullable=True),
        'country': pa.Column(pa.String, nullable=True),
        'listed_at': pa.Column(pd.DatetimeTZDtype(tz='UTC'), nullable=True),
        'first_issuance_at': pa.Column(pd.DatetimeTZDtype(tz='UTC'), nullable=True),
        'first_retirement_at': pa.Column(pd.DatetimeTZDtype(tz='UTC'), nullable=True),
        'is_compliance': pa.Column(pa.Bool, nullable=True),
        'project_url': pa.Column(pa.String, nullable=True),
    }
)


credit_without_id_schema = pa.DataFrameSchema(
    {
        'quantity': pa.Column(
            pa.Int, pa.Check.greater_than_or_equal_to(0), nullable=True, coerce=True
        ),
        'project_id': pa.Column(pa.String, nullable=False),
        'vintage': pa.Column(pa.Int, nullable=True, coerce=True),
        'transaction_date': pa.Column(pd.DatetimeTZDtype(tz='UTC'), nullable=True),
        'transaction_type': pa.Column(pa.String, nullable=True),
        'retirement_account': pa.Column(pa.String, nullable=True),
        'retirement_reason': pa.Column(pa.String, nullable=True),
        'retirement_note': pa.Column(pa.String, nullable=True),
        'retirement_beneficiary': pa.Column(pa.String, nullable=True),
        'retirement_beneficiary_harmonized': pa.Column(pa.String, nullable=True),
    }
)

credit_schema = credit_without_id_schema.add_columns({'id': pa.Column(pa.Int, nullable=False)})

