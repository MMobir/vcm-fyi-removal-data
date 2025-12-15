#!/usr/bin/env python3
"""
Fetch data from Isometric's public GraphQL API.

Isometric provides a public GraphQL API at https://edge.isometric.com/
This script fetches:
- Issuances: All credit issuance transactions
- Retirements: All credit retirement transactions  
- Projects: All carbon removal projects
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# GraphQL API endpoint
GRAPHQL_URL = "https://edge.isometric.com/"


def graphql_query(query: str, variables: dict = None) -> dict:
    """
    Execute a GraphQL query against the Isometric API.
    
    Parameters
    ----------
    query : str
        The GraphQL query string.
    variables : dict, optional
        Variables to pass to the query.
        
    Returns
    -------
    dict
        The JSON response from the API.
    """
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "VCM.fyi Removal Credits Pipeline/1.0",
    }
    
    response = requests.post(GRAPHQL_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")
    
    return data["data"]


def fetch_all_paginated(query_name: str, query_template: str, page_size: int = 100) -> list:
    """
    Fetch all records using cursor-based pagination.
    
    Parameters
    ----------
    query_name : str
        Name of the query field (e.g., 'issuances').
    query_template : str
        GraphQL query template with {first} and {after} placeholders.
    page_size : int
        Number of records per page.
        
    Returns
    -------
    list
        All records from the API.
    """
    all_records = []
    cursor = None
    page = 1
    
    while True:
        # Build the query
        after_clause = f', after: "{cursor}"' if cursor else ""
        query = query_template.format(first=page_size, after=after_clause)
        
        print(f"  Fetching page {page}...", end=" ")
        data = graphql_query(query)
        
        records = data[query_name]["nodes"]
        all_records.extend(records)
        print(f"got {len(records)} records")
        
        page_info = data[query_name]["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        
        cursor = page_info["endCursor"]
        page += 1
    
    return all_records


def fetch_issuances() -> list:
    """Fetch all issuance records from the API."""
    print("Fetching issuances...")
    
    query_template = """
    query {{
        issuances(first: {first}{after}) {{
            nodes {{
                id
                createdAt
                creditBatchSizeTotal {{ credits }}
                project {{
                    id
                    name
                    shortCode
                    durability
                    country {{ name isoAlpha3Code }}
                    status
                }}
                supplier {{
                    organisation {{ name }}
                    pathway {{ name shortName }}
                }}
            }}
            pageInfo {{ hasNextPage endCursor }}
            totalCount
        }}
    }}
    """
    
    return fetch_all_paginated("issuances", query_template)


def fetch_retirements() -> list:
    """Fetch all retirement records from the API."""
    print("Fetching retirements...")
    
    query_template = """
    query {{
        retirements(first: {first}{after}) {{
            nodes {{
                id
                createdAt
                creditBatchSizeTotal {{ credits }}
                owner {{ name }}
                beneficiary {{ name }}
                supplier {{
                    organisation {{ name }}
                    pathway {{ shortName name }}
                }}
                creditBatches {{
                    issuance {{
                        project {{
                            id
                            name
                            shortCode
                        }}
                    }}
                }}
                purposes
                notes
            }}
            pageInfo {{ hasNextPage endCursor }}
            totalCount
        }}
    }}
    """
    
    return fetch_all_paginated("retirements", query_template)


def fetch_projects() -> list:
    """Fetch all project records from the API."""
    print("Fetching projects...")
    
    query_template = """
    query {{
        projects(first: {first}{after}) {{
            nodes {{
                id
                name
                shortCode
                description
                shortDescription
                durability
                status
                country {{ name isoAlpha3Code }}
                location {{ name }}
                creditBalance {{ 
                    total {{ credits }}
                    retired {{ credits }}
                    active {{ credits }}
                }}
                supplier {{
                    organisation {{ name }}
                    pathway {{ name shortName type }}
                }}
                validatedAt
                creditingPeriodStart
                creditingPeriodEnd
            }}
            pageInfo {{ hasNextPage endCursor }}
            totalCount
        }}
    }}
    """
    
    return fetch_all_paginated("projects", query_template)


def transform_issuances_to_df(issuances: list) -> pd.DataFrame:
    """Transform issuances data to a DataFrame."""
    records = []
    for iss in issuances:
        records.append({
            "id": iss["id"],
            "transaction_date": iss["createdAt"],
            "quantity": iss["creditBatchSizeTotal"]["credits"],
            "project_id": iss["project"]["id"],
            "project_name": iss["project"]["name"],
            "project_short_code": iss["project"]["shortCode"],
            "durability": iss["project"]["durability"],
            "country": iss["project"]["country"]["name"] if iss["project"].get("country") else None,
            "country_code": iss["project"]["country"]["isoAlpha3Code"] if iss["project"].get("country") else None,
            "project_status": iss["project"]["status"],
            "supplier": iss["supplier"]["organisation"]["name"],
            "pathway": iss["supplier"]["pathway"]["shortName"],
            "pathway_full": iss["supplier"]["pathway"]["name"],
        })
    return pd.DataFrame(records)


def transform_retirements_to_df(retirements: list) -> pd.DataFrame:
    """Transform retirements data to a DataFrame."""
    records = []
    for ret in retirements:
        supplier = ret.get("supplier") or {}
        
        # Get project info from first credit batch's issuance
        project_id = None
        project_name = None
        project_short_code = None
        credit_batches = ret.get("creditBatches", [])
        if credit_batches:
            first_batch = credit_batches[0]
            issuance = first_batch.get("issuance", {}) or {}
            project = issuance.get("project", {}) or {}
            project_id = project.get("id")
            project_name = project.get("name")
            project_short_code = project.get("shortCode")
        
        records.append({
            "id": ret["id"],
            "transaction_date": ret["createdAt"],
            "quantity": ret["creditBatchSizeTotal"]["credits"],
            "project_id": project_id,
            "project_name": project_name,
            "project_short_code": project_short_code,
            "supplier": supplier.get("organisation", {}).get("name") if supplier.get("organisation") else None,
            "pathway": supplier.get("pathway", {}).get("shortName") if supplier.get("pathway") else None,
            "pathway_full": supplier.get("pathway", {}).get("name") if supplier.get("pathway") else None,
            "retired_by": ret["owner"]["name"] if ret.get("owner") else None,
            "beneficiary": ret["beneficiary"]["name"] if ret.get("beneficiary") else None,
            "purposes": ", ".join(ret.get("purposes", [])) if ret.get("purposes") else None,
            "notes": ret.get("notes"),
        })
    return pd.DataFrame(records)


def transform_projects_to_df(projects: list) -> pd.DataFrame:
    """Transform projects data to a DataFrame."""
    records = []
    for proj in projects:
        credit_balance = proj.get("creditBalance") or {}
        total = credit_balance.get("total", {})
        retired = credit_balance.get("retired", {})
        active = credit_balance.get("active", {})
        
        records.append({
            "project_id": proj["id"],
            "name": proj["name"],
            "short_code": proj["shortCode"],
            "description": proj.get("description"),
            "short_description": proj.get("shortDescription"),
            "durability": proj["durability"],
            "status": proj["status"],
            "country": proj["country"]["name"] if proj.get("country") else None,
            "country_code": proj["country"]["isoAlpha3Code"] if proj.get("country") else None,
            "location": proj["location"]["name"] if proj.get("location") else None,
            "issued": total.get("credits", 0),
            "retired": retired.get("credits", 0),
            "active": active.get("credits", 0),
            "supplier": proj["supplier"]["organisation"]["name"],
            "pathway": proj["supplier"]["pathway"]["shortName"],
            "pathway_full": proj["supplier"]["pathway"]["name"],
            "pathway_type": proj["supplier"]["pathway"].get("type"),
            "validated_at": proj.get("validatedAt"),
            "crediting_period_start": proj.get("creditingPeriodStart"),
            "crediting_period_end": proj.get("creditingPeriodEnd"),
        })
    return pd.DataFrame(records)


def fetch_all(output_dir: Path) -> dict:
    """
    Fetch all data from Isometric's public GraphQL API.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save CSV files.
        
    Returns
    -------
    dict
        Results dictionary with paths and success status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    try:
        # Fetch issuances
        issuances = fetch_issuances()
        issuances_df = transform_issuances_to_df(issuances)
        issuances_path = output_dir / "isometric_issuances.csv"
        issuances_df.to_csv(issuances_path, index=False)
        results["issuances"] = {
            "path": str(issuances_path),
            "count": len(issuances_df),
            "success": True,
        }
        print(f"  Saved {len(issuances_df)} issuances to {issuances_path}")
    except Exception as e:
        print(f"  Error fetching issuances: {e}")
        results["issuances"] = {"success": False, "error": str(e)}
    
    try:
        # Fetch retirements
        retirements = fetch_retirements()
        retirements_df = transform_retirements_to_df(retirements)
        retirements_path = output_dir / "isometric_retirements.csv"
        retirements_df.to_csv(retirements_path, index=False)
        results["retirements"] = {
            "path": str(retirements_path),
            "count": len(retirements_df),
            "success": True,
        }
        print(f"  Saved {len(retirements_df)} retirements to {retirements_path}")
    except Exception as e:
        print(f"  Error fetching retirements: {e}")
        results["retirements"] = {"success": False, "error": str(e)}
    
    try:
        # Fetch projects
        projects = fetch_projects()
        projects_df = transform_projects_to_df(projects)
        projects_path = output_dir / "isometric_projects.csv"
        projects_df.to_csv(projects_path, index=False)
        results["projects"] = {
            "path": str(projects_path),
            "count": len(projects_df),
            "success": True,
        }
        print(f"  Saved {len(projects_df)} projects to {projects_path}")
    except Exception as e:
        print(f"  Error fetching projects: {e}")
        results["projects"] = {"success": False, "error": str(e)}
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch data from Isometric's public GraphQL API"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "raw",
        help="Directory to save downloaded CSV files",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Isometric GraphQL API Fetcher")
    print("=" * 60)
    print(f"API endpoint: {GRAPHQL_URL}")
    print(f"Output directory: {args.output_dir.absolute()}")
    print()

    results = fetch_all(args.output_dir)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_success = True
    for name, result in results.items():
        if result.get("success"):
            print(f"  ✓ {name}: {result['count']} records → {result['path']}")
        else:
            print(f"  ✗ {name}: {result.get('error', 'Unknown error')}")
            all_success = False

    # Save metadata
    metadata_path = args.output_dir / "fetch_metadata.json"
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "api_endpoint": GRAPHQL_URL,
        "results": results,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata saved to: {metadata_path}")

    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
