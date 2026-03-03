"""
QCEW DATA PIPELINE FOR TERRATRENDS
====================================
Fetches BLS Quarterly Census of Employment and Wages (QCEW) data for all
159 Georgia counties across all 20 TerraTraends sectors, 2002-2023.

Data source: BLS QCEW annual flat files
  https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip

What this adds to merged_data.csv:
  - avg_employment:          annual average workers per sector per county
  - total_wages:             total annual wages paid (dollars)
  - avg_wage_per_employee:   wages / employment (productivity/quality proxy)
  - avg_establishments:      number of businesses in that sector/county
  - employment_growth_rate:  YoY % change in employment (direct growth signal)
  - wage_growth_rate:        YoY % change in total wages

Why this matters for the model:
  Current sector values are contribution-to-GDP in pp — they conflate sector
  SIZE with sector GROWTH. A county where accommodation contributes 0.15pp
  could be large-and-slow or small-and-fast. Employment/wage data breaks that
  conflation and gives the model a direct, interpretable growth rate target.

Outputs:
  data/qcew_long.csv    — long format: one row per county/year/sector
  data/qcew_wide.csv    — wide format: one row per county/year, ready to
                          merge with merged_data.csv on (GeoID, Year)

Usage:
  python fetch_qcew.py                          # fetch all years
  python fetch_qcew.py --years 2020 2021 2022   # specific years only
  python fetch_qcew.py --merge                  # also produce merged_enriched.csv

Requirements:
  pip install requests pandas tqdm

BLS API key (free): https://data.bls.gov/registrationEngine/
  Not strictly required for flat file downloads, but register anyway to
  avoid rate limiting on repeated runs. Pass via --api-key or BLS_API_KEY env var.
"""

import os
import io
import time
import zipfile
import argparse
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
YEARS        = list(range(2002, 2024))   # 2002-2023 inclusive
GA_FIPS_PREFIX = "13"                    # all Georgia county FIPS start with 13
AGGLVL_COUNTY_SUPERSECTOR = "74"         # county × NAICS supersector

# Flat file URL pattern
QCEW_URL = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip"

# Columns to read from the flat file (keeps memory usage low)
FLAT_FILE_COLS = [
    "area_fips",
    "own_code",
    "industry_code",
    "agglvl_code",
    "year",
    "annual_avg_emplvl",
    "total_annual_wages",
    "annual_avg_estabs",
    "disclosure_code",      # 'N' = suppressed, '' = ok
]

# -------------------------------------------------------------------
# NAICS / QCEW industry code mapping to TerraTraends sector names
#
# Notes:
#   - own_code 5 = private sector (used for most sectors)
#   - own_code 0 = total all ownership (used for agriculture, which is
#     largely self-employed and undercounted in private-only)
#   - own_code 2 = state & local government (for government sector)
#   - industry_code 1012/1013 are QCEW-specific aggregation codes for
#     nondurable/durable manufacturing at the county supersector level
#   - Professional and business services uses NAICS 54 (professional/
#     scientific/technical); NAICS 56 (admin) is mapped separately
# -------------------------------------------------------------------
SECTOR_MAP = {
    "Accommodation and food services": {
        "industry_code": "72",
        "own_code":      "5",
    },
    "Administrative and support and waste management and remediation services": {
        "industry_code": "56",
        "own_code":      "5",
    },
    "Agriculture, forestry, fishing and hunting": {
        "industry_code": "11",
        "own_code":      "0",   # total ownership — ag is largely self-employed
    },
    "Arts, entertainment, and recreation": {
        "industry_code": "71",
        "own_code":      "5",
    },
    "Construction": {
        "industry_code": "23",
        "own_code":      "5",
    },
    "Durable goods manufacturing": {
        "industry_code": "1013",  # QCEW county-level durable mfg aggregation
        "own_code":      "5",
    },
    "Educational services": {
        "industry_code": "61",
        "own_code":      "5",
    },
    "Finance and insurance": {
        "industry_code": "52",
        "own_code":      "5",
    },
    "Government and government enterprises": {
        "industry_code": "92",
        "own_code":      "2",   # state & local government
    },
    "Health care and social assistance": {
        "industry_code": "62",
        "own_code":      "5",
    },
    "Information": {
        "industry_code": "51",
        "own_code":      "5",
    },
    "Natural resources and mining": {
        "industry_code": "10",
        "own_code":      "5",
    },
    "Nondurable goods manufacturing": {
        "industry_code": "1012",  # QCEW county-level nondurable mfg aggregation
        "own_code":      "5",
    },
    "Other services (except government and government enterprises)": {
        "industry_code": "81",
        "own_code":      "5",
    },
    "Professional and business services": {
        "industry_code": "54",   # professional/scientific/technical services
        "own_code":      "5",
    },
    "Real estate and rental and leasing": {
        "industry_code": "53",
        "own_code":      "5",
    },
    "Retail trade": {
        "industry_code": "44-45",
        "own_code":      "5",
    },
    "Transportation and warehousing": {
        "industry_code": "48-49",
        "own_code":      "5",
    },
    "Utilities": {
        "industry_code": "22",
        "own_code":      "5",
    },
    "Wholesale trade": {
        "industry_code": "42",
        "own_code":      "5",
    },
}

# Build reverse lookup: (industry_code, own_code) -> sector name
CODE_TO_SECTOR = {
    (v["industry_code"], v["own_code"]): k
    for k, v in SECTOR_MAP.items()
}

# All unique (industry_code, own_code) pairs we want to keep
TARGET_PAIRS = set(CODE_TO_SECTOR.keys())

# FIPS -> county name lookup (built from merged_data.csv at runtime)
FIPS_TO_COUNTY = {}


# -------------------------------------------------------------------
# Download helpers
# -------------------------------------------------------------------
def _download_year(year: int, cache_dir: Path, api_key: str = None) -> pd.DataFrame:
    """
    Download and filter QCEW annual flat file for one year.
    Returns a DataFrame with Georgia rows for target sectors only.
    Caches the raw zip to disk to avoid re-downloading on reruns.
    """
    cache_path = cache_dir / f"qcew_{year}_ga.parquet"
    if cache_path.exists():
        log.info(f"  {year}: loading from cache")
        return pd.read_parquet(cache_path)

    url = QCEW_URL.format(year=year)
    log.info(f"  {year}: downloading from {url}")

    headers = {}
    if api_key:
        headers["Registration-Key"] = api_key

    # Stream download with retry
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=120, stream=True)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to download {year} after 3 attempts: {e}")
            log.warning(f"  {year}: attempt {attempt+1} failed, retrying in 5s...")
            time.sleep(5)

    # Unzip in memory and read CSV
    log.info(f"  {year}: extracting and filtering...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # The flat file inside is named {year}.annual.singlefile.csv
        csv_name = f"{year}.annual.singlefile.csv"
        if csv_name not in zf.namelist():
            # Some years use a different naming convention
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]

        with zf.open(csv_name) as f:
            # Read in chunks to keep memory low — full file is ~300MB uncompressed.
            # All filtering is vectorized — no row-by-row apply().
            chunks = []

            # Pre-build sets for cheap pre-filtering before exact pair match
            target_industry_codes = {pair[0] for pair in TARGET_PAIRS}
            target_own_codes      = {pair[1] for pair in TARGET_PAIRS}
            # Encode valid (industry, own) pairs as "ind|own" strings for isin check
            target_pair_strings   = {f"{i}|{o}" for i, o in TARGET_PAIRS}

            for chunk in pd.read_csv(
                f,
                usecols=FLAT_FILE_COLS,
                dtype=str,
                chunksize=500_000,   # larger chunks = fewer iterations = faster
            ):
                # Step 1: cheap filters first to aggressively shrink the chunk
                ga_mask     = chunk["area_fips"].str.startswith(GA_FIPS_PREFIX)
                agglvl_mask = chunk["agglvl_code"] == AGGLVL_COUNTY_SUPERSECTOR
                ind_mask    = chunk["industry_code"].isin(target_industry_codes)
                own_mask    = chunk["own_code"].isin(target_own_codes)
                chunk = chunk[ga_mask & agglvl_mask & ind_mask & own_mask]
                if chunk.empty:
                    continue

                # Step 2: exact pair match via concatenated key — still vectorized
                pair_key = chunk["industry_code"] + "|" + chunk["own_code"]
                chunk = chunk[pair_key.isin(target_pair_strings)]
                if len(chunk):
                    chunks.append(chunk)

    if not chunks:
        log.warning(f"  {year}: no matching rows found")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Convert numeric columns
    for col in ["annual_avg_emplvl", "total_annual_wages", "annual_avg_estabs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Suppress disclosed-as-confidential values (disclosure_code == 'N')
    suppress_mask = df["disclosure_code"] == "N"
    df.loc[suppress_mask, ["annual_avg_emplvl", "total_annual_wages", "annual_avg_estabs"]] = np.nan
    log.info(f"  {year}: {len(df):,} rows, {suppress_mask.sum()} suppressed")

    # Cache for next run
    df.to_parquet(cache_path, index=False)
    return df


# -------------------------------------------------------------------
# Processing
# -------------------------------------------------------------------
def _process_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw QCEW rows into clean long-format records:
    one row per (GeoID, Year, Sector).
    """
    rows = []
    for _, row in df_raw.iterrows():
        sector = CODE_TO_SECTOR.get((row["industry_code"], row["own_code"]))
        if not sector:
            continue

        fips = int(row["area_fips"])
        year = int(row["year"])
        emp  = row["annual_avg_emplvl"]
        wage = row["total_annual_wages"]
        estb = row["annual_avg_estabs"]

        # Wage per employee (NaN if either is missing or zero employment)
        if pd.notna(emp) and pd.notna(wage) and emp > 0:
            wage_per_emp = wage / emp
        else:
            wage_per_emp = np.nan

        rows.append({
            "GeoID":                fips,
            "County":               FIPS_TO_COUNTY.get(fips, f"Unknown_{fips}"),
            "Year":                 year,
            "Sector":               sector,
            "avg_employment":       emp,
            "total_wages":          wage,
            "avg_wage_per_employee": wage_per_emp,
            "avg_establishments":   estb,
        })

    return pd.DataFrame(rows)


def _add_growth_rates(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Add YoY growth rates for employment and wages.
    Computed within each (GeoID, Sector) group, sorted by Year.
    """
    df_long = df_long.sort_values(["GeoID", "Sector", "Year"]).copy()

    df_long["employment_growth_rate"] = (
        df_long.groupby(["GeoID", "Sector"])["avg_employment"]
        .pct_change()
    )
    df_long["wage_growth_rate"] = (
        df_long.groupby(["GeoID", "Sector"])["total_wages"]
        .pct_change()
    )

    # Cap extreme growth rates at ±5.0 (500%) to match winsorization philosophy
    # These occur in very small counties when one establishment opens/closes
    for col in ["employment_growth_rate", "wage_growth_rate"]:
        df_long[col] = df_long[col].clip(-5.0, 5.0)

    return df_long


def _to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long format to wide format.
    One row per (GeoID, County, Year).
    Columns: {sector}_emp, {sector}_wages, {sector}_wage_per_emp,
              {sector}_estabs, {sector}_emp_growth, {sector}_wage_growth

    Uses abbreviated column name prefixes to keep column names manageable.
    """
    # Short sector name map for column prefixes
    SECTOR_SHORT = {
        "Accommodation and food services":                                          "accom",
        "Administrative and support and waste management and remediation services": "admin",
        "Agriculture, forestry, fishing and hunting":                               "ag",
        "Arts, entertainment, and recreation":                                      "arts",
        "Construction":                                                             "const",
        "Durable goods manufacturing":                                              "durable",
        "Educational services":                                                     "edu",
        "Finance and insurance":                                                    "finance",
        "Government and government enterprises":                                    "govt",
        "Health care and social assistance":                                        "health",
        "Information":                                                              "info",
        "Natural resources and mining":                                             "mining",
        "Nondurable goods manufacturing":                                           "nondurable",
        "Other services (except government and government enterprises)":            "other_svc",
        "Professional and business services":                                       "professional",
        "Real estate and rental and leasing":                                       "realestate",
        "Retail trade":                                                             "retail",
        "Transportation and warehousing":                                           "transport",
        "Utilities":                                                                "utilities",
        "Wholesale trade":                                                          "wholesale",
    }

    value_cols = [
        "avg_employment",
        "total_wages",
        "avg_wage_per_employee",
        "avg_establishments",
        "employment_growth_rate",
        "wage_growth_rate",
    ]

    wide_frames = []
    for sector, grp in df_long.groupby("Sector"):
        prefix = SECTOR_SHORT.get(sector, sector[:10].lower().replace(" ", "_"))
        grp = grp[["GeoID", "County", "Year"] + value_cols].copy()
        rename = {col: f"{prefix}_{col}" for col in value_cols}
        grp = grp.rename(columns=rename)
        wide_frames.append(grp.set_index(["GeoID", "County", "Year"]))

    wide = pd.concat(wide_frames, axis=1).reset_index()
    wide = wide.sort_values(["GeoID", "Year"]).reset_index(drop=True)
    return wide


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def run(
    years: list,
    output_dir: str,
    cache_dir: str,
    merged_data_path: str,
    api_key: str = None,
    produce_merged: bool = False,
):
    output_path = Path(output_dir)
    cache_path  = Path(cache_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Build FIPS -> county name lookup from existing merged_data.csv
    global FIPS_TO_COUNTY
    if Path(merged_data_path).exists():
        md = pd.read_csv(merged_data_path)
        FIPS_TO_COUNTY = dict(zip(md["GeoID"], md["County"]))
        log.info(f"Loaded {len(FIPS_TO_COUNTY)} county FIPS mappings from {merged_data_path}")
    else:
        log.warning(f"merged_data.csv not found at {merged_data_path} — county names will be generic")

    # Download and process each year
    all_years = []
    log.info(f"\nFetching QCEW data for {len(years)} years ({years[0]}-{years[-1]})...")
    for year in tqdm(years, desc="Years"):
        try:
            raw = _download_year(year, cache_path, api_key)
            if raw.empty:
                continue
            processed = _process_raw(raw)
            all_years.append(processed)
        except Exception as e:
            log.error(f"  {year}: FAILED — {e}")
            continue

    if not all_years:
        raise RuntimeError("No data fetched. Check your internet connection and try again.")

    # Combine all years
    log.info("\nCombining all years...")
    df_long = pd.concat(all_years, ignore_index=True)
    df_long = df_long.sort_values(["GeoID", "Sector", "Year"]).reset_index(drop=True)

    # Add growth rates
    log.info("Computing growth rates...")
    df_long = _add_growth_rates(df_long)

    # Save long format
    long_path = output_path / "qcew_long.csv"
    df_long.to_csv(long_path, index=False)
    log.info(f"✓ Long format saved: {long_path}  ({len(df_long):,} rows)")

    # Save wide format
    log.info("Pivoting to wide format...")
    df_wide = _to_wide(df_long)
    wide_path = output_path / "qcew_wide.csv"
    df_wide.to_csv(wide_path, index=False)
    log.info(f"✓ Wide format saved: {wide_path}  ({len(df_wide):,} rows, {len(df_wide.columns)} cols)")

    # Print coverage summary
    _print_summary(df_long)

    # Optionally merge with merged_data.csv
    if produce_merged and Path(merged_data_path).exists():
        log.info("\nMerging with merged_data.csv...")
        _produce_merged(df_wide, merged_data_path, output_path)

    return df_long, df_wide


def _print_summary(df_long: pd.DataFrame):
    """Print data coverage and quality summary."""
    print("\n" + "="*65)
    print("  QCEW DATA COVERAGE SUMMARY")
    print("="*65)

    total_cells      = len(df_long)
    suppressed_emp   = df_long["avg_employment"].isna().sum()
    suppressed_wage  = df_long["total_wages"].isna().sum()

    print(f"  Total county/year/sector records:  {total_cells:,}")
    print(f"  Employment suppressed (BLS privacy): {suppressed_emp:,} ({suppressed_emp/total_cells*100:.1f}%)")
    print(f"  Wage suppressed:                     {suppressed_wage:,} ({suppressed_wage/total_cells*100:.1f}%)")
    print()

    print("  Suppression rate by sector (employment):")
    supp_by_sector = (
        df_long.groupby("Sector")["avg_employment"]
        .apply(lambda x: x.isna().mean() * 100)
        .sort_values(ascending=False)
    )
    for sector, pct in supp_by_sector.items():
        flag = " ⚠ HIGH" if pct > 30 else ""
        print(f"    {sector[:52]:<52} {pct:>5.1f}%{flag}")

    print()
    print("  Sample employment figures (2022, select counties/sectors):")
    sample = df_long[
        (df_long["Year"] == 2022) &
        (df_long["County"].isin(["Fulton, GA", "Chatham, GA", "Hall, GA"])) &
        (df_long["Sector"].isin(["Health care and social assistance",
                                  "Transportation and warehousing",
                                  "Accommodation and food services"]))
    ][["County", "Sector", "avg_employment", "avg_wage_per_employee", "employment_growth_rate"]]
    print(sample.to_string(index=False))
    print("="*65 + "\n")


def _produce_merged(df_wide: pd.DataFrame, merged_path: str, output_path: Path):
    """Merge QCEW wide data with existing merged_data.csv."""
    merged = pd.read_csv(merged_path)
    enriched = merged.merge(df_wide, on=["GeoID", "Year"], how="left", suffixes=("", "_qcew"))

    # Drop any duplicate County columns from merge
    dup_cols = [c for c in enriched.columns if c.endswith("_qcew")]
    enriched = enriched.drop(columns=dup_cols)

    out_path = output_path / "merged_enriched.csv"
    enriched.to_csv(out_path, index=False)

    new_cols = len(df_wide.columns) - 3  # subtract GeoID, County, Year
    matched  = enriched[enriched.columns[3]].notna().sum()  # rough match check

    log.info(f"✓ Enriched dataset saved: {out_path}")
    log.info(f"  Original columns: {len(merged.columns)}")
    log.info(f"  New QCEW columns: {new_cols}")
    log.info(f"  Total columns:    {len(enriched.columns)}")
    log.info(f"  Rows with QCEW data: {matched:,} / {len(enriched):,}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch QCEW employment/wage data for Georgia counties"
    )
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=YEARS,
        help="Years to fetch (default: 2002-2023)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/",
        help="Directory for output CSV files (default: data/)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/qcew_cache/",
        help="Directory for caching raw downloads (default: data/qcew_cache/)"
    )
    parser.add_argument(
        "--merged-data", type=str, default="data/merged_data.csv",
        help="Path to existing merged_data.csv for FIPS lookup"
    )
    parser.add_argument(
        "--api-key", type=str,
        default=os.environ.get("BLS_API_KEY"),
        help="BLS API key (or set BLS_API_KEY env var). Free at data.bls.gov/registrationEngine/"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Also produce merged_enriched.csv combining QCEW with merged_data.csv"
    )

    args = parser.parse_args()

    log.info("TerraTraends QCEW Pipeline")
    log.info(f"Years: {args.years[0]}-{args.years[-1]} ({len(args.years)} years)")
    log.info(f"Sectors: {len(SECTOR_MAP)}")
    log.info(f"API key: {'provided' if args.api_key else 'not provided (may hit rate limits)'}")

    run(
        years=args.years,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        merged_data_path=args.merged_data,
        api_key=args.api_key,
        produce_merged=args.merge,
    )