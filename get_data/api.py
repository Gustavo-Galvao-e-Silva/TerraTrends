import io
import os
import time
from pathlib import Path

import pandas as pd
import requests

from mapping import TARGET_INDUSTRIES

BASE_URL = "https://download.bls.gov/pub/time.series/qc/"  # placeholder; update if needed
OUT_DIR = Path("qcew_ga")
OUT_DIR.mkdir(exist_ok=True)

# Years: NAICS-based data 1990+; you can extend if you want pre-NAICS recon data
START_YEAR = 1990
END_YEAR = 2025  # adjust to latest full year available

# Build a flat set of industry prefixes (2- or 3-digit NAICS)
industry_prefixes = sorted({p for lst in TARGET_INDUSTRIES.values() for p in lst})

def industry_matches(code: str) -> bool:
    if pd.isna(code):
        return False
    code = str(code)
    for pref in industry_prefixes:
        if code.startswith(pref):
            return True
    return False

def download_annual_singlefile(year: int) -> pd.DataFrame:
    # For QCEW, the NAICS-based annual singlefile CSV URL pattern is on the data-files page.
    # Example pattern often used:
    # https://download.bls.gov/pub/time.series/qcew/qcew_YYYY_annual_singlefile.zip
    url = f"https://download.bls.gov/pub/time.series/qcew/qcew_{year}_annual_singlefile.csv"
    print(f"Year {year}: downloading {url}")
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Failed {year}: {r.status_code}")
    return pd.read_csv(io.StringIO(r.text))

def process_year(year: int) -> pd.DataFrame:
    df = download_annual_singlefile(year)

    # Normalize column names (based on QCEW layout docs)
    df.columns = [c.lower() for c in df.columns]

    # Keep GA counties
    df = df[df["area_fips"].astype(str).str.startswith("13")]

    # Filter to industry prefixes
    df = df[df["industry_code"].apply(industry_matches)]

    # Optional: handle government differently using ownership code
    # Example: if you want "government and government enterprises" as all ownership != 5 (private)
    # you can tag government rows here.

    df["year"] = year
    return df

def main():
    all_years = []
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            df_year = process_year(year)
        except Exception as e:
            print(f"Error {year}: {e}")
            continue
        all_years.append(df_year)
        time.sleep(0.3)  # be polite

    if not all_years:
        raise RuntimeError("No data collected")

    df_all = pd.concat(all_years, ignore_index=True)

    # Optional: create a high-level industry label from TARGET_INDUSTRIES
    def label_industry(code: str) -> str:
        if pd.isna(code):
            return None
        code = str(code)
        for lbl, prefs in TARGET_INDUSTRIES.items():
            for p in prefs:
                if code.startswith(p):
                    return lbl
        return "Other"

    df_all["industry_group"] = df_all["industry_code"].apply(label_industry)

    out_parquet = OUT_DIR / "ga_qcew_counties_selected_industries.parquet"
    out_csv = OUT_DIR / "ga_qcew_counties_selected_industries.csv"

    df_all.to_parquet(out_parquet, index=False)
    df_all.to_csv(out_csv, index=False)

    print(f"Saved {len(df_all):,} rows to {out_parquet} and {out_csv}")

if __name__ == "__main__":
    main()
