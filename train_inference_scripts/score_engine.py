"""
TERRATRENDS SCORE ENGINE
========================
Takes a CSV of businesses, outputs 1Y / 3Y / 5Y scores.

Score(X) = (0.5 * P_survival(X) + 0.5 * revenue_score(X)) * 100

Where:
  P_survival(X)    = adjusted survival probability at horizon X
                     (BLS base rate, adjusted for age, size, county economic outlook)
  revenue_score(X) = sigmoid of compounded sector growth rate
                     (how much the sector is expected to grow in this county)

Revenue projection:
  projected_revenue = current_revenue * compound_multiplier
  where compound_multiplier = product of (1 + predicted_annual_growth_rate)
  for each year from 2025 to the target year.

Input CSV columns (required):
  - business_name   : str
  - county          : str   must match county names in merged_data.csv (e.g. "Fulton, GA")
  - sector          : str   must match sector_columns (e.g. "Health care and social assistance")
  - current_revenue : float (USD)
  - employee_count  : int

Input CSV columns (optional):
  - founding_year   : int   used for age-based survival adjustment

Output CSV adds:
  - score_1y, score_3y, score_5y
  - survival_1y, survival_3y, survival_5y
  - revenue_score_1y, revenue_score_3y, revenue_score_5y
  - projected_revenue_1y, projected_revenue_3y, projected_revenue_5y
  - sector_growth_1y, sector_growth_3y, sector_growth_5y  (net %, e.g. 18.4)
  - status, notes

Usage:
  python score_engine.py --input businesses.csv --output scores.csv
  python score_engine.py --input businesses.csv --output scores.csv --data data/merged_data.csv --model trained_model.pkl
"""

import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from survival_base_rates import compute_survival_probability
from lstm_forecaster import forecast_multiple_horizons

CURRENT_YEAR     = 2025
BASE_DATA_YEAR   = 2023
HORIZONS         = ["1y", "3y", "5y"]
REQUIRED_COLUMNS = ["business_name", "county", "sector", "current_revenue", "employee_count"]


def _revenue_growth_to_score(total_growth: float) -> float:
    """
    Map compounded net growth (decimal) to [0, 1] via sigmoid.
    0% growth  → 0.50
    +20% total → ~0.69
    +50% total → ~0.85
    -20% total → ~0.31
    """
    return float(1 / (1 + np.exp(-4 * total_growth)))


def _validate_input(df: pd.DataFrame):
    warnings_out = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df.copy()
    df["current_revenue"] = pd.to_numeric(df["current_revenue"], errors="coerce")
    df["employee_count"]  = pd.to_numeric(df["employee_count"],  errors="coerce").fillna(5).astype(int)

    if "founding_year" in df.columns:
        df["founding_year"] = pd.to_numeric(df["founding_year"], errors="coerce")
    else:
        df["founding_year"] = np.nan
        warnings_out.append("founding_year not provided — business age defaulted to 5 years")

    null_rev = df["current_revenue"].isna().sum()
    if null_rev > 0:
        warnings_out.append(f"{null_rev} rows have missing current_revenue — projected_revenue will be blank")

    return df, warnings_out


def score_business(row: pd.Series, econ_data: pd.DataFrame, model_path: str) -> dict:
    """Score a single business across 1Y / 3Y / 5Y horizons."""

    result = {
        "business_name": row["business_name"],
        "county":        row["county"],
        "sector":        row["sector"],
        "status":        "ok",
        "notes":         "",
    }
    notes = []

    # Business age
    if pd.notna(row.get("founding_year")):
        business_age = max(0, CURRENT_YEAR - int(row["founding_year"]))
    else:
        business_age = 5
        notes.append("age defaulted to 5yr")

    employee_count  = int(row.get("employee_count", 5))
    current_revenue = row.get("current_revenue")

    # Get economic forecasts for all horizons
    try:
        forecasts = forecast_multiple_horizons(
            county=row["county"],
            sector=row["sector"],
            df=econ_data,
            base_year=BASE_DATA_YEAR,
            model_path=model_path
        )
    except Exception as e:
        result["status"] = "error"
        result["notes"]  = f"Forecast failed: {e}"
        for h in HORIZONS:
            for col in ["score", "survival", "revenue_score", "projected_revenue", "sector_growth"]:
                result[f"{col}_{h}"] = np.nan
        return result

    for h in HORIZONS:
        fc = forecasts.get(h)

        if fc is None:
            notes.append(f"no forecast for {h}")
            for col in ["score", "survival", "revenue_score", "projected_revenue", "sector_growth"]:
                result[f"{col}_{h}"] = np.nan
            continue

        # Survival probability — adjusted by economic outlook and firm characteristics
        p_survival = compute_survival_probability(
            sector=row["sector"],
            business_age_years=business_age,
            employee_count=employee_count,
            economic_adjustment=fc["economic_adjustment"],
            horizon=h
        )

        # Revenue score — sigmoid of compounded net sector growth
        total_growth  = fc["total_growth"]          # e.g. 0.18 = +18% total
        revenue_score = _revenue_growth_to_score(total_growth)

        # Projected revenue — current revenue × compound multiplier
        compound = fc["compound_multiplier"]        # e.g. 1.18
        if pd.notna(current_revenue) and current_revenue > 0:
            projected_revenue = round(current_revenue * compound, 2)
        else:
            projected_revenue = np.nan

        score = (0.5 * p_survival + 0.5 * revenue_score) * 100

        result[f"score_{h}"]             = round(score, 2)
        result[f"survival_{h}"]          = round(p_survival, 4)
        result[f"revenue_score_{h}"]     = round(revenue_score, 4)
        result[f"projected_revenue_{h}"] = projected_revenue
        result[f"sector_growth_{h}"]     = round(total_growth * 100, 2)   # as percent

    result["notes"] = "; ".join(notes) if notes else ""
    return result


def run(input_path: str, output_path: str, data_path: str, model_path: str):

    print("\n" + "="*70)
    print("  TERRATRENDS SCORE ENGINE")
    print("="*70)

    print(f"\nLoading business data:  {input_path}")
    input_df, input_warnings = _validate_input(pd.read_csv(input_path))
    print(f"✓ {len(input_df)} businesses loaded")
    for w in input_warnings:
        print(f"  ⚠ {w}")

    print(f"\nLoading economic data:  {data_path}")
    econ_data = pd.read_csv(data_path).sort_values(["County", "Year"])
    print(f"✓ {len(econ_data)} rows, {econ_data['County'].nunique()} counties")

    missing_counties = set(input_df["county"].unique()) - set(econ_data["County"].unique())
    if missing_counties:
        print(f"\n  ⚠ Counties not found in economic data: {missing_counties}")

    print(f"\nScoring across 1Y / 3Y / 5Y horizons...")
    print("-"*70)

    results = []
    for i, (_, row) in enumerate(input_df.iterrows(), 1):
        if i % 10 == 0 or i == len(input_df):
            print(f"  {i}/{len(input_df)}", end="\r")
        results.append(score_business(row, econ_data, model_path))

    print(f"\n✓ Done")

    results_df = pd.DataFrame(results)

    # Merge with input, scores first
    output_df = pd.concat([
        input_df.reset_index(drop=True),
        results_df.drop(columns=["business_name", "county", "sector"], errors="ignore").reset_index(drop=True)
    ], axis=1)

    id_cols    = ["business_name", "county", "sector", "current_revenue", "employee_count"]
    score_cols = [c for c in output_df.columns if c.startswith("score_")]
    other_cols = [c for c in output_df.columns if c not in id_cols + score_cols]
    output_df  = output_df[id_cols + score_cols + other_cols]

    output_df.to_csv(output_path, index=False)

    # Summary
    ok_count  = (results_df["status"] == "ok").sum()
    err_count = (results_df["status"] == "error").sum()

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Scored: {ok_count}   Errors: {err_count}")

    print(f"\n✓ Output saved to: {output_path}")
    print("="*70 + "\n")

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraTraends Business Score Engine")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data",   default="data/merged_data.csv")
    parser.add_argument("--model",  default="lstm_model_v2.pt")
    args = parser.parse_args()

    run(args.input, args.output, args.data, args.model)