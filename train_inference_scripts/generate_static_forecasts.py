"""
Generate static LSTM forecasts for all counties × sectors × horizons.

Runs lstm_forecaster once (model loads and caches on first call), sweeps
every (county, sector) pair, and writes a flat CSV with one row per
(county, sector, horizon).

Usage:
    python generate_static_forecasts.py \
        --data data/merged_data_v2.csv \
        --model lstm_model_v2.pt \
        --out static_forecasts.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from lstm_forecaster import _load, forecast_multiple_horizons

HORIZONS   = ["1y", "3y", "5y"]
BASE_YEAR  = 2023


def _get_county_pop(county: str, df: pd.DataFrame):
    rows = df[df["County"] == county]["TOT_POP"].dropna()
    return float(rows.iloc[-1]) if len(rows) > 0 else None


def generate(data_path: str, model_path: str, out_path: str):
    df = pd.read_csv(data_path).sort_values(["County", "Year"])
    print(f"Loaded data: {df['County'].nunique()} counties")

    # Load model once to get the full list of counties and sectors
    _, pkg = _load(model_path)
    all_counties = sorted(pkg["county2idx"].keys())
    all_sectors  = sorted(pkg["sector2idx"].keys())
    print(f"Model covers {len(all_counties)} counties × {len(all_sectors)} sectors "
          f"= {len(all_counties) * len(all_sectors):,} combinations")

    rows   = []
    errors = 0
    total  = len(all_counties) * len(all_sectors)
    done   = 0

    for county in all_counties:
        county_pop = _get_county_pop(county, df)

        for sector in all_sectors:
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total} ({100*done/total:.1f}%)  errors={errors}")

            try:
                forecasts = forecast_multiple_horizons(
                    county=county,
                    sector=sector,
                    df=df,
                    base_year=BASE_YEAR,
                    model_path=model_path,
                    county_pop=county_pop,
                )
            except Exception as e:
                errors += len(HORIZONS)
                for horizon in HORIZONS:
                    rows.append({
                        "county":              county,
                        "sector":              sector,
                        "horizon":             horizon,
                        "compound_multiplier": np.nan,
                        "total_growth":        np.nan,
                        "annual_growth_rate":  np.nan,
                        "economic_adjustment": np.nan,
                        "revenue_score":       np.nan,
                        "p_shrinking":         np.nan,
                        "p_flat":              np.nan,
                        "p_moderate":          np.nan,
                        "p_strong":            np.nan,
                        "population":          int(county_pop) if county_pop else None,
                        "pop_dampened":        bool(county_pop and county_pop < 40_000),
                        "status":              "error",
                        "notes":               str(e),
                    })
                continue

            for horizon in HORIZONS:
                fc = forecasts[horizon]
                cp = fc["class_probs"]
                rows.append({
                    "county":              county,
                    "sector":              sector,
                    "horizon":             horizon,
                    "compound_multiplier": round(fc["compound_multiplier"], 6),
                    "total_growth":        round(fc["total_growth"], 6),
                    "annual_growth_rate":  round(fc["annual_growth_rate"], 6),
                    "economic_adjustment": round(fc["economic_adjustment"], 6),
                    "revenue_score":       round(fc["revenue_score"], 6),
                    "p_shrinking":         round(cp[0], 6),
                    "p_flat":              round(cp[1], 6),
                    "p_moderate":          round(cp[2], 6),
                    "p_strong":            round(cp[3], 6),
                    "population":          int(county_pop) if county_pop else None,
                    "pop_dampened":        bool(county_pop and county_pop < 40_000),
                    "status":              "ok",
                    "notes":               "",
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    ok = (out_df["status"] == "ok").sum()
    print(f"\nDone. {ok} rows OK, {errors} errors.")
    print(f"Saved to: {out_path}  ({len(out_df):,} rows total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="data/merged_data_v2.csv")
    parser.add_argument("--model", default="lstm_model_v2.pt")
    parser.add_argument("--out",   default="static_forecasts.csv")
    args = parser.parse_args()

    generate(args.data, args.model, args.out)
