"""
TERRATRENDS SCORE ENGINE v2 (CORRECTED)
========================================
Given a business's details, scores all 159 Georgia counties and
returns them ranked best-to-worst for expansion.

Changes vs original:
  1. Passes county population to forecast_multiple_horizons() so the
     forecaster can apply small-county confidence dampening. Counties
     under 40k pop have class probabilities pulled toward uniform to
     prevent single-establishment noise from producing extreme scores.
  2. Population column added to output CSV for transparency.
  3. No other interface changes.

Flow:
  1. User inputs: sector, revenue, employee_count, founding_year
  2. For each of 159 counties:
       - Run LSTM forecast for that county + sector (with pop dampening)
       - Compute survival probability (adjusted for county outlook)
       - Compute revenue score from projected growth
       - Score = (0.5 * P_survival + 0.5 * revenue_score) * 100
  3. Return all 159 counties ranked by score

Usage:
  python score_engine.py --sector "Health care and social assistance" \
                         --revenue 500000 \
                         --employees 8 \
                         --founding-year 2015 \
                         --horizon 3y \
                         --output results.csv

  python score_engine.py --input business.csv --output results.csv
  (CSV must have: sector, current_revenue, employee_count, founding_year)
"""

import argparse
import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")

from survival_base_rates import compute_survival_probability, init_survival_model
from lstm_forecaster import forecast_multiple_horizons

CURRENT_YEAR   = 2025
BASE_DATA_YEAR = 2023
HORIZONS       = ["1y", "3y", "5y"]

# Population dampening threshold (passed to forecaster).
# Counties below this have class probabilities pulled toward uniform.
POP_DAMPEN_THRESHOLD = 40_000


def _get_county_pop(county: str, econ_data: pd.DataFrame) -> float:
    """Return the most recent non-null population for a county."""
    rows = econ_data[econ_data["County"] == county]["TOT_POP"].dropna()
    return float(rows.iloc[-1]) if len(rows) > 0 else None


def score_all_counties(
    sector: str,
    current_revenue: float,
    employee_count: int,
    founding_year: int,
    econ_data: pd.DataFrame,
    model_path: str = "lstm_model_v2.pt",
    horizon: str = "3y",
    qcew_path: str = "data/qcew_long.csv",
) -> pd.DataFrame:
    """
    Score all 159 Georgia counties for a given business profile.

    Parameters
    ----------
    sector          : str   — must match sector column names exactly
    current_revenue : float — current annual revenue in USD
    employee_count  : int
    founding_year   : int
    econ_data       : pd.DataFrame — merged_data.csv
    model_path      : str
    horizon         : str   — '1y', '3y', or '5y'
    qcew_path       : str   — path to qcew_long.csv

    Returns
    -------
    pd.DataFrame ranked by score descending, with all 159 counties
    """
    business_age = max(0, CURRENT_YEAR - founding_year)
    counties     = sorted(econ_data["County"].unique())
    results      = []

    # Initialize QCEW-powered survival model once before the county loop
    print(f"\nLoading survival model from {qcew_path}...")
    init_survival_model(qcew_path=qcew_path, merged_path="data/merged_data.csv")

    print(f"\nScoring {len(counties)} counties for '{sector}' ({horizon} horizon)...")
    print(f"Business: {employee_count} employees, ${current_revenue:,.0f} revenue, age {business_age}yr")
    print("-" * 60)

    errors = 0
    for i, county in enumerate(counties, 1):
        if i % 20 == 0 or i == len(counties):
            print(f"  {i}/{len(counties)} counties scored...", end="\r")

        try:
            # Look up population for small-county dampening
            county_pop = _get_county_pop(county, econ_data)

            forecasts = forecast_multiple_horizons(
                county=county,
                sector=sector,
                df=econ_data,
                base_year=BASE_DATA_YEAR,
                model_path=model_path,
                county_pop=county_pop,       # NEW: enables population dampening
            )

            fc = forecasts.get(horizon)
            if fc is None:
                raise ValueError(f"No forecast returned for horizon {horizon}")

            p_survival = compute_survival_probability(
                sector=sector,
                county=county,
                business_age_years=business_age,
                employee_count=employee_count,
                horizon=horizon,
                forecast_year=BASE_DATA_YEAR,
            )

            revenue_score = fc["revenue_score"]
            total_growth  = fc["total_growth"]
            compound      = fc["compound_multiplier"]

            # Multiplicative expected-value score:
            #   expected outcome = compound growth × survival probability
            # Normalised to 0-100 after all counties are scored (below).
            # Store raw expected value here; normalise after the loop.
            expected_val  = compound * p_survival

            projected_revenue = round(current_revenue * compound, 2) \
                                if current_revenue > 0 else np.nan

            class_probs = fc.get("class_probs", [None] * 4)

            results.append({
                "rank":                None,
                "county":              county,
                "population":          int(county_pop) if county_pop else None,
                "score":               None,          # filled after normalisation
                "expected_val":        expected_val,  # raw, used for normalisation
                f"score_{horizon}":    None,
                "survival_prob":       round(p_survival, 4),
                "revenue_score":       round(revenue_score, 4),
                "projected_revenue":   projected_revenue,
                "sector_growth_pct":   round(total_growth * 100, 2),
                "annual_growth_rate":  round(fc["annual_growth_rate"] * 100, 2),
                "economic_adjustment": round(fc["economic_adjustment"], 3),
                "p_shrinking":         round(class_probs[0], 3) if class_probs[0] is not None else None,
                "p_flat":              round(class_probs[1], 3) if class_probs[1] is not None else None,
                "p_moderate":          round(class_probs[2], 3) if class_probs[2] is not None else None,
                "p_strong":            round(class_probs[3], 3) if class_probs[3] is not None else None,
                "status":              "ok",
                "notes":               "pop_dampened" if county_pop and county_pop < POP_DAMPEN_THRESHOLD else "",
            })

        except Exception as e:
            errors += 1
            results.append({
                "rank":               None,
                "county":             county,
                "population":         None,
                "score":              np.nan,
                f"score_{horizon}":   np.nan,
                "survival_prob":      np.nan,
                "revenue_score":      np.nan,
                "projected_revenue":  np.nan,
                "sector_growth_pct":  np.nan,
                "annual_growth_rate": np.nan,
                "economic_adjustment": np.nan,
                "p_shrinking":        None,
                "p_flat":             None,
                "p_moderate":         None,
                "p_strong":           None,
                "status":             "error",
                "notes":              str(e),
            })

    print(f"\n✓ Scored {len(results) - errors}/159 counties ({errors} errors)")

    df_out = pd.DataFrame(results)

    # Normalise expected_val to 0-100 relative to the best county in this run
    ok_mask   = df_out["status"] == "ok"
    ev_max    = df_out.loc[ok_mask, "expected_val"].max()
    ev_min    = df_out.loc[ok_mask, "expected_val"].min()
    ev_range  = ev_max - ev_min if ev_max > ev_min else 1.0

    df_out.loc[ok_mask, "score"] = (
        (df_out.loc[ok_mask, "expected_val"] - ev_min) / ev_range * 100
    ).round(2)
    df_out[f"score_{horizon}"] = df_out["score"]

    # Tier labels based on survival and growth
    def _tier(row):
        if row["status"] != "ok":
            return "N/A"
        surv   = row["survival_prob"]
        growth = row["sector_growth_pct"]
        if surv >= 0.62 and growth >= 15:
            return "Strong Expand"
        elif surv >= 0.58 or growth >= 10:
            return "Cautious Expand"
        elif surv >= 0.50 and growth >= 0:
            return "Watch"
        else:
            return "Avoid"

    df_out["tier"] = df_out.apply(_tier, axis=1)

    df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)
    df_out["rank"] = df_out.index + 1

    cols = ["rank", "county", "population", "score", "tier", "survival_prob", "revenue_score",
            "projected_revenue", "sector_growth_pct", "annual_growth_rate",
            "economic_adjustment", "p_shrinking", "p_flat", "p_moderate", "p_strong",
            "status", "notes"]
    df_out = df_out[[c for c in cols if c in df_out.columns]]

    return df_out


def print_summary(df: pd.DataFrame, sector: str, horizon: str, top_n: int = 10):
    print(f"\n{'='*70}")
    print(f"  TOP {top_n} COUNTIES — {sector[:40]}")
    print(f"  Horizon: {horizon.upper()}")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'County':<25} {'Score':>6}  {'Survival':>8}  {'Growth':>7}  {'Tier':<16}  {'Pop':>10}")
    print(f"  {'-'*75}")
    for _, row in df.head(top_n).iterrows():
        pop_str = f"{int(row['population']):,}" if pd.notna(row.get('population')) else "N/A"
        print(f"  {int(row['rank']):<5} {row['county']:<25} {row['score']:>6.1f}  "
              f"{row['survival_prob']:>8.3f}  {row['sector_growth_pct']:>6.1f}%  "
              f"{row.get('tier',''):<16}  {pop_str:>10}")

    print(f"\n  BOTTOM 5:")
    print(f"  {'-'*75}")
    for _, row in df.tail(5).iterrows():
        if row["status"] == "ok":
            pop_str = f"{int(row['population']):,}" if pd.notna(row.get('population')) else "N/A"
            print(f"  {int(row['rank']):<5} {row['county']:<25} {row['score']:>6.1f}  "
                  f"{row['survival_prob']:>8.3f}  {row['sector_growth_pct']:>6.1f}%  "
                  f"{row.get('tier',''):<16}  {pop_str:>10}")
    print()


def run_single(
    sector: str,
    current_revenue: float,
    employee_count: int,
    founding_year: int,
    output_path: str,
    data_path: str,
    model_path: str,
    horizon: str = "3y"
):
    print("\n" + "="*60)
    print("  TERRATRENDS — COUNTY EXPANSION RANKER")
    print("="*60)

    econ_data = pd.read_csv(data_path).sort_values(["County", "Year"])
    print(f"✓ Loaded {econ_data['County'].nunique()} counties")

    ranked = score_all_counties(
        sector=sector,
        current_revenue=current_revenue,
        employee_count=employee_count,
        founding_year=founding_year,
        econ_data=econ_data,
        model_path=model_path,
        horizon=horizon
    )

    print_summary(ranked, sector, horizon)

    ranked.to_csv(output_path, index=False)
    print(f"✓ Full rankings saved to: {output_path}")
    print("="*60 + "\n")

    return ranked


def run_batch(
    input_path: str,
    output_dir: str,
    data_path: str,
    model_path: str,
    horizon: str = "3y"
):
    """
    Batch mode: read multiple businesses from CSV, output one
    ranked CSV per business into output_dir.

    Input CSV columns: business_name, sector, current_revenue,
                       employee_count, founding_year
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    businesses = pd.read_csv(input_path)
    econ_data  = pd.read_csv(data_path).sort_values(["County", "Year"])

    print(f"\n{'='*60}")
    print(f"  BATCH MODE — {len(businesses)} businesses")
    print(f"{'='*60}\n")

    for _, biz in businesses.iterrows():
        name = biz.get("business_name", f"business_{_}")
        print(f"\n>>> {name}")

        ranked = score_all_counties(
            sector=str(biz["sector"]),
            current_revenue=float(biz.get("current_revenue", 0)),
            employee_count=int(biz.get("employee_count", 5)),
            founding_year=int(biz.get("founding_year", CURRENT_YEAR - 5)),
            econ_data=econ_data,
            model_path=model_path,
            horizon=horizon
        )

        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        out_path  = os.path.join(output_dir, f"{safe_name}_rankings.csv")
        ranked.to_csv(out_path, index=False)

        print_summary(ranked, str(biz["sector"]), horizon, top_n=5)
        print(f"  Saved: {out_path}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraTraends County Expansion Ranker")

    parser.add_argument("--sector",        type=str,   help="Business sector")
    parser.add_argument("--revenue",       type=float, default=500000, help="Current annual revenue")
    parser.add_argument("--employees",     type=int,   default=10,     help="Employee count")
    parser.add_argument("--founding-year", type=int,   default=2015,   help="Year founded")
    parser.add_argument("--horizon",       type=str,   default="5y",   choices=["1y","3y","5y"])

    parser.add_argument("--input",      type=str, help="Batch input CSV")
    parser.add_argument("--output-dir", type=str, default="rankings/", help="Output dir for batch")

    parser.add_argument("--output", type=str, default="county_rankings.csv")
    parser.add_argument("--data",   type=str, default="data/merged_data.csv")
    parser.add_argument("--model",  type=str, default="lstm_model_v2.pt")

    args = parser.parse_args()

    if args.input:
        run_batch(args.input, args.output_dir, args.data, args.model, args.horizon)
    elif args.sector:
        run_single(
            sector=args.sector,
            current_revenue=args.revenue,
            employee_count=args.employees,
            founding_year=args.founding_year,
            output_path=args.output,
            data_path=args.data,
            model_path=args.model,
            horizon=args.horizon
        )
    else:
        print("Provide either --sector (single mode) or --input (batch mode)")
        print()
        print("Example:")
        print('  python score_engine.py --sector "Health care and social assistance" \\')
        print('                         --revenue 500000 --employees 8 \\')
        print('                         --founding-year 2015 --horizon 3y')
        sys.exit(1)