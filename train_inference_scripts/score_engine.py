# Score engine
# Runs inference using pre-computed static LSTM forecasts

import argparse
import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")

from survival_base_rates import compute_survival_probability, init_survival_model

CURRENT_YEAR   = 2025
BASE_DATA_YEAR = 2023
HORIZONS       = ["1y", "3y", "5y"]

# Population dampening threshold --> counties below this have class probabilities pulled toward uniform.
POP_DAMPEN_THRESHOLD = 40_000

# change this later to get population data from 2026
def _get_county_pop(county: str, econ_data: pd.DataFrame) -> float:
    rows = econ_data[econ_data["County"] == county]["TOT_POP"].dropna()
    return float(rows.iloc[-1]) if len(rows) > 0 else None

def _load_static_forecasts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"]
    df = df.set_index(["county", "sector", "horizon"])
    return df


def score_all_counties(
    sector: str,
    current_revenue: float,
    employee_count: int,
    founding_year: int,
    econ_data: pd.DataFrame,
    static_forecasts_path:  str,
    horizon: str = "3y",
    qcew_path: str = "data/qcew_long.csv",
) -> pd.DataFrame:
    """
    Score all 159 Georgia counties for a given business profile.

    Parameters
    ----------
    sector                : str   — must match sector column names exactly
    current_revenue       : float — current annual revenue in USD
    employee_count        : int
    founding_year         : int
    econ_data             : pd.DataFrame — merged_data.csv
    static_forecasts_path : str   — path to static_forecasts.csv
    horizon               : str   — '1y', '3y', or '5y'
    qcew_path             : str   — path to qcew_long.csv

    Returns
    -------
    pd.DataFrame ranked by score descending, with all 159 counties
    """
    business_age     = max(0, CURRENT_YEAR - founding_year)
    counties         = sorted(econ_data["County"].unique())
    results          = []
    static_forecasts = _load_static_forecasts(static_forecasts_path)

    init_survival_model(qcew_path=qcew_path, merged_path="data/merged_data_v2.csv")

    print(f"\nScoring {len(counties)} counties for '{sector}' ({horizon} horizon)...")
    print(f"Business: {employee_count} employees, ${current_revenue:,.0f} revenue, age {business_age}yr")
    print("-" * 60)

    county_to_geoid = (econ_data[["County", "GeoID"]].drop_duplicates().set_index("County")["GeoID"])
    unique_geoids = sorted(econ_data["GeoID"].unique())
    geoid_to_index = {g: i+1 for i, g in enumerate(unique_geoids)}

    errors = 0
    for i, county in enumerate(counties, 1):
        geo_id = county_to_geoid[county]
        id = geoid_to_index[geo_id]

        try:
            county_pop = _get_county_pop(county, econ_data)

            key = (county, sector, horizon)
            if key not in static_forecasts.index:
                raise ValueError(f"No static forecast for ({county}, {sector}, {horizon})")

            row_fc     = static_forecasts.loc[key]
            fc         = {
                "revenue_score":       row_fc["revenue_score"],
                "total_growth":        row_fc["total_growth"],
                "compound_multiplier": row_fc["compound_multiplier"],
                "annual_growth_rate":  row_fc["annual_growth_rate"],
                "economic_adjustment": row_fc["economic_adjustment"],
                "class_probs": [
                    row_fc["p_shrinking"],
                    row_fc["p_flat"],
                    row_fc["p_moderate"],
                    row_fc["p_strong"],
                ],
            }

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
            # Expected outcome = compound growth × survival probability
            # Normalized to 0-100 after all counties are scored
            # Store raw expected value here and normalize after the loop.
            expected_val  = compound * p_survival

            projected_revenue = round(current_revenue * compound, 2) \
                                if current_revenue > 0 else np.nan

            class_probs = fc.get("class_probs", [None] * 4)

            results.append({
                "rank":                None,
                "id" :                 id,
                "county":              county,
                "population":          int(county_pop) if county_pop else None,
                "score":               None,          # filled after normalization
                "expected_val":        expected_val,  # raw, used for normalization
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
                "id" :                id,
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

    print(f"\nScored {len(results) - errors}/159 counties ({errors} errors)")

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

    # Tier labels based on expected value 
    def _tier(row):
        ev = row["expected_val"]
        if ev >= 0.9:
            return "Strong Expand"
        elif ev >= 0.7:
            return "Cautious Expand"
        elif ev >= 0.5:
            return "Watch"
        else:
            return "Avoid"

    df_out["tier"] = df_out.apply(_tier, axis=1)

    df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)
    df_out["rank"] = df_out.index + 1

    cols = ["rank", "id", "county", "population", "score", "tier", "survival_prob", "revenue_score",
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


def predict(sector: str, current_revenue: float, employee_count: int, founding_year: int, data_path: str, static_forecasts_path: str, horizon: str = "3y"):
    econ_data = pd.read_csv(data_path).sort_values(["County", "Year"])
    print(f"Loaded {econ_data['County'].nunique()} counties")

    ranked = score_all_counties(
        sector=sector,
        current_revenue=current_revenue,
        employee_count=employee_count,
        founding_year=founding_year,
        econ_data=econ_data,
        static_forecasts_path=static_forecasts_path,
        horizon=horizon
    )

    ranked = ranked.replace({np.nan: None})
    print_summary(ranked, sector, horizon)
    return ranked.to_dict(orient="records")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraTrends County Expansion Ranker")

    parser.add_argument("--sector",        type=str,   required=True, help="Business sector")
    parser.add_argument("--revenue",       type=float, default=500000, help="Current annual revenue")
    parser.add_argument("--employees",     type=int,   default=10,     help="Employee count")
    parser.add_argument("--founding-year", type=int,   default=2015,   help="Year founded")
    parser.add_argument("--horizon",       type=str,   default="3y",   choices=["1y","3y","5y"])
    parser.add_argument("--data",          type=str,   default="data/merged_data_v2.csv")
    parser.add_argument("--static-forecasts", type=str, default="train_inference_scripts/static_forecasts.csv")

    args = parser.parse_args()

    predict(
        sector=args.sector,
        current_revenue=args.revenue,
        employee_count=args.employees,
        founding_year=args.founding_year,
        data_path=args.data,
        static_forecasts_path=args.static_forecasts,
        horizon=args.horizon,
    )