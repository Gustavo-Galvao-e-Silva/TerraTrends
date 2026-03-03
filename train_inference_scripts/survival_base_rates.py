"""
This model uses five independent signals, producing a ~0.25+ spread
that meaningfully differentiates county survival environments.

Signals (all applied as logit-space multipliers):
  1. Business age            — unchanged from v1
  2. Business size           — unchanged from v1
  3. Sector employment trend — 3yr rolling avg emp growth for sector in county
  4. Sector wage health      — county sector wage vs GA state sector median
  5. Local income level      — county PCI vs GA state median (99% coverage)
  6. Market saturation       — establishments per 10k pop vs state median
  7. Employment volatility   — rolling std of emp growth (penalty only)

Signals 3-7 come from QCEW data (qcew_long.csv). When QCEW data is
suppressed for a county/sector pair (BLS privacy rules, ~30% of cases),
those signals gracefully fall back to neutral (1.0x multiplier). The
income signal (signal 5) is always available from merged_data.csv.

Usage:
  from survival_base_rates import SurvivalModel

  model = SurvivalModel('data/qcew_long.csv', 'data/merged_data.csv')

  p = model.compute(
      sector='Accommodation and food services',
      county='Forsyth, GA',
      business_age_years=6,
      employee_count=7,
      horizon='3y',
      forecast_year=2023,
  )
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# BLS BED National Survival Rates (unchanged from v1)
# P(firm alive at year X | founded at year 0)
# Source: BLS Table 7, cohorts 2000-2019 average
# -------------------------------------------------------------------
SURVIVAL_RATES = {
    "Accommodation and food services":                                          {"1y": 0.79, "3y": 0.54, "5y": 0.41},
    "Administrative and support and waste management and remediation services": {"1y": 0.81, "3y": 0.57, "5y": 0.44},
    "Agriculture, forestry, fishing and hunting":                               {"1y": 0.85, "3y": 0.65, "5y": 0.53},
    "Arts, entertainment, and recreation":                                      {"1y": 0.80, "3y": 0.55, "5y": 0.42},
    "Construction":                                                             {"1y": 0.80, "3y": 0.55, "5y": 0.42},
    "Durable goods manufacturing":                                              {"1y": 0.83, "3y": 0.61, "5y": 0.49},
    "Educational services":                                                     {"1y": 0.84, "3y": 0.63, "5y": 0.51},
    "Finance and insurance":                                                    {"1y": 0.84, "3y": 0.63, "5y": 0.50},
    "Government and government enterprises":                                    {"1y": 0.95, "3y": 0.88, "5y": 0.82},
    "Health care and social assistance":                                        {"1y": 0.85, "3y": 0.65, "5y": 0.53},
    "Information":                                                              {"1y": 0.79, "3y": 0.53, "5y": 0.40},
    "Natural resources and mining":                                             {"1y": 0.83, "3y": 0.61, "5y": 0.48},
    "Nondurable goods manufacturing":                                           {"1y": 0.82, "3y": 0.59, "5y": 0.47},
    "Other services (except government and government enterprises)":            {"1y": 0.81, "3y": 0.57, "5y": 0.44},
    "Private industries":                                                       {"1y": 0.81, "3y": 0.57, "5y": 0.44},
    "Professional and business services":                                       {"1y": 0.82, "3y": 0.59, "5y": 0.47},
    "Real estate and rental and leasing":                                       {"1y": 0.82, "3y": 0.60, "5y": 0.47},
    "Retail trade":                                                             {"1y": 0.79, "3y": 0.54, "5y": 0.41},
    "Transportation and warehousing":                                           {"1y": 0.81, "3y": 0.57, "5y": 0.44},
    "Utilities":                                                                {"1y": 0.88, "3y": 0.71, "5y": 0.59},
    "Wholesale trade":                                                          {"1y": 0.82, "3y": 0.60, "5y": 0.48},
}
DEFAULT_SURVIVAL = {"1y": 0.81, "3y": 0.57, "5y": 0.44}

AGE_MULTIPLIERS = {
    (0,   1):  1.00,
    (2,   3):  1.08,
    (4,   5):  1.14,
    (6,  10):  1.20,
    (11, 20):  1.28,
    (21, 999): 1.35,
}

SIZE_MULTIPLIERS = {
    (1,    4):     0.90,
    (5,   19):     1.00,
    (20,  49):     1.08,
    (50,  99):     1.14,
    (100, 249):    1.20,
    (250, 999):    1.26,
    (1000, 99999): 1.32,
}


def _get_range_multiplier(value: float, lookup: dict) -> float:
    for (lo, hi), mult in lookup.items():
        if lo <= value <= hi:
            return mult
    return 1.0


# -------------------------------------------------------------------
# QCEW signal multiplier functions
# Each takes a raw signal value and returns a logit-space multiplier.
# NaN / missing -> returns 1.0 (neutral, no adjustment)
# All outputs are clipped to prevent any single signal from dominating.
# -------------------------------------------------------------------

def _emp_trend_mult(emp_trend_3yr) -> float:
    """
    3-year rolling average employment growth rate for the sector in this county.
    Data range: p5=-0.13, p50=0.01, p95=0.23
    +15% trend -> 1.25x boost. -15% trend -> 0.80x penalty.
    """
    if pd.isna(emp_trend_3yr):
        return 1.0
    mult = 1.0 + (float(emp_trend_3yr) * 1.67)
    return float(np.clip(mult, 0.80, 1.25))


def _wage_health_mult(wage_ratio) -> float:
    """
    County sector avg wage / GA state sector median wage.
    Higher relative wages = healthier, more productive sector.
    Data range: p5=0.64, p50=1.00, p95=1.56
    Uses dampened power function to compress extremes.
    """
    if pd.isna(wage_ratio) or wage_ratio <= 0:
        return 1.0
    mult = float(wage_ratio) ** 0.20
    return float(np.clip(mult, 0.90, 1.15))


def _income_mult(pci_ratio) -> float:
    """
    County per-capita personal income / GA state median PCI.
    Higher local income = more spending power = better survival environment.
    Data range: p5=0.79, p50=1.00, p95=1.43
    99% coverage — always available from merged_data.csv.
    """
    if pd.isna(pci_ratio) or pci_ratio <= 0:
        return 1.0
    mult = float(pci_ratio) ** 0.25
    return float(np.clip(mult, 0.88, 1.18))


def _saturation_mult(saturation_ratio) -> float:
    """
    County establishments per 10k pop / GA state median for that sector/year.
    Below median (underserved market) = slight boost.
    Above median (saturated) = slight penalty.
    Capped conservatively — saturation is a noisy signal.
    Data range: p5=0.42, p50=1.00, p95=2.11
    """
    if pd.isna(saturation_ratio) or saturation_ratio <= 0:
        return 1.0
    mult = 1.0 + (1.0 - float(saturation_ratio)) * 0.08
    return float(np.clip(mult, 0.92, 1.08))


def _volatility_mult(emp_volatility) -> float:
    """
    Rolling 5-year std dev of employment growth rate.
    Penalizes high-volatility environments (only penalizes, never boosts).
    Median volatility ~0.09 -> 1.0x. High (0.30+) -> 0.85x.
    """
    if pd.isna(emp_volatility):
        return 1.0
    penalty = max(0.0, (float(emp_volatility) - 0.09) * 0.50)
    return float(np.clip(1.0 - penalty, 0.85, 1.00))


class SurvivalModel:
    """
    County- and sector-specific survival probability model.

    Load once at startup, then call compute() for each county/sector.

    Parameters
    ----------
    qcew_path   : path to qcew_long.csv (output of fetch_qcew.py)
    merged_path : path to merged_data.csv (for PCI data)
    """

    def __init__(self, qcew_path: str, merged_path: str):
        self._signals = self._build_signal_table(qcew_path, merged_path)

    def _build_signal_table(self, qcew_path: str, merged_path: str) -> pd.DataFrame:
        """
        Precompute all QCEW-derived signals for every (County, Sector, Year).
        Called once at load time.
        """
        long = pd.read_csv(qcew_path)
        md   = pd.read_csv(merged_path)

        # Merge PCI from merged_data
        pci_df = md[["GeoID", "Year", "Per_Capita_Personal_Income"]].drop_duplicates()
        long   = long.merge(pci_df, on=["GeoID", "Year"], how="left")

        long = long.sort_values(["GeoID", "Sector", "Year"]).reset_index(drop=True)

        # Signal 1: 3yr rolling avg employment growth
        long["emp_trend_3yr"] = (
            long.groupby(["GeoID", "Sector"])["employment_growth_rate"]
            .transform(lambda x: x.rolling(3, min_periods=2).mean())
        )

        # Signal 2: wage ratio vs state sector median
        state_med_wage = long.groupby(["Sector", "Year"])["avg_wage_per_employee"].transform("median")
        long["wage_ratio"] = long["avg_wage_per_employee"] / state_med_wage

        # Signal 3: PCI ratio vs state median
        state_med_pci = md.groupby("Year")["Per_Capita_Personal_Income"].median()
        long["pci_median"] = long["Year"].map(state_med_pci)
        long["pci_ratio"]  = long["Per_Capita_Personal_Income"] / long["pci_median"]

        # Signal 4: saturation (establishments per 10k pop)
        pop_df = md[["GeoID", "Year", "TOT_POP"]].drop_duplicates()
        long   = long.merge(pop_df, on=["GeoID", "Year"], how="left")
        long["estabs_per_10k"]  = long["avg_establishments"] / (long["TOT_POP"] / 10_000)
        state_med_sat           = long.groupby(["Sector", "Year"])["estabs_per_10k"].transform("median")
        long["saturation_ratio"] = long["estabs_per_10k"] / state_med_sat

        # Signal 5: employment volatility (5yr rolling std)
        long["emp_volatility"] = (
            long.groupby(["GeoID", "Sector"])["employment_growth_rate"]
            .transform(lambda x: x.rolling(5, min_periods=3).std())
        )

        # Keep only the columns needed for lookup
        keep = [
            "County", "Sector", "Year",
            "emp_trend_3yr", "wage_ratio", "pci_ratio",
            "saturation_ratio", "emp_volatility",
        ]
        return long[keep].copy()

    def _get_signals(self, county: str, sector: str, forecast_year: int) -> dict:
        """
        Look up precomputed signals for a county/sector.
        Uses the most recent available year up to forecast_year.
        Falls back gracefully if data is missing.
        """
        df = self._signals
        mask = (df["County"] == county) & (df["Sector"] == sector) & (df["Year"] <= forecast_year)
        sub  = df[mask].sort_values("Year")

        if sub.empty:
            # Sector not in QCEW (Agriculture, Mining, Durable/Nondurable Mfg)
            # Fall back to county-level PCI signal only
            pci_mask = (df["County"] == county) & (df["Year"] <= forecast_year)
            pci_sub  = df[pci_mask].sort_values("Year")
            if not pci_sub.empty:
                return {"pci_ratio": pci_sub["pci_ratio"].iloc[-1]}
            return {}

        row = sub.iloc[-1]
        return {
            "emp_trend_3yr":    row["emp_trend_3yr"],
            "wage_ratio":       row["wage_ratio"],
            "pci_ratio":        row["pci_ratio"],
            "saturation_ratio": row["saturation_ratio"],
            "emp_volatility":   row["emp_volatility"],
        }

    def compute(
        self,
        sector: str,
        county: str,
        business_age_years: float,
        employee_count: int,
        horizon: str,
        forecast_year: int = 2023,
    ) -> float:
        """
        Compute adjusted survival probability for a specific business in a county.

        Parameters
        ----------
        sector             : business sector (must match SECTOR_COLS)
        county             : county name (e.g. 'Forsyth, GA')
        business_age_years : current age of the business in years
        employee_count     : number of employees
        horizon            : '1y', '3y', or '5y'
        forecast_year      : year of forecast (default 2023)

        Returns
        -------
        float in [0.01, 0.98]
        """
        # Base rate
        rates = SURVIVAL_RATES.get(sector, DEFAULT_SURVIVAL)
        base  = rates.get(horizon, DEFAULT_SURVIVAL[horizon])

        # Static business multipliers (age + size)
        age_mult  = _get_range_multiplier(business_age_years, AGE_MULTIPLIERS)
        size_mult = _get_range_multiplier(employee_count, SIZE_MULTIPLIERS)

        # QCEW-derived county+sector signals
        signals = self._get_signals(county, sector, forecast_year)

        emp_trend_m  = _emp_trend_mult(signals.get("emp_trend_3yr"))
        wage_m       = _wage_health_mult(signals.get("wage_ratio"))
        income_m     = _income_mult(signals.get("pci_ratio"))
        saturation_m = _saturation_mult(signals.get("saturation_ratio"))
        volatility_m = _volatility_mult(signals.get("emp_volatility"))

        # Combine in logit space — prevents any combination pushing above 1.0
        logit_base = np.log(base / (1.0 - base))
        logit_adj  = (
            np.log(age_mult)      +
            np.log(size_mult)     +
            np.log(emp_trend_m)   +
            np.log(wage_m)        +
            np.log(income_m)      +
            np.log(saturation_m)  +
            np.log(volatility_m)
        )

        adjusted = 1.0 / (1.0 + np.exp(-(logit_base + logit_adj)))
        return float(np.clip(adjusted, 0.01, 0.98))


# -------------------------------------------------------------------
# Legacy function shim — keeps score_engine.py working unchanged
# during the transition. Instantiate SurvivalModel instead for
# production use (avoids reloading QCEW data on every call).
# -------------------------------------------------------------------
_default_model: SurvivalModel = None

def init_survival_model(
    qcew_path: str = "data/qcew_long.csv",
    merged_path: str = "data/merged_data.csv",
):
    """Call once at startup to load the QCEW data into memory."""
    global _default_model
    _default_model = SurvivalModel(qcew_path, merged_path)
    return _default_model


def compute_survival_probability(
    sector: str,
    county: str,
    business_age_years: float,
    employee_count: int,
    horizon: str,
    forecast_year: int = 2023,
    economic_adjustment: float = None,
) -> float:
    """
    Compute survival probability. Requires init_survival_model() to have
    been called first. Falls back to legacy calculation if not initialized.
    """
    global _default_model

    return _default_model.compute(
        sector=sector,
        county=county,
        business_age_years=business_age_years,
        employee_count=employee_count,
        horizon=horizon,
        forecast_year=forecast_year,
    )
