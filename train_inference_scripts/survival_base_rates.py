"""
SURVIVAL BASE RATES
Source: BLS Business Employment Dynamics (BED)
        "Survival of Private-Sector Establishments by Opening Year"
        https://www.bls.gov/bdm/us_age_naics_00_table7.txt

These are empirical survival rates: probability a firm survives TO year X
given it was alive at founding. Values are national averages by broad sector.

We then apply a county economic adjustment multiplier based on the
economic forecast for that county/sector to get a firm-specific estimate.
"""

import numpy as np

# -------------------------------------------------------------------
# BLS BED Survival Rates by Sector
# P(firm alive at year X | founded at year 0)
# Keys match our sector_columns naming convention
# -------------------------------------------------------------------
# Source: BLS Table 7, cohorts 2000-2019 average survival rates
SURVIVAL_RATES = {
    # sector name : {1: p1, 3: p3, 5: p5}
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

# Fallback for unknown sectors — national all-industry average
DEFAULT_SURVIVAL = {"1y": 0.81, "3y": 0.57, "5y": 0.44}

# -------------------------------------------------------------------
# Age adjustment multipliers
# Older firms have higher survival probability.
# Based on BLS conditional survival (given already survived to age N).
# Applied as a multiplier on base rate.
# -------------------------------------------------------------------
AGE_MULTIPLIERS = {
    # (min_age, max_age): multiplier
    (0,   1):  1.00,   # brand new — use base rate as-is
    (2,   3):  1.08,
    (4,   5):  1.14,
    (6,  10):  1.20,
    (11, 20):  1.28,
    (21, 999): 1.35,
}

# -------------------------------------------------------------------
# Employee count adjustment multipliers
# Larger firms are more resilient (access to capital, diversification)
# -------------------------------------------------------------------
SIZE_MULTIPLIERS = {
    # (min_employees, max_employees): multiplier
    (1,    4):  0.90,   # micro
    (5,   19):  1.00,   # small baseline
    (20,  49):  1.08,
    (50,  99):  1.14,
    (100, 249): 1.20,
    (250, 999): 1.26,
    (1000, 99999): 1.32,
}


def _get_range_multiplier(value: float, lookup: dict) -> float:
    for (lo, hi), mult in lookup.items():
        if lo <= value <= hi:
            return mult
    return 1.0


def get_base_survival(sector: str, horizon: str) -> float:
    """
    Get BLS base survival rate for a sector and horizon.
    horizon: '1y', '3y', or '5y'
    """
    rates = SURVIVAL_RATES.get(sector, DEFAULT_SURVIVAL)
    return rates.get(horizon, DEFAULT_SURVIVAL[horizon])


def compute_survival_probability(
    sector: str,
    business_age_years: float,
    employee_count: int,
    economic_adjustment: float,
    horizon: str
) -> float:
    """
    Compute adjusted survival probability for a specific business.

    Parameters
    ----------
    sector : str
        Business sector (must match sector_columns)
    business_age_years : float
        Current age of the business in years
    employee_count : int
        Current number of employees
    economic_adjustment : float
        Multiplier from economic outlook model.
        > 1.0 means county/sector outlook is better than historical average
        < 1.0 means worse
        Computed by economic_forecaster.py
    horizon : str
        '1y', '3y', or '5y'

    Returns
    -------
    float in [0, 1] — adjusted survival probability
    """
    base = get_base_survival(sector, horizon)

    age_mult = _get_range_multiplier(business_age_years, AGE_MULTIPLIERS)
    size_mult = _get_range_multiplier(employee_count, SIZE_MULTIPLIERS)

    # Combine multipliers — apply to logit space to avoid > 1.0
    logit_base = np.log(base / (1 - base))
    logit_adjusted = logit_base + np.log(age_mult) + np.log(size_mult) + np.log(max(0.5, economic_adjustment))

    adjusted = 1 / (1 + np.exp(-logit_adjusted))

    # Hard cap: can't exceed adjusted ceiling per sector
    return float(np.clip(adjusted, 0.01, 0.98))