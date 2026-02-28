"""
LSTM FORECASTER v2
==================
Uses the v2 classification model (shrinking/flat/moderate/strong)
to forecast sector growth class probabilities, then converts to
revenue projections and economic adjustment scores.

Drop-in replacement for economic_forecaster.py — score_engine.py
needs no changes, just swap the import.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_pkg   = None
_model = None

def _load(model_path: str = "lstm_model_v2.pt"):
    global _pkg, _model
    if _model is not None:
        return _model, _pkg

    pkg = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = pkg["config"]

    class TerraLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.sector_emb = nn.Embedding(cfg["n_sectors"],  cfg["sector_emb_dim"])
            self.county_emb = nn.Embedding(cfg["n_counties"], cfg["county_emb_dim"])
            self.year_emb   = nn.Embedding(cfg["n_years"],    cfg["year_emb_dim"])

            proj_input = cfg["input_size"] + cfg["sector_emb_dim"] + cfg["county_emb_dim"] + cfg["year_emb_dim"]
            self.input_proj = nn.Linear(proj_input, cfg["hidden_size"])

            self.lstm = nn.LSTM(
                input_size=cfg["hidden_size"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                batch_first=True
            )
            self.dropout = nn.Dropout(cfg["dropout"])
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg["hidden_size"], 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, cfg["n_classes"])
                )
                for _ in range(cfg["forecast_horizon"])
            ])

        def forward(self, county_idx, sector_idx, year_idx, x_seq):
            B, T, _ = x_seq.shape
            s_emb   = self.sector_emb(sector_idx).unsqueeze(1).expand(B, T, -1)
            c_emb   = self.county_emb(county_idx).unsqueeze(1).expand(B, T, -1)
            yr_range = year_idx.unsqueeze(1) + torch.arange(T).unsqueeze(0)
            yr_range = yr_range.clamp(0, cfg["n_years"] - 1)
            y_emb   = self.year_emb(yr_range)
            x       = torch.cat([x_seq, s_emb, c_emb, y_emb], dim=-1)
            x       = torch.relu(self.input_proj(x))
            out, _  = self.lstm(x)
            last    = self.dropout(out[:, -1, :])
            logits  = [head(last) for head in self.heads]
            return torch.stack(logits, dim=1)

    m = TerraLSTM()
    m.load_state_dict(pkg["model_state"])
    m.eval()

    _pkg   = pkg
    _model = m
    return m, pkg


def _build_input(county, sector, df, pkg):
    """Build normalized input sequence for one county/sector."""
    cfg            = pkg["config"]
    seq_len        = cfg["seq_len"]
    county2idx     = pkg["county2idx"]
    sector2idx     = pkg["sector2idx"]
    year2idx       = pkg["year2idx"]
    sector_scalers = pkg["sector_scalers"]
    macro_scalers  = pkg["macro_scalers"]
    macro_features = pkg["macro_features"]
    ga_neighbors   = pkg["ga_neighbors"]
    county2fips    = pkg["county2fips"]

    if county not in county2idx:
        raise ValueError(f"County '{county}' not found. Check format matches merged_data.csv (e.g. 'Fulton, GA')")
    if sector not in sector2idx:
        raise ValueError(f"Sector '{sector}' not in model.")

    county_idx = county2idx[county]
    sector_idx = sector2idx[sector]

    county_df = df[df["County"] == county].sort_values("Year").tail(seq_len)
    if len(county_df) < seq_len:
        raise ValueError(f"Need {seq_len} years of data for {county}, only have {len(county_df)}")

    # Build neighbor county indices for spillover feature
    fips = county2fips.get(county)
    neighbor_fips = ga_neighbors.get(fips, [])[:3]
    neighbor_counties = []
    fips2county = {v: k for k, v in county2fips.items()}
    for nf in neighbor_fips:
        nc = fips2county.get(nf)
        if nc and nc in df["County"].values:
            neighbor_counties.append(nc)

    x_seq      = []
    start_year = county_df["Year"].iloc[0]
    yi_start   = year2idx.get(start_year, 0)

    for _, row in county_df.iterrows():
        macro_vec = []
        for feat in macro_features:
            val = row.get(feat, np.nan)
            if pd.isna(val):
                val = macro_scalers[feat].mean_[0]
            macro_vec.append(macro_scalers[feat].transform([[val]])[0][0])

        sv = row.get(sector, np.nan)
        if pd.isna(sv):
            sv = sector_scalers[sector].mean_[0]
        sector_norm = sector_scalers[sector].transform([[sv]])[0][0]

        # Neighbor avg for this year
        neighbor_vals = []
        for nc in neighbor_counties:
            nr = df[(df["County"] == nc) & (df["Year"] == row["Year"])]
            if not nr.empty:
                nv = nr[sector].values[0]
                if not pd.isna(nv):
                    neighbor_vals.append(sector_scalers[sector].transform([[nv]])[0][0])
        neighbor_avg = float(np.mean(neighbor_vals)) if neighbor_vals else 0.0

        x_seq.append(macro_vec + [sector_norm, neighbor_avg, 0.0])

    x_t  = torch.tensor([x_seq],   dtype=torch.float32)
    ci_t = torch.tensor([county_idx], dtype=torch.long)
    si_t = torch.tensor([sector_idx], dtype=torch.long)
    yi_t = torch.tensor([yi_start],   dtype=torch.long)

    return x_t, ci_t, si_t, yi_t


def _probs_to_expected_growth(probs, class_growth_rates):
    """
    Convert class probability distribution to expected annual growth rate.
    probs: array of shape [N_CLASSES]
    """
    return sum(probs[c] * class_growth_rates[c] for c in range(len(probs)))


def forecast_multiple_horizons(
    county: str,
    sector: str,
    df: pd.DataFrame,
    base_year: int = 2023,
    model_path: str = "lstm_model_v2.pt"
) -> dict:
    """
    Forecast 1Y, 3Y, 5Y horizons. Returns dict keyed by '1y', '3y', '5y'.
    Drop-in replacement for economic_forecaster.forecast_multiple_horizons().

    The model predicts FORECAST_HORIZON steps (currently 3).
    For 5Y we extend by repeating the last predicted distribution.
    """
    model, pkg = _load(model_path)

    cfg               = pkg["config"]
    forecast_horizon  = cfg["forecast_horizon"]
    class_growth_rates = pkg["class_growth_rates"]
    national_avg      = 0.02
    ga_premium_annual = 0.008

    x_t, ci_t, si_t, yi_t = _build_input(county, sector, df, pkg)

    with torch.no_grad():
        logits = model(ci_t, si_t, yi_t, x_t)[0]   # [FORECAST_HORIZON, N_CLASSES]
        probs  = torch.softmax(logits, dim=-1).numpy()  # [FORECAST_HORIZON, N_CLASSES]

    # probs[step] = probability distribution over [shrinking, flat, moderate, strong]
    # Expected annual growth rate for each step
    annual_rates = [
        _probs_to_expected_growth(probs[step], class_growth_rates)
        for step in range(forecast_horizon)
    ]

    # For 5Y, extend by repeating last year's distribution
    while len(annual_rates) < 5:
        annual_rates.append(annual_rates[-1])

    results     = {}
    national_avg = 0.02

    for label, n_years in [("1y", 1), ("3y", 3), ("5y", 5)]:
        rates    = annual_rates[:n_years]
        compound = 1.0
        for rate in rates:
            compound *= (1 + rate)

        # Cap total compounded growth at ±75%
        compound      = max(0.25, min(1.75, compound))
        total_growth  = compound - 1
        annual_growth = compound ** (1 / n_years) - 1 if n_years > 0 else 0

        ga_premium  = (1 + ga_premium_annual) ** n_years
        natl_compound = (1 + national_avg) ** n_years
        econ_adj    = (compound / natl_compound) * ga_premium

        # Also return class probabilities for the first step (useful for scoring)
        step_probs = probs[0].tolist() if len(probs) > 0 else [0.25] * 4

        results[label] = {
            "compound_multiplier": compound,
            "total_growth":        total_growth,
            "annual_growth_rate":  annual_growth,
            "economic_adjustment": max(0.1, econ_adj),
            "class_probs":         step_probs,    # [p_shrinking, p_flat, p_moderate, p_strong]
            "predicted_rates":     rates,
            "year_by_year":        [(2023 + i + 1, r) for i, r in enumerate(rates)],
        }

    return results