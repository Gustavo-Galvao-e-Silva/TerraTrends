"""
PRODUCTION TRAINING SCRIPT - Ultimate Model
Run this once to train all sector models with best features and techniques
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" " * 18 + "TRAINING MODEL")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('FINAL_MERGED_DATA.csv')
print(f"✓ Loaded {len(df)} records from {df['County'].nunique()} Georgia counties")

# All sectors to train
sector_columns = [
    'Accommodation and food services',
    'Administrative and support and waste management and remediation services',
    'Agriculture, forestry, fishing and hunting',
    'Arts, entertainment, and recreation',
    'Construction',
    'Durable goods manufacturing',
    'Educational services',
    'Finance and insurance',
    'Government and government enterprises',
    'Health care and social assistance',
    'Information',
    'Management of companies and enterprises',
    'Mining, quarrying, and oil and gas extraction',
    'Natural resources and mining',
    'Nondurable goods manufacturing',
    'Other services (except government and government enterprises)',
    'Private industries',
    'Professional and business services',
    'Real estate and rental and leasing',
    'Retail trade',
    'Trade',
    'Transportation and warehousing',
    'Utilities',
    'Wholesale trade'
]

print(f"Training {len(sector_columns)} sectors...")

# Store all models
all_models = {}
all_scalers = {}
all_stats = {}
counties_list = sorted(df['County'].unique())

for sector in sector_columns:
    print(f"\nTraining: {sector[:50]}...")
    
    df_sorted = df.sort_values(['County', 'Year']).copy()
    lagged_data = []
    
    # Create lagged dataset with momentum and volatility
    for county in df_sorted['County'].unique():
        county_df = df_sorted[df_sorted['County'] == county].copy()

        
        for i in range(3, len(county_df) - 1):
            prev_3 = county_df.iloc[i-3]
            prev_2 = county_df.iloc[i-2]
            prev_1 = county_df.iloc[i-1]
            current = county_df.iloc[i]
            next_year = county_df.iloc[i+1]
            
            if pd.isna(next_year[sector]) or pd.isna(current[sector]):
                continue
            
            # Calculate advanced features
            momentum = (current[sector] - prev_3[sector]) / 3
            volatility = np.std([prev_3[sector], prev_2[sector], prev_1[sector], current[sector]])
            
            row = {
                'County': county,
                'Year': current['Year'],
                'TOT_POP': current['TOT_POP'],
                'Unemployment_Rate': current['Unemployment_Rate'],
                'Per_Capita_Personal_Income': current['Per_Capita_Personal_Income'],
                'Real_GDP': current['Real_GDP'],
                'Percent_Change_Real_GDP': current['Percent_Change_Real_GDP'],
                'Bachelor_Degree_or_Higher_Pct': current['Bachelor_Degree_or_Higher_Pct'],
                'Last_Year_Growth': current[sector],
                'Two_Year_Avg': (current[sector] + prev_1[sector]) / 2,
                'Three_Year_Momentum': momentum,
                'Volatility': volatility,
                'Target': next_year[sector]
            }
            
            lagged_data.append(row)
    
    if len(lagged_data) < 50:
        print(f"  ⚠ Skipped - insufficient data")
        continue
    
    lagged_df = pd.DataFrame(lagged_data)
    
    feature_cols = [
        'Year', 'TOT_POP', 'Unemployment_Rate', 'Per_Capita_Personal_Income',
        'Real_GDP', 'Percent_Change_Real_GDP', 'Bachelor_Degree_or_Higher_Pct',
        'Last_Year_Growth', 'Two_Year_Avg', 'Three_Year_Momentum', 'Volatility'
    ]
    
    X = lagged_df[feature_cols].copy()
    y = lagged_df['Target']
    
    # Store statistics
    all_stats[sector] = {
        'mean': y.mean(),
        'std': y.std(),
        'min': y.min(),
        'max': y.max()
    }
    
    # Forward fill by county (best imputation method)
    for county in lagged_df['County'].unique():
        county_mask = lagged_df['County'] == county
        if county_mask.any():
            county_data = X[county_mask]
            X.loc[county_mask] = county_data.fillna(method='ffill').fillna(method='bfill')
    
    # Fill any remaining with median
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train ensemble with optimized hyperparameters
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    ridge = Ridge(alpha=0.5)
    
    # Fit all models
    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)
    ridge.fit(X_scaled, y)
    
    # Store models and scaler
    all_models[sector] = {
        'rf': rf,
        'gb': gb,
        'ridge': ridge
    }
    all_scalers[sector] = scaler
    
    print(f"  ✓ Complete")

print("\n" + "="*80)
print(f"✓ Successfully trained {len(all_models)} sector models")

# Save everything
print("\nSaving models...")
model_data = {
    'models': all_models,
    'scalers': all_scalers,
    'sector_stats': all_stats,
    'feature_columns': feature_cols,
    'sector_columns': list(all_models.keys()),
    'counties': counties_list,
    'model_version': 'terratrends_v1.0'
}

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Models saved to: trained_model.pkl")
print("\n" + "="*80)
