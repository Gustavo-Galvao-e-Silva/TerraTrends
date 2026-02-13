"""
PRODUCTION PREDICTION SCRIPT (FIXED)
Use trained model to predict future sector growth
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" " * 18 + "SECTOR GDP FORECASTER")
print("="*80)

# Load trained model
print("\nLoading trained model...")
try:
    with open('trained_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    models = model_data['models']
    scalers = model_data['scalers']
    sector_stats = model_data['sector_stats']
    feature_columns = model_data['feature_columns']
    counties = model_data['counties']
    
    print(f"✓ Loaded {len(models)} trained models (version: {model_data.get('model_version', 'unknown')})")
except FileNotFoundError:
    print("❌ Error: trained_model.pkl not found!")
    print("Please run train_final.py first.")
    exit(1)

# Load data
print("Loading county data...")
df = pd.read_csv('FINAL_MERGED_DATA.csv')
print(f"✓ {len(counties)} counties available")

def predict_sector_growth(county_name, sector_name, input_year):
    """
    Predict next year's sector growth based on current year data
    
    Parameters:
    -----------
    county_name : str
        County name (e.g., "Fulton, GA")
    sector_name : str
        Business sector name
    input_year : int
        Current year (model will predict input_year + 1)
    
    Returns:
    --------
    dict : Prediction results
    """
    
    if county_name not in df['County'].values:
        return {'error': f"County '{county_name}' not found"}
    
    if sector_name not in models:
        return {'error': f"Sector '{sector_name}' not available"}
    
    # Get county data for input year
    county_year_data = df[(df['County'] == county_name) & (df['Year'] == input_year)]
    
    if county_year_data.empty:
        # Use most recent year
        county_data = df[df['County'] == county_name].sort_values('Year', ascending=False).iloc[0]
        actual_input_year = int(county_data['Year'])
        print(f"\n⚠️  No data for {input_year}, using {actual_input_year}")
    else:
        county_data = county_year_data.iloc[0]
        actual_input_year = input_year
    
    # Need 3 previous years for momentum
    county_all = df[df['County'] == county_name].sort_values('Year')
    
    try:
        data_current = county_all[county_all['Year'] == actual_input_year].iloc[0]
        data_prev_1 = county_all[county_all['Year'] == actual_input_year - 1].iloc[0]
        data_prev_2 = county_all[county_all['Year'] == actual_input_year - 2].iloc[0]
        data_prev_3 = county_all[county_all['Year'] == actual_input_year - 3].iloc[0]
    except:
        return {'error': 'Insufficient historical data (need 3 years)'}
    
    # Calculate advanced features
    momentum = (data_current[sector_name] - data_prev_3[sector_name]) / 3
    volatility = np.std([
        data_prev_3[sector_name],
        data_prev_2[sector_name],
        data_prev_1[sector_name],
        data_current[sector_name]
    ])
    
    # Prepare input features
    input_features = {
        'Year': actual_input_year,
        'TOT_POP': data_current.get('TOT_POP', 50000),
        'Unemployment_Rate': data_current.get('Unemployment_Rate', 4.5),
        'Per_Capita_Personal_Income': data_current.get('Per_Capita_Personal_Income', 40000),
        'Real_GDP': data_current.get('Real_GDP', 2000000),
        'Percent_Change_Real_GDP': data_current.get('Percent_Change_Real_GDP', 2.5),
        'Bachelor_Degree_or_Higher_Pct': data_current.get('Bachelor_Degree_or_Higher_Pct', 15.0),
        'Last_Year_Growth': data_current[sector_name],
        'Two_Year_Avg': (data_current[sector_name] + data_prev_1[sector_name]) / 2,
        'Three_Year_Momentum': momentum,
        'Volatility': volatility
    }
    
    # Prepare DataFrame and handle missing values - FIXED VERSION
    input_df = pd.DataFrame([input_features])[feature_columns]
    
    # Fill missing values with reasonable defaults
    # Only use base features from df for median (not engineered features)
    base_features = ['Year', 'TOT_POP', 'Unemployment_Rate', 'Per_Capita_Personal_Income',
                     'Real_GDP', 'Percent_Change_Real_GDP', 'Bachelor_Degree_or_Higher_Pct']
    
    for col in input_df.columns:
        if col in base_features:
            input_df[col] = input_df[col].fillna(df[col].median())
        else:
            # For engineered features, fill with 0 (they shouldn't be NaN anyway)
            input_df[col] = input_df[col].fillna(0)
    
    # Scale and predict
    scaler = scalers[sector_name]
    input_scaled = scaler.transform(input_df)
    
    m = models[sector_name]
    rf_pred = m['rf'].predict(input_scaled)[0]
    gb_pred = m['gb'].predict(input_scaled)[0]
    ridge_pred = m['ridge'].predict(input_scaled)[0]
    
    # Ensemble prediction (50% RF, 35% GB, 15% Ridge)
    prediction = 0.5 * rf_pred + 0.35 * gb_pred + 0.15 * ridge_pred
    pred_std = np.std([rf_pred, gb_pred, ridge_pred])
    
    forecast_year = actual_input_year + 1
    
    # Check if actual value exists (for validation)
    actual_value = None
    next_year_data = df[(df['County'] == county_name) & (df['Year'] == forecast_year)]
    if not next_year_data.empty:
        actual_val = next_year_data[sector_name].iloc[0]
        if pd.notna(actual_val):
            actual_value = round(actual_val, 2)
    
    # Get county historical data
    county_history = df[df['County'] == county_name][sector_name].dropna()
    
    return {
        'county': county_name,
        'sector': sector_name,
        'input_year': actual_input_year,
        'forecast_year': forecast_year,
        'predicted_growth': round(prediction, 2),
        'actual_growth': actual_value,
        'confidence_interval': [
            round(prediction - 1.96*pred_std, 2),
            round(prediction + 1.96*pred_std, 2)
        ],
        'model_predictions': {
            'random_forest': round(rf_pred, 2),
            'gradient_boosting': round(gb_pred, 2),
            'ridge_regression': round(ridge_pred, 2)
        },
        'county_historical_mean': round(county_history.mean(), 2) if len(county_history) > 0 else None,
        'statewide_mean': round(sector_stats[sector_name]['mean'], 2),
        'input_features': {
            'population': int(input_features['TOT_POP']),
            'unemployment': round(input_features['Unemployment_Rate'], 1),
            'income': int(input_features['Per_Capita_Personal_Income']),
            'gdp': int(input_features['Real_GDP']),
            'last_year_growth': round(input_features['Last_Year_Growth'], 2),
            'momentum': round(input_features['Three_Year_Momentum'], 2),
            'volatility': round(input_features['Volatility'], 2)
        }
    }

# Interactive Mode
print("\n" + "="*80)
print("MAKE A PREDICTION")
print("="*80)

print("\nCounties (first 20):")
for i, county in enumerate(counties[:20], 1):
    print(f"  {i:2d}. {county}")
print(f"  ... and {len(counties)-20} more")

county_input = input("\nEnter county name [Ex. Fulton, GA]: ").strip() or "Fulton, GA"

print("\nAvailable sectors:")
sector_list = list(models.keys())
for i, sector in enumerate(sector_list, 1):
    print(f"  {i:2d}. {sector}")

sector_input = input(f"\nEnter sector number (1-{len(sector_list)}): ").strip()

if sector_input.isdigit() and 0 < int(sector_input) <= len(sector_list):
    sector_input = sector_list[int(sector_input) - 1]
else:
    print("Invalid selection, using Construction")
    sector_input = "Construction"

year_input = input("\nEnter CURRENT year (to predict next year) [Ex. 2023]: ").strip() or "2023"
current_year = int(year_input)

# Make prediction
print("\n" + "="*80)
print("FORECAST RESULTS")
print("="*80)

result = predict_sector_growth(county_input, sector_input, current_year)

if 'error' in result:
    print(f"\n❌ {result['error']}")
else:
    print(f"\n📍 County: {result['county']}")
    print(f"🏢 Sector: {result['sector']}")
    print(f"📊 Using data from: {result['input_year']}")
    print(f"🔮 Forecasting for: {result['forecast_year']}")
    
    print(f"\n🎯 PREDICTED {result['forecast_year']} Growth: {result['predicted_growth']}%")
    print(f"   95% Confidence Interval: [{result['confidence_interval'][0]}%, {result['confidence_interval'][1]}%]")
    
    # Show actual if available
    if result['actual_growth'] is not None:
        print(f"\n✅ ACTUAL {result['forecast_year']} Growth: {result['actual_growth']}%")
        error = abs(result['predicted_growth'] - result['actual_growth'])
        print(f"   Prediction Error: {error:.2f} percentage points")
    else:
        print(f"\n⚠️  No actual {result['forecast_year']} data yet (true future forecast!)")
    
    print(f"\n📊 Historical Context:")
    if result['county_historical_mean']:
        print(f"   {result['county']} Historical Avg: {result['county_historical_mean']}%")
    print(f"   Georgia Statewide Avg: {result['statewide_mean']}%")
    
    print(f"\n📈 Input Data from {result['input_year']}:")
    print(f"   Population: {result['input_features']['population']:,}")
    print(f"   Unemployment: {result['input_features']['unemployment']}%")
    print(f"   Per Capita Income: ${result['input_features']['income']:,}")
    print(f"   GDP: ${result['input_features']['gdp']:,}")
    print(f"   Last Year Growth: {result['input_features']['last_year_growth']}%")
    print(f"   3-Year Momentum: {result['input_features']['momentum']}%")
    print(f"   Volatility: {result['input_features']['volatility']}")

print("\n" + "="*80 + "\n")