import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('./data/merged_data.csv')

print(f"Original shape: {df.shape}")
print(f"Original columns: {len(df.columns)}")

# Keep geographic demographic columns
geo_demo_cols = ['GeoID', 'County', 'Year', 'TOT_POP',
                 'Unemployment_Rate', 'Per_Capita_Personal_Income', 'Real_GDP', 
                 'Percent_Change_Real_GDP', 'Bachelor_Degree_or_Higher_Pct']

# Remove redundant aggregate columns
redundant_cols = [
    'Manufacturing',  # Sum of durable nondurable
    'Trade',  # Sum of retail wholesale
]

# Remove trivial low quality
# Keep arts education columns
trivial_cols = [
    'Management of companies and enterprises',  # Low non-zero values
    'Mining, quarrying, and oil and gas extraction',  # Low non-zero outliers
]

cols_to_remove = redundant_cols + trivial_cols

print(f"\nRemoving {len(cols_to_remove)} columns:")
for col in cols_to_remove:
    print(f"  - {col}")

# Remove redundant trivial columns
df_cleaned = df.drop(columns=cols_to_remove, errors='ignore')

# Cap outliers at 3*IQR
print(f"\nHandling outliers (capping at 3*IQR bounds)...")
industry_cols = [col for col in df_cleaned.columns if col not in geo_demo_cols]

outliers_capped = {}
for col in industry_cols:
    data = df_cleaned[col].copy()
    non_null = data.dropna()
    
    if len(non_null) > 0:
        Q1 = non_null.quantile(0.25)
        Q3 = non_null.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:  # Cap if meaningful
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Count outliers before capping
            outliers_before = ((non_null < lower_bound) | (non_null > upper_bound)).sum()
            
            # Cap outliers
            df_cleaned.loc[data < lower_bound, col] = lower_bound
            df_cleaned.loc[data > upper_bound, col] = upper_bound
            
            outliers_capped[col] = outliers_before

print(f"Outliers capped per column:")
for col, count in sorted(outliers_capped.items(), key=lambda x: x[1], reverse=True)[:10]:
    if count > 0:
        print(f"  {col[:50]:<50} {count:>6} outliers capped")

print(f"\nCleaned shape: {df_cleaned.shape}")
print(f"Remaining columns: {len(df_cleaned.columns)}")
print(f"\nRemaining industry columns:")
industry_cols_remaining = [col for col in df_cleaned.columns if col not in geo_demo_cols]
for col in industry_cols_remaining:
    print(f"  - {col}")

# Save cleaned data
output_path = 'data/merged_data.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")
