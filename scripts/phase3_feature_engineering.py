"""
Phase 3: Feature Engineering
Air Quality Inspection & Prediction (Pan-India)

Input:  data/processed/city_day_cleaned.csv
Output: data/processed/city_day_features.csv

Adds:
1. Date parts: Year, Month, Day, DayOfWeek
2. Season (Indian season mapping)
3. Lag features: AQI_lag1, AQI_lag7 (per city, requires sort by date)
4. Rolling features: AQI_roll7, AQI_roll30 (per city)
5. Encoded categoricals: City_enc, State_enc (Label Encoding), Season one-hot
"""

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

IN_PATH = '/home/nv/da379/data/processed/city_day_cleaned.csv'
OUT_PATH = '/home/nv/da379/data/processed/city_day_features.csv'

df = pd.read_csv(IN_PATH, parse_dates=['Date'])
df = df.sort_values(['City', 'Date']).reset_index(drop=True)

# ---------- 1. Date parts ----------
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday

# ---------- 2. Season (Indian season mapping) ----------
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:  # 10, 11
        return 'Post-Monsoon'

df['Season'] = df['Month'].apply(get_season)

# ---------- 3. Lag features (per city, time-ordered) ----------
df['AQI_lag1'] = df.groupby('City')['AQI'].shift(1)
df['AQI_lag7'] = df.groupby('City')['AQI'].shift(7)

# ---------- 4. Rolling features (per city) ----------
df['AQI_roll7'] = df.groupby('City')['AQI'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df['AQI_roll30'] = df.groupby('City')['AQI'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())

# Lag/rolling features create NaN for each city's first few rows -> fill with that city's AQI mean
for col in ['AQI_lag1', 'AQI_lag7', 'AQI_roll7', 'AQI_roll30']:
    df[col] = df[col].fillna(df.groupby('City')['AQI'].transform('mean'))

# ---------- 5. Encode categoricals ----------
le_city = LabelEncoder()
le_state = LabelEncoder()
df['City_enc'] = le_city.fit_transform(df['City'])
df['State_enc'] = le_state.fit_transform(df['State'])

season_dummies = pd.get_dummies(df['Season'], prefix='Season')
df = pd.concat([df, season_dummies], axis=1)

# ---------- Save ----------
df.to_csv(OUT_PATH, index=False)

print("=== FEATURE ENGINEERING SUMMARY ===")
print(f"Final shape: {df.shape}")
print(f"New columns added: Year, Month, Day, DayOfWeek, Season, AQI_lag1, AQI_lag7, "
      f"AQI_roll7, AQI_roll30, City_enc, State_enc, {list(season_dummies.columns)}")
print(f"\nSample:")
print(df[['City','Date','AQI','AQI_lag1','AQI_lag7','AQI_roll7','AQI_roll30','Season','City_enc']].head(10))
print(f"\nAny NaN remaining: {df.isna().sum().sum()}")
print(f"\nSaved to {OUT_PATH}")
