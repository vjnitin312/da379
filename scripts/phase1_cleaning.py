"""
Phase 1: Data Understanding & Cleaning
Air Quality Inspection & Prediction (Pan-India)

Steps:
1. Load city_day.csv + stations.csv
2. Merge State info into main data
3. Drop Xylene (61% missing)
4. Impute missing pollutant values (city-wise + month-wise median)
5. Recompute missing AQI using official CPCB sub-index formula
6. Drop rows still missing AQI (insufficient pollutant data to compute)
7. Save cleaned dataset
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------- 1. Load ----------
cd = pd.read_csv('/home/nv/da379/data/raw/city_day.csv')
st = pd.read_csv('/home/nv/da379/data/raw/stations.csv')

cd['Date'] = pd.to_datetime(cd['Date'])

# ---------- 2. Merge State ----------
city_state = st[['City', 'State']].drop_duplicates(subset='City')
cd = cd.merge(city_state, on='City', how='left')

print("Cities without a State match:", cd.loc[cd['State'].isna(), 'City'].unique())

# ---------- 3. Drop Xylene ----------
cd = cd.drop(columns=['Xylene'])

# ---------- 4. Impute pollutants (city + month median) ----------
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
cd['Month'] = cd['Date'].dt.month

before_missing = cd[pollutants].isna().mean() * 100

for col in pollutants:
    cd[col] = cd.groupby(['City', 'Month'])[col].transform(lambda x: x.fillna(x.median()))
    # fallback: city-level median if still missing (city+month too sparse)
    cd[col] = cd.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    # final fallback: global median
    cd[col] = cd[col].fillna(cd[col].median())

after_missing = cd[pollutants].isna().mean() * 100
print("\nMissing % before -> after imputation:")
print(pd.DataFrame({'before': before_missing, 'after': after_missing}))

# ---------- 5. CPCB AQI sub-index breakpoints ----------
# Format: [(C_low, C_high, I_low, I_high), ...]
breakpoints = {
    'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
    'PM10':  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
    'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,600,401,500)],
    'O3':    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)],
    'CO':    [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)],
    'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2100,401,500)],
    'NH3':   [(0,200,0,50),(201,400,51,100),(401,800,101,200),(801,1200,201,300),(1201,1800,301,400),(1801,2400,401,500)],
}

def sub_index(pollutant, value):
    if pd.isna(value):
        return np.nan
    bps = breakpoints[pollutant]
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= value <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (value - c_lo) + i_lo
    # value above table range -> cap at last band, linear extrapolate
    c_lo, c_hi, i_lo, i_hi = bps[-1]
    if value > c_hi:
        return ((i_hi - i_lo) / (c_hi - c_lo)) * (value - c_lo) + i_lo
    return np.nan

def compute_aqi(row):
    if not pd.isna(row['AQI']):
        return row['AQI'], row['AQI_Bucket']
    sub_indices = {}
    for p in breakpoints:
        sub_indices[p] = sub_index(p, row[p])
    valid = {k: v for k, v in sub_indices.items() if not pd.isna(v)}
    # CPCB rule: need at least 3 pollutants, including PM2.5 or PM10
    has_pm = ('PM2.5' in valid) or ('PM10' in valid)
    if len(valid) >= 3 and has_pm:
        aqi = max(valid.values())
        bucket = aqi_bucket(aqi)
        return round(aqi), bucket
    return np.nan, np.nan

def aqi_bucket(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

print("\nRecomputing missing AQI values via CPCB formula...")
missing_before = cd['AQI'].isna().sum()

results = cd.apply(compute_aqi, axis=1)
cd['AQI'] = results.apply(lambda x: x[0])
cd['AQI_Bucket'] = results.apply(lambda x: x[1])

missing_after = cd['AQI'].isna().sum()
print(f"AQI missing before: {missing_before} ({missing_before/len(cd)*100:.1f}%)")
print(f"AQI missing after recomputation: {missing_after} ({missing_after/len(cd)*100:.1f}%)")
print(f"Recovered: {missing_before - missing_after} rows")

# ---------- 6. Drop rows still missing AQI ----------
cd_clean = cd.dropna(subset=['AQI', 'AQI_Bucket']).reset_index(drop=True)
print(f"\nFinal dataset shape: {cd_clean.shape}")
print(f"Rows dropped (unrecoverable): {len(cd) - len(cd_clean)} ({(len(cd)-len(cd_clean))/len(cd)*100:.1f}%)")

# ---------- 7. Save ----------
cd_clean = cd_clean.drop(columns=['Month'])
out_path = '/home/nv/da379/data/processed/city_day_cleaned.csv'
cd_clean.to_csv(out_path, index=False)
print(f"\nSaved cleaned dataset to {out_path}")
print(cd_clean.head())
