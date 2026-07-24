"""
Phase 2: Exploratory Data Analysis
Air Quality Inspection & Prediction (Pan-India)

Outputs (saved to outputs/figures/):
1. national_aqi_trend.png       - monthly national avg AQI over time
2. top_bottom_cities_aqi.png    - top10 & bottom10 cities by avg AQI
3. statewise_avg_aqi.png        - state-wise avg AQI bar chart
4. seasonal_pattern.png         - avg AQI by month (seasonality)
5. pollutant_correlation.png    - correlation heatmap of pollutants
6. aqi_bucket_distribution.png  - AQI category distribution
7. city_pollutant_boxplot.png   - PM2.5 spread, top 10 most polluted cities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sns.set_style('whitegrid')
FIG_DIR = '/home/nv/da379/outputs/figures'

df = pd.read_csv('/home/nv/da379/data/processed/city_day_cleaned.csv', parse_dates=['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['YearMonth'] = df['Date'].dt.to_period('M')

pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']

# ---------- 1. National AQI trend ----------
monthly = df.groupby('YearMonth')['AQI'].mean().reset_index()
monthly['YearMonth'] = monthly['YearMonth'].astype(str)

plt.figure(figsize=(14,5))
plt.plot(monthly['YearMonth'], monthly['AQI'], marker='o', markersize=3, color='crimson')
plt.xticks(rotation=90, fontsize=7)
plt.title('National Average AQI Trend (2015-2020)')
plt.xlabel('Month'); plt.ylabel('Avg AQI')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/national_aqi_trend.png', dpi=150)
plt.close()

# ---------- 2. Top / Bottom 10 cities ----------
city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
top10 = city_avg.head(10)
bottom10 = city_avg.tail(10)

fig, axes = plt.subplots(1,2, figsize=(14,5))
sns.barplot(x=top10.values, y=top10.index, ax=axes[0], palette='Reds_r')
axes[0].set_title('Top 10 Most Polluted Cities (avg AQI)')
axes[0].set_xlabel('Avg AQI')
sns.barplot(x=bottom10.values, y=bottom10.index, ax=axes[1], palette='Greens')
axes[1].set_title('Top 10 Least Polluted Cities (avg AQI)')
axes[1].set_xlabel('Avg AQI')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/top_bottom_cities_aqi.png', dpi=150)
plt.close()

# ---------- 3. State-wise avg AQI ----------
state_avg = df.groupby('State')['AQI'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x=state_avg.values, y=state_avg.index, palette='OrRd_r')
plt.title('State-wise Average AQI')
plt.xlabel('Avg AQI'); plt.ylabel('State')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/statewise_avg_aqi.png', dpi=150)
plt.close()

# ---------- 4. Seasonal pattern ----------
month_avg = df.groupby('Month')['AQI'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.figure(figsize=(10,5))
sns.barplot(x=[month_names[m-1] for m in month_avg.index], y=month_avg.values, palette='coolwarm')
plt.title('Seasonal Pattern: Avg AQI by Month (all years combined)')
plt.xlabel('Month'); plt.ylabel('Avg AQI')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/seasonal_pattern.png', dpi=150)
plt.close()

# ---------- 5. Pollutant correlation heatmap ----------
corr = df[pollutants + ['AQI']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Pollutant Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/pollutant_correlation.png', dpi=150)
plt.close()

# ---------- 6. AQI Bucket distribution ----------
bucket_order = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
bucket_counts = df['AQI_Bucket'].value_counts().reindex(bucket_order)
plt.figure(figsize=(8,5))
sns.barplot(x=bucket_counts.index, y=bucket_counts.values, palette='YlOrRd')
plt.title('AQI Category Distribution (All Records)')
plt.xlabel('AQI Bucket'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/aqi_bucket_distribution.png', dpi=150)
plt.close()

# ---------- 7. PM2.5 boxplot, top 10 polluted cities ----------
top10_cities = city_avg.head(10).index
plt.figure(figsize=(12,6))
sns.boxplot(data=df[df['City'].isin(top10_cities)], x='City', y='PM2.5', palette='Reds_r')
plt.xticks(rotation=45)
plt.title('PM2.5 Distribution - Top 10 Most Polluted Cities')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/city_pollutant_boxplot.png', dpi=150)
plt.close()

# ---------- Summary stats printout ----------
print("=== EDA SUMMARY ===")
print(f"\nDate range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Total records: {len(df)}, Cities: {df['City'].nunique()}, States: {df['State'].nunique()}")
print(f"\nOverall Avg AQI: {df['AQI'].mean():.1f}, Median: {df['AQI'].median():.1f}")
print(f"\nMost polluted city (avg AQI): {top10.index[0]} ({top10.iloc[0]:.1f})")
print(f"Least polluted city (avg AQI): {bottom10.index[-1]} ({bottom10.iloc[-1]:.1f})")
print(f"\nMost polluted month (avg AQI): {month_names[month_avg.idxmax()-1]} ({month_avg.max():.1f})")
print(f"Least polluted month (avg AQI): {month_names[month_avg.idxmin()-1]} ({month_avg.min():.1f})")
print(f"\nAQI Bucket distribution:\n{bucket_counts}")
print(f"\nTop 3 pollutants correlated with AQI:")
print(corr['AQI'].drop('AQI').sort_values(ascending=False).head(3))
print("\nAll figures saved to:", FIG_DIR)
