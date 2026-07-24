"""
Phase 5: Time Series Analysis
Air Quality Inspection & Prediction (Pan-India)
Top 10 most polluted cities: seasonal decomposition + Holt-Winters forecast
Plus: top 10 avg aqi map (folium)

Input:  data/processed/city_day_features.csv
Outputs:
  - outputs/tables/monthly_forecast_top10.csv
  - outputs/figures/decomposition_top10.png
  - outputs/figures/forecast_top10.png
  - outputs/maps/top10_avg_aqi_map.html
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import folium

IN_PATH = '/home/nv/da379/data/processed/city_day_features.csv'
TABLE_OUT = '/home/nv/da379/outputs/tables/monthly_forecast_top10.csv'
DECOMP_FIG = '/home/nv/da379/outputs/figures/decomposition_top10.png'
FORECAST_FIG = '/home/nv/da379/outputs/figures/forecast_top10.png'
MAP_OUT = '/home/nv/da379/outputs/maps/top10_avg_aqi_map.html'

os.makedirs('/home/nv/da379/outputs/tables', exist_ok=True)
os.makedirs('/home/nv/da379/outputs/figures', exist_ok=True)
os.makedirs('/home/nv/da379/outputs/maps', exist_ok=True)

COORDS_PATH = '/home/nv/da379/data/processed/city_coordinates.csv'
coords_df = pd.read_csv(COORDS_PATH)
city_coords = {row['City']: (row['Latitude'], row['Longitude']) for _, row in coords_df.iterrows()}

df = pd.read_csv(IN_PATH, parse_dates=['Date'])

# ---------- Top 10 most polluted cities (with >=24 months of data, for valid decomposition) ----------
city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
city_months = df.groupby('City')['Date'].agg(lambda x: (x.max()-x.min()).days/30)
eligible_cities = city_months[city_months >= 24].index
top10 = city_avg[city_avg.index.isin(eligible_cities)].head(10).index.tolist()
print("Top 10 most polluted cities (with sufficient history for decomposition):", top10)

# ---------- Monthly series per city ----------
monthly_series = {}
for city in top10:
    s = df[df['City'] == city].set_index('Date')['AQI'].resample('MS').mean()
    s = s.interpolate()  # fill any gap months
    monthly_series[city] = s

# ---------- Decomposition (grid: 10 rows x 3 cols) ----------
fig, axes = plt.subplots(10, 3, figsize=(15, 30))
for i, city in enumerate(top10):
    s = monthly_series[city]
    if len(s) >= 24:
        result = seasonal_decompose(s, model='additive', period=12)
        axes[i,0].plot(result.trend, color='blue'); axes[i,0].set_title(f'{city} - Trend', fontsize=9)
        axes[i,1].plot(result.seasonal, color='green'); axes[i,1].set_title(f'{city} - Seasonal', fontsize=9)
        axes[i,2].plot(result.resid, color='red'); axes[i,2].set_title(f'{city} - Residual', fontsize=9)
    for j in range(3):
        axes[i,j].tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(DECOMP_FIG, dpi=120)
plt.close()

# ---------- Holt-Winters forecast (6 months ahead) per city ----------
# Cities with < 24 months of data can't support seasonal_periods=12 (needs 2 full cycles)
# -> fall back to trend-only exponential smoothing for those.
forecast_rows = []
fig, axes = plt.subplots(5, 2, figsize=(14, 16))
axes = axes.flatten()

for i, city in enumerate(top10):
    s = monthly_series[city]
    model = ExponentialSmoothing(s, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(6)

    for date, val in forecast.items():
        forecast_rows.append({'City': city, 'Month': date.strftime('%Y-%m'), 'Forecast_AQI': round(val, 1)})

    axes[i].plot(s.index, s.values, label='Historical', color='steelblue')
    axes[i].plot(forecast.index, forecast.values, label='Forecast', color='crimson', linestyle='--', marker='o')
    axes[i].set_title(city, fontsize=10)
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(FORECAST_FIG, dpi=120)
plt.close()

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(TABLE_OUT, index=False)

# ---------- Folium map: avg AQI for top 10 cities (static, same style as Phase 4) ----------
def aqi_color(aqi):
    if aqi <= 50: return 'green'
    elif aqi <= 100: return 'lightgreen'
    elif aqi <= 200: return 'orange'
    elif aqi <= 300: return 'red'
    elif aqi <= 400: return 'darkred'
    else: return 'purple'

top10_avg = df[df['City'].isin(top10)].groupby('City').agg(
    avg_AQI=('AQI', 'mean'),
    max_AQI=('AQI', 'max'),
    State=('State', 'first')
).reset_index()

m = folium.Map(location=[22.5, 80], zoom_start=5, tiles='CartoDB positron')

for _, row in top10_avg.iterrows():
    city = row['City']
    if city not in city_coords:
        continue
    lat, lon = city_coords[city]
    avg_aqi = row['avg_AQI']
    color = aqi_color(avg_aqi)
    popup_html = (
        f"<b>{city}</b> ({row['State']})<br>"
        f"Avg AQI: {avg_aqi:.1f}<br>"
        f"Max AQI: {row['max_AQI']:.0f}"
    )
    folium.CircleMarker(
        location=[lat, lon],
        radius=8 + (avg_aqi / 35),
        popup=folium.Popup(popup_html, max_width=200),
        tooltip=city,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(m)

legend_html = '''
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px; font-size: 13px;">
<b>AQI Category</b><br>
<span style="color:green;">&#9679;</span> Good (0-50)<br>
<span style="color:lightgreen;">&#9679;</span> Satisfactory (51-100)<br>
<span style="color:orange;">&#9679;</span> Moderate (101-200)<br>
<span style="color:red;">&#9679;</span> Poor (201-300)<br>
<span style="color:darkred;">&#9679;</span> Very Poor (301-400)<br>
<span style="color:purple;">&#9679;</span> Severe (400+)
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))
m.save(MAP_OUT)

print("\n=== PHASE 5 SUMMARY ===")
print(f"Decomposition figure: {DECOMP_FIG}")
print(f"Forecast figure: {FORECAST_FIG}")
print(f"Forecast table: {TABLE_OUT}")
print(f"Average AQI map: {MAP_OUT}")
print("\nSample forecast (first city):")
print(forecast_df[forecast_df['City'] == top10[0]])
