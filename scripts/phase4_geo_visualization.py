"""
Phase 4: Geo-Visualization
Air Quality Inspection & Prediction (Pan-India)

Input:  data/processed/city_day_features.csv
Output: outputs/maps/india_aqi_map.html

Creates an interactive folium map of India with city markers
colored/sized by average AQI, with popup details.
"""

import pandas as pd
import folium

IN_PATH = '/home/nv/da379/data/processed/city_day_features.csv'
OUT_PATH = '/home/nv/da379/outputs/maps/india_aqi_map.html'
COORDS_PATH = '/home/nv/da379/data/processed/city_coordinates.csv'

# ---------- City coordinate lookup (loaded from CSV) ----------
coords_df = pd.read_csv(COORDS_PATH)
city_coords = {row['City']: (row['Latitude'], row['Longitude']) for _, row in coords_df.iterrows()}

def aqi_color(aqi):
    if aqi <= 50: return 'green'
    elif aqi <= 100: return 'lightgreen'
    elif aqi <= 200: return 'orange'
    elif aqi <= 300: return 'red'
    elif aqi <= 400: return 'darkred'
    else: return 'purple'

df = pd.read_csv(IN_PATH)
city_stats = df.groupby('City').agg(
    avg_AQI=('AQI', 'mean'),
    max_AQI=('AQI', 'max'),
    State=('State', 'first')
).reset_index()

missing = set(city_stats['City']) - set(city_coords.keys())
if missing:
    print(f"WARNING: no coordinates for: {missing}")

m = folium.Map(location=[22.5, 80], zoom_start=5, tiles='CartoDB positron')

for _, row in city_stats.iterrows():
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
        radius=6 + (avg_aqi / 40),
        popup=folium.Popup(popup_html, max_width=200),
        tooltip=city,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(m)

# Legend
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

import os
os.makedirs('/home/nv/da379/outputs/maps', exist_ok=True)
m.save(OUT_PATH)

print("=== GEO-VISUALIZATION SUMMARY ===")
print(f"Cities plotted: {len(city_stats) - len(missing)}")
print(f"Highest avg AQI: {city_stats.loc[city_stats['avg_AQI'].idxmax(), 'City']} "
      f"({city_stats['avg_AQI'].max():.1f})")
print(f"Lowest avg AQI: {city_stats.loc[city_stats['avg_AQI'].idxmin(), 'City']} "
      f"({city_stats['avg_AQI'].min():.1f})")
print(f"Map saved to {OUT_PATH}")
