"""
Phase 8: Streamlit Dashboard
Air Quality Inspection & Prediction (Pan-India)

Run with: streamlit run dashboard/app.py

Required files:
  - data/processed/city_day_features.csv
  - outputs/maps/india_aqi_map.html
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- Paths (update to your local da379/ structure) ----------
DATA_PATH = '/home/nv/da379/data/processed/city_day_features.csv'
MAP_INDIA_PATH = '/home/nv/da379/outputs/maps/india_aqi_map.html'

st.set_page_config(page_title="India Air Quality Dashboard", layout="wide")

FOOTER = (
    "IndiaClimate AQI Dataset · CPCB / data.gov.in (2015–2020) · "
    "IIT Guwahati — B.Sc. Data Science & AI (DA379) · Built with Streamlit + Plotly + Folium"
)

def show_footer():
    st.markdown("---")
    st.caption(FOOTER)

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    return df

df = load_data()

feature_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene',
                 'Year','Month','Day','DayOfWeek','City_enc','State_enc',
                 'AQI_lag1','AQI_lag7','AQI_roll7','AQI_roll30',
                 'Season_Monsoon','Season_Post-Monsoon','Season_Summer','Season_Winter']

# ---------- Train models once, cached ----------
@st.cache_resource
def train_models(df):
    X = df[feature_cols]
    y_reg = df['AQI']
    y_clf = df['AQI_Bucket']
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_reg_train)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_clf_train)
    return reg, clf

reg_model, clf_model = train_models(df)

city_encoding = dict(zip(df['City'], df['City_enc']))
state_encoding_lookup = df[['City','State_enc']].drop_duplicates().set_index('City')['State_enc'].to_dict()

st.title("🌫️ India Air Quality Inspection & Prediction")
st.caption("Historical analysis, city inspection, AQI prediction, and geo-visualization (2015–2020)")

# ================= SIDEBAR FILTERS =================
st.sidebar.header("Filters")

all_states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect("State(s)", all_states, default=all_states)

cities_in_states = sorted(df[df['State'].isin(selected_states)]['City'].unique())
selected_cities = st.sidebar.multiselect("City(s)", cities_in_states, default=cities_in_states)

bucket_order = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
selected_buckets = st.sidebar.multiselect("AQI Category", bucket_order, default=bucket_order)

year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

min_aqi = st.sidebar.slider("Minimum AQI", 0, int(df['AQI'].max()), 0)

st.sidebar.markdown("---")

filtered_df = df[
    (df['State'].isin(selected_states)) &
    (df['City'].isin(selected_cities)) &
    (df['AQI_Bucket'].isin(selected_buckets)) &
    (df['Year'] >= selected_years[0]) &
    (df['Year'] <= selected_years[1]) &
    (df['AQI'] >= min_aqi)
]

if filtered_df.empty:
    st.sidebar.warning("No data for this filter combination.")
    filtered_df = df  # fallback so tabs don't break

st.sidebar.markdown(
    f'<p style="margin: 8px 0; font-size: 15px;">{len(filtered_df):,} records match current filters</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset:** Air Quality Data in India (2015–2020)  \n"
    f"**Source:** [data.gov.in](https://data.gov.in) (CPCB)  \n"
    f"**Cities:** {df['City'].nunique()}  \n"
    f"**States:** {df['State'].nunique()}  \n"
    f"**Records:** {len(df):,}"
)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏙️ City Inspection", "🔮 Predict AQI", "🗺️ Map View"])

# ================= TAB 1: OVERVIEW =================
with tab1:
    st.subheader("National Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg AQI (Filtered)", f"{filtered_df['AQI'].mean():.1f}")
    col2.metric("Most Polluted City", filtered_df.groupby('City')['AQI'].mean().idxmax())
    col3.metric("Least Polluted City", filtered_df.groupby('City')['AQI'].mean().idxmin())
    col4.metric("Records (Filtered)", f"{len(filtered_df):,}")

    monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['AQI'].mean().reset_index()
    monthly['Date'] = monthly['Date'].astype(str)
    fig = px.line(monthly, x='Date', y='AQI', title='Average AQI Trend (Filtered)')
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        city_avg = filtered_df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
        fig2 = px.bar(x=city_avg.values, y=city_avg.index, orientation='h',
                      title='Top 10 Most Polluted Cities (Filtered)', labels={'x':'Avg AQI','y':'City'})
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        bucket_counts = filtered_df['AQI_Bucket'].value_counts().reindex(bucket_order)
        fig3 = px.bar(x=bucket_counts.index, y=bucket_counts.values,
                      title='AQI Category Distribution (Filtered)', labels={'x':'Category','y':'Count'})
        st.plotly_chart(fig3, use_container_width=True)

    show_footer()

# ================= TAB 2: CITY INSPECTION =================
def aqi_bucket_from_value(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

with tab2:
    st.subheader("City-wise Inspection")
    available_cities = sorted(filtered_df['City'].unique())
    selected_city = st.selectbox("Select a city", available_cities)
    city_df = filtered_df[filtered_df['City'] == selected_city].sort_values('Date')

    avg_aqi_val = city_df['AQI'].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg AQI", f"{avg_aqi_val:.1f}")
    col2.metric("Avg Category", aqi_bucket_from_value(avg_aqi_val))
    col3.metric("Max AQI", f"{city_df['AQI'].max():.0f}")
    col4.metric("Latest Day Category", city_df.iloc[-1]['AQI_Bucket'])

    fig = px.line(city_df, x='Date', y='AQI', title=f'{selected_city} - AQI Trend')
    st.plotly_chart(fig, use_container_width=True)

    seasonal = city_df.groupby('Month')['AQI'].mean()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig5 = px.bar(x=[month_names[m-1] for m in seasonal.index], y=seasonal.values,
                  title=f'{selected_city} - Seasonal Pattern', labels={'x':'Month','y':'Avg AQI'})
    st.plotly_chart(fig5, use_container_width=True)

    pollutant_cols = ['PM2.5','PM10','NO2','SO2','CO','O3']
    avg_pollutants = city_df[pollutant_cols].mean().reset_index()
    avg_pollutants.columns = ['Pollutant', 'Avg Value']
    fig4 = px.bar(avg_pollutants, x='Pollutant', y='Avg Value', title=f'{selected_city} - Avg Pollutant Levels')
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### City Statistics Table (All Filtered Cities)")
    city_stats_table = filtered_df.groupby('City').agg(
        State=('State', 'first'),
        Records=('AQI', 'count'),
        Avg_AQI=('AQI', 'mean'),
        Median_AQI=('AQI', 'median'),
        Max_AQI=('AQI', 'max'),
        Avg_PM2_5=('PM2.5', 'mean')
    ).round(1).sort_values('Avg_AQI', ascending=False).reset_index()
    st.dataframe(city_stats_table, use_container_width=True)

    show_footer()

# ================= TAB 3: PREDICTION =================
with tab3:
    st.subheader("Predict AQI from Pollutant Levels")
    st.write("Enter pollutant readings to predict AQI and its category.")

    pred_city = st.selectbox("City (for encoding context)", sorted(df['City'].unique()), key='pred_city')

    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.number_input("PM2.5", min_value=0.0, value=100.0)
        pm10 = st.number_input("PM10", min_value=0.0, value=150.0)
        no = st.number_input("NO", min_value=0.0, value=10.0)
        no2 = st.number_input("NO2", min_value=0.0, value=30.0)
    with col2:
        nox = st.number_input("NOx", min_value=0.0, value=25.0)
        nh3 = st.number_input("NH3", min_value=0.0, value=20.0)
        co = st.number_input("CO", min_value=0.0, value=1.0)
        so2 = st.number_input("SO2", min_value=0.0, value=15.0)
    with col3:
        o3 = st.number_input("O3", min_value=0.0, value=40.0)
        benzene = st.number_input("Benzene", min_value=0.0, value=3.0)
        toluene = st.number_input("Toluene", min_value=0.0, value=8.0)
        pred_month = st.slider("Month", 1, 12, 6)

    if st.button("Predict AQI"):
        city_avg_aqi = df[df['City'] == pred_city]['AQI'].mean()
        season_map = {12:'Winter',1:'Winter',2:'Winter',3:'Summer',4:'Summer',5:'Summer',
                      6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',10:'Post-Monsoon',11:'Post-Monsoon'}
        season = season_map[pred_month]

        input_row = pd.DataFrame([{
            'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': nox, 'NH3': nh3,
            'CO': co, 'SO2': so2, 'O3': o3, 'Benzene': benzene, 'Toluene': toluene,
            'Year': 2024, 'Month': pred_month, 'Day': 15, 'DayOfWeek': 2,
            'City_enc': city_encoding[pred_city], 'State_enc': state_encoding_lookup[pred_city],
            'AQI_lag1': city_avg_aqi, 'AQI_lag7': city_avg_aqi,
            'AQI_roll7': city_avg_aqi, 'AQI_roll30': city_avg_aqi,
            'Season_Monsoon': 1 if season == 'Monsoon' else 0,
            'Season_Post-Monsoon': 1 if season == 'Post-Monsoon' else 0,
            'Season_Summer': 1 if season == 'Summer' else 0,
            'Season_Winter': 1 if season == 'Winter' else 0,
        }])[feature_cols]

        pred_aqi = reg_model.predict(input_row)[0]
        pred_bucket = clf_model.predict(input_row)[0]

        st.success(f"Predicted AQI: **{pred_aqi:.1f}**")
        st.info(f"Predicted Category: **{pred_bucket}**")

        col1, col2 = st.columns(2)

        with col1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_aqi,
                title={'text': "Predicted AQI"},
                gauge={
                    'axis': {'range': [0, 500]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 50], 'color': "green"},
                        {'range': [50, 100], 'color': "lightgreen"},
                        {'range': [100, 200], 'color': "orange"},
                        {'range': [200, 300], 'color': "red"},
                        {'range': [300, 400], 'color': "darkred"},
                        {'range': [400, 500], 'color': "purple"},
                    ],
                }
            ))
            st.plotly_chart(gauge, use_container_width=True)

        with col2:
            national_avg = df['AQI'].mean()
            comparison = pd.DataFrame({
                'Category': ['Your Prediction', f'{pred_city} Avg', 'National Avg'],
                'AQI': [pred_aqi, city_avg_aqi, national_avg]
            })
            fig_comp = px.bar(comparison, x='Category', y='AQI', color='Category',
                               title='Prediction vs Historical Averages')
            st.plotly_chart(fig_comp, use_container_width=True)

    show_footer()

# ================= TAB 4: MAP VIEW =================
with tab4:
    st.subheader("Geo-Visualization")
    st.write("All 26 cities, colored and sized by average AQI (2015-2020).")

    try:
        with open(MAP_INDIA_PATH, 'r', encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=600)
    except FileNotFoundError:
        st.error(f"Map file not found at {MAP_INDIA_PATH}. Run the Phase 4 script first to generate it.")

    st.markdown("#### Regional Statistics Table")
    stats_table = df.groupby('City').agg(
        State=('State', 'first'),
        Records=('AQI', 'count'),
        Avg_AQI=('AQI', 'mean'),
        Max_AQI=('AQI', 'max')
    ).round(1).sort_values('Avg_AQI', ascending=False).reset_index()

    st.dataframe(stats_table, use_container_width=True)

    show_footer()
