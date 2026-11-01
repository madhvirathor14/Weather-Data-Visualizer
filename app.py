import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Data Visualizer", page_icon="🌦️", layout="wide")

st.title("🌦️ Weather Data Visualizer")
st.caption("Explore temperature, humidity, rainfall and wind patterns using Python, Pandas, NumPy and Matplotlib.")

@st.cache_data
def load_sample():
    return pd.read_csv("weather_data.csv")

def prepare_data(data):
    data = data.copy()
    required = ["Date", "City", "Temperature", "Feels_Like", "Humidity", "Rainfall", "Wind_Speed", "Weather"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    numeric = ["Temperature", "Feels_Like", "Humidity", "Rainfall", "Wind_Speed"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "City"])
    return data

with st.sidebar:
    st.header("📂 Weather dataset")
    uploaded = st.file_uploader("Upload your own CSV (optional)", type=["csv"])
    st.info("No upload? The project includes a sample dataset, so it works immediately.")
    st.divider()
    st.caption("Sample values are illustrative and are not live weather readings.")

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_sample()
    df = prepare_data(raw)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

if df.empty:
    st.warning("No valid records found in this dataset.")
    st.stop()

with st.sidebar:
    cities = sorted(df["City"].dropna().unique().tolist())
    selected_cities = st.multiselect("Select city", cities, default=cities)
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input("Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df[df["City"].isin(selected_cities)].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[(filtered["Date"].dt.date >= start_date) & (filtered["Date"].dt.date <= end_date)]

if filtered.empty:
    st.warning("No records match the selected filters. Change the city or date range.")
    st.stop()

# Missing-value summary
missing_total = int(filtered.isna().sum().sum())
avg_temp = filtered["Temperature"].mean()
avg_humidity = filtered["Humidity"].mean()
total_rain = filtered["Rainfall"].sum()
max_temp = filtered["Temperature"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🌡️ Average temperature", f"{avg_temp:.1f} °C" if pd.notna(avg_temp) else "N/A")
c2.metric("💧 Average humidity", f"{avg_humidity:.1f}%" if pd.notna(avg_humidity) else "N/A")
c3.metric("🌧️ Total rainfall", f"{total_rain:.1f} mm" if pd.notna(total_rain) else "N/A")
c4.metric("🔥 Highest temperature", f"{max_temp:.1f} °C" if pd.notna(max_temp) else "N/A")

st.subheader("📈 Weather trends")
tab1, tab2, tab3 = st.tabs(["Temperature", "Rainfall & humidity", "Compare cities"])

with tab1:
    daily = filtered.groupby("Date", as_index=False)["Temperature"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["Date"], daily["Temperature"], marker="o", linewidth=2, label="Average temperature")
    ax.set_title("Temperature trend over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    left, right = st.columns(2)
    with left:
        rain = filtered.groupby("Date", as_index=False)["Rainfall"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(rain["Date"], rain["Rainfall"])
        ax.set_title("Daily rainfall")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rainfall (mm)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(filtered["Temperature"], filtered["Humidity"], alpha=0.65)
        ax.set_title("Temperature vs humidity")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Humidity (%)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    city_summary = filtered.groupby("City", as_index=False).agg(
        Average_Temperature=("Temperature", "mean"),
        Average_Humidity=("Humidity", "mean"),
        Total_Rainfall=("Rainfall", "sum")
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(city_summary["City"], city_summary["Average_Temperature"])
    ax.set_title("Average temperature by city")
    ax.set_xlabel("City")
    ax.set_ylabel("Temperature (°C)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(city_summary.round(2), use_container_width=True, hide_index=True)

st.subheader("🧮 Quick data analysis")
a, b = st.columns(2)
with a:
    st.write("**Weather condition counts**")
    counts = filtered["Weather"].value_counts().rename_axis("Weather").reset_index(name="Days")
    st.dataframe(counts, use_container_width=True, hide_index=True)
with b:
    st.write("**Statistical summary**")
    summary = filtered[["Temperature", "Feels_Like", "Humidity", "Rainfall", "Wind_Speed"]].describe().round(2)
    st.dataframe(summary, use_container_width=True)

st.subheader("🧹 Data quality")
st.write(f"Missing cells in the filtered dataset: **{missing_total}**")
if missing_total:
    if st.button("Drop rows containing missing values"):
        filtered = filtered.dropna()
        st.success(f"Cleaned dataset now has {len(filtered)} rows. (This change applies to this session view.)")
else:
    st.success("No missing values in the current filtered dataset.")

st.subheader("📋 Weather records")
st.dataframe(filtered.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data as CSV", data=csv_bytes, file_name="filtered_weather_data.csv", mime="text/csv")

st.caption("Built with Python • Streamlit • Pandas • NumPy • Matplotlib")
