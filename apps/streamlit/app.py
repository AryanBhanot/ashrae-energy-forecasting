from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ASHRAE Forecast — Building 1074", page_icon="⚡", layout="wide")

# --- Data & Model ---
BASE_DIR = Path(__file__).resolve().parents[2]

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "data" / "building" / "1074" / "building_1074.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    lookup = df.set_index("timestamp")["meter_reading"]
    df["lag_24h"] = (df["timestamp"] - pd.Timedelta(hours=24)).map(lookup)
    df["lag_168h"] = (df["timestamp"] - pd.Timedelta(hours=168)).map(lookup)
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    return df

@st.cache_resource
def load_model():
    p = BASE_DIR / "models" / "building_1074_six_months_model.pkl"
    return joblib.load(p) if p.exists() else None

df = load_data()
model = load_model()

# --- August forecast data ---
FEATURES = ["hour", "dow", "month", "lag_24h", "lag_168h"]
aug = df[df["month"] == 8].copy()
aug["forecast"] = model.predict(aug[FEATURES]) if model else np.nan

# --- Header (compact) ---
st.markdown("## ⚡ Energy Forecast — Building 1074 &nbsp;·&nbsp; August 2016")
st.caption("Model trained on Jan–Jun 2016 &nbsp;|&nbsp; Office · 89,858 sq ft")

# --- KPI row ---
total_actual = aug["meter_reading"].sum()
total_forecast = aug["forecast"].sum()
peak_actual = aug["meter_reading"].max()
avg_actual = aug["meter_reading"].mean()
avg_forecast = aug["forecast"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Actual", f"{total_actual:,.0f} kWh")
c2.metric("Total Forecast", f"{total_forecast:,.0f} kWh")
c3.metric("Avg Hourly", f"{avg_actual:.1f} kWh", f"Forecast: {avg_forecast:.1f}")
c4.metric("Peak Hour", f"{peak_actual:.1f} kWh")
c5.metric("Hours", f"{len(aug)}")

# === TABS ===
tab_hourly, tab_daily, tab_weekly, tab_profile = st.tabs([
    "Hourly Forecast", "Daily Totals", "Weekly Summary", "Hour-of-Day Profile"
])

# ------ TAB 1: Hourly (full month, zoomable) ------
with tab_hourly:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aug["timestamp"], y=aug["meter_reading"],
        name="Actual", line=dict(color="#2563eb", width=1.8),
        hovertemplate="%{y:.1f} kWh<extra>Actual</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=aug["timestamp"], y=aug["forecast"],
        name="Forecast", line=dict(color="#f97316", width=1.8, dash="dash"),
        hovertemplate="%{y:.1f} kWh<extra>Forecast</extra>"
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.04),
                   showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key="hourly")

# ------ TAB 2: Daily totals bar chart ------
with tab_daily:
    daily = aug.groupby(aug["timestamp"].dt.date).agg(
        actual=("meter_reading", "sum"),
        forecast=("forecast", "sum")
    ).reset_index()
    daily.columns = ["date", "actual", "forecast"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["actual"],
        name="Actual", marker_color="#2563eb", opacity=0.85,
        hovertemplate="%{y:,.0f} kWh<extra>Actual</extra>"
    ))
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["forecast"],
        name="Forecast", marker_color="#f97316", opacity=0.85,
        hovertemplate="%{y:,.0f} kWh<extra>Forecast</extra>"
    ))
    fig.update_layout(
        barmode="group", height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis=dict(title="Date", tickformat="%b %d", dtick=86400000),
        yaxis=dict(title="Daily Total (kWh)", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key="daily")

# ------ TAB 3: Weekly summary ------
with tab_weekly:
    aug_wk = aug.copy()
    aug_wk["week_label"] = "Week " + (
        (aug_wk["timestamp"].dt.day - 1) // 7 + 1
    ).astype(str)

    weekly = aug_wk.groupby("week_label").agg(
        actual_total=("meter_reading", "sum"),
        forecast_total=("forecast", "sum"),
        actual_peak=("meter_reading", "max"),
        forecast_peak=("forecast", "max"),
        actual_avg=("meter_reading", "mean"),
        forecast_avg=("forecast", "mean"),
        hours=("meter_reading", "count")
    ).reindex([f"Week {i}" for i in range(1, 6)])

    # Two side-by-side charts: total energy + peak demand per week
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Weekly Total Energy (kWh)", "Weekly Peak Demand (kWh)")
    )
    for col_idx, (act_col, fct_col, fmt) in enumerate([
        ("actual_total", "forecast_total", ",.0f"),
        ("actual_peak", "forecast_peak", ".1f")
    ], start=1):
        fig.add_trace(go.Bar(
            x=weekly.index, y=weekly[act_col],
            name="Actual" if col_idx == 1 else None,
            marker_color="#2563eb", opacity=0.85, showlegend=(col_idx == 1),
            hovertemplate=f"%{{y:{fmt}}} kWh<extra>Actual</extra>"
        ), row=1, col=col_idx)
        fig.add_trace(go.Bar(
            x=weekly.index, y=weekly[fct_col],
            name="Forecast" if col_idx == 1 else None,
            marker_color="#f97316", opacity=0.85, showlegend=(col_idx == 1),
            hovertemplate=f"%{{y:{fmt}}} kWh<extra>Forecast</extra>"
        ), row=1, col=col_idx)

    fig.update_layout(
        barmode="group", height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        hovermode="x unified"
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    st.plotly_chart(fig, use_container_width=True, key="weekly")

    # Compact weekly stats table
    tbl = weekly[["actual_total", "forecast_total", "actual_avg", "forecast_avg", "hours"]].copy()
    tbl.columns = ["Actual Total (kWh)", "Forecast Total (kWh)", "Actual Avg (kWh/h)", "Forecast Avg (kWh/h)", "Hours"]
    st.dataframe(tbl.style.format({
        "Actual Total (kWh)": "{:,.0f}",
        "Forecast Total (kWh)": "{:,.0f}",
        "Actual Avg (kWh/h)": "{:.1f}",
        "Forecast Avg (kWh/h)": "{:.1f}",
        "Hours": "{:d}"
    }), use_container_width=True)

# ------ TAB 4: Hour-of-day profile ------
with tab_profile:
    hourly = aug.groupby("hour").agg(
        actual_avg=("meter_reading", "mean"),
        forecast_avg=("forecast", "mean")
    ).reset_index()

    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_hourly = aug.groupby(["dow", "hour"]).agg(
        actual_avg=("meter_reading", "mean"),
        forecast_avg=("forecast", "mean")
    ).reset_index()

    left, right = st.columns(2)

    with left:
        st.markdown("**Average Load by Hour (All August)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["actual_avg"],
            name="Actual", mode="lines+markers",
            line=dict(color="#2563eb", width=2), marker=dict(size=5),
            hovertemplate="Hour %{x}: %{y:.1f} kWh<extra>Actual</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["forecast_avg"],
            name="Forecast", mode="lines+markers",
            line=dict(color="#f97316", width=2, dash="dash"), marker=dict(size=5),
            hovertemplate="Hour %{x}: %{y:.1f} kWh<extra>Forecast</extra>"
        ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Hour of Day", dtick=2),
            yaxis=dict(title="Avg kWh", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, key="hourly_profile")

    with right:
        st.markdown("**Weekday vs Weekend Pattern**")
        weekday = dow_hourly[dow_hourly["dow"] < 5].groupby("hour")[["actual_avg", "forecast_avg"]].mean().reset_index()
        weekend = dow_hourly[dow_hourly["dow"] >= 5].groupby("hour")[["actual_avg", "forecast_avg"]].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekday["hour"], y=weekday["actual_avg"],
            name="Weekday Actual", line=dict(color="#2563eb", width=2),
            hovertemplate="%{y:.1f} kWh<extra>Weekday Actual</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=weekday["hour"], y=weekday["forecast_avg"],
            name="Weekday Forecast", line=dict(color="#f97316", width=2, dash="dash"),
            hovertemplate="%{y:.1f} kWh<extra>Weekday Forecast</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=weekend["hour"], y=weekend["actual_avg"],
            name="Weekend Actual", line=dict(color="#7c3aed", width=2),
            hovertemplate="%{y:.1f} kWh<extra>Weekend Actual</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=weekend["hour"], y=weekend["forecast_avg"],
            name="Weekend Forecast", line=dict(color="#e879f9", width=2, dash="dash"),
            hovertemplate="%{y:.1f} kWh<extra>Weekend Forecast</extra>"
        ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Hour of Day", dtick=2),
            yaxis=dict(title="Avg kWh", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(size=11)),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, key="wd_we_profile")
