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

# --- August forecast ---
FEATURES = ["hour", "dow", "month", "lag_24h", "lag_168h"]
aug = df[df["month"] == 8].copy()
aug["forecast"] = model.predict(aug[FEATURES]) if model else np.nan
aug["error"] = aug["meter_reading"] - aug["forecast"]

# --- Header ---
st.markdown("## ⚡ Energy Forecast — Building 1074 &nbsp;·&nbsp; August 2016")
st.caption("Model trained on Jan–Jun 2016 &nbsp;|&nbsp; Office · 89,858 sq ft")

# --- KPI row ---
total_actual = aug["meter_reading"].sum()
total_forecast = aug["forecast"].sum()
peak_actual = aug["meter_reading"].max()
avg_actual = aug["meter_reading"].mean()
avg_forecast = aug["forecast"].mean()
avg_delta = avg_forecast - avg_actual          # positive = over-forecast
total_err_abs = total_actual - total_forecast  # positive = under-forecast
total_err_pct = total_err_abs / total_actual * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Actual", f"{total_actual:,.0f} kWh")
c2.metric("Total Forecast", f"{total_forecast:,.0f} kWh")
c3.metric("Avg Hourly Actual", f"{avg_actual:.1f} kWh",
          delta=f"{avg_delta:+.1f} kWh (forecast bias)",
          delta_color="inverse")          # red = model over-predicts, green = under
c4.metric("Peak Actual", f"{peak_actual:.1f} kWh")
c5.metric("Total Forecast Error",
          f"{abs(total_err_abs):,.0f} kWh",
          delta=f"{total_err_pct:+.2f}% under-forecast" if total_err_abs > 0 else f"{abs(total_err_pct):.2f}% over-forecast",
          delta_color="inverse")

# === TABS ===
tab_hourly, tab_daily, tab_weekly, tab_profile = st.tabs([
    "Hourly Forecast", "Daily Totals", "Weekly Summary", "Hour-of-Day Profile"
])

# ------ TAB 1: Hourly with week filter ------
with tab_hourly:
    # Week filter buttons
    WEEK_RANGES = {
        "Full Month": ("2016-08-01", "2016-08-31"),
        "Week 1 (Aug 1–7)":  ("2016-08-01", "2016-08-07"),
        "Week 2 (Aug 8–14)": ("2016-08-08", "2016-08-14"),
        "Week 3 (Aug 15–21)":("2016-08-15", "2016-08-21"),
        "Week 4 (Aug 22–31)":("2016-08-22", "2016-08-31"),
    }
    selected_week = st.radio("Zoom to:", list(WEEK_RANGES.keys()), horizontal=True, key="week_radio")
    start, end = WEEK_RANGES[selected_week]
    view = aug[aug["timestamp"].between(start, end + " 23:59")]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["meter_reading"],
        name="Actual", line=dict(color="#2563eb", width=1.8),
        hovertemplate="%{y:.1f} kWh<extra>Actual</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["forecast"],
        name="Forecast", line=dict(color="#f97316", width=1.8, dash="dash"),
        hovertemplate="%{y:.1f} kWh<extra>Forecast</extra>"
    ))
    fig.update_layout(
        height=370, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.04),
                   showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key="hourly")

# ------ TAB 2: Daily totals + error bars ------
with tab_daily:
    daily = aug.groupby(aug["timestamp"].dt.date).agg(
        actual=("meter_reading", "sum"),
        forecast=("forecast", "sum")
    ).reset_index()
    daily.columns = ["date", "actual", "forecast"]
    daily["error"] = daily["actual"] - daily["forecast"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Actual bars
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["actual"],
        name="Actual", marker_color="#2563eb", opacity=0.8,
        hovertemplate="%{x|%b %d}: %{y:,.0f} kWh<extra>Actual</extra>"
    ), secondary_y=False)

    # Forecast bars
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["forecast"],
        name="Forecast", marker_color="#f97316", opacity=0.8,
        hovertemplate="%{x|%b %d}: %{y:,.0f} kWh<extra>Forecast</extra>"
    ), secondary_y=False)

    # Error line on secondary axis
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["error"],
        name="Error (Actual − Forecast)",
        mode="lines+markers",
        line=dict(color="#10b981", width=2),
        marker=dict(size=5),
        hovertemplate="%{x|%b %d}: %{y:+,.0f} kWh<extra>Error</extra>"
    ), secondary_y=True)

    # Zero line on secondary axis
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", secondary_y=True)

    fig.update_layout(
        barmode="group", height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis=dict(tickformat="%b %d"),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Daily Total (kWh)", secondary_y=False,
                     showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    fig.update_yaxes(title_text="Error (kWh)", secondary_y=True,
                     showgrid=False, zeroline=True, zerolinecolor="rgba(128,128,128,0.3)")
    st.plotly_chart(fig, use_container_width=True, key="daily")

# ------ TAB 3: Weekly summary ------
with tab_weekly:
    aug_wk = aug.copy()
    aug_wk["week_label"] = "Week " + ((aug_wk["timestamp"].dt.day - 1) // 7 + 1).astype(str)

    weekly = aug_wk.groupby("week_label").agg(
        actual_total=("meter_reading", "sum"),
        forecast_total=("forecast", "sum"),
        actual_peak=("meter_reading", "max"),
        forecast_peak=("forecast", "max"),
        actual_avg=("meter_reading", "mean"),
        forecast_avg=("forecast", "mean"),
    ).reindex([f"Week {i}" for i in range(1, 6)])
    weekly["error_total"] = weekly["actual_total"] - weekly["forecast_total"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Weekly Total Energy (kWh)", "Weekly Peak Demand (kWh)")
    )
    for col_idx, (act_col, fct_col, fmt) in enumerate([
        ("actual_total", "forecast_total", ",.0f"),
        ("actual_peak",  "forecast_peak",  ".1f")
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
        barmode="group", height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        hovermode="x unified"
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    st.plotly_chart(fig, use_container_width=True, key="weekly")

    tbl = weekly[["actual_total", "forecast_total", "error_total", "actual_avg", "forecast_avg"]].copy()
    tbl.columns = ["Actual Total (kWh)", "Forecast Total (kWh)", "Error (kWh)", "Actual Avg/h", "Forecast Avg/h"]
    st.dataframe(tbl.style.format({
        "Actual Total (kWh)": "{:,.0f}",
        "Forecast Total (kWh)": "{:,.0f}",
        "Error (kWh)": "{:+,.0f}",
        "Actual Avg/h": "{:.1f}",
        "Forecast Avg/h": "{:.1f}",
    }).background_gradient(subset=["Error (kWh)"], cmap="RdYlGn_r"),
    use_container_width=True)

# ------ TAB 4: Hour-of-day Profile ------
with tab_profile:
    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Left: Heatmap — Day-of-week × Hour of actual consumption
    heatmap_data = aug.groupby(["dow", "hour"])["meter_reading"].mean().unstack()
    heatmap_data.index = [DOW_NAMES[i] for i in heatmap_data.index]

    # Right: Weekday vs Weekend ± std dev bands
    def get_stats(mask):
        g = aug[mask].groupby("hour")["meter_reading"]
        return g.mean().reset_index(name="mean"), g.std().reset_index(name="std")

    wd_mean, wd_std = get_stats(aug["dow"] < 5)
    we_mean, we_std = get_stats(aug["dow"] >= 5)

    left, right = st.columns(2)

    with left:
        st.markdown("**Avg Consumption Heatmap — Day of Week × Hour**")
        fig = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=list(range(24)),
            y=heatmap_data.index.tolist(),
            colorscale="Blues",
            colorbar=dict(title="kWh", thickness=12, len=0.8),
            hovertemplate="Hour %{x} · %{y}: %{z:.1f} kWh<extra></extra>"
        ))
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Hour of Day", dtick=2, tickmode="linear"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True, key="heatmap")

    with right:
        st.markdown("**Weekday vs Weekend — Actual ± 1 Std Dev &amp; Forecast**")
        fig = go.Figure()

        # Weekday std dev band
        fig.add_trace(go.Scatter(
            x=pd.concat([wd_mean["hour"], wd_mean["hour"][::-1]]),
            y=pd.concat([wd_mean["mean"] + wd_std["std"], (wd_mean["mean"] - wd_std["std"])[::-1]]),
            fill="toself", fillcolor="rgba(37,99,235,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Weekday ±1σ", showlegend=True,
            hoverinfo="skip"
        ))

        # Weekend std dev band
        fig.add_trace(go.Scatter(
            x=pd.concat([we_mean["hour"], we_mean["hour"][::-1]]),
            y=pd.concat([we_mean["mean"] + we_std["std"], (we_mean["mean"] - we_std["std"])[::-1]]),
            fill="toself", fillcolor="rgba(124,58,237,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Weekend ±1σ", showlegend=True,
            hoverinfo="skip"
        ))

        # Mean lines
        fig.add_trace(go.Scatter(
            x=wd_mean["hour"], y=wd_mean["mean"],
            name="Weekday Actual", line=dict(color="#2563eb", width=2),
            hovertemplate="Hour %{x}: %{y:.1f} kWh<extra>Weekday Actual</extra>"
        ))

        wd_fc = aug[aug["dow"] < 5].groupby("hour")["forecast"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=wd_fc["hour"], y=wd_fc["forecast"],
            name="Weekday Forecast", line=dict(color="#f97316", width=2, dash="dash"),
            hovertemplate="Hour %{x}: %{y:.1f} kWh<extra>Weekday Forecast</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=we_mean["hour"], y=we_mean["mean"],
            name="Weekend Actual", line=dict(color="#7c3aed", width=2),
            hovertemplate="Hour %{x}: %{y:.1f} kWh<extra>Weekend Actual</extra>"
        ))
        we_fc = aug[aug["dow"] >= 5].groupby("hour")["forecast"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=we_fc["hour"], y=we_fc["forecast"],
            name="Weekend Forecast", line=dict(color="#c026d3", width=2, dash="dash"),
            hovertemplate="Hour %{x}: %{y:.1f} kWh<extra>Weekend Forecast</extra>"
        ))

        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Hour of Day", dtick=2, tickmode="linear"),
            yaxis=dict(title="Avg kWh", showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
            legend=dict(orientation="h", y=1.18, x=1, xanchor="right", font=dict(size=10)),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, key="wd_we_profile")
