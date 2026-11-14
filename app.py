import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Orbit Analytics | Punjab Earth System Trend Detective",
    page_icon="🛰️",
    layout="wide",
)

# --- METADATA & DEFINITIONS ---
VARIABLES = {
    "Aerosol Optical Depth (Smog & Air Quality)": {
        "key": "aerosol_modis",
        "unit": "AOD (Unitless, 550 nm)",
        "sensor": "NASA MODIS-Terra (MOD08_M3 v6.1)",
        "description": (
            "Measures how much sunlight is blocked by airborne aerosols (smoke, fine dust, "
            "and industrial/crop-burning smog). Higher values indicate heavier atmospheric pollution."
        ),
        "expected": "Increasing trend driven by urban-industrial emissions and seasonal paddy stubble burning.",
    },
    "Root Zone Soil Moisture (Water Storage)": {
        "key": "soil_moisture_gldas",
        "unit": "kg m⁻²",
        "sensor": "NASA GLDAS Noah Land Surface Model (v2.1)",
        "description": (
            "Measures total water stored in the upper soil layers accessible to crops and vegetation. "
            "Tracks groundwater recharge, irrigation intensity, and agricultural drought stress."
        ),
        "expected": "Decreasing or stressed trend in intensive rice-wheat irrigation zones like Rachna Doab.",
    },
}

REGIONS = {
    "Gujranwala (Featured Case Study)": {
        "slug": "gujranwala",
        "profile": "Central-North Punjab | Dense Industrial Hub & Intensive Rice-Wheat Belt (Rachna Doab)",
        "coords": "31.85°N–32.35°N, 73.85°E–74.45°E",
    },
    "Lahore": {
        "slug": "lahore",
        "profile": "Central-East Punjab | Mega-City Urban Sprawl & Heavy Vehicular/Industrial Corridor",
        "coords": "31.35°N–31.70°N, 74.15°E–74.55°E",
    },
    "Multan": {
        "slug": "multan",
        "profile": "South Punjab | Semi-Arid Cotton/Mango Belt with High Heat & Dust Exposure",
        "coords": "29.95°N–30.40°N, 71.25°E–71.75°E",
    },
    "Rawalpindi / Murree": {
        "slug": "rawalpindi",
        "profile": "North Potohar Foothills | Rain-Fed (Barani) Terrain & Higher Elevation Forest Cover",
        "coords": "33.45°N–33.95°N, 72.85°E–73.55°E",
    },
}


# --- ROBUST NASA GIOVANNI CSV PARSER ---
@st.cache_data
def load_giovanni_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=["Date", "Value"])

    skip_rows = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            lower = line.strip().lower()
            if lower.startswith("time,") or lower.startswith("date,") or line[:4].isdigit():
                skip_rows = i if not line[:4].isdigit() else max(0, i - 1)
                break

    df = pd.read_csv(filepath, skiprows=skip_rows)
    df = df.iloc[:, :2].copy()
    df.columns = ["Date", "Value"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Date", "Value"])
    df = df[df["Value"] > -900]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# --- MANN-KENDALL & LINEAR TREND ENGINE ---
def analyze_trend(df: pd.DataFrame):
    n = len(df)
    if n < 12:
        return None

    y = df["Value"].values
    t_years = (df["Date"] - df["Date"].iloc[0]).dt.days.values / 365.25

    slope, intercept, r_val, p_ols, std_err = stats.linregress(t_years, y)
    trend_line = intercept + slope * t_years

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(y[k + 1 :] - y[k]))

    unique_y, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_mk = 2 * (1 - stats.norm.cdf(abs(z)))

    start_est = trend_line[0]
    end_est = trend_line[-1]
    pct_change = ((end_est - start_est) / abs(start_est)) * 100 if start_est != 0 else 0.0

    return {
        "slope_per_year": slope,
        "p_mk": p_mk,
        "p_ols": p_ols,
        "z_score": z,
        "pct_change": pct_change,
        "trend_line": trend_line,
        "significant": p_mk < 0.05,
        "direction": "Increasing" if slope > 0 else "Decreasing",
    }


# --- HEADER ---
st.title("🛰️ Orbit Analytics: Punjab Earth System Trend Detective")
st.markdown(
    "Detecting **20 years of opposing environmental shifts** across Punjab, Pakistan using "
    "**NASA MODIS** and **NASA GLDAS** satellite datasets."
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Detective Controls")

selected_var_label = st.sidebar.selectbox(
    "1. Select Earth Science Variable",
    list(VARIABLES.keys()),
)
var_info = VARIABLES[selected_var_label]

selected_region_label = st.sidebar.selectbox(
    "2. Select Primary District",
    list(REGIONS.keys()),
)
region_info = REGIONS[selected_region_label]

primary_path = os.path.join("data", f"{region_info['slug']}_{var_info['key']}.csv")
df_raw = load_giovanni_csv(primary_path)

if df_raw.empty:
    st.error(
        f"Could not find `{primary_path}`. Make sure your 8 NASA Giovanni CSV files are inside the `data/` folder."
    )
    st.stop()

min_year = int(df_raw["Date"].dt.year.min())
max_year = int(df_raw["Date"].dt.year.max())

year_range = st.sidebar.slider(
    "3. Select Time Period (Years)",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

show_rolling = st.sidebar.checkbox("Show 12-Month Rolling Mean (De-seasonalized)", value=True)
compare_all = st.sidebar.checkbox("Compare All 4 Punjab Districts", value=False)

# --- CONTEXT BANNER ---
with st.expander("ℹ️ What is this satellite measuring & why does it matter?", expanded=True):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**Sensor:** `{var_info['sensor']}`")
        st.markdown(f"**Measurement:** {var_info['description']}")
        st.markdown(f"**Regional Context ({selected_region_label}):** {region_info['profile']}")
    with c2:
        st.info(f"**Bounding Box:**\n`{region_info['coords']}`\n\n**Hypothesis:** {var_info['expected']}")

# Filter by selected time window
df_filtered = df_raw[
    (df_raw["Date"].dt.year >= year_range[0]) & (df_raw["Date"].dt.year <= year_range[1])
].copy()
df_filtered["Rolling_12M"] = df_filtered["Value"].rolling(window=12, min_periods=6).mean()

# --- STEP 5: STATISTICAL SIGNIFICANCE VERDICT ---
result = analyze_trend(df_filtered)

st.subheader(f"📊 Trend Verdict: {selected_region_label} ({year_range[0]}–{year_range[1]})")

if result:
    m1, m2, m3, m4 = st.columns(4)
    arrow = "📈" if result["direction"] == "Increasing" else "📉"
    m1.metric("Trend Direction", f"{arrow} {result['direction']}")
    m2.metric(
        "Annual Rate of Change",
        f"{result['slope_per_year']:+.4f} / yr",
        f"{result['pct_change']:+.1f}% total shift",
    )
    m3.metric(
        "Mann-Kendall p-value",
        f"{result['p_mk']:.4f}",
        "Significant (p < 0.05)" if result["significant"] else "Not Significant (p ≥ 0.05)",
        delta_color="normal" if result["significant"] else "off",
    )
    m4.metric("Observations Analyzed", f"{len(df_filtered)} months")

    if result["significant"]:
        st.success(
            f"**Scientific Detective Finding:** There is a **statistically significant {result['direction'].lower()} trend** "
            f"($p = {result['p_mk']:.4f} < 0.05$, Mann-Kendall $Z = {result['z_score']:.2f}$) in "
            f"**{selected_var_label}** over **{selected_region_label}** between {year_range[0]} and {year_range[1]}."
        )
    else:
        st.warning(
            f"**Scientific Detective Finding:** Over {year_range[0]}–{year_range[1]}, the observed slope "
            f"($p = {result['p_mk']:.4f} \\ge 0.05$) does **not** pass the 95% statistical significance threshold, "
            "indicating high seasonal/interannual variability rather than a confirmed monotonic shift."
        )

# --- STEP 4: TIME-SERIES VISUALIZATION ---
st.subheader("📈 Time-Series Analysis")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_filtered["Date"],
        y=df_filtered["Value"],
        mode="lines+markers",
        name="Monthly Satellite Observation",
        line=dict(color="rgba(100, 149, 237, 0.45)", width=1.5),
        marker=dict(size=4),
    )
)

if show_rolling:
    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["Rolling_12M"],
            mode="lines",
            name="12-Month Rolling Mean",
            line=dict(color="#1f77b4", width=3),
        )
    )

if result:
    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=result["trend_line"],
            mode="lines",
            name=f"Linear Trend ({result['slope_per_year']:+.4f}/yr)",
            line=dict(color="#d62728", width=2.5, dash="dash"),
        )
    )

fig.update_layout(
    title=f"{selected_var_label} — {selected_region_label}",
    xaxis_title="Year",
    yaxis_title=var_info["unit"],
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# --- MULTI-DISTRICT PUNJAB COMPARISON ---
if compare_all:
    st.subheader("🗺️ Cross-Regional Comparison Across Punjab (12-Month Rolling Mean)")
    comp_fig = go.Figure()
    summary_rows = []

    for r_label, r_meta in REGIONS.items():
        path = os.path.join("data", f"{r_meta['slug']}_{var_info['key']}.csv")
        d = load_giovanni_csv(path)
        d = d[(d["Date"].dt.year >= year_range[0]) & (d["Date"].dt.year <= year_range[1])].copy()
        if not d.empty:
            d["Rolling_12M"] = d["Value"].rolling(window=12, min_periods=6).mean()
            comp_fig.add_trace(
                go.Scatter(
                    x=d["Date"],
                    y=d["Rolling_12M"],
                    mode="lines",
                    name=r_label.replace(" (Featured Case Study)", ""),
                )
            )
            res = analyze_trend(d)
            if res:
                summary_rows.append(
                    {
                        "District": r_label.replace(" (Featured Case Study)", ""),
                        "Mean Value": round(d["Value"].mean(), 4),
                        "Trend Direction": res["direction"],
                        "Slope / Year": f"{res['slope_per_year']:+.4f}",
                        "Mann-Kendall p-value": round(res["p_mk"], 4),
                        "Significant (α=0.05)?": "Yes ✅" if res["significant"] else "No ⚠️",
                    }
                )

    comp_fig.update_layout(
        xaxis_title="Year",
        yaxis_title=var_info["unit"],
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    if summary_rows:
        st.markdown("**Regional Statistical Significance Table:**")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# --- FOOTER ---
st.divider()
st.caption(
    "Data Source: NASA Earthdata Giovanni (MODIS-Terra MOD08_M3 v6.1 & GLDAS Noah v2.1). "
    "Built for the 2026 NASA Space Apps Challenge — 'Be An Earth System Trend Detective!' "
    "(AI tools assisted with code structuring and documentation)."
)