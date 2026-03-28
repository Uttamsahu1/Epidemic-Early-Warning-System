import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from data  import load_data, get_country_list, get_latest_snapshot, get_country_data
from model import forecast, get_alert_level

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EpiWatch AI",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS (MindTrack dark style) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #080b12 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0 !important; }

.block-container { padding: 2rem 2.5rem 3rem 2.5rem !important; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, #0f1923 0%, #111827 50%, #0a0f1a 100%);
    border: 1px solid #1e2d45;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(239,68,68,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
.hero-title span { color: #10b981; }
.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.2px;
}

.metric-card {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
    margin-bottom: 1rem;
}
.metric-card:hover { border-color: #2d4a6b; transform: translateY(-2px); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #63b3ed); }
.metric-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.metric-card.red::before   { background: linear-gradient(90deg, #ef4444, #f87171); }
.metric-label {
    font-size: 0.75rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-sub { font-size: 0.8rem; color: #475569; margin-top: 0.3rem; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2535;
    margin-left: 0.5rem;
}

.alert-box {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    border: 1px solid;
    margin: 1rem 0;
}
.alert-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.alert-desc { color: #94a3b8; font-size: 0.95rem; }

.prob-row { margin: 0.4rem 0; }
.prob-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.2rem; display:flex; justify-content:space-between; }
.prob-bar-bg { background: #1e2535; border-radius: 999px; height: 8px; }
.prob-bar-fill { height: 8px; border-radius: 999px; }

hr { border-color: #1e2535 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-bottom: 1px solid #1e2535 !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.4rem !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #10b981 !important;
    border-bottom: 2px solid #10b981 !important;
    background: transparent !important;
}

[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#1e2535",
    "axes.labelcolor":   "#64748b",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#94a3b8",
    "grid.color":        "#1e2535",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

RISK_COLORS = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load():
    return load_data()

with st.spinner("⏳ Loading global epidemic data..."):
    df      = cached_load()
    snap    = get_latest_snapshot(df)
    countries = get_country_list(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#f1f5f9'>
            🦠 EpiWatch
        </div>
        <div style='font-size:0.78rem;color:#334155;margin-top:0.2rem;letter-spacing:0.5px'>
            AI EPIDEMIC EARLY WARNING SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["Overview", "Country Analysis", "Forecast", "Risk Alerts"], label_visibility="collapsed")
    st.divider()

    total_countries = snap["Country/Region"].nunique()
    high_risk = (snap["Risk_Level"] == "HIGH").sum()
    total_cases = int(snap["Confirmed"].sum())

    st.markdown(f"""
    <div style='padding:1rem;background:#0d1117;border:1px solid #1e2535;border-radius:10px;margin-top:0.5rem'>
        <div style='font-size:0.7rem;color:#334155;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem'>LIVE STATS</div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'>
            <span style='color:#475569;font-size:0.82rem'>Countries Tracked</span>
            <span style='color:#e2e8f0;font-weight:600;font-size:0.82rem'>{total_countries}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'>
            <span style='color:#475569;font-size:0.82rem'>High Risk Nations</span>
            <span style='color:#ef4444;font-weight:600;font-size:0.82rem'>{high_risk}</span>
        </div>
        <div style='display:flex;justify-content:space-between'>
            <span style='color:#475569;font-size:0.82rem'>Total Cases</span>
            <span style='color:#e2e8f0;font-weight:600;font-size:0.82rem'>{total_cases/1e6:.1f}M</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>AI Epidemic<br><span>Early Warning System</span></div>
        <div class='hero-subtitle'>Real-time outbreak detection across 195+ countries · Powered by Johns Hopkins CSSE data</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    low_n    = (snap["Risk_Level"] == "LOW").sum()
    mid_n    = (snap["Risk_Level"] == "MEDIUM").sum()
    high_n   = (snap["Risk_Level"] == "HIGH").sum()
    total_deaths = int(snap["Deaths"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card blue'>
            <div class='metric-label'>Total Confirmed Cases</div>
            <div class='metric-value'>{total_cases/1e6:.1f}M</div>
            <div class='metric-sub'>Global cumulative</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card green'>
            <div class='metric-label'>Low Risk Countries</div>
            <div class='metric-value'>{low_n}</div>
            <div class='metric-sub'>Stable transmission</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card amber'>
            <div class='metric-label'>Medium Risk Countries</div>
            <div class='metric-value'>{mid_n}</div>
            <div class='metric-sub'>Enhanced surveillance</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card red'>
            <div class='metric-label'>High Risk Countries</div>
            <div class='metric-value'>{high_n}</div>
            <div class='metric-sub'>Critical alert</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("<div class='section-header'>Risk Distribution by Country</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        labels  = ["Low Risk", "Medium Risk", "High Risk"]
        counts  = [low_n, mid_n, high_n]
        colors  = ["#10b981", "#f59e0b", "#ef4444"]
        bars = ax.bar(labels, counts, color=colors, width=0.5, zorder=3)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val}", ha="center", fontsize=11, color="#94a3b8", fontweight="600")
        ax.set_ylabel("Number of Countries", fontsize=10)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("<div class='section-header'>Risk Breakdown</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.8), subplot_kw=dict(aspect="equal"))
        wedge_colors = ["#059669", "#d97706", "#dc2626"]
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels,
            colors=wedge_colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="#0d1117", linewidth=2)
        )
        for t in texts:     t.set_color("#64748b"); t.set_fontsize(9)
        for t in autotexts: t.set_color("#f1f5f9"); t.set_fontsize(9); t.set_fontweight("bold")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-header'>Top 10 Countries by Total Cases</div>", unsafe_allow_html=True)
    top10 = snap.nlargest(10, "Confirmed")[["Country/Region","Confirmed","Daily_Cases","Growth_Rate","Risk_Level"]]
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(7, 4))
        top10_sorted = top10.sort_values("Confirmed")
        risk_bar_colors = [RISK_COLORS[r] for r in top10_sorted["Risk_Level"]]
        ax.barh(top10_sorted["Country/Region"], top10_sorted["Confirmed"]/1e6,
                color=risk_bar_colors, height=0.6, zorder=3)
        ax.set_xlabel("Confirmed Cases (Millions)")
        ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(["#10b981","#f59e0b","#ef4444"],["Low","Medium","High"])]
        ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.1)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(7, 4))
        top10_gr = top10.sort_values("Growth_Rate")
        gr_colors = [RISK_COLORS[r] for r in top10_gr["Risk_Level"]]
        ax.barh(top10_gr["Country/Region"], top10_gr["Growth_Rate"],
                color=gr_colors, height=0.6, zorder=3)
        ax.set_xlabel("7-Day Growth Rate (%)")
        ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COUNTRY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Country Analysis":

    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.3rem'>
        Country Analysis
    </div>
    <div style='color:#475569;font-size:0.95rem;margin-bottom:2rem'>
        Deep-dive into individual country epidemic trajectories
    </div>
    """, unsafe_allow_html=True)

    default_idx = countries.index("US") if "US" in countries else 0
    country = st.selectbox("Select Country", countries, index=default_idx)
    cdf = get_country_data(df, country)
    latest = cdf.iloc[-1]

    alert = get_alert_level(latest["Growth_Rate"], latest.get("Doubling_Time", np.nan))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card blue'>
            <div class='metric-label'>Total Confirmed</div>
            <div class='metric-value'>{int(latest['Confirmed'])/1e6:.2f}M</div>
            <div class='metric-sub'>Cumulative cases</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card green'>
            <div class='metric-label'>7-Day Avg Cases</div>
            <div class='metric-value'>{int(latest['Rolling_7day']):,}</div>
            <div class='metric-sub'>Daily average</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card amber'>
            <div class='metric-label'>Growth Rate</div>
            <div class='metric-value'>{latest['Growth_Rate']:.1f}%</div>
            <div class='metric-sub'>Last 7 days</div></div>""", unsafe_allow_html=True)
    with c4:
        dt = latest.get("Doubling_Time", np.nan)
        dt_str = f"{dt:.0f}d" if pd.notna(dt) and dt > 0 else "Stable"
        st.markdown(f"""<div class='metric-card red'>
            <div class='metric-label'>Doubling Time</div>
            <div class='metric-value'>{dt_str}</div>
            <div class='metric-sub'>At current rate</div></div>""", unsafe_allow_html=True)

    # Alert box
    st.markdown(f"""
    <div class='alert-box' style='background:{alert["bg"]};border-color:{alert["border"]}'>
        <div class='alert-label' style='color:{alert["color"]}'>{alert["emoji"]} {alert["level"]} RISK</div>
        <div class='alert-desc'>{alert["msg"]}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  Case Trend  ", "  Daily Cases  ", "  Deaths  "])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(cdf["Date"], cdf["Confirmed"]/1e6, color="#3b82f6", lw=2)
        ax.fill_between(cdf["Date"], cdf["Confirmed"]/1e6, alpha=0.15, color="#3b82f6")
        ax.set_ylabel("Confirmed Cases (Millions)"); ax.set_xlabel("")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(cdf["Date"], cdf["Daily_Cases"], color="#1e3a5f", width=1, zorder=2, label="Daily Cases")
        ax.plot(cdf["Date"], cdf["Rolling_7day"], color="#f59e0b", lw=2, label="7-Day Avg")
        ax.set_ylabel("Daily New Cases"); ax.set_xlabel("")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.legend(framealpha=0.1, fontsize=9)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(cdf["Date"], cdf["Deaths"]/1e3, color="#ef4444", lw=2)
        ax.fill_between(cdf["Date"], cdf["Deaths"]/1e3, alpha=0.15, color="#ef4444")
        ax.set_ylabel("Deaths (Thousands)"); ax.set_xlabel("")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":

    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.3rem'>
        Outbreak Forecast
    </div>
    <div style='color:#475569;font-size:0.95rem;margin-bottom:2rem'>
        ML-powered 30-day case projection using Ridge Regression with lag features
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Settings</div>", unsafe_allow_html=True)
        default_idx = countries.index("US") if "US" in countries else 0
        sel_country  = st.selectbox("Country", countries, index=default_idx)
        forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30)
        run_btn = st.button("🔮  Generate Forecast", use_container_width=True)

        cdf = get_country_data(df, sel_country)
        latest = cdf.iloc[-1]
        alert  = get_alert_level(latest["Growth_Rate"], latest.get("Doubling_Time", np.nan))

        st.markdown(f"""
        <div style='margin-top:1.5rem'>
            <div class='metric-card {"red" if alert["level"]=="HIGH" else "amber" if alert["level"]=="MEDIUM" else "green"}'>
                <div class='metric-label'>Current Risk</div>
                <div class='metric-value' style='font-size:1.4rem'>{alert["emoji"]} {alert["level"]}</div>
                <div class='metric-sub'>{latest["Growth_Rate"]:.1f}% growth rate</div>
            </div>
        </div>""", unsafe_allow_html=True)

        for label, val, color, max_val in [
            ("Growth Rate", min(abs(latest["Growth_Rate"]), 100), "#ef4444" if latest["Growth_Rate"]>20 else "#f59e0b" if latest["Growth_Rate"]>5 else "#10b981", 100),
            ("7-Day Avg Cases", min(latest["Rolling_7day"], 100000), "#3b82f6", 100000),
        ]:
            pct = int(val / max_val * 100)
            display = f"{latest['Growth_Rate']:.1f}%" if "Growth" in label else f"{int(latest['Rolling_7day']):,}"
            st.markdown(f"""
            <div style='margin-bottom:1rem;margin-top:0.5rem'>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                    <span style='color:#94a3b8;font-size:0.85rem'>{label}</span>
                    <span style='color:{color};font-size:0.85rem;font-weight:600'>{display}</span>
                </div>
                <div style='background:#1e2535;border-radius:99px;height:6px'>
                    <div style='background:{color};width:{pct}%;height:6px;border-radius:99px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-header'>30-Day Case Forecast</div>", unsafe_allow_html=True)

        if run_btn or True:
            with st.spinner("Running ML forecast..."):
                fc = forecast(cdf, days=forecast_days)

            if fc is not None:
                recent = cdf[cdf["Date"] >= cdf["Date"].max() - pd.Timedelta(days=60)]

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(recent["Date"], recent["Daily_Cases"],
                        color="#475569", lw=1.5, label="Historical Daily Cases", alpha=0.7)
                ax.plot(recent["Date"], recent["Rolling_7day"],
                        color="#3b82f6", lw=2.5, label="7-Day Rolling Avg")
                ax.plot(fc["Date"], fc["Forecast"],
                        color="#10b981", lw=2.5, linestyle="--", label=f"{forecast_days}-Day Forecast")
                ax.fill_between(fc["Date"],
                                fc["Forecast"] * 0.7,
                                fc["Forecast"] * 1.3,
                                alpha=0.12, color="#10b981", label="Confidence Band (±30%)")
                ax.axvline(x=cdf["Date"].max(), color="#334155", linestyle=":", lw=1.5)
                ax.text(cdf["Date"].max(), ax.get_ylim()[1]*0.95,
                        " Forecast start", color="#64748b", fontsize=8)
                ax.set_ylabel("Daily Cases"); ax.set_xlabel("")
                ax.legend(framealpha=0.1, fontsize=9, loc="upper left")
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

                peak_forecast = fc["Forecast"].max()
                peak_date     = fc.loc[fc["Forecast"].idxmax(), "Date"].strftime("%b %d, %Y")
                total_forecast = int(fc["Forecast"].sum())

                f1, f2, f3 = st.columns(3)
                with f1:
                    st.markdown(f"""<div class='metric-card blue'>
                        <div class='metric-label'>Peak Forecast</div>
                        <div class='metric-value'>{int(peak_forecast):,}</div>
                        <div class='metric-sub'>cases/day</div></div>""", unsafe_allow_html=True)
                with f2:
                    st.markdown(f"""<div class='metric-card amber'>
                        <div class='metric-label'>Peak Date</div>
                        <div class='metric-value' style='font-size:1.2rem'>{peak_date}</div>
                        <div class='metric-sub'>projected peak</div></div>""", unsafe_allow_html=True)
                with f3:
                    st.markdown(f"""<div class='metric-card green'>
                        <div class='metric-label'>Total Projected</div>
                        <div class='metric-value'>{total_forecast/1e3:.1f}K</div>
                        <div class='metric-sub'>over {forecast_days} days</div></div>""", unsafe_allow_html=True)
            else:
                st.warning("Not enough data to forecast for this country.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK ALERTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Alerts":

    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.3rem'>
        Risk Alerts
    </div>
    <div style='color:#475569;font-size:0.95rem;margin-bottom:2rem'>
        Real-time early warning signals across all monitored regions
    </div>
    """, unsafe_allow_html=True)

    tab_high, tab_med, tab_all = st.tabs(["  🔴 High Risk  ", "  🟡 Medium Risk  ", "  📊 All Countries  "])

    def risk_table(risk_df, color):
        cols = ["Country/Region","Confirmed","Daily_Cases","Rolling_7day","Growth_Rate","Doubling_Time","Deaths"]
        display = risk_df[cols].copy()
        display["Confirmed"]   = display["Confirmed"].apply(lambda x: f"{x/1e6:.2f}M")
        display["Daily_Cases"] = display["Daily_Cases"].apply(lambda x: f"{int(x):,}")
        display["Rolling_7day"]= display["Rolling_7day"].apply(lambda x: f"{int(x):,}")
        display["Growth_Rate"] = display["Growth_Rate"].apply(lambda x: f"{x:.1f}%")
        display["Doubling_Time"]= display["Doubling_Time"].apply(lambda x: f"{x:.0f}d" if pd.notna(x) and x>0 else "Stable")
        display["Deaths"]      = display["Deaths"].apply(lambda x: f"{x/1e3:.1f}K")
        display.columns = ["Country","Total Cases","Daily Cases","7-Day Avg","Growth Rate","Doubling Time","Deaths"]
        st.dataframe(display.reset_index(drop=True), use_container_width=True, height=400)

    with tab_high:
        high_df = snap[snap["Risk_Level"] == "HIGH"].sort_values("Growth_Rate", ascending=False)
        st.markdown(f"<div style='color:#ef4444;font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;margin-bottom:1rem'>⚠️ {len(high_df)} countries at HIGH risk</div>", unsafe_allow_html=True)
        if len(high_df):
            risk_table(high_df, "#ef4444")
            fig, ax = plt.subplots(figsize=(10, 4))
            top_high = high_df.head(15).sort_values("Growth_Rate")
            ax.barh(top_high["Country/Region"], top_high["Growth_Rate"], color="#ef4444", height=0.6, zorder=3)
            ax.set_xlabel("Growth Rate (%)"); ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        else:
            st.success("No countries currently at HIGH risk.")

    with tab_med:
        med_df = snap[snap["Risk_Level"] == "MEDIUM"].sort_values("Growth_Rate", ascending=False)
        st.markdown(f"<div style='color:#f59e0b;font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;margin-bottom:1rem'>⚡ {len(med_df)} countries at MEDIUM risk</div>", unsafe_allow_html=True)
        if len(med_df):
            risk_table(med_df, "#f59e0b")
        else:
            st.success("No countries currently at MEDIUM risk.")

    with tab_all:
        st.markdown("<div class='section-header'>All Countries — Risk Snapshot</div>", unsafe_allow_html=True)
        search = st.text_input("🔍 Search country", placeholder="Type country name...")
        filtered = snap[snap["Country/Region"].str.contains(search, case=False)] if search else snap
        filtered = filtered.sort_values("Growth_Rate", ascending=False)
        risk_table(filtered, "#94a3b8")
