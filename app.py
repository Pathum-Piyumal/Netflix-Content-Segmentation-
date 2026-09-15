import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
from html import escape

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import load_data, clean_data, create_features, prepare_feature_matrix
from src.clustering import find_optimal_k, fit_kmeans, fit_hierarchical, fit_dbscan, apply_pca, profile_clusters

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Content Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES — Dark Netflix Glassmorphism Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

/* ══════════════════════════════════════════════════════════════
   BASE & RESET
══════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080810;
    color: #EFEFEF;
    scroll-behavior: smooth;
}

/* Animated top progress bar */
@keyframes progressSlide {
    0%   { width: 0%; opacity: 1; }
    85%  { width: 100%; opacity: 1; }
    100% { width: 100%; opacity: 0; }
}
.top-progress-bar {
    position: fixed;
    top: 0; left: 0;
    height: 3px;
    background: linear-gradient(90deg, #E50914, #FF6B35, #FFD700, #E50914);
    background-size: 200% 100%;
    animation: progressSlide 1.8s ease-out forwards, shimmerMove 1s linear infinite;
    z-index: 9999;
    border-radius: 0 2px 2px 0;
    box-shadow: 0 0 12px rgba(229,9,20,0.8), 0 0 24px rgba(229,9,20,0.4);
}
@keyframes shimmerMove {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1440px; }

/* ══════════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0e0e18 0%, #150808 55%, #0a0a12 100%);
    border-right: 1px solid rgba(229,9,20,0.12);
    position: relative;
    overflow: hidden;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(229,9,20,0.08) 0%, transparent 70%);
    pointer-events: none;
}
[data-testid="stSidebar"] .block-container { padding: 1rem; }

/* Sidebar logo */
.sidebar-logo-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.sidebar-n-badge {
    width: 38px; height: 38px;
    background: #E50914;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; font-weight: 900; color: #fff;
    font-family: 'Inter', sans-serif;
    letter-spacing: -2px;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(229,9,20,0.5);
    animation: nPulse 3s ease-in-out infinite;
}
@keyframes nPulse {
    0%, 100% { box-shadow: 0 0 16px rgba(229,9,20,0.5); }
    50%       { box-shadow: 0 0 28px rgba(229,9,20,0.85), 0 0 48px rgba(229,9,20,0.3); }
}
.sidebar-logo-text {
    font-size: 1.3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #fff 30%, #E50914);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.sidebar-sub {
    font-size: 0.62rem;
    color: #555;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    padding-left: 48px;
}
/* Live status dot */
.status-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.62rem;
    color: #4ade80;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 1.2rem;
    padding-left: 48px;
}
.status-dot {
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    animation: livePulse 1.5s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(74,222,128,0.6);
}
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ══════════════════════════════════════════════════════════════
   HERO BANNER
══════════════════════════════════════════════════════════════ */
@keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes scanLine {
    0%   { transform: translateY(-100%); opacity: 0.04; }
    100% { transform: translateY(1000%); opacity: 0.04; }
}
@keyframes floatOrb {
    0%, 100% { transform: translateY(0px) scale(1); }
    50%       { transform: translateY(-12px) scale(1.06); }
}
.hero-banner {
    background: linear-gradient(135deg, #1c0202 0%, #0f0f1c 45%, #080810 100%);
    border: 1px solid rgba(229,9,20,0.22);
    border-radius: 20px;
    padding: 2.4rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: heroFadeIn 0.7s ease-out both;
}
/* Animated scan-line */
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(229,9,20,0.6), transparent);
    animation: scanLine 4s linear infinite;
    pointer-events: none;
}
/* Glowing orb top-right */
.hero-orb-tr {
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(229,9,20,0.18) 0%, transparent 70%);
    border-radius: 50%;
    animation: floatOrb 6s ease-in-out infinite;
    pointer-events: none;
}
/* Glowing orb bottom-left */
.hero-orb-bl {
    position: absolute;
    bottom: -50px; left: 20px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(255,107,53,0.1) 0%, transparent 70%);
    border-radius: 50%;
    animation: floatOrb 8s ease-in-out infinite reverse;
    pointer-events: none;
}
/* grid overlay */
.hero-banner-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(229,9,20,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(229,9,20,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
}
.hero-content { position: relative; z-index: 2; }
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #ffcccc 30%, #E50914 65%, #FF6B35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.08;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: #777;
    letter-spacing: 0.4px;
    max-width: 520px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(229,9,20,0.12);
    border: 1px solid rgba(229,9,20,0.35);
    color: #E50914;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
    animation: badgePulse 2s ease-in-out infinite;
}
@keyframes badgePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(229,9,20,0); }
    50%       { box-shadow: 0 0 0 4px rgba(229,9,20,0.12); }
}
.hero-divider {
    width: 60px; height: 3px;
    background: linear-gradient(90deg, #E50914, #FF6B35);
    border-radius: 2px;
    margin: 1rem 0;
}

/* ══════════════════════════════════════════════════════════════
   KPI CARDS
══════════════════════════════════════════════════════════════ */
@keyframes cardReveal {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes shimmerSweep {
    0%   { left: -100%; }
    100% { left: 200%; }
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
}
.kpi-card {
    background: linear-gradient(145deg, #13131c, #1a1a28);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.28s cubic-bezier(.34,1.56,.64,1),
                border-color 0.28s ease,
                box-shadow 0.28s ease;
    cursor: default;
    animation: cardReveal 0.6s ease both;
}
.kpi-card:hover {
    transform: translateY(-6px) scale(1.01);
    border-color: rgba(229,9,20,0.4);
    box-shadow: 0 20px 48px rgba(229,9,20,0.12), 0 4px 16px rgba(0,0,0,0.4);
}
/* Shimmer sweep on hover */
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
    transition: none;
    pointer-events: none;
}
.kpi-card:hover::after {
    animation: shimmerSweep 0.7s ease;
}
/* Top accent bar */
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #E50914, #FF6B35, #FFD700);
    border-radius: 16px 16px 0 0;
    opacity: 0.85;
}
.kpi-icon {
    font-size: 1.8rem;
    margin-bottom: 0.6rem;
    display: block;
    filter: drop-shadow(0 0 8px rgba(229,9,20,0.4));
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 900;
    color: #fff;
    line-height: 1;
    margin-bottom: 0.3rem;
    font-variant-numeric: tabular-nums;
    letter-spacing: -1px;
}
.kpi-label {
    font-size: 0.68rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 600;
    margin-top: 0.2rem;
}
.kpi-delta {
    font-size: 0.72rem;
    color: #4ade80;
    font-weight: 600;
    margin-top: 0.55rem;
    display: flex;
    align-items: center;
    gap: 3px;
}
.kpi-sparkline {
    position: absolute;
    bottom: 0; right: 0;
    width: 80px; height: 40px;
    opacity: 0.15;
}

/* ══════════════════════════════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════════════════════════════ */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.1rem 0;
}
.section-header-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, rgba(229,9,20,0.2), rgba(255,107,53,0.08));
    border: 1px solid rgba(229,9,20,0.28);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    box-shadow: 0 0 12px rgba(229,9,20,0.1);
}
.section-header-text {
    font-size: 1.18rem;
    font-weight: 800;
    color: #F2F2F2;
    letter-spacing: -0.3px;
}

/* ══════════════════════════════════════════════════════════════
   GLASS CARD
══════════════════════════════════════════════════════════════ */
.glass-card {
    background: rgba(18, 18, 26, 0.85);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    border-color: rgba(229,9,20,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* ══════════════════════════════════════════════════════════════
   SCORE BADGE
══════════════════════════════════════════════════════════════ */
.score-badge {
    background: linear-gradient(135deg, rgba(74,222,128,0.12), rgba(34,197,94,0.04));
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    display: inline-block;
    margin-bottom: 1rem;
}
.score-badge-label { font-size: 0.62rem; color: #4ade80; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
.score-badge-value { font-size: 2rem; font-weight: 900; color: #4ade80; }

/* ══════════════════════════════════════════════════════════════
   CLUSTER PILL
══════════════════════════════════════════════════════════════ */
.cluster-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(229,9,20,0.09);
    border: 1px solid rgba(229,9,20,0.18);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.73rem; font-weight: 700; color: #E50914;
    margin: 2px;
    transition: background 0.2s ease;
}
.cluster-pill:hover { background: rgba(229,9,20,0.18); }

/* ══════════════════════════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
══════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSlider"] > div > div > div { background: #E50914 !important; }
[data-testid="stSlider"] > div > div > div > div {
    box-shadow: 0 0 10px rgba(229,9,20,0.5) !important;
}
[data-testid="stRadio"] label {
    font-size: 0.85rem !important;
    padding: 4px 0 !important;
    transition: color 0.2s ease;
}
[data-testid="stSelectbox"] > div {
    background: #13131c !important;
    border-color: rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stSpinner"] > div {
    border-top-color: #E50914 !important;
}

/* ══════════════════════════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: rgba(18,18,26,0.9) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    padding: 0.8rem 1rem !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(229,9,20,0.3) !important;
    box-shadow: 0 4px 16px rgba(229,9,20,0.08) !important;
}
.streamlit-expanderContent {
    background: rgba(14,14,20,0.7) !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ══════════════════════════════════════════════════════════════
   ALERTS & TAGS
══════════════════════════════════════════════════════════════ */
[data-testid="stSuccess"] {
    border-radius: 12px;
    background: rgba(74,222,128,0.06);
    border-color: rgba(74,222,128,0.22);
}
.tag {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 7px;
    padding: 3px 9px;
    font-size: 0.7rem;
    color: #bbb;
    margin: 2px;
    transition: background 0.2s ease, color 0.2s ease;
}
.tag:hover { background: rgba(229,9,20,0.1); color: #E50914; border-color: rgba(229,9,20,0.25); }

/* ══════════════════════════════════════════════════════════════
   DIVIDERS
══════════════════════════════════════════════════════════════ */
.red-divider {
    height: 1px;
    background: linear-gradient(90deg, #E50914 0%, rgba(255,107,53,0.4) 40%, transparent 100%);
    margin: 1.5rem 0;
    border: none;
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ══════════════════════════════════════════════════════════════
   METRIC CALLOUT
══════════════════════════════════════════════════════════════ */
.metric-callout {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(145deg, #13131c, #1a1224);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-callout:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}
.metric-callout-val {
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -2px;
}
.metric-callout-label {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #555;
    margin-top: 6px;
    font-weight: 600;
}
.metric-callout-sub {
    font-size: 0.72rem;
    color: #444;
    margin-top: 4px;
}

/* ══════════════════════════════════════════════════════════════
   PAGE FADE-IN
══════════════════════════════════════════════════════════════ */
@keyframes pageFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.page-wrapper {
    animation: pageFadeIn 0.5s ease-out both;
}

/* ══════════════════════════════════════════════════════════════
   CHART CONTAINERS
══════════════════════════════════════════════════════════════ */
[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    overflow: hidden;
    background: rgba(14,14,20,0.5);
    border: 1px solid rgba(255,255,255,0.04);
    transition: box-shadow 0.25s ease;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* ══════════════════════════════════════════════════════════════
   GLOW TEXT UTILITY
══════════════════════════════════════════════════════════════ */
.glow-red   { color: #E50914; text-shadow: 0 0 20px rgba(229,9,20,0.5); }
.glow-green { color: #4ade80; text-shadow: 0 0 20px rgba(74,222,128,0.5); }
.glow-blue  { color: #60a5fa; text-shadow: 0 0 20px rgba(96,165,250,0.5); }

/* ══════════════════════════════════════════════════════════════
   FOOTER
══════════════════════════════════════════════════════════════ */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.62rem;
    color: #333;
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
# Base layout — no 'title' key here to avoid duplicate-kwarg conflicts
_BASE_LAYOUT = dict(
    font=dict(family="Inter", color="#CCCCCC", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(0,0,0,0)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#E50914", "#FF6B35", "#F7C59F", "#4ade80", "#60a5fa", "#a78bfa", "#f472b6", "#facc15", "#fb923c", "#34d399"]
)

# Keep for backward-compat references (e.g. fig.update_layout(**PLOTLY_TEMPLATE["layout"]))
PLOTLY_TEMPLATE = dict(layout=_BASE_LAYOUT)

_TITLE_STYLE = dict(font=dict(size=15, color="#EFEFEF"), x=0.01)

def theme(title: str = "", **extra):
    """Return a merged update_layout kwargs dict that safely includes a styled title."""
    layout = dict(_BASE_LAYOUT)  # shallow copy — avoids mutation
    if title:
        layout["title"] = dict(text=title, **_TITLE_STYLE)
    layout.update(extra)
    return layout

NETFLIX_COLORS = ["#E50914", "#FF6B35", "#FF9F1C", "#FFBF69", "#CBF3F0", "#2EC4B6", "#4361EE", "#7209B7", "#F72585", "#3A0CA3"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_processed_data():
    df_raw = load_data(BASE_DIR / "data" / "netflix_titles.csv")
    df_clean = clean_data(df_raw)
    df_feat = create_features(df_clean)
    X, X_scaled, scaler = prepare_feature_matrix(df_feat)
    return df_raw, df_feat, X, X_scaled


@st.cache_data(show_spinner=False, max_entries=24)
def cached_optimal_k(matrix, max_k):
    return find_optimal_k(matrix, max_k=max_k)

@st.cache_resource(show_spinner=False, max_entries=16)
def cached_cluster(matrix, algorithm, k=5, eps=2.0, min_samples=5):
    if algorithm == "K-Means":
        return fit_kmeans(matrix, n_clusters=k)
    if algorithm == "Agglomerative Hierarchical":
        return fit_hierarchical(matrix, n_clusters=k)
    return fit_dbscan(matrix, eps=eps, min_samples=min_samples)

@st.cache_data(show_spinner=False, max_entries=3)
def cached_pca(matrix, dimensions):
    return apply_pca(matrix, n_components=dimensions)

# Top progress bar injection
st.markdown('<div class="top-progress-bar"></div>', unsafe_allow_html=True)

with st.spinner("🎬 Loading Netflix dataset..."):
    df_raw, df_feat, X, X_scaled = get_processed_data()

# ── JavaScript: Animated number counters ──────────────────────────────────────
st.markdown("""
<script>
(function() {
  function animateCounter(el, target, duration, prefix, suffix) {
    var start = 0;
    var startTime = null;
    var isFloat = target % 1 !== 0;
    function step(timestamp) {
      if (!startTime) startTime = timestamp;
      var progress = Math.min((timestamp - startTime) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      var current = eased * target;
      el.textContent = prefix + (isFloat ? current.toFixed(3) : Math.floor(current).toLocaleString()) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  function initCounters() {
    document.querySelectorAll('[data-counter]').forEach(function(el) {
      var target = parseFloat(el.getAttribute('data-counter'));
      var prefix = el.getAttribute('data-prefix') || '';
      var suffix = el.getAttribute('data-suffix') || '';
      animateCounter(el, target, 1400, prefix, suffix);
    });
  }
  // Wait for Streamlit to inject the elements
  var observer = new MutationObserver(function(mutations, obs) {
    if (document.querySelectorAll('[data-counter]').length > 0) {
      initCounters();
      obs.disconnect();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
  setTimeout(initCounters, 800);
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div class="sidebar-logo-wrap">
        <div class="sidebar-n-badge">N</div>
        <div class="sidebar-logo-text">NETFLIX</div>
    </div>
    <div class="sidebar-sub">Content Intelligence</div>
    <div class="status-live"><div class="status-dot"></div> Live Analysis</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📊  Exploratory Analysis",
            "📉  Cluster Selection",
            "🤖  Interactive Clustering",
            "🏷️  Cluster Profiles",
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

    # Dataset stats panel
    n_movies_s = len(df_raw[df_raw["type"] == "Movie"])
    n_shows_s  = len(df_raw[df_raw["type"] == "TV Show"])
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:0.9rem 1rem; font-size:0.72rem; line-height:2;">
        <div style="color:#444; font-size:0.58rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px; font-weight:700;">Dataset Stats</div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#555;">Total Titles</span>
            <b style="color:#E50914;">{len(df_raw):,}</b>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#555;">Movies</span>
            <b style="color:#FF6B35;">{n_movies_s:,}</b>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#555;">TV Shows</span>
            <b style="color:#facc15;">{n_shows_s:,}</b>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#555;">Features</span>
            <b style="color:#60a5fa;">{X.shape[1]}</b>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#555;">Algorithms</span>
            <b style="color:#a78bfa;">3</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:6px;">
        <div style="display:flex; align-items:center; gap:8px; background:rgba(229,9,20,0.06); border:1px solid rgba(229,9,20,0.12); border-radius:8px; padding:6px 10px; font-size:0.68rem;">
            <span>🧠</span><span style="color:#888;">K-Means</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; background:rgba(255,107,53,0.06); border:1px solid rgba(255,107,53,0.12); border-radius:8px; padding:6px 10px; font-size:0.68rem;">
            <span>🌳</span><span style="color:#888;">Hierarchical</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; background:rgba(96,165,250,0.06); border:1px solid rgba(96,165,250,0.12); border-radius:8px; padding:6px 10px; font-size:0.68rem;">
            <span>🔭</span><span style="color:#888;">DBSCAN</span>
        </div>
    </div>
    <br>
    <div style="font-size:0.58rem; color:#2a2a3a; text-align:center; letter-spacing:1.5px;">
        UNSUPERVISED ML · 2024
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if "Overview" in page:
    # Hero
    n_movies = len(df_raw[df_raw["type"] == "Movie"])
    n_shows = len(df_raw[df_raw["type"] == "TV Show"])
    n_countries = df_raw["country"].dropna().str.split(",").explode().str.strip().nunique()
    n_genres = df_feat["main_genre"].nunique()

    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-banner-grid"></div>
        <div class="hero-orb-tr"></div>
        <div class="hero-orb-bl"></div>
        <div class="hero-content">
            <div class="hero-badge">⚡ Netflix · Unsupervised ML · 2024</div>
            <div class="hero-title">Content Segmentation<br>Intelligence Platform</div>
            <div class="hero-divider"></div>
            <div class="hero-subtitle">
                Unsupervised Machine Learning applied to <strong style="color:#fff;">{len(df_raw):,}</strong> Netflix titles
                across <strong style="color:#fff;">{n_countries}</strong> countries — uncovering hidden content clusters and behavioral patterns.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards with animated counters
    movie_pct = n_movies / len(df_raw) * 100
    show_pct  = n_shows  / len(df_raw) * 100
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">🎬</div>
            <div class="kpi-value" data-counter="{len(df_raw)}">{len(df_raw):,}</div>
            <div class="kpi-label">Total Titles</div>
            <div class="kpi-delta">▲ Full Catalog</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🎥</div>
            <div class="kpi-value" data-counter="{n_movies}">{n_movies:,}</div>
            <div class="kpi-label">Movies</div>
            <div class="kpi-delta">▲ {movie_pct:.1f}% of catalog</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📺</div>
            <div class="kpi-value" data-counter="{n_shows}">{n_shows:,}</div>
            <div class="kpi-label">TV Shows</div>
            <div class="kpi-delta">▲ {show_pct:.1f}% of catalog</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🌍</div>
            <div class="kpi-value" data-counter="{n_countries}">{n_countries:,}</div>
            <div class="kpi-label">Countries</div>
            <div class="kpi-delta">▲ {n_genres} genres</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Raw Data Preview
    st.markdown('<div class="section-header"><div class="section-header-icon">📋</div><div class="section-header-text">Raw Dataset Preview</div></div>', unsafe_allow_html=True)
    st.dataframe(
        df_raw.head(80).style.set_properties(**{
            "background-color": "#16161E",
            "color": "#DEDEDE",
            "border-color": "rgba(255,255,255,0.04)"
        }),
        use_container_width=True,
        height=320
    )

    # Missing Values + Dtypes
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header"><div class="section-header-icon">❓</div><div class="section-header-text">Missing Values</div></div>', unsafe_allow_html=True)
        miss = df_raw.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing Count"]
        miss["Missing %"] = (miss["Missing Count"] / len(df_raw) * 100).round(2)
        miss = miss[miss["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

        fig_miss = px.bar(
            miss, x="Missing Count", y="Column", orientation="h",
            color="Missing %", color_continuous_scale=["#E50914", "#FF6B35"],
            text="Missing Count",
            title="Missing Values per Column",
        )
        fig_miss.update_traces(textposition="outside", textfont_size=11)
        fig_miss.update_layout(**theme("Missing Values per Column", height=280, coloraxis_showscale=False))
        st.plotly_chart(fig_miss, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header"><div class="section-header-icon">🗂</div><div class="section-header-text">Data Types</div></div>', unsafe_allow_html=True)
        dtypes_df = pd.DataFrame(df_raw.dtypes.astype(str), columns=["Data Type"]).reset_index()
        dtypes_df.columns = ["Column", "Data Type"]
        dtype_counts = dtypes_df["Data Type"].value_counts().reset_index()
        dtype_counts.columns = ["Type", "Count"]
        fig_dtype = px.pie(
            dtype_counts, names="Type", values="Count",
            color_discrete_sequence=NETFLIX_COLORS,
            title="Column Data Types",
            hole=0.55
        )
        fig_dtype.update_traces(textposition="outside", textinfo="label+value")
        fig_dtype.update_layout(**theme("Column Data Types", height=280, showlegend=True))
        st.plotly_chart(fig_dtype, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
elif "Exploratory" in page:
    n_total = len(df_feat)
    n_genres_eda = df_feat["main_genre"].nunique()
    n_countries_eda = df_feat["main_country"].nunique()
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-banner-grid"></div>
        <div class="hero-orb-tr"></div>
        <div class="hero-orb-bl"></div>
        <div class="hero-content">
            <div class="hero-badge">📊 Exploratory Analysis</div>
            <div class="hero-title">Data Exploration &amp;<br>Visual Discovery</div>
            <div class="hero-divider"></div>
            <div class="hero-subtitle">Uncover the distribution, trends, and composition of Netflix’s content library &mdash;
            <strong style="color:#fff;">{n_genres_eda} genres</strong> across <strong style="color:#fff;">{n_countries_eda} countries</strong>.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Content type donut
        type_counts = df_feat["type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig_type = px.pie(
            type_counts, names="Type", values="Count",
            color_discrete_sequence=["#E50914", "#FF6B35"],
            hole=0.6
        )
        fig_type.update_traces(textinfo="label+percent", textfont_size=12, pull=[0.03, 0])
        fig_type.update_layout(**theme("📽 Movies vs TV Shows", height=340,
                               annotations=[dict(text=f"<b>{len(df_feat):,}</b><br>Titles", x=0.5, y=0.5,
                                                  font_size=14, showarrow=False, font_color="#fff")]))
        st.plotly_chart(fig_type, use_container_width=True)

    with c2:
        # Top genres bar
        top_genres = df_feat["main_genre"].value_counts().head(12).reset_index()
        top_genres.columns = ["Genre", "Count"]
        fig_genre = px.bar(
            top_genres, x="Count", y="Genre", orientation="h",
            color="Count", color_continuous_scale=["#3A0CA3", "#7209B7", "#E50914"],
            text="Count"
        )
        fig_genre.update_traces(textposition="outside", textfont_size=10)
        fig_genre.update_layout(**theme("🎭 Top 12 Genres", height=340, coloraxis_showscale=False))
        fig_genre.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_genre, use_container_width=True)

    # Release year trend (full width)
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)
    year_data = df_feat[df_feat["release_year"] >= 2000]["release_year"].value_counts().sort_index().reset_index()
    year_data.columns = ["Year", "Count"]

    fig_year = go.Figure()
    fig_year.add_trace(go.Scatter(
        x=year_data["Year"], y=year_data["Count"],
        mode="lines+markers",
        line=dict(color="#E50914", width=2.5),
        marker=dict(size=5, color="#FF6B35"),
        fill="tozeroy",
        fillcolor="rgba(229,9,20,0.07)",
        name="Titles Released",
        hovertemplate="<b>%{x}</b><br>Titles: %{y}<extra></extra>"
    ))
    fig_year.update_layout(**theme("📅 Content Release Trend (2000–Present)", height=300))
    st.plotly_chart(fig_year, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Rating distribution
        ratings = df_feat["rating"].value_counts().head(10).reset_index()
        ratings.columns = ["Rating", "Count"]
        fig_rating = px.bar(
            ratings, x="Rating", y="Count",
            color="Count", color_continuous_scale=["#1a0a2e", "#7209B7", "#F72585"],
            text="Count"
        )
        fig_rating.update_traces(textposition="outside", textfont_size=10)
        fig_rating.update_layout(**theme("🔞 Rating Distribution", height=320, coloraxis_showscale=False))
        st.plotly_chart(fig_rating, use_container_width=True)

    with c4:
        # Top countries
        country_data = df_feat["main_country"].value_counts().head(10).reset_index()
        country_data.columns = ["Country", "Count"]
        fig_country = px.bar(
            country_data, x="Count", y="Country", orientation="h",
            color="Count", color_continuous_scale=["#0d1b2a", "#1b4332", "#52b788"],
            text="Count"
        )
        fig_country.update_traces(textposition="outside", textfont_size=10)
        fig_country.update_layout(**theme("🌍 Top Production Countries", height=320, coloraxis_showscale=False))
        fig_country.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_country, use_container_width=True)

    # Year added heatmap by content type
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)
    heat_df = df_feat.dropna(subset=["year_added"])
    heat_df = heat_df[heat_df["year_added"] >= 2015]
    heat_pivot = pd.crosstab(heat_df["year_added"].astype(int), heat_df["main_genre"])
    top_g_list = df_feat["main_genre"].value_counts().head(8).index.tolist()
    heat_pivot = heat_pivot[[c for c in top_g_list if c in heat_pivot.columns]]

    fig_heat = px.imshow(
        heat_pivot.T,
        color_continuous_scale=["#0A0A0F", "#E50914", "#FF6B35"],
        aspect="auto"
    )
    fig_heat.update_layout(**theme("🔥 Genre Activity Heatmap (Year Added vs Genre)", height=340))
    st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CLUSTER SELECTION
# ─────────────────────────────────────────────────────────────────────────────
elif "Cluster Selection" in page:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-banner-grid"></div>
        <div class="hero-orb-tr"></div>
        <div class="hero-orb-bl"></div>
        <div class="hero-content">
            <div class="hero-badge">📉 Model Selection</div>
            <div class="hero-title">Optimal Cluster<br>Discovery</div>
            <div class="hero-divider"></div>
            <div class="hero-subtitle">Use <strong style="color:#fff;">Elbow Method</strong> and <strong style="color:#fff;">Silhouette Analysis</strong> to scientifically determine the ideal K for your segmentation.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    max_k = st.slider("🔢 Maximum K to Evaluate", min_value=5, max_value=14, value=10,
                       help="Evaluates K-Means from K=2 up to this value")

    with st.spinner("⚙️ Computing inertia and silhouette scores across all K values..."):
        k_range, inertia, silhouette_scores = cached_optimal_k(X_scaled, max_k=max_k)

    best_k = k_range[int(np.nanargmax([v if v is not None else np.nan for v in silhouette_scores]))]
    best_sil = max(v for v in silhouette_scores if v is not None)

    # Best K highlight — premium metric callouts
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:2rem;">
        <div class="metric-callout">
            <div class="metric-callout-label">Best K (Silhouette)</div>
            <div class="metric-callout-val glow-red">{best_k}</div>
            <div class="metric-callout-sub">optimal clusters</div>
        </div>
        <div class="metric-callout">
            <div class="metric-callout-label">Silhouette Score</div>
            <div class="metric-callout-val glow-green">{best_sil:.3f}</div>
            <div class="metric-callout-sub">cluster cohesion</div>
        </div>
        <div class="metric-callout">
            <div class="metric-callout-label">K Values Tested</div>
            <div class="metric-callout-val glow-blue">{len(k_range)}</div>
            <div class="metric-callout-sub">from K=2 to K={max_k}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_range, y=inertia,
            mode="lines+markers",
            line=dict(color="#E50914", width=2.5),
            marker=dict(size=8, color="#FF6B35", line=dict(width=2, color="#E50914")),
            fill="tozeroy", fillcolor="rgba(229,9,20,0.06)",
            name="Inertia",
            hovertemplate="K=%{x}<br>Inertia=%{y:.0f}<extra></extra>"
        ))
        fig_elbow.update_layout(**theme(
            "🦵 Elbow Curve (Inertia)",
            xaxis_title="K — Number of Clusters",
            yaxis_title="Inertia (WCSS)",
            height=340
        ))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=k_range, y=silhouette_scores,
            marker=dict(
                color=silhouette_scores,
                colorscale=[[0, "#1a1a2e"], [0.5, "#7209B7"], [1.0, "#4ade80"]],
                line=dict(width=0)
            ),
            hovertemplate="K=%{x}<br>Score=%{y:.4f}<extra></extra>",
            name="Silhouette"
        ))
        # Mark best
        fig_sil.add_vline(x=best_k, line_dash="dash", line_color="rgba(74,222,128,0.5)",
                           annotation_text=f"  Best K={best_k}", annotation_font_color="#4ade80")
        fig_sil.update_layout(**theme(
            "✨ Silhouette Score vs K",
            xaxis_title="K — Number of Clusters",
            yaxis_title="Silhouette Score",
            height=340
        ))
        st.plotly_chart(fig_sil, use_container_width=True)

    # Score table
    st.markdown('<div class="section-header"><div class="section-header-icon">📋</div><div class="section-header-text">Detailed Scores Table</div></div>', unsafe_allow_html=True)
    score_df = pd.DataFrame({
        "K (Clusters)": k_range,
        "Inertia": [round(i, 2) for i in inertia],
        "Silhouette Score": [round(s, 5) for s in silhouette_scores],
    })
    score_df["🏆 Best?"] = score_df["K (Clusters)"].apply(lambda x: "⭐ Best" if x == best_k else "")
    st.dataframe(score_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — INTERACTIVE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif "Interactive" in page:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-banner-grid"></div>
        <div class="hero-orb-tr"></div>
        <div class="hero-orb-bl"></div>
        <div class="hero-content">
            <div class="hero-badge">🤖 Interactive ML</div>
            <div class="hero-title">Interactive Content<br>Clustering Lab</div>
            <div class="hero-divider"></div>
            <div class="hero-subtitle">Choose your algorithm, tune parameters, and visualize content segments in real-time — <strong style="color:#fff;">K-Means</strong>, <strong style="color:#fff;">Hierarchical</strong>, or <strong style="color:#fff;">DBSCAN</strong>.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([1, 3])

    with col_ctrl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**⚙️ Algorithm Controls**")
        algo = st.selectbox("Algorithm", ["K-Means", "Agglomerative Hierarchical", "DBSCAN"])

        if algo in ["K-Means", "Agglomerative Hierarchical"]:
            n_clusters = st.slider("Clusters (K)", min_value=2, max_value=10, value=5)
            eps, min_samples = None, None
        else:
            eps = st.slider("DBSCAN eps", min_value=0.3, max_value=5.0, value=2.0, step=0.1)
            min_samples = st.slider("DBSCAN min_samples", min_value=2, max_value=25, value=5)
            n_clusters = None

        n_pca = st.radio("PCA Dimensions", [2, 3], horizontal=True, index=0)
        color_scale = st.selectbox("Color Palette", ["Vivid", "Plasma", "Turbo", "Spectral"])
        st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner(f"🔄 Fitting {algo} and projecting PCA..."):
        if algo == "K-Means":
            model, labels, score = cached_cluster(X_scaled, "K-Means", k=n_clusters)
        elif algo == "Agglomerative Hierarchical":
            model, labels, score = cached_cluster(X_scaled, "Agglomerative Hierarchical", k=n_clusters)
        else:
            model, labels, score = cached_cluster(X_scaled, "DBSCAN", eps=eps, min_samples=min_samples)

        df_disp = df_feat.copy()
        df_disp["Cluster"] = labels.astype(str)
        df_disp["Cluster_Int"] = labels

        pca, X_pca = cached_pca(X_scaled, n_pca)
        df_disp["PCA1"] = X_pca[:, 0]
        df_disp["PCA2"] = X_pca[:, 1]
        if n_pca == 3:
            df_disp["PCA3"] = X_pca[:, 2]

        unique_clusters = sorted(df_disp["Cluster_Int"].unique())
        n_noise = sum(labels == -1)

    with col_viz:
        # Score display
        if score is not None:
            quality = "Excellent" if score > 0.5 else "Good" if score > 0.3 else "Fair"
            q_color = "#4ade80" if score > 0.5 else "#facc15" if score > 0.3 else "#f87171"
            st.markdown(f"""
            <div style="display:flex; gap:1rem; margin-bottom:1rem; align-items:center;">
                <div style="background:rgba(74,222,128,0.08); border:1px solid {q_color}40; border-radius:10px; padding:0.6rem 1.2rem;">
                    <span style="font-size:0.65rem; color:#777; text-transform:uppercase; letter-spacing:1.2px;">Silhouette Score</span><br>
                    <span style="font-size:1.6rem; font-weight:800; color:{q_color};">{score:.4f}</span>
                    <span style="font-size:0.8rem; color:{q_color}; margin-left:6px;">({quality})</span>
                </div>
                <div style="background:rgba(229,9,20,0.08); border:1px solid rgba(229,9,20,0.25); border-radius:10px; padding:0.6rem 1.2rem;">
                    <span style="font-size:0.65rem; color:#777; text-transform:uppercase; letter-spacing:1.2px;">Clusters Found</span><br>
                    <span style="font-size:1.6rem; font-weight:800; color:#E50914;">{len(unique_clusters) - (1 if -1 in unique_clusters else 0)}</span>
                </div>
                {"<div style='background:rgba(255,107,53,0.08); border:1px solid rgba(255,107,53,0.25); border-radius:10px; padding:0.6rem 1.2rem;'><span style='font-size:0.65rem; color:#777; text-transform:uppercase; letter-spacing:1.2px;'>Noise Points</span><br><span style='font-size:1.6rem; font-weight:800; color:#FF6B35;'>" + str(n_noise) + "</span></div>" if algo == "DBSCAN" else ""}
            </div>
            """, unsafe_allow_html=True)

        # 2D or 3D scatter
        color_map = {"Vivid": NETFLIX_COLORS, "Plasma": px.colors.sequential.Plasma,
                     "Turbo": px.colors.sequential.Turbo, "Spectral": px.colors.diverging.Spectral}
        c_seq = color_map.get(color_scale, NETFLIX_COLORS)

        if n_pca == 2:
            fig_pca = px.scatter(
                df_disp, x="PCA1", y="PCA2",
                color="Cluster",
                color_discrete_sequence=c_seq,
                hover_data={"title": True, "type": True, "main_genre": True, "release_year": True,
                            "PCA1": False, "PCA2": False, "Cluster": False},
                opacity=0.75,
                title=f"🔬 PCA 2D — {algo} ({len(unique_clusters)} Clusters)"
            )
            fig_pca.update_traces(marker=dict(size=5, line=dict(width=0.3, color="rgba(0,0,0,0.4)")))
            fig_pca.update_layout(**theme(height=520))
        else:
            fig_pca = px.scatter_3d(
                df_disp, x="PCA1", y="PCA2", z="PCA3",
                color="Cluster",
                color_discrete_sequence=c_seq,
                hover_data={"title": True, "type": True, "main_genre": True},
                opacity=0.75,
                title=f"🔬 PCA 3D — {algo} ({len(unique_clusters)} Clusters)"
            )
            fig_pca.update_traces(marker=dict(size=3))
            fig_pca.update_layout(**theme(height=520,
                                   scene=dict(
                                       xaxis=dict(backgroundcolor="#0A0A0F", gridcolor="rgba(255,255,255,0.05)"),
                                       yaxis=dict(backgroundcolor="#0A0A0F", gridcolor="rgba(255,255,255,0.05)"),
                                       zaxis=dict(backgroundcolor="#0A0A0F", gridcolor="rgba(255,255,255,0.05)"),
                                   )))
        st.plotly_chart(fig_pca, use_container_width=True)

    # Cluster size distribution
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)
    cluster_size = df_disp["Cluster"].value_counts().reset_index()
    cluster_size.columns = ["Cluster", "Count"]
    fig_size = px.bar(
        cluster_size.sort_values("Cluster"), x="Cluster", y="Count",
        color="Count",
        color_continuous_scale=["#1a0a2e", "#E50914", "#FF6B35"],
        text="Count",
        title="📦 Cluster Size Distribution"
    )
    fig_size.update_traces(textposition="outside")
    fig_size.update_layout(**theme("📦 Cluster Size Distribution", height=280, coloraxis_showscale=False))
    st.plotly_chart(fig_size, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — CLUSTER PROFILES
# ─────────────────────────────────────────────────────────────────────────────
elif "Profiles" in page:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🏷️ Profiles & Insights</div>
        <div class="hero-title">Cluster Profiling &<br>Business Intelligence</div>
        <div class="hero-subtitle">Deep-dive into each discovered segment to understand genre composition, ratings, countries, and content dynamics.</div>
    </div>
    """, unsafe_allow_html=True)

    n_clusters = st.sidebar.slider("Clusters (K) for Profiling", min_value=2, max_value=8, value=5)

    with st.spinner(f"🔄 Running K-Means with K={n_clusters}..."):
        model, labels, score = cached_cluster(X_scaled, "K-Means", k=n_clusters)
        df_prof = df_feat.copy()
        df_prof["cluster"] = labels
        summary, type_dist, profiles = profile_clusters(df_prof, "cluster")

    if score:
        quality = "Excellent" if score > 0.5 else "Good" if score > 0.3 else "Fair"
        q_color = "#4ade80" if score > 0.5 else "#facc15" if score > 0.3 else "#f87171"
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem;">
            <div class="metric-callout">
                <div class="metric-callout-label">Silhouette Score</div>
                <div class="metric-callout-val" style="color:{q_color}; text-shadow:0 0 20px {q_color}66;">{score:.4f}</div>
                <div class="metric-callout-sub">{quality} separation</div>
            </div>
            <div class="metric-callout">
                <div class="metric-callout-label">Clusters (K)</div>
                <div class="metric-callout-val glow-red">{n_clusters}</div>
                <div class="metric-callout-sub">content segments</div>
            </div>
            <div class="metric-callout">
                <div class="metric-callout-label">Titles Clustered</div>
                <div class="metric-callout-val glow-blue">{len(df_prof):,}</div>
                <div class="metric-callout-sub">total titles</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header"><div class="section-header-icon">📊</div><div class="section-header-text">Feature Means per Cluster</div></div>', unsafe_allow_html=True)

        radar_fig = go.Figure()
        categories = ["release_year", "year_added", "duration_value"]
        norm_summary = summary.copy()
        for col in categories:
            norm_summary[col] = (summary[col] - summary[col].min()) / (summary[col].max() - summary[col].min() + 1e-9)

        for i, cluster_id in enumerate(norm_summary.index):
            vals = norm_summary.loc[cluster_id, categories].tolist()
            vals += [vals[0]]  # close polygon
            cats = categories + [categories[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats,
                fill="toself",
                name=f"Cluster {cluster_id}",
                line=dict(color=NETFLIX_COLORS[i % len(NETFLIX_COLORS)], width=2),
                fillcolor=NETFLIX_COLORS[i % len(NETFLIX_COLORS)].replace(")", ",0.1)").replace("rgb", "rgba")
                if "rgb" in NETFLIX_COLORS[i % len(NETFLIX_COLORS)] else
                f"rgba(229,9,20,0.07)",
                opacity=0.8
            ))
        radar_fig.update_layout(**theme(
            "🕸 Normalized Feature Radar",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.06)", color="#555"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#888")
            ),
            height=380,
            showlegend=True
        ))
        st.plotly_chart(radar_fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header"><div class="section-header-icon">🎬</div><div class="section-header-text">Movie vs TV Show Mix</div></div>', unsafe_allow_html=True)
        type_dist_reset = type_dist.reset_index().melt(id_vars="cluster", var_name="Content Type", value_name="Proportion")
        fig_td = px.bar(
            type_dist_reset, x="cluster", y="Proportion", color="Content Type",
            barmode="stack", color_discrete_sequence=["#E50914", "#FF6B35"],
            text=type_dist_reset["Proportion"].apply(lambda x: f"{x:.0%}"),
            title="🎥 Content Type Mix per Cluster"
        )
        fig_td.update_traces(textposition="inside", textfont_size=11)
        fig_td.update_layout(**theme("🎥 Content Type Mix per Cluster", height=380,
                              xaxis_title="Cluster", yaxis_title="Proportion", yaxis_tickformat=".0%"))
        st.plotly_chart(fig_td, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header"><div class="section-header-icon">📋</div><div class="section-header-text">Numeric Feature Means</div></div>', unsafe_allow_html=True)
    st.caption("Duration measures minutes for movies and seasons for TV shows; cross-type duration averages are not comparable.")
    st.dataframe(summary.style.format("{:.1f}").highlight_max(axis=0, color="#3d1a00").highlight_min(axis=0, color="#0a1f0a"),
                 use_container_width=True)

    # Cluster detail cards
    st.markdown('<div class="section-header"><div class="section-header-icon">🔍</div><div class="section-header-text">Cluster Deep Dive</div></div>', unsafe_allow_html=True)
    colors_cycle = NETFLIX_COLORS

    for c_id, info in profiles.items():
        c_color = colors_cycle[c_id % len(colors_cycle)]
        with st.expander(f"📌 Cluster {c_id}  —  {info['count']:,} titles", expanded=(c_id == 0)):
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown("**🎭 Top Genres**")
                for genre, cnt in info["top_genres"].items():
                    st.markdown(f'<span class="tag">{escape(str(genre))}</span> <span style="color:#777; font-size:0.72rem;">{cnt}</span>', unsafe_allow_html=True)
            with col_b:
                st.markdown("**🔞 Top Ratings**")
                for rating, cnt in info["top_ratings"].items():
                    st.markdown(f'<span class="tag">{escape(str(rating))}</span> <span style="color:#777; font-size:0.72rem;">{cnt}</span>', unsafe_allow_html=True)
            with col_c:
                st.markdown("**🌍 Top Countries**")
                for country, cnt in info["top_countries"].items():
                    st.markdown(f'<span class="tag">{escape(str(country))}</span> <span style="color:#777; font-size:0.72rem;">{cnt}</span>', unsafe_allow_html=True)
            with col_d:
                st.markdown("**📊 Quick Stats**")
                movie_pct = type_dist.loc[c_id, "Movie"] * 100 if "Movie" in type_dist.columns and c_id in type_dist.index else 0
                show_pct = type_dist.loc[c_id, "TV Show"] * 100 if "TV Show" in type_dist.columns and c_id in type_dist.index else 0
                st.markdown(f"""
                <div style="font-size:0.78rem; line-height:2;">
                    <div>🎥 Movies: <b style="color:#E50914;">{movie_pct:.1f}%</b></div>
                    <div>📺 TV Shows: <b style="color:#FF6B35;">{show_pct:.1f}%</b></div>
                    <div>📌 Total: <b style="color:#ccc;">{info['count']:,}</b></div>
                    <div>📈 Share: <b style="color:#60a5fa;">{info['count']/len(df_prof)*100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

    # Mini genre sunburst across clusters
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)
    sunburst_data = df_prof.groupby(["cluster", "main_genre"]).size().reset_index(name="count")
    sunburst_data["cluster"] = "Cluster " + sunburst_data["cluster"].astype(str)
    top_genre_list = df_prof["main_genre"].value_counts().head(10).index.tolist()
    sunburst_data = sunburst_data[sunburst_data["main_genre"].isin(top_genre_list)]
    fig_sun = px.sunburst(
        sunburst_data, path=["cluster", "main_genre"], values="count",
        color_discrete_sequence=NETFLIX_COLORS,
        title="🌐 Cluster × Genre Sunburst"
    )
    fig_sun.update_layout(**theme("🌐 Cluster × Genre Sunburst", height=480))
    st.plotly_chart(fig_sun, use_container_width=True)
