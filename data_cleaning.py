"""
╔══════════════════════════════════════════════════════════════╗
║          ML ANALYTICS PRO — v4.0 ULTRA                       ║
║    Professional • Fast • Accurate • Beautiful                ║
╚══════════════════════════════════════════════════════════════╝
Run:  streamlit run data_cleaning.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
import ast
import html
import time
import pickle
import sqlite3
import inspect
import functools
import traceback
from datetime import datetime
import warnings

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
                                   QuantileTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              VotingClassifier, VotingRegressor, IsolationForest)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, KFold)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             mean_squared_error, r2_score, mean_absolute_error,
                             mean_absolute_percentage_error, silhouette_score)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import (shapiro, gaussian_kde, ttest_ind, chi2_contingency, f_oneway,
                         mannwhitneyu, kruskal)
import requests

# ── Optional libraries: the app keeps working even if any of these is missing ──
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb, XGB_AVAILABLE = None, False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb, LGB_AVAILABLE = None, False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap, SHAP_AVAILABLE = None, False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:
    optuna, OPTUNA_AVAILABLE = None, False

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# STREAMLIT VERSION COMPATIBILITY
# Newer Streamlit uses width='stretch'; older versions only know use_container_width.
# This shim makes the same code run on both, so version differences can't crash the app.
# ─────────────────────────────────────────────────────────────
def _install_compat():
    if not hasattr(st, 'rerun') and hasattr(st, 'experimental_rerun'):
        st.rerun = st.experimental_rerun
    for _name in ['button', 'download_button', 'dataframe', 'plotly_chart', 'data_editor']:
        _fn = getattr(st, _name, None)
        if _fn is None or getattr(_fn, '_compat_wrapped', False):
            continue
        try:
            _p = inspect.signature(_fn).parameters
        except (TypeError, ValueError):
            continue
        _ann = str(_p['width'].annotation) if 'width' in _p else ''
        if 'width' in _p and ('Width' in _ann or 'stretch' in _ann):
            continue                                   # modern Streamlit: nothing to do

        def _make(fn, params):
            @functools.wraps(fn)
            def inner(*a, **k):
                w = k.get('width')
                if w in ('stretch', 'content'):
                    k.pop('width')
                    if 'use_container_width' in params:
                        k['use_container_width'] = (w == 'stretch')
                return fn(*a, **k)
            inner._compat_wrapped = True
            return inner
        setattr(st, _name, _make(_fn, _p))


_install_compat()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="ML Analytics Pro v4",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
_state_defaults = {
    'df': None, 'df2': None, 'history': [], 'trained_models': {},
    'best_model': None, 'last_change': None, 'auto_insights': [],
    'feature_importance': None, 'data_quality_score': None,
    'chat_history': [], 'sql_history': [], 'shap_values': None,
    'ai_api_key': '', 'ml_results': None, 'aml_results': None,
    'upload_sig': None,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# ELITE CSS — Ultra-professional dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #6C63FF;
    --primary-glow: rgba(108, 99, 255, 0.4);
    --secondary: #FF6584;
    --accent: #43E97B;
    --accent2: #38F9D7;
    --bg-deep: #07080D;
    --bg-card: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --border-active: rgba(108, 99, 255, 0.5);
    --text: #E8E9F0;
    --text-muted: rgba(232, 233, 240, 0.55);
    --success: #43E97B;
    --warning: #F9AB00;
    --error: #FF4757;
    --gradient-main: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
    --gradient-cool: linear-gradient(135deg, #43E97B 0%, #38F9D7 100%);
    --gradient-dark: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

* { box-sizing: border-box; }
html, body, .main { background: var(--bg-deep) !important; }

/* Main layout */
.main .block-container {
    padding: 1.5rem 2rem 3rem;
    max-width: 1600px;
}

/* ──── HERO HEADER ──── */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border: 1px solid var(--border-active);
    border-radius: 24px;
    padding: 48px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 50% at 50% 0%, rgba(108,99,255,0.2) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 52px; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #fff 0%, #a8a4ff 50%, #6C63FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -1px;
    position: relative; z-index: 1;
}
.hero-sub {
    font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 400;
    color: var(--text-muted); margin-top: 12px; letter-spacing: 0.5px;
    position: relative; z-index: 1;
}
.hero-badges { margin-top: 20px; display: flex; gap: 10px; justify-content: center; position: relative; z-index: 1; }
.badge {
    background: rgba(108,99,255,0.15); border: 1px solid rgba(108,99,255,0.3);
    color: #a8a4ff; padding: 5px 14px; border-radius: 100px;
    font-size: 12px; font-weight: 600; font-family: 'Inter', sans-serif; letter-spacing: 0.5px;
}
.badge-green { background: rgba(67,233,123,0.12); border-color: rgba(67,233,123,0.25); color: #43E97B; }
.badge-orange { background: rgba(249,171,0,0.12); border-color: rgba(249,171,0,0.25); color: #F9AB00; }

/* ──── GLASS CARD ──── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 28px;
    margin: 16px 0;
    border: 1px solid var(--border);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-active);
    box-shadow: 0 0 30px var(--primary-glow);
}

/* ──── METRIC CARDS ──── */
.metric-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.08) 0%, rgba(255,101,132,0.05) 100%);
    border: 1px solid var(--border);
    border-radius: 18px; padding: 24px 20px; text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--gradient-main); opacity: 0; transition: opacity 0.3s;
}
.metric-card:hover { border-color: var(--border-active); transform: translateY(-4px); }
.metric-card:hover::before { opacity: 1; }
.metric-icon { font-size: 28px; margin-bottom: 10px; }
.metric-label { font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--text-muted); font-family: 'Inter', sans-serif; }
.metric-value { font-size: 32px; font-weight: 800; color: var(--text);
    font-family: 'Space Grotesk', sans-serif; margin-top: 6px; line-height: 1; }
.metric-sub { font-size: 12px; color: var(--text-muted); margin-top: 6px; }

/* ──── ALERT CARDS ──── */
.alert-success {
    background: rgba(67,233,123,0.08); border: 1px solid rgba(67,233,123,0.25);
    border-left: 4px solid var(--success); border-radius: 12px; padding: 16px 20px;
    color: #a8f5c8; font-family: 'Inter', sans-serif; font-size: 14px; margin: 12px 0;
}
.alert-info {
    background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.25);
    border-left: 4px solid var(--primary); border-radius: 12px; padding: 16px 20px;
    color: #c5c2ff; font-family: 'Inter', sans-serif; font-size: 14px; margin: 12px 0;
}
.alert-warning {
    background: rgba(249,171,0,0.08); border: 1px solid rgba(249,171,0,0.25);
    border-left: 4px solid var(--warning); border-radius: 12px; padding: 16px 20px;
    color: #fce28a; font-family: 'Inter', sans-serif; font-size: 14px; margin: 12px 0;
}

/* ──── CHANGE SUMMARY ──── */
.change-banner {
    background: linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(255,101,132,0.08) 100%);
    border: 1px solid var(--border-active); border-radius: 16px; padding: 20px 24px;
    margin: 16px 0; font-family: 'Inter', sans-serif;
}
.change-banner h4 { color: #a8a4ff; font-size: 14px; font-weight: 700; margin: 0 0 12px; letter-spacing: 0.5px; }
.change-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.change-item-label { font-size: 11px; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; }
.change-item-val { font-size: 18px; font-weight: 700; color: var(--text); margin-top: 4px; }
.change-item-diff { font-size: 12px; margin-top: 3px; }
.diff-pos { color: var(--success); } .diff-neg { color: var(--error); } .diff-zero { color: var(--text-muted); }

/* ──── SIDEBAR ──── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0e1a 0%, #111225 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif !important; font-size: 13px !important;
    font-weight: 700 !important; letter-spacing: 1px !important; text-transform: uppercase !important;
    color: var(--text-muted) !important; margin: 0 0 10px !important;
}

/* ──── BUTTONS ──── */
.stButton > button {
    background: var(--gradient-main) !important; color: white !important;
    border: none !important; border-radius: 12px !important;
    padding: 12px 24px !important; font-weight: 700 !important; font-size: 14px !important;
    font-family: 'Inter', sans-serif !important; letter-spacing: 0.3px !important;
    width: 100% !important; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.5) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ──── TABS ──── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important; padding: 6px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 13px !important;
    transition: all 0.2s ease !important; padding: 10px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient-main) !important; color: white !important;
    box-shadow: 0 2px 12px rgba(108,99,255,0.4) !important;
}

/* ──── INPUTS ──── */
.stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    color: var(--text) !important; font-family: 'Inter', sans-serif !important;
}
.stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {
    border-color: var(--border-active) !important;
}

/* ──── SLIDERS ──── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--gradient-main) !important;
}

/* ──── DATAFRAME ──── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] > div { background: rgba(255,255,255,0.03) !important; }

/* ──── PROGRESS ──── */
.stProgress > div > div { background: var(--gradient-main) !important; border-radius: 999px !important; }
.stProgress > div { background: rgba(255,255,255,0.06) !important; border-radius: 999px !important; }

/* ──── TEXT OVERRIDES ──── */
h1, h2, h3, h4, p, span, div, label {
    font-family: 'Inter', sans-serif; color: var(--text);
}
h2 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; font-size: 24px !important; }
h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; font-size: 18px !important; }

/* ──── QUALITY RING ──── */
.quality-ring {
    text-align: center; padding: 20px 16px;
    background: rgba(255,255,255,0.03); border: 1px solid var(--border);
    border-radius: 16px; margin: 12px 0;
}
.quality-score {
    font-family: 'Space Grotesk', sans-serif; font-size: 52px; font-weight: 900;
    line-height: 1; margin: 8px 0;
}
.quality-label { font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--text-muted); }

/* ──── INSIGHT BOX ──── */
.insight-item {
    background: rgba(108,99,255,0.06); border: 1px solid rgba(108,99,255,0.15);
    border-radius: 12px; padding: 14px 16px; margin: 8px 0; font-size: 14px; color: #c5c2ff;
}

/* ──── PREDICTION RESULT ──── */
.pred-result {
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(67,233,123,0.08));
    border: 2px solid var(--border-active); border-radius: 20px;
    padding: 48px 40px; text-align: center; margin: 20px 0;
}
.pred-value {
    font-family: 'Space Grotesk', sans-serif; font-size: 72px; font-weight: 900;
    background: var(--gradient-main); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; background-clip: text; line-height: 1;
}
.pred-conf { font-size: 20px; color: var(--accent); margin-top: 12px; font-weight: 600; }

/* ──── SECTION DIVIDER ──── */
.section-divider {
    height: 1px; background: var(--border); margin: 24px 0;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* ──── MODEL RESULT ROW ──── */
.model-row {
    background: rgba(255,255,255,0.03); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px; margin: 8px 0;
    display: flex; align-items: center; gap: 16px; transition: all 0.2s;
}
.model-row:hover { border-color: var(--border-active); background: rgba(108,99,255,0.06); }
.model-row.best { border-color: var(--success); background: rgba(67,233,123,0.05); }

/* ──── FOOTER ──── */
.footer {
    text-align: center; padding: 40px 20px; margin-top: 48px;
    border-top: 1px solid var(--border);
}
.footer-title { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 800;
    background: var(--gradient-main); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; background-clip: text; }
.footer-sub { color: var(--text-muted); font-size: 13px; margin-top: 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# GENERIC HELPERS
# ─────────────────────────────────────────────────────────────
def get_num_cols(d: pd.DataFrame):
    """Numeric (non-bool) columns — works on pandas 2.x and 3.x."""
    return [c for c in d.columns
            if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]


def get_cat_cols(d: pd.DataFrame):
    """Text / category / bool columns (everything that is neither numeric nor datetime)."""
    return [c for c in d.columns
            if (not pd.api.types.is_numeric_dtype(d[c]) or pd.api.types.is_bool_dtype(d[c]))
            and not pd.api.types.is_datetime64_any_dtype(d[c])]


def get_dt_cols(d: pd.DataFrame):
    return [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]


def fmt_class(c):
    if isinstance(c, (float, np.floating)) and float(c).is_integer():
        return str(int(c))
    return str(c)


def esc(x):
    return html.escape(str(x))


def push_history(df: pd.DataFrame, action: str):
    prev_df = st.session_state.df
    changes = {
        "action": action, "timestamp": datetime.now(),
        "rows_before": len(prev_df) if prev_df is not None else 0,
        "rows_after": len(df),
        "cols_before": len(prev_df.columns) if prev_df is not None else 0,
        "cols_after": len(df.columns)
    }
    st.session_state.history.append({"time": datetime.now(), "action": action,
                                     "df": df.copy(), "shape": df.shape, "changes": changes})
    st.session_state.last_change = changes
    if len(st.session_state.history) > 30:
        st.session_state.history = st.session_state.history[-30:]


def apply_df(new_df: pd.DataFrame, action: str, rescore=True):
    """Single place that commits a dataframe change (history + quality score + insights)."""
    push_history(new_df, action)
    st.session_state.df = new_df
    if rescore:
        try:
            st.session_state.data_quality_score = calculate_data_quality_score(new_df)
            st.session_state.auto_insights = auto_generate_insights(new_df)
        except Exception:
            pass


def show_change_summary():
    if not st.session_state.last_change:
        return
    ch = st.session_state.last_change
    rd = ch['rows_after'] - ch['rows_before']
    cd = ch['cols_after'] - ch['cols_before']
    rd_class = "diff-pos" if rd >= 0 else "diff-neg"
    cd_class = "diff-pos" if cd >= 0 else "diff-neg"
    st.markdown(f"""
    <div class="change-banner">
        <h4>⚡ {esc(ch['action'])}</h4>
        <div class="change-grid">
            <div>
                <div class="change-item-label">Rows</div>
                <div class="change-item-val">{ch['rows_before']:,} → {ch['rows_after']:,}</div>
                <div class="change-item-diff {rd_class}">{'+' if rd >= 0 else ''}{rd:,}</div>
            </div>
            <div>
                <div class="change-item-label">Columns</div>
                <div class="change-item-val">{ch['cols_before']:,} → {ch['cols_after']:,}</div>
                <div class="change-item-diff {cd_class}">{'+' if cd >= 0 else ''}{cd:,}</div>
            </div>
            <div>
                <div class="change-item-label">Time</div>
                <div class="change-item-val">{ch['timestamp'].strftime('%H:%M:%S')}</div>
                <div class="change-item-diff diff-zero">{ch['timestamp'].strftime('%b %d')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def calculate_data_quality_score(df: pd.DataFrame):
    score = 100.0
    issues = []
    if df.shape[0] == 0 or df.shape[1] == 0:
        return 0.0, ["🔴 Dataset is empty"]
    mp = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if mp > 0:
        pen = min(30, mp * 3)
        score -= pen
        issues.append(f"🔴 Missing data: {mp:.1f}% (−{pen:.0f} pts)")
    dp = df.duplicated().sum() / len(df) * 100
    if dp > 0:
        pen = min(20, dp * 5)
        score -= pen
        issues.append(f"🟡 Duplicates: {dp:.1f}% (−{pen:.0f} pts)")
    ti = 0
    for c in get_cat_cols(df):
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c].dropna()
        if len(s) and pd.to_numeric(s, errors='coerce').notna().mean() > 0.5:
            ti += 1
    if ti:
        pen = min(15, ti * 3)
        score -= pen
        issues.append(f"🔵 Type mismatches: {ti} cols (−{pen:.0f} pts)")
    nc = get_num_cols(df)
    if nc:
        oc = 0
        for c in nc:
            sd = df[c].std()
            if sd and not np.isnan(sd) and sd > 0:
                oc += int(((df[c] - df[c].mean()).abs() > 3 * sd).sum())
        op = oc / (len(df) * len(nc)) * 100
        if op > 5:
            pen = min(10, (op - 5) * 1.5)
            score -= pen
            issues.append(f"🟠 Outliers: {op:.1f}% (−{pen:.0f} pts)")
    return max(0.0, score), issues


def auto_generate_insights(df: pd.DataFrame):
    insights = []
    if df is None or df.empty:
        return insights
    nc = get_num_cols(df)
    cc = get_cat_cols(df)
    for col in nc[:6]:
        sk = df[col].skew()
        if pd.notna(sk) and abs(sk) > 1.5:
            insights.append(f"📊 <b>{esc(col)}</b> is {'heavily right' if sk > 0 else 'heavily left'}-skewed (skewness={sk:.2f}). Consider log transform.")
        if df[col].nunique() < 12:
            insights.append(f"🎯 <b>{esc(col)}</b> has only {df[col].nunique()} unique values — suitable as a classification target.")
    if len(nc) > 1:
        corr = df[nc].corr()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                v = corr.iloc[i, j]
                if pd.notna(v) and abs(v) > 0.75:
                    insights.append(f"🔗 <b>{esc(corr.columns[i])}</b> ↔ <b>{esc(corr.columns[j])}</b> highly correlated (r={v:.2f}). Consider dropping one.")
    mc = df.isnull().sum()
    big_miss = mc[mc > len(df) * 0.2]
    for col, cnt in big_miss.items():
        insights.append(f"❓ <b>{esc(col)}</b> has {cnt / len(df) * 100:.1f}% missing values — investigate before imputing.")
    for col in cc[:4]:
        vc = df[col].value_counts(normalize=True)
        if len(vc) and vc.iloc[0] > 0.6:
            insights.append(f"📈 <b>{esc(col)}</b> is dominated by '{esc(vc.index[0])}' ({vc.iloc[0] * 100:.0f}%) — possible class imbalance.")
    return insights[:8]


def download_button(df, fmt, label, key):
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if fmt == "csv":
            st.download_button(label, df.to_csv(index=False).encode('utf-8'), f"data_{ts}.csv", "text/csv", key=key, width='stretch')
        elif fmt == "excel":
            # Excel generation is slow → only build it when the user asks (cached in session state)
            ck = f"_xl_{key}"
            cached = st.session_state.get(ck)
            sig = (df.shape, int(pd.util.hash_pandas_object(df.head(50), index=False).sum()) if len(df) else 0)
            if cached and cached[0] == sig:
                st.download_button(label, cached[1], f"data_{ts}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=key, width='stretch')
            elif st.button(f"⚙️ Prepare {label}", key=key + "_prep", width='stretch'):
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    df.to_excel(w, index=False, sheet_name='Data')
                st.session_state[ck] = (sig, buf.getvalue())
                st.rerun()
        elif fmt == "json":
            st.download_button(label, df.to_json(orient='records', indent=2, date_format='iso'), f"data_{ts}.json",
                               "application/json", key=key, width='stretch')
    except Exception as e:
        st.error(f"Download error: {e}")


def plotly_dark_layout(**kwargs):
    base = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E9F0', family='Inter'), margin=dict(t=50, b=20, l=20, r=20),
                legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)'))
    base.update(kwargs)
    return base


def color_scale():
    return [[0, '#6C63FF'], [0.5, '#FF6584'], [1.0, '#43E97B']]


def read_uploaded(uploaded):
    """Robust file reader (CSV encodings / delimiters, Excel, JSON & JSON-lines, Parquet)."""
    ext = uploaded.name.split('.')[-1].lower()
    raw_bytes = uploaded.getvalue()
    if ext in ('csv', 'txt', 'tsv'):
        last_err = None
        for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
            try:
                try:
                    return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
                except pd.errors.ParserError:
                    return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=None, engine='python')
            except UnicodeDecodeError as e:
                last_err = e
        raise last_err
    if ext in ('xlsx', 'xls'):
        return pd.read_excel(io.BytesIO(raw_bytes))
    if ext == 'json':
        try:
            return pd.read_json(io.BytesIO(raw_bytes))
        except ValueError:
            return pd.read_json(io.BytesIO(raw_bytes), lines=True)
    if ext == 'parquet':
        return pd.read_parquet(io.BytesIO(raw_bytes))
    raise ValueError(f"Unsupported file type: .{ext}")


# ─────────────────────────────────────────────────────────────
# SAFE FORMULA EVALUATION (replaces raw eval)
# ─────────────────────────────────────────────────────────────
_NP_FUNCS = {'log', 'log1p', 'log2', 'log10', 'exp', 'sqrt', 'abs', 'sin', 'cos', 'tan', 'floor', 'ceil',
             'round', 'clip', 'where', 'minimum', 'maximum', 'square', 'sign', 'power', 'mean', 'median',
             'std', 'sum', 'min', 'max'}
_SAFE_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Constant, ast.Name,
               ast.Load, ast.Call, ast.Attribute, ast.IfExp, ast.Tuple, ast.List, ast.operator,
               ast.unaryop, ast.cmpop, ast.boolop, ast.keyword)


def safe_formula_eval(d: pd.DataFrame, formula: str):
    """Evaluate a column formula such as  Hours_Studied * Attendance / 100  or  np.log1p(`My Col`).
    Only arithmetic, comparisons and a whitelist of np.* functions are allowed."""
    names = {}

    def _repl(m):
        key = f"__c{len(names)}"
        names[key] = m.group(1)
        return key
    expr = re.sub(r"`([^`]+)`", _repl, formula).strip()
    tree = ast.parse(expr, mode='eval')
    env = {c: d[c] for c in d.columns if isinstance(c, str) and c.isidentifier()}
    for k, v in names.items():
        if v not in d.columns:
            raise ValueError(f"Unknown column `{v}`")
        env[k] = d[v]
    env['np'] = np
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr in _NP_FUNCS):
                raise ValueError("Only np.<function> calls from the allowed list are supported")
        elif isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"Unknown name '{node.id}' — use column names (wrap names with spaces in `backticks`)")
        elif not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
    return eval(compile(tree, '<formula>', 'eval'), {'__builtins__': {}}, env)


# ─────────────────────────────────────────────────────────────
# SQL (sqlite3 — zero extra dependencies, no SQLAlchemy version issues)
# ─────────────────────────────────────────────────────────────
def run_sql(d: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.strip().rstrip(';').strip()
    if not re.match(r'^(select|with)\b', q, re.I):
        raise ValueError("Only SELECT / WITH queries are allowed.")
    safe = d.copy()
    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = safe[c].astype(str)
        elif isinstance(safe[c].dtype, pd.CategoricalDtype):
            safe[c] = safe[c].astype(object)
    conn = sqlite3.connect(':memory:')
    try:
        safe.to_sql('df', conn, index=False, if_exists='replace')
        return pd.read_sql_query(q, conn)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# ML ENGINE — one consistent pipeline for training, prediction, SHAP and export
# ─────────────────────────────────────────────────────────────
_ORDINAL_SCALES = [
    ['none', 'very low', 'low', 'medium', 'moderate', 'high', 'very high'],
    ['negative', 'neutral', 'positive'],
    ['near', 'moderate', 'far'], ['close', 'moderate', 'far'],
    ['high school', 'college', 'bachelor', 'master', 'postgraduate', 'phd', 'doctorate'],
    ['poor', 'fair', 'average', 'good', 'very good', 'excellent'],
    ['never', 'rarely', 'sometimes', 'often', 'always'],
    ['small', 'medium', 'large'], ['beginner', 'intermediate', 'advanced', 'expert'],
    ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree'],
    ['no', 'yes'], ['false', 'true'], ['n', 'y'],
]


def detect_ordinal(series: pd.Series):
    """If the column's values form a known ordered scale (Low<Medium<High, No<Yes ...) return the ordered list."""
    vals = [v for v in pd.unique(series.dropna()) if isinstance(v, str)]
    if len(vals) < 2:
        return None
    lower = {}
    for v in vals:
        lower.setdefault(v.strip().lower(), []).append(v)
    if any(len(v) > 1 for v in lower.values()):
        return None
    for scale in _ORDINAL_SCALES:
        if set(lower) <= set(scale):
            return [lower[s][0] for s in scale if s in lower]
    return None


def coerce_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes so training and prediction always see identical types."""
    X = X.copy()
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            X[c] = ((s - pd.Timestamp('1970-01-01')).dt.total_seconds() / 86400.0).astype('float64')
        elif pd.api.types.is_bool_dtype(s):
            X[c] = s.astype('float64')
        elif pd.api.types.is_numeric_dtype(s):
            X[c] = pd.Series(s).to_numpy(dtype='float64', na_value=np.nan)
        else:
            X[c] = s.astype(str).astype(object).where(s.notna(), np.nan)
    return X


def build_feature_meta(X: pd.DataFrame):
    meta = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            sn = s.dropna()
            if sn.empty:
                lo = hi = med = 0.0
                is_int = False
            else:
                lo, hi, med = float(sn.min()), float(sn.max()), float(sn.median())
                is_int = bool((sn % 1 == 0).all())
            meta[c] = {'kind': 'num', 'min': lo, 'max': hi, 'default': med, 'int': is_int}
        else:
            vc = s.dropna().astype(str).value_counts()
            meta[c] = {'kind': 'cat', 'options': sorted(vc.index.tolist()),
                       'default': vc.index[0] if len(vc) else ''}
    return meta


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:                                   # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def make_preprocessor(X: pd.DataFrame, scaler='standard'):
    """Numeric → median-impute (+scale) · ordinal text → ordered codes · text → one-hot."""
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cats = [c for c in X.columns if c not in num]
    ord_cols, ord_cats, nom_low, nom_high = [], [], [], []
    for c in cats:
        o = detect_ordinal(X[c])
        if o:
            ord_cols.append(c); ord_cats.append(o)
        elif X[c].nunique(dropna=True) <= 30:
            nom_low.append(c)
        else:
            nom_high.append(c)

    def _scaler():
        return RobustScaler() if scaler == 'robust' else StandardScaler()
    use_scale = scaler in ('standard', 'robust')
    tfs = []
    if num:
        steps = [('imp', SimpleImputer(strategy='median'))]
        if use_scale: steps.append(('sc', _scaler()))
        tfs.append(('num', Pipeline(steps), num))
    if ord_cols:
        steps = [('imp', SimpleImputer(strategy='most_frequent')),
                 ('enc', OrdinalEncoder(categories=ord_cats, handle_unknown='use_encoded_value', unknown_value=-1))]
        if use_scale: steps.append(('sc', _scaler()))
        tfs.append(('ord', Pipeline(steps), ord_cols))
    if nom_low:
        tfs.append(('oh', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', _make_ohe())]), nom_low))
    if nom_high:
        tfs.append(('hc', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                    ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), nom_high))
    if not tfs:
        raise ValueError("No usable feature columns.")
    return ColumnTransformer(tfs, remainder='drop', verbose_feature_names_out=False)


def build_pipeline(X_sample, estimator, ptype, scaler='standard', k_select=None):
    prep = make_preprocessor(X_sample, scaler)
    steps = [('prep', prep)]
    if k_select:
        n_out = len(clone(prep).fit(X_sample).get_feature_names_out())
        if k_select < n_out:
            steps.append(('select', SelectKBest(f_classif if ptype == 'classification' else f_regression, k=int(k_select))))
    steps.append(('model', estimator))
    return Pipeline(steps)


def final_estimator(pipe):
    m = pipe.named_steps['model'] if hasattr(pipe, 'named_steps') else pipe
    return m.regressor_ if hasattr(m, 'regressor_') else m


def transformed_feature_names(pipe):
    names = list(pipe.named_steps['prep'].get_feature_names_out())
    if 'select' in pipe.named_steps:
        names = [n for n, keep in zip(names, pipe.named_steps['select'].get_support()) if keep]
    return names


def _ttr(est):
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def get_model_catalog(ptype, n_est=200, seed=42):
    """name → factory (fresh estimator each call). Defaults are tuned for accuracy on tabular data."""
    if ptype == 'classification':
        cat = {
            "🔗 Logistic Regression": lambda: LogisticRegression(max_iter=3000, random_state=seed),
            "🌲 Random Forest": lambda: RandomForestClassifier(n_estimators=n_est, random_state=seed, n_jobs=-1),
            "📈 Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=max(100, n_est // 2), learning_rate=0.08, random_state=seed),
            "🌳 Extra Trees": lambda: ExtraTreesClassifier(n_estimators=n_est, random_state=seed, n_jobs=-1),
            "🧠 Neural Network": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, early_stopping=True, random_state=seed),
            "📍 KNN": lambda: KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
            "🔵 SVM (RBF)": lambda: SVC(kernel='rbf', probability=True, random_state=seed),
            "📊 Naive Bayes": lambda: GaussianNB(),
        }
        if XGB_AVAILABLE:
            cat["⚡ XGBoost"] = lambda: xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.08, max_depth=5, subsample=0.9,
                                                        colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbosity=0)
        if LGB_AVAILABLE:
            cat["💡 LightGBM"] = lambda: lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.08, num_leaves=31,
                                                           random_state=seed, n_jobs=-1, verbose=-1)
    else:
        cat = {
            "📏 Linear Regression": lambda: LinearRegression(),
            "🔷 Ridge": lambda: Ridge(alpha=1.0),
            "🔹 Lasso": lambda: Lasso(alpha=0.01, max_iter=10000),
            "⚖️ ElasticNet": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
            "🌲 Random Forest": lambda: RandomForestRegressor(n_estimators=n_est, random_state=seed, n_jobs=-1),
            "📈 Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=max(150, n_est), learning_rate=0.05, max_depth=3, random_state=seed),
            "🌳 Extra Trees": lambda: ExtraTreesRegressor(n_estimators=n_est, random_state=seed, n_jobs=-1),
            "🧠 Neural Network": lambda: _ttr(MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, early_stopping=True, random_state=seed)),
            "📍 KNN": lambda: KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1),
            "🔵 SVR": lambda: _ttr(SVR(kernel='rbf', C=10.0)),
        }
        if XGB_AVAILABLE:
            cat["⚡ XGBoost"] = lambda: xgb.XGBRegressor(n_estimators=n_est, learning_rate=0.05, max_depth=4, subsample=0.9,
                                                       colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbosity=0)
        if LGB_AVAILABLE:
            cat["💡 LightGBM"] = lambda: lgb.LGBMRegressor(n_estimators=n_est, learning_rate=0.05, num_leaves=31,
                                                          random_state=seed, n_jobs=-1, verbose=-1)
    return cat


def default_model_names(ptype, catalog):
    pref = (["🔗 Logistic Regression", "🌲 Random Forest", "⚡ XGBoost", "💡 LightGBM", "📈 Gradient Boosting"]
            if ptype == 'classification' else
            ["🔷 Ridge", "🌲 Random Forest", "⚡ XGBoost", "💡 LightGBM", "📈 Gradient Boosting"])
    out = [m for m in pref if m in catalog]
    return out or list(catalog.keys())[:4]


def infer_problem_type(y: pd.Series, override='Auto'):
    if override and override != 'Auto':
        return override.lower()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return 'classification'
    yn = y.dropna()
    nun = yn.nunique()
    is_int = bool((yn % 1 == 0).all()) if len(yn) else False
    if nun <= 2:
        return 'classification'
    if is_int and nun <= 10:
        return 'classification'
    if is_int and nun < 25 and nun / max(len(yn), 1) < 0.05:
        return 'classification'
    return 'regression'


def _sorted_classes(vals):
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=lambda v: str(v))


def iqr_outlier_mask(y: pd.Series, k=1.5):
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return pd.Series(False, index=y.index)
    return (y < q1 - k * iqr) | (y > q3 + k * iqr)


def prepare_ml_data(d, target_col, feature_cols, problem_type='Auto', remove_outliers=False):
    """Returns X (coerced raw features), y, label_encoder, problem_type, info dict."""
    d = d.dropna(subset=[target_col]).copy()
    info = {'dropped_features': [], 'outliers_found': 0, 'outliers_removed': 0, 'X_out': None, 'y_out': None,
            'rare_dropped': 0}
    keep = []
    for c in feature_cols:
        if c not in d.columns or c == target_col:
            continue
        nun = d[c].nunique(dropna=True)
        if nun <= 1:
            info['dropped_features'].append((c, 'constant')); continue
        if (not pd.api.types.is_numeric_dtype(d[c])) and nun == len(d) and len(d) > 20:
            info['dropped_features'].append((c, 'ID-like')); continue
        keep.append(c)
    if not keep:
        raise ValueError("No usable features left (all selected columns are constant or ID-like).")

    y_raw = d[target_col]
    ptype = infer_problem_type(y_raw, problem_type)
    t_enc = None
    if ptype == 'regression':
        y = pd.to_numeric(y_raw, errors='coerce')
        ok = y.notna()
        d, y = d[ok], y[ok]
        if len(y) < 20:
            raise ValueError("Need at least 20 rows with a numeric target for regression.")
        om = iqr_outlier_mask(y)
        info['outliers_found'] = int(om.sum())
        if remove_outliers and om.any():
            info['X_out'] = coerce_frame(d.loc[om, keep])
            info['y_out'] = y[om].to_numpy()
            info['outliers_removed'] = int(om.sum())
            d, y = d[~om], y[~om]
    else:
        vc = y_raw.value_counts()
        rare = vc[vc < 2].index
        if len(rare):
            info['rare_dropped'] = int(y_raw.isin(rare).sum())
            d = d[~y_raw.isin(rare)]
            y_raw = d[target_col]
        classes = _sorted_classes(list(pd.unique(y_raw)))
        if len(classes) > 50:
            raise ValueError(f"Target has {len(classes)} distinct classes — too many for classification. "
                             f"Choose 'Regression' or a different target.")
        if len(classes) < 2:
            raise ValueError("Target needs at least 2 classes for classification.")
        t_enc = LabelEncoder()
        t_enc.classes_ = np.array([fmt_class(c) for c in classes], dtype=object)
        y = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int)
    X = coerce_frame(d[keep])
    info['features'] = keep
    return X, y, t_enc, ptype, info


def regression_scores(y_true, y_pred, tol):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = None
    return {'R²': float(r2_score(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'MAPE': mape,
            'Tol Acc': float(np.mean(err <= tol))}


def _cv_splitter(ptype, y_tr, folds):
    if ptype == 'classification':
        mc = int(pd.Series(np.asarray(y_tr)).value_counts().min())
        if mc >= folds:
            return StratifiedKFold(folds, shuffle=True, random_state=42)
    return KFold(folds, shuffle=True, random_state=42)


def train_one(name, estimator, ctx):
    """Fit one full pipeline on raw X, evaluate on the hold-out set (+ optional CV)."""
    ptype = ctx['ptype']
    t0 = time.time()
    pipe = build_pipeline(ctx['X_tr'], estimator, ptype, ctx.get('scaler', 'standard'), ctx.get('k_select'))
    pipe.fit(ctx['X_tr'], ctx['y_tr'])
    y_pred = pipe.predict(ctx['X_te'])
    cv_txt = None
    if ctx.get('cv_folds'):
        try:
            cv = _cv_splitter(ptype, ctx['y_tr'], ctx['cv_folds'])
            sc = cross_val_score(pipe, ctx['X_tr'], ctx['y_tr'], cv=cv, n_jobs=1,
                                 scoring='accuracy' if ptype == 'classification' else 'r2')
            if np.isfinite(sc).any():
                cv_txt = f"{np.nanmean(sc):.4f} ± {np.nanstd(sc):.4f}"
        except Exception:
            cv_txt = None
    y_te = np.asarray(ctx['y_te'])
    if ptype == 'classification':
        n_cls = len(np.unique(ctx['y_all']))
        avg = 'binary' if n_cls == 2 else 'weighted'
        auc = None
        try:
            if hasattr(pipe, 'predict_proba'):
                pp = pipe.predict_proba(ctx['X_te'])
                auc = float(roc_auc_score(y_te, pp[:, 1]) if n_cls == 2 else
                            roc_auc_score(y_te, pp, multi_class='ovr', average='weighted', labels=list(range(n_cls))))
        except Exception:
            auc = None
        acc = float(accuracy_score(y_te, y_pred))
        row = {'Model': name, 'Accuracy': acc,
               'Precision': float(precision_score(y_te, y_pred, average=avg, zero_division=0)),
               'Recall': float(recall_score(y_te, y_pred, average=avg, zero_division=0)),
               'F1': float(f1_score(y_te, y_pred, average=avg, zero_division=0)),
               'AUC': auc, 'CV Score': cv_txt, 'Score': acc}
    else:
        m = regression_scores(y_te, y_pred, ctx['tol'])
        r2_all = None
        if ctx.get('X_out') is not None and len(ctx['X_out']):
            try:
                yo = np.concatenate([y_te, ctx['y_out']])
                po = np.concatenate([y_pred, pipe.predict(ctx['X_out'])])
                r2_all = float(r2_score(yo, po))
            except Exception:
                r2_all = None
        row = {'Model': name, 'R²': m['R²'], 'RMSE': m['RMSE'], 'MAE': m['MAE'], 'MAPE': m['MAPE'],
               'Tol Acc': m['Tol Acc'], 'R² (+outliers)': r2_all, 'CV Score': cv_txt, 'Score': m['R²']}
    row['Time (s)'] = round(time.time() - t0, 1)
    return pipe, y_pred, row


def make_model_info(pipe, y_pred, row, ctx, target, source='ML'):
    resid_q = None
    if ctx['ptype'] == 'regression':
        resid_q = float(np.quantile(np.abs(np.asarray(ctx['y_te'], dtype=float) - np.asarray(y_pred, dtype=float)), 0.9))
    return {'model': pipe, 'pipeline': pipe, 'features': list(ctx['X_tr'].columns), 'target': target,
            'type': ctx['ptype'], 't_enc': ctx['t_enc'], 'feature_meta': build_feature_meta(ctx['X_tr']),
            'X_test': ctx['X_te'].reset_index(drop=True), 'y_test': np.asarray(ctx['y_te']),
            'y_pred': np.asarray(y_pred), 'metrics': row, 'source': source, 'resid_q90': resid_q,
            'tol': ctx.get('tol'), 'shap_pipeline': ctx.get('shap_pipeline', pipe)}


def perm_importance_df(pipe, X_te, y_te, ptype, n_repeats=5, max_rows=1500):
    X_te = X_te.reset_index(drop=True)
    y_te = np.asarray(y_te)
    if len(X_te) > max_rows:
        idx = np.random.RandomState(42).choice(len(X_te), max_rows, replace=False)
        X_te, y_te = X_te.iloc[idx], y_te[idx]
    r = permutation_importance(pipe, X_te, y_te, n_repeats=n_repeats, random_state=42, n_jobs=1,
                               scoring='accuracy' if ptype == 'classification' else 'r2')
    return (pd.DataFrame({'Feature': X_te.columns, 'Importance': r.importances_mean, 'Std': r.importances_std})
            .sort_values('Importance', ascending=False).reset_index(drop=True))


def tune_with_optuna(base_name, ptype, ctx, n_trials=15, timeout=90):
    """Light Optuna search for a boosted-tree model. Returns (estimator, best_cv_score) or (None, None)."""
    if not OPTUNA_AVAILABLE:
        return None, None
    cls = ptype == 'classification'
    is_x, is_l, is_g = 'XGBoost' in base_name, 'LightGBM' in base_name, 'Gradient Boosting' in base_name
    if not (is_x or is_l or is_g):
        return None, None
    if (is_x and not XGB_AVAILABLE) or (is_l and not LGB_AVAILABLE):
        return None, None
    X_fit, y_fit = ctx['X_tr'], ctx['y_tr']
    if len(X_fit) > 4000:                                     # tune on a subset for speed
        idx = np.random.RandomState(0).choice(len(X_fit), 4000, replace=False)
        X_fit, y_fit = X_fit.iloc[idx], np.asarray(y_fit)[idx]
    cv = _cv_splitter(ptype, y_fit, 3)

    def make(p):
        if is_x:
            C = xgb.XGBClassifier if cls else xgb.XGBRegressor
            return C(random_state=42, n_jobs=-1, verbosity=0, **p)
        if is_l:
            C = lgb.LGBMClassifier if cls else lgb.LGBMRegressor
            return C(random_state=42, n_jobs=-1, verbose=-1, **p)
        C = GradientBoostingClassifier if cls else GradientBoostingRegressor
        return C(random_state=42, **p)

    def objective(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 100, 500),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True)}
        if is_x:
            p.update(max_depth=trial.suggest_int('max_depth', 3, 8), subsample=trial.suggest_float('subsample', 0.6, 1.0),
                     colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                     reg_lambda=trial.suggest_float('reg_lambda', 0.1, 10.0, log=True))
        elif is_l:
            p.update(num_leaves=trial.suggest_int('num_leaves', 8, 64), subsample=trial.suggest_float('subsample', 0.6, 1.0),
                     colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                     reg_lambda=trial.suggest_float('reg_lambda', 0.1, 10.0, log=True), subsample_freq=1)
        else:
            p.update(max_depth=trial.suggest_int('max_depth', 2, 5), subsample=trial.suggest_float('subsample', 0.6, 1.0))
        pipe = build_pipeline(X_fit, make(p), ptype, ctx.get('scaler', 'standard'), ctx.get('k_select'))
        sc = cross_val_score(pipe, X_fit, y_fit, cv=cv, n_jobs=1, scoring='accuracy' if cls else 'r2')
        return float(np.mean(sc))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    if not study.trials:
        return None, None
    return make(study.best_params), float(study.best_value)


def align_input(info, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Bring any raw frame into the exact column set/dtypes the model was trained on."""
    X = X_raw.reindex(columns=info['features'])
    meta = info['feature_meta']
    for c in X.columns:
        if meta[c]['kind'] == 'num':
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return coerce_frame(X)


def split_data(X, y, ptype, test_size):
    strat = y if (ptype == 'classification' and pd.Series(np.asarray(y)).value_counts().min() >= 2) else None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=42)


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <div class="hero-title">🚀 ML Analytics Pro</div>
    <div class="hero-sub">Advanced Machine Learning · Real-time Insights · Production Ready</div>
    <div class="hero-badges">
        <span class="badge">v4.0 ULTRA</span>
        <span class="badge badge-green">XGBoost · LightGBM · SHAP</span>
        <span class="badge badge-orange">AutoML · AI Assistant</span>
        <span class="badge">SQL · Stats Tests</span>
    </div>
</div>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def make_sample_dataset(n=3000, seed=42):
    """Synthetic churn dataset with real signal (a good classifier reaches ~88-92 % accuracy)."""
    rng = np.random.RandomState(seed)
    s = pd.DataFrame({
        'Age': rng.randint(18, 75, n),
        'Income': rng.lognormal(10.8, 0.6, n).astype(int),
        'CreditScore': rng.randint(300, 850, n),
        'Experience': rng.randint(0, 35, n),
        'LoanAmount': rng.randint(5000, 500000, n),
        'Education': rng.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.25, 0.45, 0.22, 0.08]),
        'Department': rng.choice(['Engineering', 'Sales', 'Marketing', 'Finance', 'HR'], n),
        'Region': rng.choice(['North', 'South', 'East', 'West'], n),
        'Satisfaction': rng.randint(1, 11, n),
    })
    logit = (-0.9 * (s['Satisfaction'] - 5.5) - 0.035 * (s['CreditScore'] - 575) / 10
             + 0.9 * (s['Experience'] < 4) + 0.6 * (s['Department'] == 'Sales') - 0.9 + rng.normal(0, 0.8, n))
    s['Churned'] = (rng.rand(n) < 1 / (1 + np.exp(-1.4 * logit))).astype(int)
    for col in ['Income', 'CreditScore', 'LoanAmount']:
        s[col] = s[col].astype(float)
        s.loc[rng.rand(n) < 0.04, col] = np.nan
    return s


def commit_new_dataset(raw: pd.DataFrame, action: str):
    apply_df(raw, action)
    st.session_state.ml_results = None
    st.session_state.aml_results = None
    st.session_state.shap_values = None


with st.sidebar:
    st.markdown("### 📁 Dataset")
    uploaded = st.file_uploader("Upload file", type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                                key="main_upload", label_visibility="collapsed")
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        try:
            if st.session_state.df is None or not st.session_state.history:
                raw = read_uploaded(uploaded)
                st.session_state.upload_sig = sig
                commit_new_dataset(raw, "📁 Dataset uploaded")
                st.rerun()
            elif st.session_state.upload_sig != sig:
                st.info(f"📎 New file: **{uploaded.name}**")
                if st.button("🔄 Replace current data with this file", width='stretch', key="replace_data"):
                    raw = read_uploaded(uploaded)
                    st.session_state.upload_sig = sig
                    commit_new_dataset(raw, "📁 Dataset replaced")
                    st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.markdown("### 📊 Second Dataset")
    up2 = st.file_uploader("For merging", type=['csv', 'xlsx', 'json'], key="second_upload", label_visibility="collapsed")
    if up2 is not None:
        try:
            df2_raw = read_uploaded(up2)
            st.session_state.df2 = df2_raw
            st.success(f"✅ {df2_raw.shape[0]:,} rows loaded")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("### 🎲 Quick Start")
    if st.button("🎲 Load Sample Dataset", width='stretch', key="load_sample"):
        commit_new_dataset(make_sample_dataset(), "🎲 Sample data loaded")
        st.rerun()

    _student_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'StudentPerformanceFactors.csv') \
        if '__file__' in globals() else 'StudentPerformanceFactors.csv'
    if os.path.exists(_student_csv):
        if st.button("🎓 Load Student Performance Data", width='stretch', key="load_student"):
            commit_new_dataset(pd.read_csv(_student_csv), "🎓 Student performance data loaded")
            st.rerun()

    # Data quality score display
    if st.session_state.data_quality_score:
        score, issues = st.session_state.data_quality_score
        color = '#43E97B' if score >= 80 else '#F9AB00' if score >= 60 else '#FF4757'
        label = 'EXCELLENT' if score >= 85 else 'GOOD' if score >= 70 else 'FAIR' if score >= 55 else 'POOR'
        st.markdown(f"""
        <div class="quality-ring">
            <div class="quality-label">Data Quality</div>
            <div class="quality-score" style="color: {color};">{score:.0f}</div>
            <div style="font-size:11px;color:{color};font-weight:700;letter-spacing:1px;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
        if issues:
            with st.expander("⚠️ Issues found", expanded=False):
                for iss in issues:
                    st.markdown(f"<div style='font-size:12px;color:#fce28a;padding:4px 0'>{iss}</div>", unsafe_allow_html=True)

    # History in sidebar
    if st.session_state.history:
        with st.expander(f"📜 History ({len(st.session_state.history)})", expanded=False):
            _recent = list(reversed(st.session_state.history[-10:]))
            for i, h in enumerate(_recent):
                if st.button(f"↩️ {h['action'][:30]}", key=f"hist_{i}", width='stretch'):
                    st.session_state.df = h['df'].copy()
                    st.session_state.data_quality_score = calculate_data_quality_score(st.session_state.df)
                    st.session_state.auto_insights = auto_generate_insights(st.session_state.df)
                    st.rerun()

    st.markdown("---")
    _env = [("XGBoost", XGB_AVAILABLE), ("LightGBM", LGB_AVAILABLE), ("SHAP", SHAP_AVAILABLE), ("Optuna", OPTUNA_AVAILABLE)]
    st.caption("Engines: " + " · ".join(f"{'🟢' if ok else '⚪'} {n}" for n, ok in _env))
    if not all(ok for _, ok in _env):
        st.caption("Missing ones are optional → `pip install -r requirements.txt`")




if st.session_state.df is None:
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:80px 40px;margin-top:20px;">
        <div style="font-size:72px;margin-bottom:24px;">🤖</div>
        <h2 style="font-size:28px;margin-bottom:16px;">Upload your dataset to begin</h2>
        <p style="color:var(--text-muted);font-size:16px;line-height:1.8;max-width:500px;margin:0 auto;">
            Supports CSV, Excel, JSON, Parquet · AI-powered insights · 
            Advanced ML with XGBoost, LightGBM · AutoML · Real-time visualization
        </p>
        <div style="margin-top:28px;display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
            <span class="badge">📊 Data Cleaning</span>
            <span class="badge badge-green">🤖 AutoML</span>
            <span class="badge badge-orange">📈 10+ Viz Types</span>
            <span class="badge">🎯 Prediction Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df
show_change_summary()

# AI Insights
if st.session_state.auto_insights:
    with st.expander("🧠 AI-Powered Insights", expanded=True):
        cols_ins = st.columns(2)
        for i, insight in enumerate(st.session_state.auto_insights[:6]):
            with cols_ins[i % 2]:
                st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "🔧 Clean", "📈 Visualize", "🤖 ML Models", "🎯 Predict", "🧬 Features", "⚙️ Advanced", "🏆 AutoML", "🔬 Stats Tests", "🗄️ SQL Query", "🧠 SHAP", "💬 AI Assistant", "💾 Export"])



# ═══════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 📊 Dataset Overview")
    n_rows_, n_cols_ = df.shape
    missing_pct = df.isnull().sum().sum() / max(n_rows_ * n_cols_, 1) * 100
    dup_n_ = int(df.duplicated().sum())
    dup_pct = dup_n_ / max(n_rows_, 1) * 100
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    num_cols_n = len(get_num_cols(df))

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "📝", "ROWS", f"{n_rows_:,}", "Total records"),
        (c2, "🔢", "COLUMNS", f"{n_cols_:,}", f"{num_cols_n} numeric"),
        (c3, "❓", "MISSING", f"{missing_pct:.1f}%", '✅ Clean' if missing_pct < 5 else '⚠️ Needs attention'),
        (c4, "💾", "MEMORY", f"{mem_mb:.1f} MB", "In memory"),
        (c5, "🔄", "DUPES", f"{dup_pct:.1f}%", f"{dup_n_:,} rows"),
    ]
    for col, icon, label, val, sub in metrics:
        with col:
            mc = 'var(--success)' if (label == 'MISSING' and missing_pct < 5) or (label == 'DUPES' and dup_pct == 0) else 'var(--text)'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{mc}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### 👀 Data Preview")
        if len(df) > 5:
            n_prev = st.slider("Rows to show", 5, min(200, len(df)), min(20, len(df)), key="prev_rows")
        else:
            n_prev = len(df)
        st.dataframe(df.head(n_prev), width='stretch', height=400)
    with c2:
        st.markdown("### 🗂️ Column Summary")
        info = pd.DataFrame({'Column': df.columns.astype(str), 'Type': df.dtypes.astype(str).values,
                             'Non-Null': df.notnull().sum().values,
                             'Missing %': (df.isnull().sum() / max(len(df), 1) * 100).round(1).values,
                             'Unique': df.nunique().values})
        st.dataframe(info, width='stretch', height=400, hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📈 Statistics")
        nc_ov = df[get_num_cols(df)]
        if nc_ov.shape[1]:
            s = nc_ov.describe().T.round(3)
            s['cv%'] = (s['std'] / s['mean'].abs().replace(0, np.nan) * 100).round(1)
            s['range'] = s['max'] - s['min']
            st.dataframe(s, width='stretch', height=350)
        else:
            st.info("No numeric columns.")
    with c2:
        st.markdown("### 🗃️ Data Types")
        tc = df.dtypes.astype(str).value_counts()
        fig = go.Figure(data=[go.Pie(labels=tc.index.tolist(), values=tc.values.tolist(), hole=0.55,
                                     marker=dict(colors=['#6C63FF', '#FF6584', '#43E97B', '#38F9D7', '#F9AB00']),
                                     textfont=dict(size=13, color='white'))])
        fig.update_layout(**plotly_dark_layout(height=350, showlegend=True))
        st.plotly_chart(fig, width='stretch', key="ov_pie")


# ═══════════════════════════════════════════════════════════
# TAB 3: VISUALIZE
# ═══════════════════════════════════════════════════════════
def _kde_curve(data, n=200):
    try:
        if data.nunique() < 3:
            return None, None
        kde = gaussian_kde(data)
        xr = np.linspace(data.min(), data.max(), n)
        return xr, kde(xr)
    except Exception:
        return None, None


with tabs[2]:
    st.markdown("## 📈 Advanced Visualizations")
    viz_type = st.selectbox("Chart Type", ["Distribution Analysis", "Correlation Matrix", "Scatter Plot",
                                           "Box / Violin Plot", "3D Scatter", "Time Series / Line",
                                           "Categorical Analysis", "Pair Plot Heatmap"], key="viz_type")
    nc = get_num_cols(df)
    cc = get_cat_cols(df)

    if viz_type == "Distribution Analysis":
        if nc:
            col = st.selectbox("Column", nc, key="dist_col")
            data = df[col].dropna()
            if len(data) < 3:
                st.warning("Not enough data in this column.")
            else:
                fig = make_subplots(rows=2, cols=2, subplot_titles=("Histogram + KDE", "Box Plot", "Q-Q Plot", "ECDF"))
                fig.add_trace(go.Histogram(x=data, nbinsx=50, marker_color='#6C63FF', opacity=0.8, name='Hist'), row=1, col=1)
                xr, yk = _kde_curve(data)
                if xr is not None:
                    hist_scale = len(data) * (data.max() - data.min()) / 50
                    fig.add_trace(go.Scatter(x=xr, y=yk * hist_scale, mode='lines', line=dict(color='#43E97B', width=2.5), name='KDE'), row=1, col=1)
                fig.add_trace(go.Box(y=data, marker_color='#FF6584', boxmean='sd', name='Box'), row=1, col=2)
                qq = stats.probplot(data, dist="norm")
                fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='#38F9D7', size=4), name='Q-Q'), row=2, col=1)
                fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0] * qq[0][0], mode='lines', line=dict(color='#F9AB00', dash='dash'), name='Ref'), row=2, col=1)
                xs = np.sort(data.to_numpy())
                fig.add_trace(go.Scatter(x=xs, y=np.arange(1, len(xs) + 1) / len(xs), mode='lines', line=dict(color='#6C63FF', width=2), name='ECDF'), row=2, col=2)
                fig.update_layout(**plotly_dark_layout(height=650, showlegend=False))
                st.plotly_chart(fig, width='stretch', key="viz_dist")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Mean", f"{data.mean():.3f}"); st.metric("Median", f"{data.median():.3f}")
                with c2:
                    st.metric("Std Dev", f"{data.std():.3f}"); st.metric("IQR", f"{data.quantile(0.75) - data.quantile(0.25):.3f}")
                with c3:
                    st.metric("Skewness", f"{data.skew():.3f}"); st.metric("Kurtosis", f"{data.kurtosis():.3f}")
                with c4:
                    try:
                        _, p = shapiro(data.iloc[:5000])
                        st.metric("Shapiro p-val", f"{p:.4f}")
                        st.markdown('<div class="alert-success">✅ Likely Normal</div>' if p > 0.05
                                    else '<div class="alert-warning">⚠️ Not Normal</div>', unsafe_allow_html=True)
                    except Exception:
                        st.info("Shapiro test unavailable")
        else:
            st.info("No numeric columns.")

    elif viz_type == "Correlation Matrix":
        if len(nc) > 1:
            c1, c2 = st.columns([2, 1])
            with c1: meth = st.selectbox("Method", ["Pearson", "Spearman", "Kendall"], key="corr_meth")
            with c2: show_all = st.checkbox("Show full matrix (not triangular)", False, key="corr_full")
            corr = df[nc].corr(method=meth.lower())
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=0) if not show_all else np.zeros(corr.shape, dtype=bool)
            z = np.where(mask, np.nan, corr.values)
            txt = np.where(mask, '', np.round(corr.values, 2).astype(str))
            fig = go.Figure(go.Heatmap(z=z, x=corr.columns.astype(str), y=corr.columns.astype(str), colorscale='RdBu_r', zmid=0,
                                       text=txt, texttemplate='%{text}', textfont=dict(size=9), colorbar=dict(title="r")))
            fig.update_layout(**plotly_dark_layout(height=620, title=f"{meth} Correlation Matrix"))
            fig.update_yaxes(autorange='reversed')
            st.plotly_chart(fig, width='stretch', key="viz_corr")
            pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j]) for i in range(len(nc)) for j in range(i + 1, len(nc))]
            pairs = [p for p in pairs if pd.notna(p[2])]
            top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:8]
            st.markdown("### 🔗 Strongest Correlations")
            st.dataframe(pd.DataFrame(top_pairs, columns=['Feature A', 'Feature B', 'Correlation']).round(4), width='stretch', hide_index=True)
        else:
            st.info("Need at least 2 numeric columns.")

    elif viz_type == "Scatter Plot":
        if len(nc) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1: xc = st.selectbox("X-axis", nc, key="scat_x")
            with c2: yc = st.selectbox("Y-axis", [c for c in nc if c != xc], key="scat_y")
            with c3: color_c = st.selectbox("Color by", ["None"] + df.columns.tolist(), key="scat_color")
            sub = df.dropna(subset=[xc, yc])
            if len(sub) > 8000:
                sub = sub.sample(8000, random_state=42)
            fig = px.scatter(sub, x=xc, y=yc, color=color_c if color_c != "None" else None, opacity=0.65,
                             marginal_x="histogram", marginal_y="violin", title=f"{xc} vs {yc}")
            if len(sub) > 2 and sub[xc].nunique() > 1:                       # numpy trendline — no statsmodels needed
                m_, b_ = np.polyfit(sub[xc], sub[yc], 1)
                xs_ = np.array([sub[xc].min(), sub[xc].max()])
                fig.add_trace(go.Scatter(x=xs_, y=m_ * xs_ + b_, mode='lines', name='OLS trend',
                                         line=dict(color='#F9AB00', width=3)), row=2, col=1)
            fig.update_layout(**plotly_dark_layout(height=620))
            st.plotly_chart(fig, width='stretch', key="viz_scatter")
            cv_ = sub[[xc, yc]].corr().iloc[0, 1]
            st.metric("Pearson Correlation", f"{cv_:.4f}", f"{'Strong' if abs(cv_) > 0.7 else 'Moderate' if abs(cv_) > 0.4 else 'Weak'} correlation", delta_color="off")
        else:
            st.info("Need at least 2 numeric columns.")

    elif viz_type == "Box / Violin Plot":
        if nc:
            c1, c2, c3 = st.columns(3)
            with c1: yc = st.selectbox("Value column", nc, key="bv_y")
            with c2: grp = st.selectbox("Group by", ["None"] + cc, key="bv_grp")
            with c3: chart_t = st.radio("Type", ["Box", "Violin", "Both"], key="bv_type", horizontal=True)
            gc = grp if grp != "None" else None
            if chart_t == "Box":
                fig = px.box(df, y=yc, x=gc, color=gc, points="outliers")
            elif chart_t == "Violin":
                fig = px.violin(df, y=yc, x=gc, color=gc, box=True)
            else:
                fig = go.Figure()
                groups = sorted(df[gc].dropna().unique().tolist(), key=str) if gc else [None]
                colors = px.colors.qualitative.Plotly
                for i, g in enumerate(groups):
                    gdata = df.loc[df[gc] == g, yc].dropna() if gc else df[yc].dropna()
                    fig.add_trace(go.Violin(y=gdata, name=str(g) if g is not None else yc, box_visible=True,
                                            fillcolor=colors[i % len(colors)], opacity=0.75, line_color='white'))
            fig.update_layout(**plotly_dark_layout(height=500))
            st.plotly_chart(fig, width='stretch', key="viz_box")
        else:
            st.info("No numeric columns.")

    elif viz_type == "3D Scatter":
        if len(nc) >= 3:
            c1, c2, c3 = st.columns(3)
            with c1: xc = st.selectbox("X", nc, key="3d_x")
            with c2: yc = st.selectbox("Y", [c for c in nc if c != xc], key="3d_y")
            with c3: zc = st.selectbox("Z", [c for c in nc if c not in [xc, yc]], key="3d_z")
            color_c = st.selectbox("Color by", ["None"] + df.columns.tolist(), key="3d_color")
            sub = df.dropna(subset=[xc, yc, zc])
            if len(sub) > 5000: sub = sub.sample(5000, random_state=42)
            fig = px.scatter_3d(sub, x=xc, y=yc, z=zc, color=color_c if color_c != "None" else None, opacity=0.65,
                                title=f"3D: {xc} × {yc} × {zc}", height=650)
            fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)',
                                         xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                                         yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                                         zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)')),
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E8E9F0'))
            st.plotly_chart(fig, width='stretch', key="viz_3d")
        else:
            st.info("Need at least 3 numeric columns.")

    elif viz_type == "Categorical Analysis":
        if cc:
            c1, c2 = st.columns(2)
            with c1: cat_col = st.selectbox("Category", cc, key="cat_col")
            with c2: num_col = st.selectbox("Metric (optional)", ["Count"] + nc, key="cat_num")
            if num_col == "Count":
                vc = df[cat_col].value_counts().head(25)
                fig = px.bar(x=vc.index.astype(str), y=vc.values, color=vc.values, color_continuous_scale='Viridis',
                             labels={'x': cat_col, 'y': 'Count'}, title=f"Distribution of {cat_col}")
            else:
                agg = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(25)
                fig = px.bar(x=agg.index.astype(str), y=agg.values, color=agg.values, color_continuous_scale='Plasma',
                             labels={'x': cat_col, 'y': f'Mean {num_col}'}, title=f"Mean {num_col} by {cat_col}")
            fig.update_layout(**plotly_dark_layout(height=450, coloraxis_showscale=False))
            st.plotly_chart(fig, width='stretch', key="viz_cat")
        else:
            st.info("No categorical columns found.")

    elif viz_type == "Time Series / Line":
        date_cols = get_dt_cols(df) + [c for c in df.columns if ('date' in str(c).lower() or 'time' in str(c).lower()) and c not in get_dt_cols(df)]
        if nc:
            c1, c2 = st.columns(2)
            with c1: x_c = st.selectbox("X axis (time/index)", ["Index"] + date_cols + [c for c in df.columns if c not in date_cols], key="ts_x")
            with c2: y_cols = st.multiselect("Y columns", nc, default=nc[:min(3, len(nc))], key="ts_y")
            if y_cols:
                tmp = df.copy()
                if x_c != "Index":
                    tmp = tmp.sort_values(x_c)
                xdata = tmp.index if x_c == "Index" else tmp[x_c]
                if len(tmp) > 5000:
                    step = len(tmp) // 5000 + 1
                    tmp, xdata = tmp.iloc[::step], xdata[::step]
                fig = go.Figure()
                clrs = ['#6C63FF', '#FF6584', '#43E97B', '#38F9D7', '#F9AB00']
                for i, yc in enumerate(y_cols):
                    fig.add_trace(go.Scatter(x=xdata, y=tmp[yc], mode='lines', name=str(yc), line=dict(color=clrs[i % len(clrs)], width=2)))
                fig.update_layout(**plotly_dark_layout(height=500, title="Time Series"))
                st.plotly_chart(fig, width='stretch', key="viz_ts")
        else:
            st.info("No numeric columns.")

    elif viz_type == "Pair Plot Heatmap":
        if len(nc) >= 2:
            sel = st.multiselect("Select columns (max 6)", nc, default=nc[:min(5, len(nc))], key="pair_cols")[:6]
            if len(sel) >= 2:
                sub = df[sel].dropna()
                if len(sub) > 1200: sub = sub.sample(1200, random_state=42)
                fig = make_subplots(rows=len(sel), cols=len(sel))
                for i, r in enumerate(sel):
                    for j, c in enumerate(sel):
                        if i == j:
                            xr, yk = _kde_curve(sub[r], 100)
                            if xr is not None:
                                fig.add_trace(go.Scatter(x=xr, y=yk, mode='lines', line=dict(color='#6C63FF', width=2), showlegend=False), row=i + 1, col=j + 1)
                        else:
                            fig.add_trace(go.Scattergl(x=sub[c], y=sub[r], mode='markers', marker=dict(size=3, color='#FF6584', opacity=0.4), showlegend=False), row=i + 1, col=j + 1)
                fig.update_layout(**plotly_dark_layout(height=max(500, 170 * len(sel)), title="Pair Plot Matrix"))
                st.plotly_chart(fig, width='stretch', key="viz_pair")
        else:
            st.info("Need at least 2 numeric columns.")




# ═══════════════════════════════════════════════════════════
# TAB 2: CLEANING
# ═══════════════════════════════════════════════════════════
def outlier_bounds(s: pd.Series, method: str):
    """Return (lo, hi) for bound-based methods, or None (Isolation Forest)."""
    s = s.dropna()
    if s.empty:
        return None
    if method.startswith("IQR"):
        k = 1.5 if "1.5" in method else 3.0
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (q1 - k * iqr, q3 + k * iqr)
    if method.startswith("Z-Score"):
        k = 2.0 if "2σ" in method else 3.0
        sd = s.std()
        return (s.mean() - k * sd, s.mean() + k * sd)
    if method.startswith("Modified"):
        med = s.median()
        mad = np.median(np.abs(s - med))
        if mad == 0:
            mad = s.std() * 0.6745 if s.std() else 0
        if mad == 0:
            return None
        return (med - 3.5 * mad / 0.6745, med + 3.5 * mad / 0.6745)
    return None


with tabs[1]:
    st.markdown("## 🔧 Advanced Data Cleaning")

    # ── One-click smart clean ──
    with st.expander("✨ Smart Auto-Clean (one click)", expanded=True):
        st.caption("Removes duplicate rows and fills gaps: median for numeric columns, most-frequent value for text columns.")
        a1, a2 = st.columns(2)
        with a1: sc_dups = st.checkbox("Remove duplicate rows", True, key="sc_dups")
        with a2: sc_fill = st.checkbox("Fill missing values", True, key="sc_fill")
        if st.button("✨ Run Smart Auto-Clean", key="smart_clean", width='stretch'):
            dc, notes = df.copy(), []
            if sc_dups:
                n0 = len(dc); dc = dc.drop_duplicates(); notes.append(f"{n0 - len(dc)} dups removed")
            if sc_fill:
                filled = 0
                for c in dc.columns:
                    if dc[c].isna().any():
                        if pd.api.types.is_numeric_dtype(dc[c]) and not pd.api.types.is_bool_dtype(dc[c]):
                            if dc[c].notna().any():
                                filled += int(dc[c].isna().sum()); dc[c] = dc[c].fillna(dc[c].median())
                        else:
                            m = dc[c].mode()
                            if len(m):
                                filled += int(dc[c].isna().sum()); dc[c] = dc[c].fillna(m.iloc[0])
                notes.append(f"{filled} values filled")
            apply_df(dc, "✨ Smart Auto-Clean: " + ", ".join(notes))
            st.rerun()

    # ── Missing Values ──
    with st.expander("❓ Missing Values", expanded=True):
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss):
            miss_df = pd.DataFrame({'Column': miss.index.astype(str), 'Missing': miss.values,
                                    'Percent': (miss.values / len(df) * 100).round(2)})
            fig = px.bar(miss_df, x='Column', y='Percent', color='Percent', color_continuous_scale='Reds', title='Missing Data %',
                         text=miss_df['Percent'].apply(lambda x: f'{x:.1f}%'))
            fig.update_traces(textposition='outside')
            fig.update_layout(**plotly_dark_layout(height=350, coloraxis_showscale=False))
            st.plotly_chart(fig, width='stretch', key="miss_fig")

            c1, c2, c3 = st.columns(3)
            with c1:
                mc = st.selectbox("Column", miss_df['Column'].tolist(), key="miss_col")
            mc_is_num = pd.api.types.is_numeric_dtype(df[mc]) and not pd.api.types.is_bool_dtype(df[mc])
            strategies = (["Drop rows", "Fill mean", "Fill median", "Fill mode", "Forward fill", "Backward fill", "KNN Imputer", "Custom value"]
                          if mc_is_num else ["Drop rows", "Fill mode", "Forward fill", "Backward fill", "Custom value"])
            with c2:
                strat = st.selectbox("Strategy", strategies, key="miss_strat")
            with c3:
                cval = st.text_input("Custom value", "", key="miss_cval") if strat == "Custom value" else None

            if st.button("🔧 Apply", key="miss_apply", width='stretch'):
                dc = df.copy()
                try:
                    if strat == "Drop rows":
                        dc = dc.dropna(subset=[mc])
                    elif strat == "Fill mean":
                        dc[mc] = dc[mc].fillna(dc[mc].mean())
                    elif strat == "Fill median":
                        dc[mc] = dc[mc].fillna(dc[mc].median())
                    elif strat == "Fill mode":
                        m = dc[mc].mode()
                        if len(m): dc[mc] = dc[mc].fillna(m.iloc[0])
                    elif strat == "Forward fill":
                        dc[mc] = dc[mc].ffill()
                    elif strat == "Backward fill":
                        dc[mc] = dc[mc].bfill()
                    elif strat == "KNN Imputer":
                        nums = get_num_cols(dc)
                        if len(nums) < 2:
                            raise ValueError("KNN imputation needs at least 2 numeric columns (it uses the other columns as neighbours).")
                        base = dc[nums]
                        keep_cols = [c for c in nums if base[c].notna().any()]
                        imputed = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(base[keep_cols]), columns=keep_cols, index=dc.index)
                        dc[mc] = imputed[mc]
                    elif strat == "Custom value":
                        if cval is None or cval == "":
                            raise ValueError("Enter a custom value first.")
                        val = float(cval) if mc_is_num else cval
                        dc[mc] = dc[mc].fillna(val)
                    apply_df(dc, f"🔧 {strat}: {mc}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown('<div class="alert-success">✅ No missing values found! Dataset is complete.</div>', unsafe_allow_html=True)

    # ── Duplicates ──
    with st.expander("🔄 Duplicates", expanded=True):
        dup_n = int(df.duplicated().sum())
        if dup_n > 0:
            st.markdown(f'<div class="alert-warning">⚠️ Found {dup_n:,} duplicate rows ({dup_n / len(df) * 100:.2f}%)</div>', unsafe_allow_html=True)
            st.dataframe(df[df.duplicated(keep=False)].head(10), width='stretch')
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Remove All Duplicates", key="rm_dups", width='stretch'):
                    apply_df(df.drop_duplicates(), f"🗑️ Removed {dup_n} duplicates")
                    st.rerun()
            with c2:
                sub = st.multiselect("Remove by subset", df.columns.tolist(), key="dup_sub")
                if sub and st.button("🗑️ Remove by Subset", key="rm_sub_dups", width='stretch'):
                    dc = df.drop_duplicates(subset=sub)
                    apply_df(dc, f"🗑️ Removed {len(df) - len(dc)} duplicates (subset)")
                    st.rerun()
        else:
            st.markdown('<div class="alert-success">✅ No duplicate rows found!</div>', unsafe_allow_html=True)

    # ── Outliers ──
    with st.expander("🎯 Outlier Detection & Treatment", expanded=True):
        nc_o = get_num_cols(df)
        if nc_o:
            c1, c2, c3 = st.columns(3)
            with c1:
                oc = st.selectbox("Column", nc_o, key="out_col")
            with c2:
                om = st.selectbox("Detection Method", ["IQR (1.5×)", "IQR (3×)", "Z-Score (2σ)", "Z-Score (3σ)",
                                                       "Isolation Forest", "Modified Z-Score"], key="out_meth")
            with c3:
                show_viz = st.checkbox("Show visualization", True, key="out_viz")
            s_o = df[oc]
            valid_o = s_o.notna()

            def _outlier_flags(series, method):
                v = series.notna()
                flags = pd.Series(False, index=series.index)
                if method == "Isolation Forest":
                    if v.sum() >= 10:
                        pred = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1).fit_predict(series[v].to_frame())
                        flags.loc[v] = (pred == -1)
                    return flags, None
                b = outlier_bounds(series, method)
                if b is None:
                    return flags, None
                flags = v & ((series < b[0]) | (series > b[1]))
                return flags, b
            flags_o, bounds_o = _outlier_flags(s_o, om)
            n_out = int(flags_o.sum())
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Min", f"{s_o.min():.2f}" if valid_o.any() else "—"); m2.metric("Max", f"{s_o.max():.2f}" if valid_o.any() else "—")
            m3.metric("Mean", f"{s_o.mean():.2f}" if valid_o.any() else "—"); m4.metric("Std", f"{s_o.std():.2f}" if valid_o.sum() > 1 else "—")
            m5.metric("Outliers found", f"{n_out:,}", f"{n_out / max(len(df), 1) * 100:.1f}% of rows", delta_color="off")

            if show_viz and valid_o.any():
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Box Plot", "Distribution"))
                fig.add_trace(go.Box(y=s_o, name=oc, marker_color='#6C63FF', boxmean=True), row=1, col=1)
                fig.add_trace(go.Histogram(x=s_o, name=oc, marker_color='#FF6584', nbinsx=60, opacity=0.8), row=1, col=2)
                if bounds_o:
                    for bv in bounds_o:
                        fig.add_vline(x=bv, line_dash='dash', line_color='#F9AB00', row=1, col=2)
                fig.update_layout(**plotly_dark_layout(height=380, showlegend=False))
                st.plotly_chart(fig, width='stretch', key="out_fig")

            can_cap = bounds_o is not None
            action = st.radio("Action", ["Remove rows", "Cap at bounds (winsorize)"] if can_cap else ["Remove rows"],
                              horizontal=True, key="out_action")
            if st.button("🎯 Apply", key="rm_out", width='stretch', disabled=(n_out == 0)):
                try:
                    dc = df.copy()
                    if action == "Remove rows":
                        dc = dc[~flags_o]
                        apply_df(dc, f"🎯 Removed {n_out} outliers ({oc})")
                    else:
                        dc[oc] = dc[oc].clip(lower=bounds_o[0], upper=bounds_o[1])
                        apply_df(dc, f"🎯 Capped {n_out} outliers ({oc})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            if n_out == 0:
                st.caption("No outliers detected with this method.")
        else:
            st.info("No numeric columns found for outlier detection.")

    # ── Encoding ──
    with st.expander("🏷️ Categorical Encoding", expanded=False):
        cat_c = get_cat_cols(df)
        if cat_c:
            c1, c2 = st.columns(2)
            with c1:
                enc_col = st.selectbox("Column", cat_c, key="enc_col")
            with c2:
                enc_type = st.selectbox("Encoding", ["Label Encoding", "Ordinal (auto order: Low<Medium<High…)", "One-Hot Encoding", "Frequency Encoding"], key="enc_type")
            uniq = df[enc_col].nunique()
            st.caption(f"Unique values: {uniq}")
            if uniq <= 25:
                vc = df[enc_col].value_counts().head(25)
                fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x': enc_col, 'y': 'Count'}, color=vc.values, color_continuous_scale='Viridis')
                fig.update_layout(**plotly_dark_layout(height=300, coloraxis_showscale=False))
                st.plotly_chart(fig, width='stretch', key="enc_fig")
            if enc_type == "One-Hot Encoding" and uniq > 50:
                st.warning(f"One-hot would create {uniq} new columns — consider Label or Frequency encoding.")
            if st.button("🏷️ Encode Column", key="enc_btn", width='stretch'):
                dc = df.copy()
                try:
                    if enc_type == "Label Encoding":
                        cats_ = sorted(dc[enc_col].dropna().astype(str).unique())
                        dc[enc_col] = dc[enc_col].astype(str).map({c: i for i, c in enumerate(cats_)}).where(dc[enc_col].notna())
                    elif enc_type.startswith("Ordinal"):
                        order = detect_ordinal(dc[enc_col].astype(object))
                        if not order:
                            raise ValueError("No known ordered scale detected for this column — use Label Encoding.")
                        dc[enc_col] = dc[enc_col].map({v: i for i, v in enumerate(order)})
                    elif enc_type == "One-Hot Encoding":
                        dc = pd.get_dummies(dc, columns=[enc_col], prefix=enc_col, dtype=int)
                    else:
                        freq = dc[enc_col].value_counts(normalize=True)
                        dc[f'{enc_col}_freq'] = dc[enc_col].map(freq)
                    apply_df(dc, f"🏷️ {enc_type.split(' (')[0]}: {enc_col}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No categorical columns detected.")

    # ── Scaling ──
    with st.expander("⚖️ Feature Scaling", expanded=False):
        nc_s = get_num_cols(df)
        if nc_s:
            st.caption("💡 The ML tabs already scale features automatically inside their pipeline — scale here only if you need scaled data for export. Don't scale your target column.")
            c1, c2 = st.columns(2)
            with c1:
                scale_cols = st.multiselect("Columns to scale", nc_s, default=nc_s[:min(5, len(nc_s))], key="scale_cols")
            with c2:
                scaler_type = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler", "QuantileTransformer", "PowerTransformer"], key="scale_type")
            if scale_cols and st.button("⚖️ Apply Scaling", key="scale_btn", width='stretch'):
                dc = df.copy()
                try:
                    scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler(),
                               "QuantileTransformer": QuantileTransformer(output_distribution='normal', n_quantiles=max(10, min(1000, len(dc)))),
                               "PowerTransformer": PowerTransformer()}
                    dc[scale_cols] = scalers[scaler_type].fit_transform(dc[scale_cols].astype(float))
                    apply_df(dc, f"⚖️ {scaler_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No numeric columns to scale.")

    # ── Auto Data Type Fixer ──
    with st.expander("🔠 Auto Data Type Fixer", expanded=False):
        st.markdown("**Automatically detects and fixes wrong data types in your dataset.**")
        suggestions = []
        for col in df.columns:
            s_all = df[col]
            is_text = (s_all.dtype == object) or pd.api.types.is_string_dtype(s_all)
            if is_text and not isinstance(s_all.dtype, pd.CategoricalDtype):
                sample = s_all.dropna().astype(str).head(200)
                if sample.empty:
                    continue
                if pd.to_numeric(sample, errors='coerce').notna().all():
                    suggestions.append({'Column': col, 'Current': str(s_all.dtype), 'Suggested': 'numeric (float/int)', 'Reason': 'Numeric values stored as text', 'fix': 'numeric'}); continue
                if sample.str.contains(r'[-/:]').mean() > 0.8 and sample.str.contains(r'\d').mean() > 0.8:
                    try:
                        if pd.to_datetime(sample, errors='coerce', format='mixed').notna().mean() > 0.95:
                            suggestions.append({'Column': col, 'Current': str(s_all.dtype), 'Suggested': 'datetime', 'Reason': 'Date/time values stored as text', 'fix': 'datetime'}); continue
                    except Exception:
                        pass
                uq = set(sample.str.lower().unique())
                if len(uq) == 2 and uq <= {'true', 'false', 'yes', 'no', 'y', 'n'}:
                    suggestions.append({'Column': col, 'Current': str(s_all.dtype), 'Suggested': 'boolean', 'Reason': 'Boolean values stored as text', 'fix': 'bool'}); continue
                nu = s_all.nunique()
                if nu / len(df) < 0.05 and nu < 50:
                    suggestions.append({'Column': col, 'Current': str(s_all.dtype), 'Suggested': 'category', 'Reason': f'Only {nu} unique values — category saves memory', 'fix': 'category'})
            elif pd.api.types.is_float_dtype(s_all) and s_all.notna().all() and len(s_all):
                if (s_all % 1 == 0).all():
                    suggestions.append({'Column': col, 'Current': str(s_all.dtype), 'Suggested': 'int64', 'Reason': 'Float column contains only whole numbers', 'fix': 'int'})

        if suggestions:
            st.dataframe(pd.DataFrame(suggestions)[['Column', 'Current', 'Suggested', 'Reason']], width='stretch', hide_index=True)
            cols_to_fix = st.multiselect("Select columns to fix", [s['Column'] for s in suggestions],
                                         default=[s['Column'] for s in suggestions], key="dtype_fix_cols")
            if st.button("🔧 Apply Type Fixes", width='stretch', key="dtype_fix_btn"):
                dc, fixed = df.copy(), []
                bmap = {'true': True, 'false': False, 'yes': True, 'no': False, 'y': True, 'n': False}
                for s in suggestions:
                    if s['Column'] not in cols_to_fix:
                        continue
                    cn = s['Column']
                    try:
                        if s['fix'] == 'numeric':    dc[cn] = pd.to_numeric(dc[cn], errors='coerce')
                        elif s['fix'] == 'datetime': dc[cn] = pd.to_datetime(dc[cn], format='mixed', errors='coerce')
                        elif s['fix'] == 'bool':     dc[cn] = dc[cn].astype(str).str.lower().map(bmap)
                        elif s['fix'] == 'category': dc[cn] = dc[cn].astype('category')
                        elif s['fix'] == 'int':      dc[cn] = dc[cn].astype('int64')
                        fixed.append(cn)
                    except Exception as ex:
                        st.warning(f"Could not fix {cn}: {ex}")
                if fixed:
                    apply_df(dc, f"🔠 Auto dtype fix: {', '.join(fixed)[:60]}")
                    st.rerun()
        else:
            st.markdown('<div class="alert-success">✅ All data types look correct! No fixes needed.</div>', unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════
# SHARED RESULT RENDERERS (used by ML Models + AutoML tabs)
# ═══════════════════════════════════════════════════════════
def render_metrics_table(rdf, ptype, score_col):
    # ── Beautiful HTML Comparison Table ──
    st.markdown("### 📊 Detailed Metrics Comparison Table")

    if ptype == 'classification':
        headers    = ['#', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Score (5-fold)']
        data_keys  = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'CV Score']
        higher_better = {'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'}
    else:
        _hmap = {'R²': 'R²', 'Tol Acc': 'Tol. Accuracy', 'RMSE': 'RMSE', 'MAE': 'MAE', 'MAPE': 'MAPE',
                 'R² (+outliers)': 'R² incl. outliers', 'CV Score': 'CV Score (5-fold)'}
        data_keys = [k for k in _hmap if k in rdf.columns and rdf[k].notna().any()]
        headers   = ['#', 'Model'] + [_hmap[k] for k in data_keys]
        higher_better = {'R²', 'Tol Acc', 'R² (+outliers)'}

    # ── Pre-compute per-column stats: sorted ranks, min, max ──
    # col_rank[col][val_str] = rank_index (0 = best)
    col_stats  = {}   # {col: {min, max, sorted_vals (best first)}}
    col_best   = {}   # {col: best_val}  (highest for HB, lowest for LB)
    col_worst  = {}   # {col: worst_val}

    for k in data_keys:
        if k == 'CV Score': continue
        vals = []
        for _, row in rdf.iterrows():
            v = row.get(k, None)
            if v is not None and v != 'N/A' and not (isinstance(v, float) and np.isnan(v)):
                try: vals.append(float(v))
                except: pass
        if not vals: continue
        is_hb = k in higher_better
        sorted_vals = sorted(vals, reverse=is_hb)   # best first
        col_stats[k] = {
            'min': min(vals), 'max': max(vals),
            'sorted': sorted_vals,
            'n': len(sorted_vals)
        }
        col_best[k]  = sorted_vals[0]
        col_worst[k] = sorted_vals[-1]

    def get_rank_idx(val, col_name):
        """Return 0-based rank index in sorted list (0=best). None if unavailable."""
        cs = col_stats.get(col_name)
        if cs is None: return None, 1
        try:
            v = float(val)
        except:
            return None, cs['n']
        # Use index in the pre-sorted list (best-first)
        try:
            idx = cs['sorted'].index(v)
        except ValueError:
            # float not exact match — find closest
            idx = min(range(cs['n']), key=lambda i: abs(cs['sorted'][i] - v))
        return idx, cs['n']

    def get_rank_pct(val, col_name):
        """0.0 = best, 1.0 = worst."""
        idx, n = get_rank_idx(val, col_name)
        if idx is None or n <= 1: return 0.0 if idx == 0 else None
        return idx / (n - 1)

    def bar_width(val, col_name):
        rp = get_rank_pct(val, col_name)
        if rp is None: return 50
        return max(5, int((1.0 - rp) * 100))   # rank1=100%, last=5%

    def rank_color(rank_pct):
        """Color based on rank position: 0=best → green, 1=worst → red."""
        if rank_pct is None: return '#6C63FF', 'rgba(108,99,255,0.10)'
        if rank_pct == 0.0:               return '#43E97B', 'rgba(67,233,123,0.13)'
        elif rank_pct < 0.5:              return '#38F9D7', 'rgba(56,249,215,0.09)'
        elif rank_pct < 1.0:              return '#F9AB00', 'rgba(249,171,0,0.09)'
        else:                             return '#FF4757', 'rgba(255,71,87,0.10)'

    def rank_label(val, col_name):
        """▲ Best label ONLY for rank-1, ▼ Worst ONLY for rank-last. Nothing else."""
        idx, n = get_rank_idx(val, col_name)
        if idx is None or n < 2: return '', ''
        if idx == 0:     return '▲ Best',  '#43E97B'
        if idx == n - 1: return '▼ Worst', '#FF4757'
        return '', ''

    def fmt_number(v):
        if abs(v) >= 1000:  return f'{v:,.1f}'
        elif abs(v) >= 100: return f'{v:.2f}'
        elif abs(v) >= 1:   return f'{v:.4f}'
        else:               return f'{v:.4f}'

    def format_cell(val, col_name):
        """Unified cell renderer — works for any value range."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '<span style="color:#444;font-size:13px">—</span>'

        # ── CV Score: special treatment ──
        if col_name == 'CV Score':
            try:
                parts  = str(val).split('±')
                mean_v = float(parts[0].strip())
                std_v  = float(parts[1].strip()) if len(parts) > 1 else None
                # Color based on absolute quality (CV mean is comparable across runs)
                if mean_v >= 0.85:   cv_c = '#43E97B'
                elif mean_v >= 0.70: cv_c = '#38F9D7'
                elif mean_v >= 0.50: cv_c = '#F9AB00'
                else:                cv_c = '#FF4757'
                std_str = f'<span style="color:#555;font-size:11px"> ± {std_v:.4f}</span>' if std_v is not None else ''
                return (f'<div style="font-family:JetBrains Mono,monospace">'
                        f'<span style="font-size:14px;font-weight:700;color:{cv_c}">{mean_v:.4f}</span>'
                        f'{std_str}</div>')
            except:
                return f'<span style="color:#aaa;font-size:12px">{val}</span>'

        # ── Numeric metric cell ──
        try:
            v        = float(val)
            rp       = get_rank_pct(val, col_name)
            bw       = bar_width(val, col_name)
            b_col, cell_bg = rank_color(rp)
            lbl_text, lbl_c = rank_label(val, col_name)
            disp     = fmt_number(v)
            lbl_html = (f'<span style="font-size:9px;font-weight:800;color:{lbl_c};'
                        f'background:rgba(255,255,255,0.07);padding:1px 6px;'
                        f'border-radius:3px;letter-spacing:0.3px">{lbl_text}</span>'
                        if lbl_text else '')
            return (f'<div style="display:flex;flex-direction:column;gap:6px">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;gap:6px">'
                    f'<span style="font-weight:700;font-size:13px;color:#E8E9F0;letter-spacing:-0.3px">{disp}</span>'
                    f'{lbl_html}'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.06);border-radius:3px;height:5px">'
                    f'<div style="width:{bw}%;background:{b_col};height:5px;border-radius:3px"></div>'
                    f'</div>'
                    f'</div>')
        except:
            return f'<span style="color:#aaa">{val}</span>'

    medals   = ['🥇', '🥈', '🥉']
    rows_html = ''
    for i, row in rdf.iterrows():
        is_best  = (i == 0)
        medal    = medals[i] if i < 3 else f'<span style="color:#666;font-size:12px">#{i+1}</span>'
        row_bg   = 'rgba(67,233,123,0.04)' if is_best else ('rgba(255,255,255,0.02)' if i % 2 == 0 else 'rgba(0,0,0,0)')
        border   = 'border-left:3px solid #43E97B;' if is_best else 'border-left:3px solid transparent;'
        best_badge = "<div style='margin-top:5px'><span style='background:rgba(67,233,123,0.15);color:#43E97B;font-size:10px;padding:2px 9px;border-radius:20px;font-weight:700;letter-spacing:0.5px'>★ BEST</span></div>" if is_best else ""
        model_td = (f'<td style="padding:14px 18px;min-width:160px">'
                    f'<div style="font-weight:700;font-size:14px;color:#E8E9F0">{esc(row["Model"])}</div>'
                    f'{best_badge}'
                    f'</td>')
        cells = f'<td style="padding:14px 16px;text-align:center;font-size:18px">{medal}</td>' + model_td
        for k in data_keys:
            val    = row.get(k, None)
            rp     = get_rank_pct(val, k) if k != 'CV Score' else None
            _, bg  = rank_color(rp) if rp is not None else ('#fff', 'transparent')
            cells += f'<td style="padding:11px 16px;background:{bg};min-width:130px;vertical-align:middle">{format_cell(val, k)}</td>'
        rows_html += f'<tr style="background:{row_bg};{border}">{cells}</tr>'

    header_cells = ''.join(
        f'<th style="padding:13px {"14px" if h=="#" else "16px"};text-align:{"center" if h=="#" else "left"};'
        f'font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#6C63FF;'
        f'white-space:nowrap;border-bottom:2px solid rgba(108,99,255,0.3)">{h}</th>'
        for h in headers
    )

    # Legend — always show score range for context
    score_vals = [float(r) for r in rdf[score_col] if r is not None]
    sc_lo, sc_hi = min(score_vals), max(score_vals)

    legend_items = (
        f'<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:11px;color:rgba(232,233,240,0.6)">'
        f'<div style="display:flex;gap:5px;align-items:center"><div style="width:9px;height:9px;background:#43E97B;border-radius:2px"></div><span>▲ Best in column</span></div>'
        f'<div style="display:flex;gap:5px;align-items:center"><div style="width:9px;height:9px;background:#38F9D7;border-radius:2px"></div><span>2nd tier</span></div>'
        f'<div style="display:flex;gap:5px;align-items:center"><div style="width:9px;height:9px;background:#F9AB00;border-radius:2px"></div><span>3rd tier</span></div>'
        f'<div style="display:flex;gap:5px;align-items:center"><div style="width:9px;height:9px;background:#FF4757;border-radius:2px"></div><span>▼ Worst in column</span></div>'
        f'<span style="opacity:0.5;font-style:italic">{score_col} range: {sc_lo:.4f} → {sc_hi:.4f}</span>'
        f'</div>'
    )

    _metrics_html = (
        f'<div style="border-radius:16px;overflow:hidden;border:1px solid rgba(108,99,255,0.2);margin:16px 0;">'
        f'<div style="background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(255,101,132,0.06));padding:16px 20px;border-bottom:1px solid rgba(255,255,255,0.06);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">'
        f'<div>'
        f'<div style="font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:700;color:#E8E9F0">Algorithm Performance Dashboard</div>'
        f'<div style="font-size:12px;color:rgba(232,233,240,0.5);margin-top:2px">{len(rdf)} models trained · sorted by best {score_col}</div>'
        f'</div>'
        f'<div style="display:flex;gap:16px;font-size:11px;color:rgba(232,233,240,0.55);flex-wrap:wrap">{legend_items}</div>'
        f'</div>'
        f'<div style="overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead style="background:rgba(0,0,0,0.3)"><tr>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
        f'</div>'
        f'<div style="background:rgba(0,0,0,0.2);padding:12px 20px;border-top:1px solid rgba(255,255,255,0.04);font-size:11px;color:rgba(232,233,240,0.4)">'
        f'💡 Bar height = relative rank within each column independently · ▲ Best / ▼ Worst labels = only the #1 and #last ranked model per metric · CV = 5-fold mean ± std'
        f'</div>'
        f'</div>'
    )
    st.markdown(_metrics_html, unsafe_allow_html=True)



def render_model_kpis(best_row, ptype, tol, outlier_note=None):
    """Headline numbers + a clear verdict against the 85–90 % goal."""
    if ptype == 'classification':
        headline = best_row['Accuracy'] * 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Best Model", best_row['Model'])
        k2.metric("🎯 Accuracy", f"{headline:.2f}%")
        k3.metric("F1-Score", f"{best_row['F1'] * 100:.2f}%")
        auc = best_row.get('AUC')
        k4.metric("AUC-ROC", f"{auc:.4f}" if auc is not None and not pd.isna(auc) else "—")
        label = "Accuracy"
    else:
        headline = best_row['R²'] * 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Best Model", best_row['Model'])
        k2.metric("🎯 Accuracy (R² = variance explained)", f"{headline:.2f}%")
        k3.metric(f"✅ Accuracy within ±{tol:.2f}", f"{best_row['Tol Acc'] * 100:.2f}%")
        k4.metric("Avg error (MAE)", f"{best_row['MAE']:.3f}")
        label = "R² accuracy"
    if headline >= 90:
        st.markdown(f'<div class="alert-success">🚀 <b>Excellent — {label} {headline:.1f}%</b> (target 85–90% achieved).</div>', unsafe_allow_html=True)
    elif headline >= 85:
        st.markdown(f'<div class="alert-success">✅ <b>Target met — {label} {headline:.1f}%</b> (≥ 85%).</div>', unsafe_allow_html=True)
    else:
        tips = ["tick <b>Remove target outliers</b>" if ptype == 'regression' else "check for class imbalance / leakage-free features",
                "run <b>🏆 AutoML → Accurate</b> (Optuna tuning + ensemble)",
                "add engineered features in <b>🧬 Features</b>", "clean missing values / outliers in <b>🔧 Clean</b>"]
        st.markdown(f'<div class="alert-warning">⚠️ {label} is {headline:.1f}% (< 85%). Try: ' + " · ".join(tips) +
                    '. Note: accuracy is capped by how much signal the data actually contains.</div>', unsafe_allow_html=True)
    if outlier_note:
        st.markdown(f'<div class="alert-info">{outlier_note}</div>', unsafe_allow_html=True)


def outlier_note_text(info, best_row):
    if not info or not info.get('outliers_removed'):
        return None
    n = info['outliers_removed']
    extra = ''
    r2_all = best_row.get('R² (+outliers)')
    if r2_all is not None and not pd.isna(r2_all):
        extra = f" Honest check: R² on test rows <b>including</b> those outliers = <b>{r2_all * 100:.1f}%</b>."
    return (f"🛡️ <b>{n:,} target outliers</b> (IQR 1.5×) were excluded from training/evaluation — they are values the "
            f"features cannot explain.{extra}")


def render_model_diagnostics(minfo, ptype, imp_df=None, key_prefix="ml"):
    if ptype == 'classification':
        c1, c2 = st.columns(2)
        classes = list(minfo['t_enc'].classes_)
        labels = list(range(len(classes)))
        with c1:
            st.markdown("### 🎯 Confusion Matrix")
            cm = confusion_matrix(minfo['y_test'], minfo['y_pred'], labels=labels)
            fig2 = go.Figure(go.Heatmap(z=cm, x=classes, y=classes, colorscale='Blues',
                                        text=cm, texttemplate='%{text}', textfont=dict(size=16)))
            fig2.update_layout(**plotly_dark_layout(height=380, xaxis_title="Predicted", yaxis_title="Actual"))
            st.plotly_chart(fig2, width='stretch', key=f"{key_prefix}_cm")
        with c2:
            st.markdown("### 📋 Classification Report")
            rep = classification_report(minfo['y_test'], minfo['y_pred'], labels=labels, target_names=classes,
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3), width='stretch', height=380)
    else:
        c1, c2 = st.columns(2)
        act = np.asarray(minfo['y_test'], dtype=float); pre = np.asarray(minfo['y_pred'], dtype=float)
        with c1:
            st.markdown("### 📈 Actual vs Predicted")
            pdf = pd.DataFrame({'Actual': act, 'Predicted': pre})
            fig2 = px.scatter(pdf, x='Actual', y='Predicted', opacity=0.5)
            mn, mx = float(min(act.min(), pre.min())), float(max(act.max(), pre.max()))
            if len(pdf) > 2 and np.ptp(act) > 0:
                _m, _b = np.polyfit(act, pre, 1)
                fig2.add_trace(go.Scatter(x=[mn, mx], y=[_m * mn + _b, _m * mx + _b], mode='lines',
                                          name='Trend (OLS)', line=dict(color='#FF6B6B', width=2)))
            fig2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Perfect',
                                      line=dict(color='#43E97B', dash='dash', width=2)))
            fig2.update_layout(**plotly_dark_layout(height=420))
            st.plotly_chart(fig2, width='stretch', key=f"{key_prefix}_avp")
        with c2:
            st.markdown("### 📉 Residual Distribution")
            fig3 = px.histogram(x=act - pre, nbins=40, color_discrete_sequence=['#6C63FF'],
                                labels={'x': 'Residual (actual − predicted)'})
            fig3.add_vline(x=0, line_dash='dash', line_color='#43E97B')
            fig3.update_layout(**plotly_dark_layout(height=420, showlegend=False))
            st.plotly_chart(fig3, width='stretch', key=f"{key_prefix}_res")
    if imp_df is not None and len(imp_df):
        st.markdown("### 🎯 Feature Importance (permutation — works for every model)")
        top = imp_df.head(15).sort_values('Importance', ascending=True)
        fig4 = px.bar(top, x='Importance', y='Feature', orientation='h', color='Importance',
                      color_continuous_scale='Viridis', title="Drop in score when the feature is shuffled")
        fig4.update_layout(**plotly_dark_layout(height=max(320, 28 * len(top) + 120), coloraxis_showscale=False))
        st.plotly_chart(fig4, width='stretch', key=f"{key_prefix}_imp")


def run_training(model_names, catalog, ctx, target, prefix='', source='ML', progress_label="Training"):
    """Train a list of models, store them in session state, return (rows, errors)."""
    rows, errors = [], []
    prog = st.progress(0.0)
    status = st.empty()
    total = max(len(model_names), 1)
    for idx, mname in enumerate(model_names):
        status.markdown(f'<div class="alert-info">⚙️ {progress_label} <b>{esc(mname)}</b> ({idx + 1}/{total})...</div>', unsafe_allow_html=True)
        try:
            pipe, y_pred, row = train_one(mname, catalog[mname](), ctx)
            rows.append(row)
            st.session_state.trained_models[prefix + mname] = make_model_info(pipe, y_pred, row, ctx, target, source)
        except Exception as e:
            errors.append(f"{mname}: {e}")
        prog.progress((idx + 1) / total)
    status.empty(); prog.empty()
    return rows, errors


# ═══════════════════════════════════════════════════════════
# TAB 4: ML MODELS
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🤖 Machine Learning")

    all_cols = df.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Need at least 2 columns (1 target + 1 feature).")
    else:
        _default_tgt = len(all_cols) - 1
        c1, c2, c3 = st.columns([2, 4, 1.4])
        with c1:
            target = st.selectbox("🎯 Target Variable", all_cols, index=_default_tgt, key="ml_tgt")
        with c2:
            _opts = [c for c in all_cols if c != target]
            feats = st.multiselect("📊 Features", _opts, default=_opts[:min(40, len(_opts))], key=f"ml_feats_{target}")
        with c3:
            ptype_override = st.selectbox("Problem type", ["Auto", "Classification", "Regression"], key="ml_ptype")

        if not feats:
            st.warning("⚠️ Select at least one feature column.")
        else:
            _guess = infer_problem_type(df[target].dropna(), ptype_override) if df[target].notna().any() else 'classification'
            _out_frac, _out_n = 0.0, 0
            if _guess == 'regression':
                _yy = pd.to_numeric(df[target], errors='coerce').dropna()
                if len(_yy):
                    _out_n = int(iqr_outlier_mask(_yy).sum()); _out_frac = _out_n / len(_yy)

            with st.expander("⚙️ Advanced Training Options", expanded=False):
                a1, a2, a3 = st.columns(3)
                with a1:
                    use_cv = st.checkbox("Cross-validation (5-fold)", True, key="use_cv")
                    test_sz = st.slider("Test Size %", 10, 40, 20, key="test_sz") / 100
                with a2:
                    n_est = st.slider("Trees / boosting rounds", 50, 600, 200, 50, key="ml_nest")
                    feat_sel_k = st.number_input("Auto feature selection — keep top K (0 = all)", 0, max(len(feats), 1) * 5, 0, key="ml_k")
                with a3:
                    scale_before = st.selectbox("Feature scaling", ["StandardScaler", "RobustScaler", "None"], key="pre_scale")
                    tol_sigma = st.slider("Tolerance for 'accuracy within ±' (× target std)", 0.05, 1.0, 0.25, 0.05, key="ml_tol",
                                          help="Regression only: a prediction counts as correct when it is within ± this many standard deviations of the true value.")
                if _guess == 'regression':
                    rm_out = st.checkbox(
                        f"🛡️ Remove target outliers before training (IQR 1.5×) — {_out_n:,} rows ({_out_frac * 100:.1f}%) detected",
                        value=bool(0 < _out_frac <= 0.05), key=f"ml_rmout_{target}",
                        help="Rows whose target value is far outside the normal range are usually noise the features cannot explain. "
                             "The honest R² including them is still reported.")
                else:
                    rm_out = False

            ready = True
            try:
                X, y, t_enc, ptype, pinfo = prepare_ml_data(df, target, feats, ptype_override, rm_out)
            except Exception as e:
                ready = False
                st.error(f"Data preparation error: {e}")

            if ready:
                extra = ''
                if ptype == 'classification':
                    extra = f"· Classes: {len(t_enc.classes_)}"
                else:
                    extra = f"· Target range: [{float(y.min()):.2f}, {float(y.max()):.2f}]"
                st.markdown(f"""
                <div class="alert-info">
                    <strong>🔍 Problem Type: {ptype.upper()}</strong> · Target: <code>{esc(target)}</code> ·
                    Features: {len(pinfo['features'])} · Samples: {len(X):,} {extra}
                </div>
                """, unsafe_allow_html=True)
                if pinfo['dropped_features']:
                    st.caption("Auto-dropped: " + ", ".join(f"{c} ({why})" for c, why in pinfo['dropped_features']))
                if pinfo['rare_dropped']:
                    st.caption(f"Removed {pinfo['rare_dropped']} rows whose class has only 1 sample.")

                catalog = get_model_catalog(ptype, n_est=n_est)
                sel_models = st.multiselect("📋 Select Models to Train", list(catalog.keys()),
                                            default=default_model_names(ptype, catalog), key=f"sel_models_{ptype}")

                if st.button("🚀 Train Selected Models", key="train_btn", width='stretch', type="primary"):
                    if not sel_models:
                        st.warning("Select at least one model.")
                    else:
                        X_tr, X_te, y_tr, y_te = split_data(X, y, ptype, test_sz)
                        tol = tol_sigma * float(np.std(np.asarray(y_tr, dtype=float))) if ptype == 'regression' else None
                        ctx = dict(ptype=ptype, X_tr=X_tr, y_tr=np.asarray(y_tr), X_te=X_te, y_te=np.asarray(y_te), y_all=np.asarray(y),
                                   t_enc=t_enc, scaler={'StandardScaler': 'standard', 'RobustScaler': 'robust', 'None': 'none'}[scale_before],
                                   k_select=int(feat_sel_k) or None, cv_folds=5 if use_cv else 0, tol=tol,
                                   X_out=pinfo['X_out'], y_out=pinfo['y_out'])
                        rows, errs = run_training(sel_models, catalog, ctx, target)
                        for e_ in errs:
                            st.error(f"❌ {e_}")
                        if rows:
                            rdf = pd.DataFrame(rows).sort_values('Score', ascending=False).reset_index(drop=True)
                            best_name = rdf.iloc[0]['Model']
                            st.session_state.best_model = best_name
                            bi = st.session_state.trained_models[best_name]
                            imp_df = None
                            try:
                                with st.spinner("Computing feature importance..."):
                                    imp_df = perm_importance_df(bi['pipeline'], bi['X_test'], bi['y_test'], ptype)
                                st.session_state.feature_importance = {'features': imp_df['Feature'].tolist(),
                                                                       'importance': imp_df['Importance'].to_numpy()}
                            except Exception:
                                imp_df = None
                            st.session_state.ml_results = {
                                'rdf': rdf, 'ptype': ptype, 'best': best_name, 'target': target, 'tol': tol,
                                'imp_df': imp_df, 'info': {k: v for k, v in pinfo.items() if k not in ('X_out', 'y_out')},
                                'n_train': len(X_tr), 'n_test': len(X_te)}
                            st.session_state.shap_values = None

                res = st.session_state.get('ml_results')
                if res:
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown(f"### 🏆 Results — target `{esc(res['target'])}` ({res['ptype']}) · "
                                f"{res['n_train']:,} train / {res['n_test']:,} test rows")
                    rdf = res['rdf']
                    best_row = rdf.iloc[0].to_dict()
                    render_model_kpis(best_row, res['ptype'], res['tol'], outlier_note_text(res['info'], best_row))

                    score_col = 'Accuracy' if res['ptype'] == 'classification' else 'R²'
                    fig = px.bar(rdf, x='Model', y=score_col, color=score_col, color_continuous_scale='Viridis',
                                 text=rdf[score_col].apply(lambda x: f'{float(x):.4f}'), title=f"Model Performance — {score_col}")
                    fig.update_traces(textposition='outside', textfont_size=11)
                    fig.update_layout(**plotly_dark_layout(height=420, coloraxis_showscale=False))
                    st.plotly_chart(fig, width='stretch', key="ml_bar")

                    st.markdown("### 📊 Detailed Metrics Comparison Table")
                    render_metrics_table(rdf, res['ptype'], score_col)

                    if res['best'] in st.session_state.trained_models:
                        render_model_diagnostics(st.session_state.trained_models[res['best']], res['ptype'], res['imp_df'], "ml")




# ═══════════════════════════════════════════════════════════
# TAB 5: PREDICT
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🎯 Prediction Engine")

    if not st.session_state.trained_models:
        st.markdown('<div class="alert-warning">⚠️ No trained models found. Go to the <b>ML Models</b> or <b>AutoML</b> tab first.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-info">
            💡 <b>Tip:</b> You can still <b>download your cleaned data</b> anytime from the <b>💾 Export</b> tab — no model training needed!
        </div>
        """, unsafe_allow_html=True)
    else:
        _names = list(st.session_state.trained_models.keys())
        _best = st.session_state.best_model
        _idx = _names.index(_best) if _best in _names else 0
        mname = st.selectbox("Select Model", _names, index=_idx, key="pred_m")
        minfo = st.session_state.trained_models[mname]
        pipe, f_cols, p_type, p_tenc, meta = minfo['pipeline'], minfo['features'], minfo['type'], minfo['t_enc'], minfo['feature_meta']
        _mt = minfo.get('metrics', {})
        _score_txt = ''
        if _mt:
            _score_txt = (f" · Accuracy {_mt['Accuracy'] * 100:.1f}%" if p_type == 'classification'
                          else f" · R² {_mt['R²'] * 100:.1f}%")
        st.markdown(f"""
        <div class="alert-info">🤖 <b>{esc(mname)}</b> · Type: {p_type.title()} · Target: <code>{esc(minfo['target'])}</code> ·
        Features: {len(f_cols)}{_score_txt}</div>
        """, unsafe_allow_html=True)

        mode = st.radio("Mode", ["Single Prediction", "Batch Prediction (CSV / Excel)"], horizontal=True, key="pred_mode")

        if mode == "Single Prediction":
            st.markdown("### 📝 Enter Feature Values")
            input_data = {}
            cols4 = st.columns(4)
            for i, col in enumerate(f_cols):
                m = meta[col]
                with cols4[i % 4]:
                    wkey = f"inp_{mname}_{col}"
                    if m['kind'] == 'num':
                        rng_txt = f"Seen in training: {m['min']:g} – {m['max']:g}"
                        if m['int']:
                            input_data[col] = st.number_input(col, value=int(round(m['default'])), step=1, key=wkey, help=rng_txt)
                        else:
                            span = m['max'] - m['min']
                            input_data[col] = st.number_input(col, value=float(m['default']), step=float(span / 100) if span > 0 else 0.1,
                                                              format="%.3f", key=wkey, help=rng_txt)
                    else:
                        opts = m['options'] or ['']
                        di = opts.index(m['default']) if m['default'] in opts else 0
                        input_data[col] = st.selectbox(col, opts, index=di, key=wkey)

            if st.button("🔮 Predict", key="pred_single", width='stretch', type="primary"):
                try:
                    inp_df = align_input(minfo, pd.DataFrame([input_data]))
                    pred = pipe.predict(inp_df)[0]
                    if p_type == 'classification':
                        pred_show = p_tenc.classes_[int(pred)]
                        if hasattr(pipe, 'predict_proba'):
                            proba = pipe.predict_proba(inp_df)[0]
                            conf = float(np.max(proba) * 100)
                            st.markdown(f"""
                            <div class="pred-result">
                                <div style="font-size:16px;color:var(--text-muted);margin-bottom:12px;">PREDICTION RESULT</div>
                                <div class="pred-value">{esc(pred_show)}</div>
                                <div class="pred-conf">⚡ Confidence: {conf:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            pf = pd.DataFrame({'Class': list(p_tenc.classes_), 'Probability': proba * 100}).sort_values('Probability', ascending=False)
                            fig = px.bar(pf, x='Class', y='Probability', color='Probability', color_continuous_scale='Viridis',
                                         text=pf['Probability'].apply(lambda x: f'{x:.2f}%'))
                            fig.update_traces(textposition='outside')
                            fig.update_layout(**plotly_dark_layout(height=380, coloraxis_showscale=False))
                            st.plotly_chart(fig, width='stretch', key="pred_proba")
                        else:
                            st.markdown(f'<div class="pred-result"><div class="pred-value">{esc(pred_show)}</div></div>', unsafe_allow_html=True)
                    else:
                        band = ''
                        if minfo.get('resid_q90') is not None:
                            band = (f'<div class="pred-conf">Likely range (90% of test errors): '
                                    f'{pred - minfo["resid_q90"]:.2f} – {pred + minfo["resid_q90"]:.2f}</div>')
                        st.markdown(f"""
                        <div class="pred-result">
                            <div style="font-size:16px;color:var(--text-muted);margin-bottom:12px;">PREDICTION RESULT · {esc(minfo['target'])}</div>
                            <div class="pred-value">{pred:.2f}</div>{band}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        else:
            st.markdown("### 📊 Batch Prediction")
            up_pred = st.file_uploader("Upload CSV / Excel", type=['csv', 'xlsx'], key="batch_pred")
            if up_pred is not None:
                try:
                    bdf = read_uploaded(up_pred)
                except Exception as e:
                    bdf = None
                    st.error(f"Could not read file: {e}")
                if bdf is not None:
                    st.caption(f"Loaded: {len(bdf):,} rows × {bdf.shape[1]} columns")
                    st.dataframe(bdf.head(10), width='stretch')
                    missing = [c for c in f_cols if c not in bdf.columns]
                    if missing:
                        st.warning(f"Missing columns will be imputed: {missing}")
                    if st.button("🔮 Predict All", width='stretch', type="primary", key="batch_btn"):
                        try:
                            Xb = align_input(minfo, bdf)
                            preds = pipe.predict(Xb)
                            out = bdf.copy()
                            if p_type == 'classification':
                                out['Prediction'] = p_tenc.classes_[preds.astype(int)]
                                if hasattr(pipe, 'predict_proba'):
                                    out['Confidence_%'] = (pipe.predict_proba(Xb).max(axis=1) * 100).round(2)
                            else:
                                out['Prediction'] = np.round(preds, 4)
                            st.success(f"✅ Predicted {len(out):,} samples!")
                            st.dataframe(out, width='stretch')
                            download_button(out, "csv", "📥 Download Predictions", "dl_pred")
                        except Exception as e:
                            st.error(f"Error: {e}")




# ═══════════════════════════════════════════════════════════
# TAB 6: FEATURES
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 🧬 Feature Engineering & Analysis")

    # ── Feature importance & selection (documented in README) ──
    with st.expander("🏅 Feature Importance & Selection", expanded=True):
        if len(df.columns) < 2:
            st.info("Need at least 2 columns.")
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: fi_tgt = st.selectbox("Target", df.columns.tolist(), index=len(df.columns) - 1, key="fi_tgt")
            with c2: fi_method = st.selectbox("Method", ["Mutual Information", "F-test (SelectKBest)", "Random Forest"], key="fi_method")
            with c3: fi_k = st.number_input("Show top K", 3, 100, 15, key="fi_k")
            if st.button("🏅 Rank Features", key="fi_btn", width='stretch'):
                try:
                    with st.spinner("Ranking features..."):
                        Xf, yf, _, pt, _ = prepare_ml_data(df, fi_tgt, [c for c in df.columns if c != fi_tgt])
                        prep_f = make_preprocessor(Xf, 'standard')
                        Xt_f = prep_f.fit_transform(Xf)
                        names_f = list(prep_f.get_feature_names_out())
                        if fi_method == "Mutual Information":
                            sc_ = (mutual_info_classif if pt == 'classification' else mutual_info_regression)(Xt_f, yf, random_state=42)
                        elif fi_method.startswith("F-test"):
                            sc_ = (f_classif if pt == 'classification' else f_regression)(Xt_f, yf)[0]
                        else:
                            rf_ = (RandomForestClassifier if pt == 'classification' else RandomForestRegressor)(n_estimators=150, random_state=42, n_jobs=-1)
                            sc_ = rf_.fit(Xt_f, yf).feature_importances_
                        sc_ = np.nan_to_num(np.asarray(sc_, dtype=float))
                        st.session_state['_fi_result'] = pd.DataFrame({'Feature': names_f, 'Score': sc_}).sort_values('Score', ascending=False).reset_index(drop=True)
                except Exception as e:
                    st.session_state['_fi_result'] = None
                    st.error(f"Error: {e}")
            fi_res = st.session_state.get('_fi_result')
            if fi_res is not None:
                top = fi_res.head(int(fi_k)).sort_values('Score')
                fig = px.bar(top, x='Score', y='Feature', orientation='h', color='Score', color_continuous_scale='Viridis', title="Feature ranking")
                fig.update_layout(**plotly_dark_layout(height=max(320, 26 * len(top) + 120), coloraxis_showscale=False))
                st.plotly_chart(fig, width='stretch', key="fi_fig")
                st.caption("Categorical columns appear as one-hot parts (e.g. Gender_Male). Low-score features can usually be dropped without hurting accuracy.")

    with st.expander("🤖 Auto Feature Engineering", expanded=False):
        tgt_fe = st.selectbox("Target (for correlation-based selection)", ["None"] + df.columns.tolist(), key="fe_tgt")
        if st.button("🧬 Generate Features", key="gen_fe", width='stretch'):
            dfc = df.copy()
            nc_f = get_num_cols(dfc)
            tgt = tgt_fe if tgt_fe != "None" else None
            if tgt in nc_f: nc_f.remove(tgt)
            new_feats = []
            if len(nc_f) >= 2 and tgt and tgt in dfc.columns and pd.api.types.is_numeric_dtype(dfc[tgt]):
                top = dfc[nc_f].corrwith(dfc[tgt]).abs().dropna().sort_values(ascending=False).head(4).index.tolist()
            else:
                top = nc_f[:4]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    a, b = top[i], top[j]
                    dfc[f'{a}_x_{b}'] = dfc[a] * dfc[b]; new_feats.append(f'{a}_x_{b}')
                    if (dfc[b] != 0).all():
                        dfc[f'{a}_div_{b}'] = dfc[a] / dfc[b]; new_feats.append(f'{a}_div_{b}')
            if len(nc_f) >= 3:
                dfc['row_mean'] = dfc[nc_f].mean(axis=1); dfc['row_std'] = dfc[nc_f].std(axis=1)
                dfc['row_max'] = dfc[nc_f].max(axis=1); dfc['row_min'] = dfc[nc_f].min(axis=1)
                new_feats += ['row_mean', 'row_std', 'row_max', 'row_min']
            if new_feats:
                st.success(f"✅ Generated {len(new_feats)} new features!")
                st.write(", ".join(new_feats))
                st.session_state['_pending_fe'] = (dfc, new_feats)
            else:
                st.info("Not enough numeric columns to generate features.")
        if '_pending_fe' in st.session_state:
            _dfc, _new_feats = st.session_state['_pending_fe']
            if st.button("➕ Add to Dataset", key="add_fe", width='stretch'):
                apply_df(_dfc, f"🧬 Added {len(_new_feats)} engineered features")
                del st.session_state['_pending_fe']
                st.rerun()

    with st.expander("🎯 Clustering Analysis", expanded=False):
        nc_c = get_num_cols(df)
        if len(nc_c) >= 2:
            cl_feats = st.multiselect("Features", nc_c, default=nc_c[:min(5, len(nc_c))], key="cl_feats")
            c1, c2, c3 = st.columns(3)
            with c1: cl_meth = st.selectbox("Method", ["K-Means", "DBSCAN", "Agglomerative"], key="cl_meth")
            with c2:
                if cl_meth != "DBSCAN": n_cl = st.slider("Clusters", 2, 12, 4, key="n_cl")
                else: eps_v = st.slider("DBSCAN eps", 0.1, 5.0, 0.5, key="dbscan_eps")
            with c3: use_pca_cl = st.checkbox("PCA reduction", True, key="pca_cl")
            if len(cl_feats) < 2:
                st.caption("Select at least 2 features.")
            elif st.button("🎯 Run Clustering", key="cl_btn", width='stretch'):
                try:
                    Xc = df[cl_feats].dropna()
                    if len(Xc) < 10: raise ValueError("Not enough complete rows.")
                    Xs = StandardScaler().fit_transform(Xc)
                    if use_pca_cl and len(cl_feats) > 2:
                        Xs = PCA(n_components=min(3, len(cl_feats)), random_state=42).fit_transform(Xs)
                    if cl_meth == "K-Means": cl = KMeans(n_clusters=n_cl, random_state=42, n_init=10).fit_predict(Xs)
                    elif cl_meth == "DBSCAN": cl = DBSCAN(eps=eps_v, min_samples=5).fit_predict(Xs)
                    else:
                        samp = Xs if len(Xs) <= 6000 else Xs[:6000]
                        if len(Xs) > 6000: Xc = Xc.iloc[:6000]; Xs = samp
                        cl = AgglomerativeClustering(n_clusters=n_cl).fit_predict(Xs)
                    try:
                        if len(set(cl)) > 1:
                            sil = silhouette_score(Xs, cl, sample_size=min(len(Xs), 3000), random_state=42)
                            st.metric("Silhouette Score", f"{sil:.4f}", 'Excellent' if sil > 0.7 else 'Good' if sil > 0.5 else 'Fair', delta_color="off")
                    except Exception:
                        pass
                    fig = px.scatter(x=Xs[:, 0], y=Xs[:, 1], color=cl.astype(str), title=f"{cl_meth} Clustering",
                                     labels={'x': 'PC1' if (use_pca_cl and len(cl_feats) > 2) else cl_feats[0],
                                             'y': 'PC2' if (use_pca_cl and len(cl_feats) > 2) else cl_feats[1]})
                    fig.update_layout(**plotly_dark_layout(height=450))
                    st.plotly_chart(fig, width='stretch', key="cl_fig")
                    u = np.unique(cl)
                    st.dataframe(pd.DataFrame({'Cluster': u, 'Size': [(cl == c).sum() for c in u], '%': [round((cl == c).mean() * 100, 1) for c in u]}),
                                 width='stretch', hide_index=True)
                    st.session_state['_pending_cl'] = (Xc.index, cl, cl_meth)
                except Exception as e:
                    st.error(f"Clustering error: {e}")
            if '_pending_cl' in st.session_state:
                _ci, _cl, _cm = st.session_state['_pending_cl']
                if st.button("➕ Add Clusters to Dataset", key="add_cl", width='stretch'):
                    dfc = df.copy(); dfc['Cluster'] = -1
                    dfc.loc[_ci, 'Cluster'] = _cl
                    apply_df(dfc, f"🎯 {_cm} clustering")
                    del st.session_state['_pending_cl']
                    st.rerun()
        else:
            st.info("Need at least 2 numeric columns.")

    with st.expander("📉 Dimensionality Reduction", expanded=False):
        nc_d = get_num_cols(df)
        if len(nc_d) >= 3:
            c1, c2 = st.columns(2)
            with c1: dr_meth = st.selectbox("Method", ["PCA", "t-SNE", "ICA"], key="dr_meth")
            with c2: dr_n = st.slider("Components", 2, min(10, len(nc_d)), min(3, len(nc_d)), key="dr_n")
            color_dr = st.selectbox("Color points by (optional)", ["None"] + df.columns.tolist(), key="dr_color")
            if st.button("📉 Apply Reduction", key="dr_btn", width='stretch'):
                try:
                    Xd = df[nc_d].copy()
                    Xd = Xd.loc[:, Xd.notna().any()]
                    Xd = Xd.fillna(Xd.median())
                    if len(Xd) > 3000 and dr_meth == "t-SNE":
                        Xd = Xd.sample(3000, random_state=42); st.caption("t-SNE uses a 3,000-row sample for speed.")
                    Xds = StandardScaler().fit_transform(Xd)
                    k_ = min(dr_n, Xds.shape[1])
                    if dr_meth == "PCA":
                        r = PCA(n_components=k_, random_state=42); Xr = r.fit_transform(Xds)
                        ve = r.explained_variance_ratio_
                        fig2 = px.bar(x=[f'PC{i + 1}' for i in range(k_)], y=ve * 100, title="Explained Variance per Component", labels={'x': 'Component', 'y': 'Variance %'})
                        fig2.update_layout(**plotly_dark_layout(height=300))
                        st.plotly_chart(fig2, width='stretch', key="dr_var")
                        st.metric("Total Variance Explained", f"{ve.sum() * 100:.1f}%")
                    elif dr_meth == "t-SNE":
                        Xr = TSNE(n_components=min(k_, 3), random_state=42, perplexity=max(5, min(30, len(Xds) // 4))).fit_transform(Xds)
                    else:
                        Xr = FastICA(n_components=k_, random_state=42, max_iter=500).fit_transform(Xds)
                    colr = df.loc[Xd.index, color_dr].astype(str) if color_dr != "None" else None
                    fig = px.scatter(x=Xr[:, 0], y=Xr[:, 1], color=colr, opacity=0.6, title=f"{dr_meth} — 2D Projection",
                                     labels={'x': f'{dr_meth} 1', 'y': f'{dr_meth} 2'})
                    fig.update_layout(**plotly_dark_layout(height=500))
                    st.plotly_chart(fig, width='stretch', key="dr_fig")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Need at least 3 numeric columns.")


# ═══════════════════════════════════════════════════════════
# TAB 7: ADVANCED
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## ⚙️ Advanced Operations")

    with st.expander("🎲 Smart Sampling", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            samp_t = st.selectbox("Method", ["Random %", "Fixed N", "Stratified", "Bootstrap", "Systematic"], key="samp_t")
        with c2:
            if samp_t == "Random %": samp_pct = st.slider("Percentage", 1, 99, 30, key="samp_pct")
            elif samp_t == "Fixed N": samp_n = st.number_input("N rows", 1, max(len(df), 1), min(1000, max(len(df), 1)), key="samp_n")
            elif samp_t == "Stratified":
                strat_c = st.selectbox("Stratify by", df.columns.tolist(), key="strat_c")
                samp_pct2 = st.slider("%", 1, 99, 30, key="samp_pct2")
            elif samp_t == "Bootstrap": boot_n = st.number_input("N samples", 1, max(len(df) * 3, 1), min(1000, max(len(df), 1)), key="boot_n")
            else: step_v = st.number_input("Step size", 1, 100, 5, key="step_v")
        if st.button("🎲 Apply Sampling", key="samp_btn", width='stretch'):
            try:
                if samp_t == "Random %": ds = df.sample(frac=samp_pct / 100, random_state=42)
                elif samp_t == "Fixed N": ds = df.sample(n=min(int(samp_n), len(df)), random_state=42)
                elif samp_t == "Stratified": ds = df.groupby(strat_c, group_keys=False).sample(frac=samp_pct2 / 100, random_state=42)
                elif samp_t == "Bootstrap": ds = df.sample(n=int(boot_n), replace=True, random_state=42)
                else: ds = df.iloc[::int(step_v)]
                apply_df(ds, f"🎲 {samp_t} sampling")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("🔍 Advanced Filtering", expanded=True):
        fc = st.selectbox("Column to filter", df.columns.tolist(), key="flt_col")
        if pd.api.types.is_numeric_dtype(df[fc]) and not pd.api.types.is_bool_dtype(df[fc]) and df[fc].notna().any():
            c1, c2 = st.columns(2)
            with c1: ft = st.selectbox("Filter", ["Range", "Greater than", "Less than", "Between percentiles"], key="flt_t")
            mn, mx = float(df[fc].min()), float(df[fc].max())
            with c2:
                if mn == mx:
                    st.caption(f"Column is constant ({mn:g}) — nothing to filter.")
                    fv = None
                elif ft == "Range": fv = st.slider("Range", mn, mx, (mn, mx), key=f"flt_rng_{fc}")
                elif ft == "Between percentiles": fv = st.slider("Percentiles", 0, 100, (10, 90), key="flt_pct")
                else: fv = st.number_input("Threshold", value=float(df[fc].median()), key=f"flt_val_{fc}")
            if fv is not None and st.button("🔍 Apply Filter", key="flt_btn", width='stretch'):
                if ft == "Range": dff = df[(df[fc] >= fv[0]) & (df[fc] <= fv[1])]
                elif ft == "Greater than": dff = df[df[fc] > fv]
                elif ft == "Less than": dff = df[df[fc] < fv]
                else:
                    lo, hi = df[fc].quantile(fv[0] / 100), df[fc].quantile(fv[1] / 100)
                    dff = df[(df[fc] >= lo) & (df[fc] <= hi)]
                if dff.empty: st.error("Filter would remove every row.")
                else:
                    apply_df(dff, f"🔍 Filter {fc}"); st.rerun()
        else:
            uv = sorted(df[fc].dropna().unique().tolist(), key=str)
            sv = st.multiselect("Select values", uv, default=uv[:min(5, len(uv))], key=f"flt_sel_{fc}")
            if sv and st.button("🔍 Apply Filter", key="flt_cat_btn", width='stretch'):
                apply_df(df[df[fc].isin(sv)], f"🔍 Filter {fc}"); st.rerun()

    with st.expander("🔗 Merge Datasets", expanded=False):
        if st.session_state.df2 is not None:
            df2 = st.session_state.df2
            st.caption(f"Second dataset: {df2.shape[0]:,} × {df2.shape[1]:,}")
            c1, c2, c3 = st.columns(3)
            with c1: jt = st.selectbox("Join type", ["inner", "left", "right", "outer"], key="jt")
            with c2: lk = st.selectbox("Left key", df.columns.tolist(), key="lk")
            with c3: rk = st.selectbox("Right key", df2.columns.tolist(), key="rk")
            if st.button("🔗 Merge", key="mrg_btn", width='stretch'):
                try:
                    merged = pd.merge(df, df2, left_on=lk, right_on=rk, how=jt, suffixes=('', '_2'))
                    apply_df(merged, f"🔗 {jt.title()} merge on {lk}"); st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Upload a second dataset in the sidebar to enable merging.")

    with st.expander("✏️ Column Operations", expanded=False):
        c1, c2 = st.columns(2)
        with c1: op_type = st.selectbox("Operation", ["Rename Column", "Drop Column", "Change Dtype", "Create from Formula"], key="col_op")
        with c2: op_col = st.selectbox("Column", df.columns.tolist(), key="op_col") if op_type != "Create from Formula" else None

        if op_type == "Rename Column":
            new_name = st.text_input("New name", op_col or "", key=f"new_name_{op_col}")
            if st.button("✅ Rename", width='stretch', key="rename_btn"):
                if not new_name.strip(): st.error("Name can't be empty.")
                else:
                    apply_df(df.rename(columns={op_col: new_name.strip()}), f"✏️ Renamed {op_col} → {new_name.strip()}"); st.rerun()
        elif op_type == "Drop Column":
            drop_multi = st.multiselect("Columns to drop", df.columns.tolist(), key="drop_cols")
            if drop_multi and st.button("🗑️ Drop Columns", width='stretch', key="drop_btn"):
                if len(drop_multi) >= df.shape[1]: st.error("Can't drop every column.")
                else:
                    apply_df(df.drop(columns=drop_multi), f"🗑️ Dropped {len(drop_multi)} columns"); st.rerun()
        elif op_type == "Change Dtype":
            new_type = st.selectbox("New type", ["int64", "float64", "str", "category", "datetime64[ns]"], key="new_type")
            if st.button("🔄 Convert", width='stretch', key="conv_btn"):
                try:
                    dfc = df.copy()
                    if new_type in ("int64", "float64"):
                        dfc[op_col] = pd.to_numeric(dfc[op_col], errors='raise')
                    if new_type == "int64" and dfc[op_col].isna().any():
                        raise ValueError("Column has missing values — fill them before converting to int64.")
                    dfc[op_col] = dfc[op_col].astype(new_type)
                    apply_df(dfc, f"🔄 {op_col}: → {new_type}"); st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            new_col = st.text_input("New column name", "new_feature", key="formula_name")
            formula = st.text_input("Formula — e.g. Hours_Studied * Attendance / 100   or   np.log1p(`My Column`)", key="formula")
            st.caption("Columns: " + ", ".join(str(c) for c in df.columns) + " · allowed functions: np." + ", np.".join(sorted(['log', 'log1p', 'exp', 'sqrt', 'abs', 'clip', 'where', 'round'])))
            if formula and st.button("✅ Create Column", width='stretch', key="formula_btn"):
                try:
                    dfc = df.copy()
                    dfc[new_col] = safe_formula_eval(dfc, formula)
                    apply_df(dfc, f"➕ Created {new_col}"); st.rerun()
                except Exception as e:
                    st.error(f"Formula error: {e}")




# ═══════════════════════════════════════════════════════════
# TAB 8: AUTOML
# ═══════════════════════════════════════════════════════════
def render_automl_leaderboard(rdf, score_label):
    medals = ['🥇', '🥈', '🥉']
    sc = rdf['Score'].astype(float)
    lo, hi = float(sc.min()), float(sc.max())
    rng_ = (hi - lo) if hi != lo else 1.0

    def _cv_mean(v):
        try:
            return float(str(v).split('±')[0].strip())
        except (ValueError, TypeError):
            return None
    rows_html = ''
    for i, row in rdf.reset_index(drop=True).iterrows():
        best = (i == 0)
        s = float(row['Score'])
        rel = (s - lo) / rng_
        col = '#43E97B' if rel >= 0.75 else '#38F9D7' if rel >= 0.5 else '#F9AB00' if rel >= 0.25 else '#FF4757'
        bar_w = max(4, int(rel * 100))
        cv = _cv_mean(row.get('CV Score'))
        cv_html = (f'<span style="font-size:14px;font-weight:700;color:{"#43E97B" if cv >= 0.85 else "#38F9D7" if cv >= 0.7 else "#F9AB00"}">{cv:.4f}</span>'
                   if cv is not None else '<span style="color:#666">—</span>')
        badge = ("<div style='margin-top:5px'><span style='background:rgba(67,233,123,0.18);color:#43E97B;font-size:10px;"
                 "padding:2px 9px;border-radius:20px;font-weight:700'>★ BEST MODEL</span></div>") if best else ''
        extra = ''
        if 'Tol Acc' in row.index and not pd.isna(row.get('Tol Acc')):
            extra = f'<div style="font-size:10px;color:rgba(232,233,240,0.5)">within ±tol: {row["Tol Acc"] * 100:.1f}%</div>'
        rows_html += (
            f'<tr style="background:{"rgba(67,233,123,0.05)" if best else "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"};'
            f'border-left:3px solid {"#43E97B" if best else "transparent"}">'
            f'<td style="padding:14px;text-align:center;font-size:20px">{medals[i] if i < 3 else f"#{i + 1}"}</td>'
            f'<td style="padding:12px 18px"><div style="font-weight:700;font-size:15px;color:#E8E9F0">{esc(row["Model"])}</div>{badge}</td>'
            f'<td style="padding:12px 18px;min-width:220px"><div style="display:flex;justify-content:space-between">'
            f'<span style="font-size:19px;font-weight:800;color:#E8E9F0">{s:.4f}</span>'
            f'<span style="font-size:11px;color:{col}">{s * 100:.1f}%</span></div>'
            f'<div style="background:rgba(255,255,255,0.08);border-radius:5px;height:7px;margin-top:6px">'
            f'<div style="width:{bar_w}%;background:{col};height:7px;border-radius:5px"></div></div>{extra}</td>'
            f'<td style="padding:12px 18px;text-align:center">{cv_html}</td>'
            f'<td style="padding:12px 18px;text-align:center;color:#999;font-size:12px">{row.get("Time (s)", "")}s</td></tr>')
    th = 'padding:12px 16px;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#6C63FF;border-bottom:2px solid rgba(108,99,255,0.3)'
    st.markdown(
        f'<div style="border-radius:16px;overflow:hidden;border:1px solid rgba(108,99,255,0.2);margin:16px 0">'
        f'<div style="background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(255,101,132,0.06));padding:16px 20px">'
        f'<div style="font-size:16px;font-weight:700;color:#E8E9F0">AutoML Algorithm Leaderboard</div>'
        f'<div style="font-size:12px;color:rgba(232,233,240,0.5)">Sorted by hold-out {score_label}</div></div>'
        f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse"><thead style="background:rgba(0,0,0,0.3)"><tr>'
        f'<th style="{th};text-align:center">Rank</th><th style="{th};text-align:left">Model</th>'
        f'<th style="{th};text-align:left">{score_label}</th><th style="{th};text-align:center">CV Mean</th>'
        f'<th style="{th};text-align:center">Time</th></tr></thead><tbody>{rows_html}</tbody></table></div></div>',
        unsafe_allow_html=True)


with tabs[7]:
    st.markdown("## 🏆 AutoML — Automated Machine Learning")
    st.markdown("""
    <div class="alert-info">
        <strong>🤖 AutoML Pipeline:</strong> Smart preprocessing (ordinal / one-hot / scaling) → outlier guard → multi-model training
        → <b>Optuna tuning</b> (Accurate mode) → <b>Top-3 Ensemble</b> → best model saved to Predict / SHAP / Export
    </div>
    """, unsafe_allow_html=True)

    if len(df.columns) < 2:
        st.warning("Need at least 2 columns.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            aml_tgt = st.selectbox("🎯 Target Variable", df.columns.tolist(), index=len(df.columns) - 1, key="aml_tgt")
        with c2:
            _nfeat = len(df.columns) - 1
            if _nfeat > 5:
                aml_feats_k = st.slider("Max features to keep (feature selection)", 5, min(60, _nfeat) if _nfeat > 5 else 6,
                                        min(60, _nfeat), key="aml_k")
            else:
                aml_feats_k = _nfeat
                st.caption(f"All {_nfeat} features are used.")
        with c3:
            speed_mode = st.selectbox("⚡ Speed Mode", ["⚡ Fast (recommended)", "⚖️ Balanced", "🎯 Accurate (slow)"], key="aml_speed")

        _cfg_all = {
            "⚡ Fast (recommended)": dict(n_est=150, cv_folds=3, max_rows=6000, ensemble=False, tune=0),
            "⚖️ Balanced": dict(n_est=250, cv_folds=3, max_rows=15000, ensemble=True, tune=0),
            "🎯 Accurate (slow)": dict(n_est=400, cv_folds=5, max_rows=None, ensemble=True, tune=20),
        }
        cfg = _cfg_all[speed_mode]

        o1, o2, o3 = st.columns(3)
        with o1:
            use_aml_cv = st.checkbox("Cross-validation in AutoML", True, key="aml_cv")
        with o2:
            _g = infer_problem_type(df[aml_tgt].dropna()) if df[aml_tgt].notna().any() else 'classification'
            _of = 0.0
            if _g == 'regression':
                _yy = pd.to_numeric(df[aml_tgt], errors='coerce').dropna()
                _of = float(iqr_outlier_mask(_yy).mean()) if len(_yy) else 0.0
            aml_rm_out = st.checkbox("🛡️ Remove target outliers (regression)", value=bool(0 < _of <= 0.05),
                                     key=f"aml_rmout_{aml_tgt}", disabled=(_g != 'regression'),
                                     help="IQR 1.5× on the target. The honest R² including those rows is still reported.")
        with o3:
            aml_tol = st.slider("Tolerance (× target std)", 0.05, 1.0, 0.25, 0.05, key="aml_tol")

        st.markdown(f"""<div style="padding:8px 12px;background:rgba(108,99,255,0.08);border-radius:8px;font-size:12px;color:rgba(232,233,240,0.7)">
            🌲 Trees: <b>{cfg['n_est']}</b> &nbsp;|&nbsp; CV folds: <b>{cfg['cv_folds']}</b> &nbsp;|&nbsp;
            Ensemble: <b>{'Yes' if cfg['ensemble'] else 'No'}</b> &nbsp;|&nbsp;
            Optuna tuning: <b>{(str(cfg['tune']) + ' trials') if cfg['tune'] and OPTUNA_AVAILABLE else ('unavailable (pip install optuna)' if cfg['tune'] else 'No')}</b> &nbsp;|&nbsp;
            Max rows: <b>{cfg['max_rows'] or 'All'}</b></div>""", unsafe_allow_html=True)

        if st.button("🚀 Launch AutoML", width='stretch', type="primary", key="aml_btn"):
            avail = [c for c in df.columns if c != aml_tgt]
            try:
                with st.spinner("🤖 AutoML running — preparing data..."):
                    X, y, t_enc, ptype, pinfo = prepare_ml_data(df, aml_tgt, avail, 'Auto', aml_rm_out)
                    if cfg['max_rows'] and len(X) > cfg['max_rows']:
                        sidx = np.random.RandomState(42).choice(len(X), cfg['max_rows'], replace=False)
                        X, y = X.iloc[sidx].reset_index(drop=True), pd.Series(np.asarray(y)[sidx])
                    X_tr, X_te, y_tr, y_te = split_data(X, y, ptype, 0.2)
                    tol = aml_tol * float(np.std(np.asarray(y_tr, dtype=float))) if ptype == 'regression' else None
                    ctx = dict(ptype=ptype, X_tr=X_tr, y_tr=np.asarray(y_tr), X_te=X_te, y_te=np.asarray(y_te), y_all=np.asarray(y),
                               t_enc=t_enc, scaler='standard', k_select=int(aml_feats_k) if aml_feats_k < X.shape[1] else None,
                               cv_folds=cfg['cv_folds'] if use_aml_cv else 0, tol=tol, X_out=pinfo['X_out'], y_out=pinfo['y_out'])

                catalog = get_model_catalog(ptype, n_est=cfg['n_est'])
                wanted = ["🔗 Logistic Regression" if ptype == 'classification' else "🔷 Ridge",
                          "⚡ XGBoost", "💡 LightGBM", "🌲 Random Forest", "🌳 Extra Trees", "📈 Gradient Boosting"]
                names = [n for n in wanted if n in catalog]
                rows, errs = run_training(names, catalog, ctx, aml_tgt, prefix="AutoML · ", source='AutoML', progress_label="AutoML training")
                for e_ in errs:
                    st.warning(f"⚠️ {e_}")
                if not rows:
                    raise RuntimeError("No model could be trained: " + "; ".join(errs))

                # ── Optuna tuning on the best boosted model ──
                if cfg['tune'] and OPTUNA_AVAILABLE:
                    boosted = [r for r in sorted(rows, key=lambda r: -r['Score']) if any(k in r['Model'] for k in ('XGBoost', 'LightGBM', 'Gradient Boosting'))]
                    if boosted:
                        base = boosted[0]['Model']
                        with st.spinner(f"🔧 Optuna tuning {base} ({cfg['tune']} trials)..."):
                            try:
                                tuned_est, _ = tune_with_optuna(base, ptype, ctx, n_trials=cfg['tune'], timeout=90)
                                if tuned_est is not None:
                                    nm = f"🔧 Tuned {base.split(' ', 1)[1]}"
                                    pipe_t, yp_t, row_t = train_one(nm, tuned_est, ctx)
                                    rows.append(row_t)
                                    st.session_state.trained_models["AutoML · " + nm] = make_model_info(pipe_t, yp_t, row_t, ctx, aml_tgt, 'AutoML')
                            except Exception as e_:
                                st.warning(f"Tuning skipped: {e_}")

                # ── Top-3 ensemble ──
                if cfg['ensemble'] and len(rows) >= 3:
                    with st.spinner("🎯 Building top-3 ensemble..."):
                        try:
                            top = sorted(rows, key=lambda r: -r['Score'])[:3]
                            members = [(r['Model'], st.session_state.trained_models["AutoML · " + r['Model']]['pipeline']) for r in top]
                            ens = (VotingClassifier(estimators=members, voting='soft') if ptype == 'classification'
                                   else VotingRegressor(estimators=members))
                            t0 = time.time()
                            ens.fit(ctx['X_tr'], ctx['y_tr'])
                            yp_e = ens.predict(ctx['X_te'])
                            if ptype == 'classification':
                                acc_e = float(accuracy_score(ctx['y_te'], yp_e))
                                row_e = {'Model': '🎯 Ensemble (Top-3)', 'Accuracy': acc_e, 'F1': float(f1_score(ctx['y_te'], yp_e, average='weighted', zero_division=0)),
                                         'CV Score': None, 'Score': acc_e}
                            else:
                                m_ = regression_scores(ctx['y_te'], yp_e, tol)
                                r2a = None
                                if ctx['X_out'] is not None and len(ctx['X_out']):
                                    r2a = float(r2_score(np.concatenate([ctx['y_te'], ctx['y_out']]),
                                                         np.concatenate([yp_e, ens.predict(ctx['X_out'])])))
                                row_e = {'Model': '🎯 Ensemble (Top-3)', 'R²': m_['R²'], 'RMSE': m_['RMSE'], 'MAE': m_['MAE'], 'MAPE': m_['MAPE'],
                                         'Tol Acc': m_['Tol Acc'], 'R² (+outliers)': r2a, 'CV Score': None, 'Score': m_['R²']}
                            row_e['Time (s)'] = round(time.time() - t0, 1)
                            rows.append(row_e)
                            ctx_e = dict(ctx, shap_pipeline=members[0][1])
                            st.session_state.trained_models["AutoML · 🎯 Ensemble (Top-3)"] = make_model_info(ens, yp_e, row_e, ctx_e, aml_tgt, 'AutoML')
                        except Exception as e_:
                            st.warning(f"Ensemble skipped: {e_}")

                rdf = pd.DataFrame(rows).sort_values('Score', ascending=False).reset_index(drop=True)
                best_name = rdf.iloc[0]['Model']
                st.session_state.best_model = "AutoML · " + best_name
                bi = st.session_state.trained_models["AutoML · " + best_name]
                imp_df = None
                try:
                    with st.spinner("Computing feature importance..."):
                        imp_df = perm_importance_df(bi['pipeline'], bi['X_test'], bi['y_test'], ptype)
                except Exception:
                    imp_df = None
                st.session_state.aml_results = {
                    'rdf': rdf, 'ptype': ptype, 'best': best_name, 'target': aml_tgt, 'tol': tol, 'imp_df': imp_df,
                    'info': {k: v for k, v in pinfo.items() if k not in ('X_out', 'y_out')},
                    'n_train': len(X_tr), 'n_test': len(X_te)}
                st.session_state.shap_values = None
                st.session_state['_aml_flash'] = "✅ AutoML complete! Every model is saved — pick any in Predict / SHAP / Export."
                _aml_done = True
            except Exception as e:
                st.error(f"AutoML error: {e}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
            if locals().get('_aml_done'):
                st.rerun()      # Predict / SHAP / Export tabs are rendered earlier in the script → refresh them

        if st.session_state.get('_aml_flash'):
            st.success(st.session_state.pop('_aml_flash'))

        ares = st.session_state.get('aml_results')
        if ares:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"### 🏆 AutoML Results — target `{esc(ares['target'])}` ({ares['ptype']}) · "
                        f"{ares['n_train']:,} train / {ares['n_test']:,} test rows")
            rdf = ares['rdf']
            best_row = rdf.iloc[0].to_dict()
            if ares['ptype'] == 'classification':
                for k in ('Precision', 'Recall'):
                    best_row.setdefault(k, np.nan)
            render_model_kpis(best_row, ares['ptype'], ares['tol'], outlier_note_text(ares['info'], best_row))
            score_label = 'Accuracy' if ares['ptype'] == 'classification' else 'R²'
            fig = px.bar(rdf, x='Model', y='Score', color='Score', color_continuous_scale='Viridis',
                         text=rdf['Score'].apply(lambda x: f'{x:.4f}'), title=f"AutoML — {score_label} Comparison")
            fig.update_traces(textposition='outside')
            fig.update_layout(**plotly_dark_layout(height=420, coloraxis_showscale=False))
            st.plotly_chart(fig, width='stretch', key="aml_bar")
            render_automl_leaderboard(rdf, score_label)
            key_ = "AutoML · " + ares['best']
            if key_ in st.session_state.trained_models:
                render_model_diagnostics(st.session_state.trained_models[key_], ares['ptype'], ares['imp_df'], "aml")




# ═══════════════════════════════════════════════════════════
# TAB 9: STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════
def stat_card(title, items, conclusion, color):
    cells = ''.join(
        f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">{esc(lbl)}</div>'
        f'<div style="font-size:30px;font-weight:800;color:{c or "#E8E9F0"}">{esc(val)}</div></div>' for lbl, val, c in items)
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.3);border:1px solid {color};border-radius:16px;padding:26px;margin:16px 0;text-align:center">'
        f'<div style="font-size:13px;color:rgba(232,233,240,0.6)">{esc(title)}</div>'
        f'<div style="display:flex;justify-content:center;gap:44px;margin:16px 0;flex-wrap:wrap">{cells}</div>'
        f'<div style="font-size:15px;font-weight:600;color:{color}">{conclusion}</div></div>', unsafe_allow_html=True)


with tabs[8]:
    st.markdown("## 🔬 Statistical Tests & Hypothesis Testing")
    nc = get_num_cols(df)
    cc = get_cat_cols(df)
    grp_cands = cc + [c for c in nc if df[c].nunique() <= 10]
    chi_cands = cc + [c for c in nc if df[c].nunique() <= 20]

    test_type = st.selectbox("Select Test", [
        "📊 T-Test (2 group means comparison)",
        "📊 Mann-Whitney U (non-parametric T-Test)",
        "📊 ANOVA (3+ group means comparison)",
        "📊 Chi-Square (categorical independence)",
        "📊 Shapiro-Wilk (normality check)",
        "📊 Correlation Matrix with P-values",
    ], key="stat_test_type")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if ("T-Test" in test_type or "Mann-Whitney" in test_type):
        if not nc or not grp_cands:
            st.info("Needs at least one numeric column and one grouping (categorical / low-cardinality) column.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: num_col = st.selectbox("Numeric Column", nc, key="tt_num")
            with c2: grp_col = st.selectbox("Group Column", grp_cands, key="tt_grp")
            with c3: alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01, key="tt_alpha")
            levels = sorted(df[grp_col].dropna().unique().tolist(), key=lambda v: str(v))
            if len(levels) < 2:
                st.warning("The group column needs at least 2 distinct values.")
            else:
                g1c, g2c = st.columns(2)
                with g1c: ga = st.selectbox("Group A", levels, index=0, key="tt_ga")
                with g2c: gb = st.selectbox("Group B", [l for l in levels if l != ga], index=0, key="tt_gb")
                if st.button("▶ Run Test", width='stretch', key="tt_run"):
                    try:
                        g1 = df.loc[df[grp_col] == ga, num_col].dropna().to_numpy(dtype=float)
                        g2 = df.loc[df[grp_col] == gb, num_col].dropna().to_numpy(dtype=float)
                        if len(g1) < 2 or len(g2) < 2:
                            st.error("Each group needs at least 2 non-missing values.")
                        else:
                            if "Mann" in test_type:
                                stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                                test_name = "Mann-Whitney U"
                                eff_lbl, eff = "RANK-BISERIAL r", 1 - 2 * stat / (len(g1) * len(g2))
                            else:
                                stat, p = ttest_ind(g1, g2, equal_var=False)
                                test_name = "Welch's T-Test (unequal variances)"
                                sp = np.sqrt(((len(g1) - 1) * g1.var(ddof=1) + (len(g2) - 1) * g2.var(ddof=1)) / (len(g1) + len(g2) - 2))
                                eff_lbl, eff = "COHEN'S d", (g1.mean() - g2.mean()) / sp if sp > 0 else 0.0
                            col_ = "#43E97B" if p > alpha else "#FF4757"
                            concl = ("✅ Fail to reject H₀ — No significant difference" if p > alpha
                                     else "❌ Reject H₀ — Significant difference exists")
                            stat_card(f"{test_name}: {ga} (n={len(g1)}, mean={g1.mean():.3f}) vs {gb} (n={len(g2)}, mean={g2.mean():.3f})",
                                      [("TEST STATISTIC", f"{stat:.4f}", None), ("P-VALUE", f"{p:.4g}", col_),
                                       (eff_lbl, f"{eff:.3f}", None), ("ALPHA", f"{alpha}", None)], concl, col_)
                            fig = go.Figure()
                            fig.add_trace(go.Box(y=g1, name=str(ga), boxpoints='outliers', marker_color='#6C63FF'))
                            fig.add_trace(go.Box(y=g2, name=str(gb), boxpoints='outliers', marker_color='#FF6584'))
                            fig.update_layout(**plotly_dark_layout(title=f"{num_col} by {grp_col}", height=380))
                            st.plotly_chart(fig, width='stretch', key="tt_fig")
                    except Exception as e:
                        st.error(f"Test error: {e}")

    elif "ANOVA" in test_type:
        if not nc or not grp_cands:
            st.info("Needs at least one numeric column and one grouping column.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: num_col = st.selectbox("Numeric Column", nc, key="anova_num")
            with c2: grp_col = st.selectbox("Group Column", grp_cands, key="anova_grp")
            with c3: alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="anova_alpha")
            if st.button("▶ Run ANOVA", width='stretch', key="anova_run"):
                try:
                    sub = df[[grp_col, num_col]].dropna()
                    groups = [g[num_col].to_numpy(dtype=float) for _, g in sub.groupby(grp_col) if len(g) >= 2]
                    if len(groups) < 2:
                        st.error("Need at least 2 groups with ≥ 2 observations each.")
                    elif len(groups) > 40:
                        st.error("Too many groups (> 40). Choose a column with fewer categories.")
                    else:
                        f_stat, p = f_oneway(*groups)
                        allv = np.concatenate(groups)
                        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
                        ss_t = ((allv - allv.mean()) ** 2).sum()
                        eta = ss_b / ss_t if ss_t > 0 else 0.0
                        try: kp = kruskal(*groups)[1]
                        except Exception: kp = float('nan')
                        col_ = "#43E97B" if p > alpha else "#FF4757"
                        concl = ("✅ Fail to reject H₀ — Group means are equal" if p > alpha
                                 else "❌ Reject H₀ — At least one group mean differs")
                        stat_card(f"One-Way ANOVA · {len(groups)} groups", [("F-STATISTIC", f"{f_stat:.4f}", None), ("P-VALUE", f"{p:.4g}", col_),
                                  ("η² EFFECT", f"{eta:.3f}", None), ("KRUSKAL p", f"{kp:.4g}", None)], concl, col_)
                        fig = px.violin(sub, x=grp_col, y=num_col, box=True, color=grp_col)
                        fig.update_layout(**plotly_dark_layout(height=400))
                        st.plotly_chart(fig, width='stretch', key="anova_fig")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif "Chi-Square" in test_type:
        if len(chi_cands) < 2:
            st.info("Chi-square needs at least two categorical columns.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: col1 = st.selectbox("Column 1", chi_cands, key="chi_c1")
            with c2: col2 = st.selectbox("Column 2", [c for c in chi_cands if c != col1], key="chi_c2")
            with c3: alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="chi_alpha")
            if st.button("▶ Run Chi-Square", width='stretch', key="chi_run"):
                try:
                    ct = pd.crosstab(df[col1], df[col2])
                    if ct.shape[0] < 2 or ct.shape[1] < 2:
                        st.error("Both columns need at least 2 categories.")
                    else:
                        chi2, p, dof, expected = chi2_contingency(ct)
                        n = ct.to_numpy().sum()
                        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n else 0.0
                        col_ = "#43E97B" if p > alpha else "#FF4757"
                        concl = "✅ Variables are INDEPENDENT" if p > alpha else "❌ Variables are DEPENDENT (significant association)"
                        stat_card("Chi-Square Test of Independence", [("CHI² STATISTIC", f"{chi2:.4f}", None), ("P-VALUE", f"{p:.4g}", col_),
                                  ("DOF", f"{dof}", None), ("CRAMÉR'S V", f"{v:.3f}", None)], concl, col_)
                        if (expected < 5).mean() > 0.2:
                            st.caption("⚠️ More than 20% of expected counts are < 5 — interpret the p-value with caution.")
                        fig = px.imshow(ct, text_auto=True, color_continuous_scale='Viridis', title="Contingency Table", aspect='auto')
                        fig.update_layout(**plotly_dark_layout(height=400))
                        st.plotly_chart(fig, width='stretch', key="chi_fig")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif "Shapiro" in test_type:
        if not nc:
            st.info("No numeric columns available.")
        else:
            col = st.selectbox("Select Column", nc, key="sw_col")
            if st.button("▶ Run Shapiro-Wilk", width='stretch', key="sw_run"):
                try:
                    data = df[col].dropna()
                    if len(data) < 3:
                        st.error("Need at least 3 values.")
                    elif data.nunique() < 2:
                        st.error("Column is constant — normality test is undefined.")
                    else:
                        if len(data) > 5000: data = data.sample(5000, random_state=42)
                        stat, p = shapiro(data)
                        ok = p > 0.05
                        col_ = "#43E97B" if ok else "#F9AB00"
                        concl = ("✅ Data looks NORMAL (Gaussian)" if ok else "⚠️ Data is NOT normal — consider log/sqrt transform")
                        stat_card(f"Shapiro-Wilk · n={len(data):,}", [("W STATISTIC", f"{stat:.4f}", None), ("P-VALUE", f"{p:.4g}", col_)], concl, col_)
                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.histogram(data, title=f"{col} Distribution", color_discrete_sequence=['#6C63FF'])
                            fig.update_layout(**plotly_dark_layout(height=350, showlegend=False))
                            st.plotly_chart(fig, width='stretch', key="sw_h")
                        with c2:
                            qq = stats.probplot(data)
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='#6C63FF', size=4), name='Data'))
                            fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * np.array(qq[0][0]) + qq[1][1], mode='lines',
                                                      line=dict(color='#43E97B', width=2), name='Normal line'))
                            fig2.update_layout(**plotly_dark_layout(title="Q-Q Plot", height=350))
                            st.plotly_chart(fig2, width='stretch', key="sw_q")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif "Correlation" in test_type:
        if len(nc) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            c1, c2 = st.columns([4, 1])
            with c1: sel_cols = st.multiselect("Select Numeric Columns", nc, default=nc[:min(8, len(nc))], key="corr_cols")
            with c2: cmeth = st.selectbox("Method", ["Pearson", "Spearman"], key="corr_pv_meth")
            if len(sel_cols) >= 2:
                corr = df[sel_cols].corr(method=cmeth.lower())
                p_matrix = pd.DataFrame(np.ones((len(sel_cols), len(sel_cols))), columns=sel_cols, index=sel_cols)
                for i in range(len(sel_cols)):
                    for j in range(i + 1, len(sel_cols)):
                        pair = df[[sel_cols[i], sel_cols[j]]].dropna()          # aligned pairwise-complete rows
                        try:
                            fn = stats.pearsonr if cmeth == "Pearson" else stats.spearmanr
                            pv = float(fn(pair.iloc[:, 0], pair.iloc[:, 1])[1])
                        except Exception:
                            pv = np.nan
                        p_matrix.iloc[i, j] = p_matrix.iloc[j, i] = pv
                fig = px.imshow(corr.round(3), text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                                title=f"{cmeth} Correlation Matrix")
                fig.update_layout(**plotly_dark_layout(height=520))
                st.plotly_chart(fig, width='stretch', key="corr_pv_fig")
                sig_rows = []
                for i in range(len(sel_cols)):
                    for j in range(i + 1, len(sel_cols)):
                        pv = p_matrix.iloc[i, j]
                        if pd.notna(pv) and pv < 0.05:
                            sig_rows.append({'Feature A': sel_cols[i], 'Feature B': sel_cols[j],
                                             'r': round(float(corr.iloc[i, j]), 3), 'p-value': float(pv)})
                if sig_rows:
                    st.markdown("**📌 Significant correlations (p < 0.05):**")
                    st.dataframe(pd.DataFrame(sig_rows).sort_values('r', key=lambda s: s.abs(), ascending=False),
                                 width='stretch', hide_index=True)
                else:
                    st.info("No statistically significant correlations at α = 0.05.")




# ═══════════════════════════════════════════════════════════
# TAB 10: SQL QUERY
# ═══════════════════════════════════════════════════════════
def _qid(c):
    return '"' + str(c).replace('"', '""') + '"'


def _sql_pick_example():
    ex = st.session_state.get('_sql_examples', {})
    choice = st.session_state.get('sql_example', 'Custom')
    if choice != 'Custom' and choice in ex:
        st.session_state['sql_input'] = ex[choice]


def _sql_set_query(q):
    st.session_state['sql_input'] = q


with tabs[9]:
    st.markdown("## 🗄️ SQL Query on DataFrame")
    st.markdown('<div class="alert-info">💡 Use <b>df</b> as the table name · SQLite syntax (SELECT / WITH, JOIN, GROUP BY, window functions…) · '
                'wrap column names containing spaces or symbols in <code>"double quotes"</code>.</div>', unsafe_allow_html=True)

    _nc, _cc = get_num_cols(df), get_cat_cols(df)
    examples = {"Preview 10 rows": "SELECT * FROM df LIMIT 10",
                "First 3 columns": f"SELECT {', '.join(_qid(c) for c in df.columns[:3])} FROM df LIMIT 20"}
    if _nc:
        examples["Count & average"] = f"SELECT COUNT(*) AS total, AVG({_qid(_nc[0])}) AS avg_val FROM df"
    if _nc and _cc:
        examples["Group by category"] = (f"SELECT {_qid(_cc[0])}, COUNT(*) AS n, ROUND(AVG({_qid(_nc[0])}), 2) AS avg_{re.sub(r'[^0-9A-Za-z_]', '_', _nc[0])} "
                                         f"FROM df GROUP BY {_qid(_cc[0])} ORDER BY n DESC")
    st.session_state['_sql_examples'] = examples
    if 'sql_input' not in st.session_state:
        st.session_state['sql_input'] = "SELECT * FROM df LIMIT 10"

    st.selectbox("📋 Example Queries", ["Custom"] + list(examples.keys()), key="sql_example", on_change=_sql_pick_example)
    sql_query = st.text_area("Write your SQL query:", height=130, key="sql_input", placeholder="SELECT * FROM df WHERE column > 100 LIMIT 50")
    c1, c2 = st.columns([3, 1])
    with c1:
        run_sql_btn = st.button("▶ Execute Query", width='stretch', type="primary", key="sql_run")
    with c2:
        save_result = st.checkbox("Save result as new dataset", key="sql_save")

    if run_sql_btn:
        if not sql_query.strip():
            st.warning("Write a query first.")
        else:
            try:
                with st.spinner("Running query..."):
                    result = run_sql(df, sql_query)
                st.session_state.sql_history.append({'query': sql_query, 'rows': len(result), 'time': datetime.now()})
                st.session_state['sql_last'] = {'query': sql_query, 'result': result}
                if save_result:
                    apply_df(result, "🗄️ SQL Query result")
                    st.rerun()
            except Exception as e:
                st.session_state['sql_last'] = None
                st.error(f"SQL Error: {e}")

    _last = st.session_state.get('sql_last')
    if _last:
        res_df = _last['result']
        st.markdown(f'<div class="alert-success">✅ Query returned <b>{len(res_df):,} rows</b> × <b>{len(res_df.columns)} columns</b></div>', unsafe_allow_html=True)
        st.dataframe(res_df, width='stretch', height=400)
        download_button(res_df, "csv", "📥 Download Result", "sql_dl")

    if st.session_state.sql_history:
        st.markdown("### 📜 Query History")
        for i, h in enumerate(reversed(st.session_state.sql_history[-10:])):
            with st.expander(f"{h['time'].strftime('%H:%M:%S')} · {h['rows']} rows · {h['query'][:60].replace(chr(10), ' ')}", expanded=False):
                st.code(h['query'], language='sql')
                st.button("↩️ Re-run", key=f"sql_rerun_{i}", on_click=_sql_set_query, args=(h['query'],))




# ═══════════════════════════════════════════════════════════
# TAB 11: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
def normalize_shap(sv, est, Xt):
    """Return a 2-D (samples × features) array whatever SHAP version / model type produced."""
    if isinstance(sv, list):
        sv = np.stack([np.asarray(s) for s in sv], axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 3:                                           # (samples, features, classes) → predicted class
        try:
            pred = np.asarray(est.predict(Xt)).astype(int)
        except Exception:
            pred = np.zeros(len(sv), dtype=int)
        pred = np.clip(pred, 0, sv.shape[2] - 1)
        sv = sv[np.arange(len(sv)), :, pred]
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv


def compute_shap(pipe, X_raw, ptype, n_rows):
    X_raw = X_raw.iloc[:n_rows]
    Xt = np.asarray(pipe[:-1].transform(X_raw), dtype=float)
    names = transformed_feature_names(pipe)
    est = final_estimator(pipe)
    method = 'Tree'
    try:
        sv = shap.TreeExplainer(est).shap_values(Xt, check_additivity=False)
    except Exception:
        try:
            if not hasattr(est, 'coef_'):
                raise ValueError('not linear')
            method = 'Linear'
            sv = shap.LinearExplainer(est, Xt).shap_values(Xt)
        except Exception:
            method = 'Kernel'
            bg = shap.sample(Xt, min(30, len(Xt)), random_state=0)
            fn = est.predict_proba if (ptype == 'classification' and hasattr(est, 'predict_proba')) else est.predict
            Xt = Xt[:min(60, len(Xt))]
            sv = shap.KernelExplainer(fn, bg).shap_values(Xt, nsamples=100)
    sv = normalize_shap(sv, est, Xt)
    return sv, pd.DataFrame(Xt, columns=names), names, method


with tabs[10]:
    st.markdown("## 🧠 SHAP — Model Explainability")

    if not SHAP_AVAILABLE:
        st.markdown('<div class="alert-warning">⚠️ <b>SHAP</b> not installed. Run: <code>pip install shap</code></div>', unsafe_allow_html=True)
    elif not st.session_state.trained_models:
        st.markdown('<div class="alert-warning">⚠️ No trained models found. Train a model in the ML Models or AutoML tab first.</div>', unsafe_allow_html=True)
    else:
        _names = list(st.session_state.trained_models.keys())
        _best = st.session_state.best_model
        sel_model_name = st.selectbox("🤖 Select Model to Explain", _names, index=_names.index(_best) if _best in _names else 0, key="shap_model")
        model_info = st.session_state.trained_models[sel_model_name]
        _sp = model_info.get('shap_pipeline', model_info['pipeline'])
        if not hasattr(_sp, 'named_steps'):
            _sp = model_info['pipeline']
        if _sp is not model_info['pipeline']:
            st.caption("ℹ️ Ensemble selected — explaining its strongest member model.")
        n_rows = st.slider("Rows to explain", 20, 500, min(150, max(20, len(model_info['X_test']))), 10, key="shap_n")

        if st.button("🧠 Generate SHAP Explanation", width='stretch', type="primary", key="shap_run"):
            try:
                with st.spinner("Computing SHAP values... (Kernel fallback can take up to a minute)"):
                    sv, Xt_df, fnames, method = compute_shap(_sp, model_info['X_test'], model_info['type'], n_rows)
                st.session_state.shap_values = {'vals': sv, 'data': Xt_df, 'features': fnames,
                                                'model': sel_model_name, 'method': method, 'ptype': model_info['type']}
                st.success(f"✅ SHAP values computed ({method} explainer).")
            except Exception as e:
                st.error(f"SHAP error: {e}")

        sres = st.session_state.shap_values
        if sres and sres.get('model') == sel_model_name:
            sv_arr, features, Xd = sres['vals'], sres['features'], sres['data']
            t1, t2, t3, t4 = st.tabs(["📊 Feature Importance", "🔍 Sample Explanation", "🌡️ Heatmap", "🐝 Beeswarm"])

            with t1:
                imp = pd.DataFrame({'Feature': features, 'SHAP Importance': np.abs(sv_arr).mean(axis=0)})
                imp = imp.sort_values('SHAP Importance', ascending=True).tail(20)
                fig = px.bar(imp, x='SHAP Importance', y='Feature', orientation='h', color='SHAP Importance',
                             color_continuous_scale='Viridis', title="Mean |SHAP| — Global Feature Importance")
                fig.update_layout(**plotly_dark_layout(height=max(380, 26 * len(imp) + 120), coloraxis_showscale=False))
                st.plotly_chart(fig, width='stretch', key="shap_imp")

            with t2:
                n_s = len(sv_arr)
                sample_idx = st.slider("Select Sample Index", 0, n_s - 1, 0, key="shap_sample") if n_s > 1 else 0
                sdf = pd.DataFrame({'Feature': features, 'SHAP Value': sv_arr[sample_idx], 'Feature Value': Xd.iloc[sample_idx].to_numpy()})
                sdf['Direction'] = np.where(sdf['SHAP Value'] > 0, '▲ Increases prediction', '▼ Decreases prediction')
                sdf = sdf.reindex(sdf['SHAP Value'].abs().sort_values(ascending=False).index).head(15)
                fig2 = go.Figure(go.Bar(x=sdf['SHAP Value'][::-1], y=sdf['Feature'][::-1], orientation='h',
                                        marker_color=['#43E97B' if v > 0 else '#FF4757' for v in sdf['SHAP Value'][::-1]]))
                fig2.update_layout(**plotly_dark_layout(title=f"Sample #{sample_idx} — Top Feature Contributions", height=500))
                st.plotly_chart(fig2, width='stretch', key="shap_sample_fig")
                st.dataframe(sdf[['Feature', 'Feature Value', 'SHAP Value', 'Direction']].reset_index(drop=True), width='stretch', hide_index=True)

            with t3:
                top_n = min(12, len(features))
                top_idx = np.argsort(np.abs(sv_arr).mean(axis=0))[-top_n:][::-1]
                fig3 = px.imshow(sv_arr[:, top_idx].T, x=list(range(len(sv_arr))), y=[features[i] for i in top_idx],
                                 color_continuous_scale='RdBu_r', color_continuous_midpoint=0, aspect='auto',
                                 title="SHAP Values Heatmap (features × samples)",
                                 labels={'x': 'Sample Index', 'y': 'Feature', 'color': 'SHAP'})
                fig3.update_layout(**plotly_dark_layout(height=500))
                st.plotly_chart(fig3, width='stretch', key="shap_heat")

            with t4:
                top_n = min(10, len(features))
                top_idx = np.argsort(np.abs(sv_arr).mean(axis=0))[-top_n:]
                fig4 = go.Figure()
                rng_ = np.random.RandomState(0)
                for rank, j in enumerate(top_idx):
                    v = Xd.iloc[:, j].to_numpy(dtype=float)
                    norm = (v - v.min()) / (np.ptp(v) if np.ptp(v) > 0 else 1.0)
                    fig4.add_trace(go.Scatter(x=sv_arr[:, j], y=rank + rng_.uniform(-0.28, 0.28, len(v)), mode='markers',
                                              marker=dict(size=5, color=norm, colorscale='Bluered', opacity=0.75,
                                                          showscale=(rank == len(top_idx) - 1),
                                                          colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["low", "high"])),
                                              showlegend=False, hovertemplate=f"{features[j]}<br>SHAP=%{{x:.3f}}<extra></extra>"))
                fig4.update_yaxes(tickvals=list(range(len(top_idx))), ticktext=[features[j] for j in top_idx])
                fig4.update_layout(**plotly_dark_layout(title="Beeswarm — impact vs feature value", height=520, xaxis_title="SHAP value"))
                st.plotly_chart(fig4, width='stretch', key="shap_bee")




# ═══════════════════════════════════════════════════════════
# TAB 12: AI ASSISTANT (Groq — free)
# ═══════════════════════════════════════════════════════════
def _get_api_key():
    k = st.session_state.get('ai_api_key', '')
    if k:
        return k
    k = os.environ.get('GROQ_API_KEY', '')
    if k:
        return k
    # only touch st.secrets if a secrets file exists (older Streamlit shows a red error box otherwise)
    if any(os.path.exists(os.path.join(d, '.streamlit', 'secrets.toml')) for d in (os.getcwd(), os.path.expanduser('~'))):
        try:
            return st.secrets.get('GROQ_API_KEY', '')
        except Exception:
            return ''
    return ''


with tabs[11]:
    st.markdown("## 💬 AI Data Assistant")
    with st.expander("🔑 Groq API Key Setup (Free)", expanded=not _get_api_key()):
        st.markdown('<div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.25);border-radius:12px;padding:14px;margin-bottom:12px">'
                    '<div style="font-weight:700;color:#a8a4ff;margin-bottom:6px">🆓 Groq — Free & Fast</div>'
                    '<div style="font-size:13px;color:rgba(232,233,240,0.7)">✅ Completely FREE &nbsp;·&nbsp; Llama 3.3 70B &nbsp;·&nbsp; Super fast<br>'
                    '🔗 Key: <a href="https://console.groq.com/keys" target="_blank" style="color:#a8a4ff"><b>console.groq.com/keys</b></a> → Sign up → Create API Key<br>'
                    'Tip: you can also set the <code>GROQ_API_KEY</code> environment variable or Streamlit secret.</div></div>', unsafe_allow_html=True)
        key_input = st.text_input("Groq API Key", type="password", value=st.session_state.get('ai_api_key', ''), placeholder="gsk_...", key="api_key_field")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Key", width='stretch', key="save_key_btn"):
                if key_input and len(key_input) > 10:
                    st.session_state['ai_api_key'] = key_input.strip(); st.rerun()
                else:
                    st.error("❌ Enter a valid key")
        with c2:
            if st.button("🗑️ Clear Key", width='stretch', key="clear_key_btn"):
                st.session_state['ai_api_key'] = ''; st.rerun()

    api_key = _get_api_key()
    if not api_key:
        st.markdown('<div class="alert-warning">⚠️ <b>Groq API key not set.</b> Add it in the setup section above. '
                    '<a href="https://console.groq.com/keys" target="_blank" style="color:#F9AB00"><b>Get a free key</b></a></div>', unsafe_allow_html=True)
    else:
        nc_cols, cc_cols = get_num_cols(df), get_cat_cols(df)
        miss_cols = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
        try: desc_stats = df[nc_cols[:20]].describe().round(3).to_dict() if nc_cols else {}
        except Exception: desc_stats = {}
        model_lines = []
        for n_, mi_ in list(st.session_state.trained_models.items())[:8]:
            mt_ = mi_.get('metrics', {})
            model_lines.append(f"{n_} ({mi_['type']}, target={mi_['target']}, score={mt_.get('Score', float('nan')):.4f})")
        data_ctx = (f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.\n"
                    f"Numeric columns ({len(nc_cols)}): {', '.join(map(str, nc_cols[:25]))}.\n"
                    f"Categorical columns ({len(cc_cols)}): {', '.join(map(str, cc_cols[:25]))}.\n"
                    f"Missing values: {int(df.isnull().sum().sum())} {miss_cols if miss_cols else '(none)'}.\n"
                    f"Duplicates: {int(df.duplicated().sum())}.\nStats: {desc_stats}.\n"
                    f"Trained models: {'; '.join(model_lines) or 'None'}.\n")
        system_prompt = ("You are an expert data scientist embedded in an ML Analytics app. Dataset context:\n\n" + data_ctx +
                         "\nBe concise and actionable. Reference actual column names. Use markdown, bullet points and code blocks where helpful.")

        st.markdown("**💡 Quick questions:**")
        qcols = st.columns(3)
        quick_prompts = ["What are the main patterns in this data?", "Which columns have data quality issues?", "What ML model would you recommend?",
                         "Which features are most important?", "Are there outliers I should handle?", "What transforms to apply before modeling?",
                         "Give me a full EDA summary.", "How to handle the missing values?", "Classification or regression problem?"]

        def _prefill(q):
            st.session_state['ai_chat_input'] = q
        for i, qp in enumerate(quick_prompts):
            with qcols[i % 3]:
                st.button(qp[:36] + ("…" if len(qp) > 36 else ""), key=f"qp_{i}", width='stretch', on_click=_prefill, args=(qp,))

        if st.session_state.chat_history:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div style="display:flex;justify-content:flex-end;margin:10px 0"><div style="background:rgba(108,99,255,0.12);'
                                f'border:1px solid rgba(108,99,255,0.3);border-radius:16px 16px 4px 16px;padding:12px 18px;max-width:80%;font-size:14px">'
                                f'👤 <b>You</b><br><span style="color:#E8E9F0">{esc(msg["content"])}</span></div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:rgba(67,233,123,0.06);border:1px solid rgba(67,233,123,0.2);border-radius:4px 16px 16px 16px;'
                                'padding:12px 18px;margin:10px 0;font-size:14px">🤖 <b>AI Assistant</b></div>', unsafe_allow_html=True)
                    st.markdown(msg["content"])
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        user_input = st.text_area("Ask anything about your data:", height=100, placeholder="e.g. What patterns exist? Which model is best?", key="ai_chat_input")
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1: send_btn = st.button("📤 Send", width='stretch', type="primary", key="ai_send")
        with c2: clear_btn = st.button("🗑️ Clear", width='stretch', key="ai_clear")
        with c3: pass
        if st.session_state.chat_history:
            chat_txt = "\n\n".join(f"{'YOU' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in st.session_state.chat_history)
            with c3:
                st.download_button("📥 Export", chat_txt, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain", key="chat_dl", width='stretch')
        if clear_btn:
            st.session_state.chat_history = []; st.rerun()

        if send_btn and user_input.strip():
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            ai_reply = ""
            try:
                with st.spinner("🤖 AI is thinking..."):
                    msgs = [{"role": "system", "content": system_prompt}] + \
                           [{"role": m['role'], "content": m['content']} for m in st.session_state.chat_history[-12:]]
                    resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                                         json={"model": "llama-3.3-70b-versatile", "messages": msgs, "max_tokens": 1500}, timeout=60)
                    if resp.status_code == 200: ai_reply = resp.json()['choices'][0]['message']['content']
                    elif resp.status_code == 401: ai_reply = "❌ **Invalid Groq API Key.** Check at [console.groq.com/keys](https://console.groq.com/keys)"
                    elif resp.status_code == 429: ai_reply = "⚠️ **Rate limit hit.** Wait a moment and retry."
                    else: ai_reply = f"❌ **Groq Error {resp.status_code}:** {resp.text[:300]}"
            except requests.exceptions.Timeout:
                ai_reply = "⏱️ **Request timed out.** Try again."
            except requests.exceptions.ConnectionError:
                ai_reply = "🌐 **Connection error.** Check your internet."
            except Exception as e:
                ai_reply = f"❌ **Error:** {e}"
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_reply})
            st.rerun()


# ═══════════════════════════════════════════════════════════
# TAB 13: EXPORT
# ═══════════════════════════════════════════════════════════
with tabs[12]:
    st.markdown("## 💾 Export Data & Models")
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    st.markdown("""
    <div class="glass-card" style="border-color:rgba(67,233,123,0.4);background:rgba(67,233,123,0.05);">
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="font-size:36px;">📥</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:#43E97B;">Download Cleaned Data — No Model Training Required</div>
                <div style="font-size:13px;color:var(--text-muted);margin-top:4px;">Export your current (cleaned) dataset at any time.</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;">
            <div><div class="metric-label">ROWS</div><div class="metric-value">{df.shape[0]:,}</div></div>
            <div><div class="metric-label">COLUMNS</div><div class="metric-value">{df.shape[1]:,}</div></div>
            <div><div class="metric-label">MEMORY</div><div class="metric-value">{mem:.1f} MB</div></div>
            <div><div class="metric-label">MISSING</div><div class="metric-value">{int(df.isnull().sum().sum()):,}</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📄 Export Formats")
    c1, c2, c3, c4 = st.columns(4)
    with c1: download_button(df, "csv", "📄 CSV", "exp_csv")
    with c2: download_button(df, "excel", "📊 Excel", "exp_excel")
    with c3: download_button(df, "json", "📋 JSON", "exp_json")
    with c4:
        _pq = st.session_state.get('_pq_cache')
        _psig = (df.shape, tuple(map(str, df.columns)))
        if _pq and _pq[0] == _psig:
            st.download_button("📦 Parquet", _pq[1], f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                               "application/octet-stream", key="exp_parquet", width='stretch')
        elif st.button("⚙️ Prepare Parquet", key="exp_parquet_prep", width='stretch'):
            try:
                pb = io.BytesIO(); df.to_parquet(pb, index=False)
                st.session_state['_pq_cache'] = (_psig, pb.getvalue()); st.rerun()
            except Exception as e:
                st.error(f"Parquet export failed: {str(e)[:200]}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("🔧 Export Options — Filter Columns / Rows", expanded=False):
        exp_cols = st.multiselect("Select columns to export (default = all)", df.columns.tolist(), default=df.columns.tolist(), key="exp_col_sel")
        exp_rows = st.radio("Row filter", ["All rows", "Remove missing rows", "Custom sample %"], horizontal=True, key="exp_row_filter")
        exp_df = df[exp_cols] if exp_cols else df
        if exp_rows == "Remove missing rows":
            exp_df = exp_df.dropna(); st.caption(f"After filter: {len(exp_df):,} rows")
        elif exp_rows == "Custom sample %":
            pct = st.slider("Sample %", 10, 100, 80, key="exp_sample_pct")
            exp_df = exp_df.sample(frac=pct / 100, random_state=42); st.caption(f"Sample: {len(exp_df):,} rows")
        c1f, c2f = st.columns(2)
        with c1f: download_button(exp_df, "csv", "📄 Filtered CSV", "exp_filt_csv")
        with c2f: download_button(exp_df, "excel", "📊 Filtered Excel", "exp_filt_excel")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    if st.session_state.trained_models:
        st.markdown(f"### 🤖 Trained Models ({len(st.session_state.trained_models)})")
        for mn, mi in st.session_state.trained_models.items():
            is_best = (mn == st.session_state.best_model)
            sc_ = mi.get('metrics', {}).get('Score')
            st.markdown(f"""
            <div class="model-row {'best' if is_best else ''}">
                <span>{'🏆 ' if is_best else '✅ '}<b>{esc(mn)}</b></span>
                <span style="color:var(--text-muted);font-size:12px;">Type: {mi.get('type', '?')} · Target: {esc(mi.get('target', '?'))} ·
                Features: {len(mi.get('features', []))}{f' · Score: {sc_:.4f}' if sc_ is not None else ''}</span>
            </div>""", unsafe_allow_html=True)
        _mnames = list(st.session_state.trained_models.keys())
        _bi = _mnames.index(st.session_state.best_model) if st.session_state.best_model in _mnames else 0
        exp_model = st.selectbox("Model to export", _mnames, index=_bi, key="exp_model_sel")
        try:
            _mi = st.session_state.trained_models[exp_model]
            payload = {k: _mi[k] for k in ('pipeline', 'features', 'target', 'type', 't_enc', 'feature_meta', 'metrics')}
            st.download_button("💾 Download model (.pkl — full preprocessing + model)", pickle.dumps(payload),
                               f"model_{re.sub(r'[^0-9A-Za-z]+', '_', exp_model)}.pkl", "application/octet-stream", key="exp_model_dl", width='stretch')
            st.caption("Load later with:  `import pickle; m = pickle.load(open('model.pkl','rb')); m['pipeline'].predict(raw_dataframe)` — only unpickle files you trust.")
        except Exception as e:
            st.error(f"Model export failed: {e}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    if st.session_state.history:
        st.markdown("### 📜 Operation History")
        for i, h in enumerate(reversed(st.session_state.history[-20:])):
            with st.expander(f"{h['time'].strftime('%H:%M:%S')} · {h['action']}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Rows", f"{h['shape'][0]:,}")
                with c2: st.metric("Cols", f"{h['shape'][1]:,}")
                with c3: st.metric("Memory", f"{h['df'].memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
                with c4:
                    if st.button("↩️ Restore", key=f"rst_{i}", width='stretch'):
                        st.session_state.df = h['df'].copy()
                        st.session_state.data_quality_score = calculate_data_quality_score(st.session_state.df)
                        st.session_state.auto_insights = auto_generate_insights(st.session_state.df)
                        st.rerun()


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-title">🚀 ML Analytics Pro v4.0 ULTRA</div>
    <div class="footer-sub">
        XGBoost · LightGBM · Scikit-learn · SHAP · Optuna · Plotly · Streamlit<br>
        Statistical Tests · SQL Query · Auto Dtype Fixer · AI Assistant · AutoML
    </div>
</div>
""", unsafe_allow_html=True)
