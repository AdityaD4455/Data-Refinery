"""
╔══════════════════════════════════════════════════════════════╗
║          ML ANALYTICS PRO — v3.0 ELITE EDITION              ║
║    Professional • Fast • Accurate • Beautiful               ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
from functools import lru_cache
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   LabelEncoder, QuantileTransformer, PowerTransformer)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor,
                               VotingClassifier, VotingRegressor, IsolationForest)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, KFold)
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              mean_squared_error, r2_score, mean_absolute_error,
                              mean_absolute_percentage_error, silhouette_score)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer, KNNImputer
import xgboost as xgb
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import json
from scipy import stats
from scipy.stats import shapiro, gaussian_kde, ttest_ind, chi2_contingency, f_oneway, mannwhitneyu
import requests

# Optional advanced imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import pandasql as pdsql
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="ML Analytics Pro v3",
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
    'chat_history': [], 'sql_history': [], 'shap_values': None

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
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
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
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]


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
        <h4>⚡ {ch['action']}</h4>
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
    ti = sum(1 for c in df.columns if df[c].dtype == 'object' and pd.to_numeric(df[c], errors='coerce').notna().mean() > 0.5)
    if ti:
        pen = min(15, ti * 3)
        score -= pen
        issues.append(f"🔵 Type mismatches: {ti} cols (−{pen:.0f} pts)")
    nc = df.select_dtypes(include=np.number).columns
    if len(nc):
        oc = sum(((df[c] - df[c].mean()).abs() > 3 * df[c].std()).sum() for c in nc)
        op = oc / (len(df) * len(nc)) * 100
        if op > 5:
            pen = min(10, (op - 5) * 1.5)
            score -= pen
            issues.append(f"🟠 Outliers: {op:.1f}% (−{pen:.0f} pts)")
    return max(0.0, score), issues


def auto_generate_insights(df: pd.DataFrame):
    insights = []
    nc = df.select_dtypes(include=np.number).columns.tolist()
    cc = df.select_dtypes(include='object').columns.tolist()
    for col in nc[:6]:
        sk = df[col].skew()
        if abs(sk) > 1.5:
            insights.append(f"📊 <b>{col}</b> is {'heavily right' if sk > 0 else 'heavily left'}-skewed (skewness={sk:.2f}). Consider log transform.")
        if df[col].nunique() < 12:
            insights.append(f"🎯 <b>{col}</b> has only {df[col].nunique()} unique values — suitable as a classification target.")
    if len(nc) > 1:
        corr = df[nc].corr()
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                v = corr.iloc[i, j]
                if abs(v) > 0.75:
                    insights.append(f"🔗 <b>{corr.columns[i]}</b> ↔ <b>{corr.columns[j]}</b> highly correlated (r={v:.2f}). Consider dropping one.")
    mc = df.isnull().sum()
    big_miss = mc[mc > len(df) * 0.2]
    for col, cnt in big_miss.items():
        insights.append(f"❓ <b>{col}</b> has {cnt/len(df)*100:.1f}% missing values — investigate before imputing.")
    for col in cc[:4]:
        vc = df[col].value_counts(normalize=True)
        if len(vc) and vc.iloc[0] > 0.6:
            insights.append(f"📈 <b>{col}</b> is dominated by '{vc.index[0]}' ({vc.iloc[0]*100:.0f}%) — possible class imbalance.")
    return insights[:8]


def prepare_ml_data(df, target_col, feature_cols, use_knn=False):
    df_c = df.dropna(subset=[target_col]).copy()
    for col in feature_cols:
        if col not in df_c.columns:
            continue
        if df_c[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            if use_knn:
                pass  # handled below
            else:
                df_c[col] = df_c[col].fillna(df_c[col].median())
        else:
            df_c[col] = df_c[col].fillna('__missing__')
    if use_knn:
        num_feats = [c for c in feature_cols if df_c[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        if num_feats:
            imp = KNNImputer(n_neighbors=5)
            df_c[num_feats] = imp.fit_transform(df_c[num_feats])
    X = df_c[feature_cols].copy()
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    y = df_c[target_col].copy()
    t_enc = None
    n_unique = y.nunique()
    is_int = (y.dropna() % 1 == 0).all() if pd.api.types.is_numeric_dtype(y) else False
    if y.dtype == 'object' or (n_unique < 25 and (is_int or y.dtype == 'object') and n_unique / len(y) < 0.05):
        problem_type = 'classification'
    else:
        problem_type = 'regression'
    if problem_type == 'classification' and y.dtype == 'object':
        t_enc = LabelEncoder()
        y = t_enc.fit_transform(y.astype(str))
    elif problem_type == 'classification' and pd.api.types.is_numeric_dtype(y):
        y = y.astype(int)
    return X, y, encoders, t_enc, problem_type


def download_button(df, fmt, label, key):
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if fmt == "csv":
            data = df.to_csv(index=False)
            st.download_button(label, data, f"data_{ts}.csv", "text/csv", key=key, use_container_width=True)
        elif fmt == "excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as w:
                df.to_excel(w, index=False, sheet_name='Data')
            st.download_button(label, buf.getvalue(), f"data_{ts}.xlsx", "application/vnd.ms-excel", key=key, use_container_width=True)
        elif fmt == "json":
            st.download_button(label, df.to_json(orient='records', indent=2), f"data_{ts}.json", "application/json", key=key, use_container_width=True)
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
with st.sidebar:
    st.markdown("### 📁 Dataset")
    uploaded = st.file_uploader("Upload file", type=['csv', 'xlsx', 'json', 'parquet'], key="main_upload", label_visibility="collapsed")
    if uploaded:
        try:
            ext = uploaded.name.split('.')[-1].lower()
            readers = {'csv': pd.read_csv, 'xlsx': pd.read_excel, 'parquet': pd.read_parquet, 'json': pd.read_json}
            raw = readers.get(ext, pd.read_csv)(uploaded)
            if st.session_state.df is None or not st.session_state.history:
                st.session_state.df = raw
                push_history(raw, "📁 Dataset uploaded")
                st.session_state.auto_insights = auto_generate_insights(raw)
                st.session_state.data_quality_score = calculate_data_quality_score(raw)
                st.rerun()
            else:
                if st.button("🔄 Replace current data", use_container_width=True):
                    st.session_state.df = raw
                    push_history(raw, "📁 Dataset replaced")
                    st.session_state.auto_insights = auto_generate_insights(raw)
                    st.session_state.data_quality_score = calculate_data_quality_score(raw)
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("### 📊 Second Dataset")
    up2 = st.file_uploader("For merging", type=['csv', 'xlsx', 'json'], key="second_upload", label_visibility="collapsed")
    if up2:
        try:
            ext2 = up2.name.split('.')[-1].lower()
            df2_raw = {'csv': pd.read_csv, 'xlsx': pd.read_excel, 'json': pd.read_json}.get(ext2, pd.read_csv)(up2)
            st.session_state.df2 = df2_raw
            st.success(f"✅ {df2_raw.shape[0]:,} rows loaded")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🎲 Load Sample Dataset", use_container_width=True):
        np.random.seed(42)
        n = 3000
        sample = pd.DataFrame({
            'Age': np.random.randint(18, 75, n),
            'Income': (np.random.lognormal(10.8, 0.6, n)).astype(int),
            'CreditScore': np.random.randint(300, 850, n),
            'Experience': np.random.randint(0, 35, n),
            'LoanAmount': np.random.randint(5000, 500000, n),
            'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.25, 0.45, 0.22, 0.08]),
            'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'Finance', 'HR'], n),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
            'Satisfaction': np.random.randint(1, 11, n),
            'Churned': np.random.choice([0, 1], n, p=[0.70, 0.30])
        })
        # Add some realistic missing values
        for col in ['Income', 'CreditScore', 'LoanAmount']:
            mask = np.random.rand(n) < 0.04
            sample.loc[mask, col] = np.nan
        st.session_state.df = sample
        push_history(sample, "🎲 Sample data loaded")
        st.session_state.auto_insights = auto_generate_insights(sample)
        st.session_state.data_quality_score = calculate_data_quality_score(sample)
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
            for i, h in enumerate(reversed(st.session_state.history[-10:])):
                if st.button(f"↩️ {h['action'][:30]}", key=f"hist_{i}", use_container_width=True):
                    st.session_state.df = h['df'].copy()
                    st.rerun()


# ─────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────
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

    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    dup_pct = df.duplicated().sum() / len(df) * 100
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    num_cols = df.select_dtypes(include=np.number).shape[1]

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "📝", "ROWS", f"{df.shape[0]:,}", "Total records"),
        (c2, "🔢", "COLUMNS", f"{df.shape[1]:,}", f"{num_cols} numeric"),
        (c3, "❓", "MISSING", f"{missing_pct:.1f}%", f"{'✅ Clean' if missing_pct < 5 else '⚠️ Needs attention'}"),
        (c4, "💾", "MEMORY", f"{mem_mb:.1f} MB", "In memory"),
        (c5, "🔄", "DUPES", f"{dup_pct:.1f}%", f"{df.duplicated().sum():,} rows"),
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
        n = st.slider("Rows to show", 5, min(200, len(df)), 20, key="prev_rows")
        st.dataframe(df.head(n), use_container_width=True, height=400)
    with c2:
        st.markdown("### 🗂️ Column Summary")
        info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(1),
            'Unique': df.nunique(),
        })
        st.dataframe(info, use_container_width=True, height=400)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📈 Statistics")
        nc = df.select_dtypes(include=np.number)
        if len(nc.columns):
            s = nc.describe().T.round(3)
            s['cv%'] = (s['std'] / s['mean'].abs() * 100).round(1)
            s['range'] = s['max'] - s['min']
            st.dataframe(s, use_container_width=True, height=350)
    with c2:
        st.markdown("### 🗃️ Data Types")
        tc = df.dtypes.value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=tc.index.astype(str), values=tc.values, hole=0.55,
            marker=dict(colors=['#6C63FF', '#FF6584', '#43E97B', '#38F9D7', '#F9AB00']),
            textfont=dict(size=13, color='white')
        )])
        fig.update_layout(**plotly_dark_layout(height=350, showlegend=True))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: CLEANING
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🔧 Advanced Data Cleaning")

    # Missing Values
    with st.expander("❓ Missing Values", expanded=True):
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss):
            miss_df = pd.DataFrame({'Column': miss.index, 'Missing': miss.values,
                                    'Percent': (miss.values / len(df) * 100).round(2)})
            fig = px.bar(miss_df, x='Column', y='Percent', color='Percent',
                        color_continuous_scale='Reds', title='Missing Data %',
                        text=miss_df['Percent'].apply(lambda x: f'{x:.1f}%'))
            fig.update_traces(textposition='outside')
            fig.update_layout(**plotly_dark_layout(height=350, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                mc = st.selectbox("Column", miss_df['Column'].tolist(), key="miss_col")
            with c2:
                strat = st.selectbox("Strategy", ["Drop rows", "Fill mean", "Fill median", "Fill mode",
                                                   "Forward fill", "Backward fill", "KNN Imputer", "Custom value"], key="miss_strat")
            with c3:
                cval = st.text_input("Custom value", "", key="miss_cval") if strat == "Custom value" else None

            if st.button("🔧 Apply", key="miss_apply", use_container_width=True):
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
                        if len(m): dc[mc] = dc[mc].fillna(m[0])
                    elif strat == "Forward fill":
                        dc[mc] = dc[mc].ffill()
                    elif strat == "Backward fill":
                        dc[mc] = dc[mc].bfill()
                    elif strat == "KNN Imputer":
                        if dc[mc].dtype in [np.float64, np.int64]:
                            dc[mc] = KNNImputer(n_neighbors=5).fit_transform(dc[[mc]])
                        else:
                            st.warning("KNN only works on numeric columns")
                            st.stop()
                    elif strat == "Custom value":
                        dc[mc] = dc[mc].fillna(cval)
                    push_history(dc, f"🔧 {strat}: {mc}")
                    st.session_state.df = dc
                    st.session_state.data_quality_score = calculate_data_quality_score(dc)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown('<div class="alert-success">✅ No missing values found! Dataset is complete.</div>', unsafe_allow_html=True)

    # Duplicates
    with st.expander("🔄 Duplicates", expanded=True):
        dup_n = df.duplicated().sum()
        if dup_n > 0:
            st.markdown(f'<div class="alert-warning">⚠️ Found {dup_n:,} duplicate rows ({dup_n/len(df)*100:.2f}%)</div>', unsafe_allow_html=True)
            st.dataframe(df[df.duplicated(keep=False)].head(10), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Remove All Duplicates", key="rm_dups", use_container_width=True):
                    dc = df.drop_duplicates()
                    push_history(dc, f"🗑️ Removed {dup_n} duplicates")
                    st.session_state.df = dc
                    st.session_state.data_quality_score = calculate_data_quality_score(dc)
                    st.rerun()
            with c2:
                sub = st.multiselect("Remove by subset", df.columns.tolist(), key="dup_sub")
                if sub and st.button("🗑️ Remove by Subset", key="rm_sub_dups", use_container_width=True):
                    dc = df.drop_duplicates(subset=sub)
                    removed = len(df) - len(dc)
                    push_history(dc, f"🗑️ Removed {removed} duplicates (subset)")
                    st.session_state.df = dc
                    st.session_state.data_quality_score = calculate_data_quality_score(dc)
                    st.rerun()
        else:
            st.markdown('<div class="alert-success">✅ No duplicate rows found!</div>', unsafe_allow_html=True)

    # Outliers
    with st.expander("🎯 Outlier Detection & Removal", expanded=True):
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if nc:
            c1, c2, c3 = st.columns(3)
            with c1:
                oc = st.selectbox("Column", nc, key="out_col")
            with c2:
                om = st.selectbox("Detection Method", ["IQR (1.5×)", "IQR (3×)", "Z-Score (2σ)", "Z-Score (3σ)",
                                                        "Isolation Forest", "Modified Z-Score"], key="out_meth")
            with c3:
                show_viz = st.checkbox("Show visualization", True, key="out_viz")

            if show_viz:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Box Plot", "Distribution"))
                fig.add_trace(go.Box(y=df[oc], name=oc, marker_color='#6C63FF', boxmean=True), row=1, col=1)
                fig.add_trace(go.Histogram(x=df[oc], name=oc, marker_color='#FF6584', nbinsx=60, opacity=0.8), row=1, col=2)
                fig.update_layout(**plotly_dark_layout(height=380, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            for col_ref, label, val in [(c1, "Min", df[oc].min()), (c2, "Max", df[oc].max()),
                                         (c3, "Mean", df[oc].mean()), (c4, "Std", df[oc].std())]:
                with col_ref:
                    st.metric(label, f"{val:.2f}")

            if st.button("🗑️ Remove Outliers", key="rm_out", use_container_width=True):
                dc = df.copy()
                try:
                    if "IQR (1.5" in om:
                        q1, q3 = dc[oc].quantile(0.25), dc[oc].quantile(0.75)
                        iqr = q3 - q1
                        dc = dc[(dc[oc] >= q1 - 1.5*iqr) & (dc[oc] <= q3 + 1.5*iqr)]
                    elif "IQR (3" in om:
                        q1, q3 = dc[oc].quantile(0.25), dc[oc].quantile(0.75)
                        iqr = q3 - q1
                        dc = dc[(dc[oc] >= q1 - 3*iqr) & (dc[oc] <= q3 + 3*iqr)]
                    elif "Z-Score (2" in om:
                        z = np.abs((dc[oc] - dc[oc].mean()) / dc[oc].std())
                        dc = dc[z < 2]
                    elif "Z-Score (3" in om:
                        z = np.abs((dc[oc] - dc[oc].mean()) / dc[oc].std())
                        dc = dc[z < 3]
                    elif "Isolation" in om:
                        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
                        pred = iso.fit_predict(dc[[oc]])
                        dc = dc[pred == 1]
                    elif "Modified" in om:
                        med = dc[oc].median()
                        mad = np.median(np.abs(dc[oc] - med))
                        mz = 0.6745 * (dc[oc] - med) / (mad + 1e-8)
                        dc = dc[np.abs(mz) < 3.5]
                    removed = len(df) - len(dc)
                    push_history(dc, f"🎯 Removed {removed} outliers ({oc})")
                    st.session_state.df = dc
                    st.session_state.data_quality_score = calculate_data_quality_score(dc)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No numeric columns found for outlier detection.")

    # Encoding
    with st.expander("🏷️ Categorical Encoding", expanded=False):
        cat_c = df.select_dtypes(include='object').columns.tolist()
        if cat_c:
            c1, c2 = st.columns(2)
            with c1:
                enc_col = st.selectbox("Column", cat_c, key="enc_col")
            with c2:
                enc_type = st.selectbox("Encoding", ["Label Encoding", "One-Hot Encoding", "Frequency Encoding"], key="enc_type")
            uniq = df[enc_col].nunique()
            st.caption(f"Unique values: {uniq}")
            if uniq <= 25:
                vc = df[enc_col].value_counts().head(25)
                fig = px.bar(x=vc.index, y=vc.values, labels={'x': enc_col, 'y': 'Count'},
                             color=vc.values, color_continuous_scale='Viridis')
                fig.update_layout(**plotly_dark_layout(height=300, coloraxis_showscale=False))
                st.plotly_chart(fig, use_container_width=True)
            if st.button("🏷️ Encode Column", key="enc_btn", use_container_width=True):
                dc = df.copy()
                try:
                    if enc_type == "Label Encoding":
                        dc[enc_col] = LabelEncoder().fit_transform(dc[enc_col].astype(str))
                    elif enc_type == "One-Hot Encoding":
                        dc = pd.get_dummies(dc, columns=[enc_col], prefix=enc_col)
                    elif enc_type == "Frequency Encoding":
                        freq = dc[enc_col].value_counts(normalize=True)
                        dc[f'{enc_col}_freq'] = dc[enc_col].map(freq)
                    push_history(dc, f"🏷️ {enc_type}: {enc_col}")
                    st.session_state.df = dc
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No categorical columns detected.")

    # Scaling
    with st.expander("⚖️ Feature Scaling", expanded=False):
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if nc:
            c1, c2 = st.columns(2)
            with c1:
                scale_cols = st.multiselect("Columns to scale", nc, default=nc[:min(5, len(nc))], key="scale_cols")
            with c2:
                scaler_type = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler",
                                                       "QuantileTransformer", "PowerTransformer"], key="scale_type")
            if scale_cols and st.button("⚖️ Apply Scaling", key="scale_btn", use_container_width=True):
                dc = df.copy()
                try:
                    scalers = {
                        "StandardScaler": StandardScaler(),
                        "MinMaxScaler": MinMaxScaler(),
                        "RobustScaler": RobustScaler(),
                        "QuantileTransformer": QuantileTransformer(output_distribution='normal'),
                        "PowerTransformer": PowerTransformer()
                    }
                    dc[scale_cols] = scalers[scaler_type].fit_transform(dc[scale_cols])
                    push_history(dc, f"⚖️ {scaler_type}")
                    st.session_state.df = dc
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Auto Data Type Fixer ──
    with st.expander("🔠 Auto Data Type Fixer", expanded=False):
        st.markdown("**Automatically detects and fixes wrong data types in your dataset.**")
        suggestions = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(100)
            if dtype == 'object':
                # Check if it's numeric stored as string
                try:
                    pd.to_numeric(sample)
                    suggestions.append({'Column': col, 'Current': dtype, 'Suggested': 'numeric (float/int)', 'Reason': 'Numeric values stored as text', 'fix': 'numeric'})
                    continue
                except: pass
                # Check if it's datetime
                try:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    suggestions.append({'Column': col, 'Current': dtype, 'Suggested': 'datetime', 'Reason': 'Date/time values stored as text', 'fix': 'datetime'})
                    continue
                except: pass
                # Check if boolean
                uniq = set(sample.astype(str).str.lower().unique())
                if uniq <= {'true','false','yes','no','1','0','y','n'}:
                    suggestions.append({'Column': col, 'Current': dtype, 'Suggested': 'boolean', 'Reason': 'Boolean values stored as text', 'fix': 'bool'})
                    continue
                # Check if low-cardinality → category
                if df[col].nunique() / len(df) < 0.05 and df[col].nunique() < 50:
                    suggestions.append({'Column': col, 'Current': dtype, 'Suggested': 'category', 'Reason': f'Only {df[col].nunique()} unique values — category saves memory', 'fix': 'category'})
            elif dtype in ['float64', 'float32']:
                if (sample.dropna() % 1 == 0).all() and df[col].notna().all():
                    suggestions.append({'Column': col, 'Current': dtype, 'Suggested': 'int64', 'Reason': 'Float column contains only whole numbers', 'fix': 'int'})

        if suggestions:
            sdf = pd.DataFrame(suggestions)[['Column','Current','Suggested','Reason']]
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            cols_to_fix = st.multiselect("Select columns to fix", [s['Column'] for s in suggestions],
                                         default=[s['Column'] for s in suggestions], key="dtype_fix_cols")
            if st.button("🔧 Apply Type Fixes", use_container_width=True, key="dtype_fix_btn"):
                dc = df.copy()
                fixed = []
                for s in suggestions:
                    if s['Column'] not in cols_to_fix: continue
                    try:
                        if s['fix'] == 'numeric':   dc[s['Column']] = pd.to_numeric(dc[s['Column']], errors='coerce')
                        elif s['fix'] == 'datetime': dc[s['Column']] = pd.to_datetime(dc[s['Column']], infer_datetime_format=True, errors='coerce')
                        elif s['fix'] == 'bool':    dc[s['Column']] = dc[s['Column']].astype(str).str.lower().map({'true':True,'false':False,'yes':True,'no':False,'1':True,'0':False,'y':True,'n':False})
                        elif s['fix'] == 'category': dc[s['Column']] = dc[s['Column']].astype('category')
                        elif s['fix'] == 'int':     dc[s['Column']] = dc[s['Column']].astype('int64')
                        fixed.append(s['Column'])
                    except Exception as ex:
                        st.warning(f"Could not fix {s['Column']}: {ex}")
                if fixed:
                    push_history(dc, f"🔠 Auto dtype fix: {', '.join(fixed)}")
                    st.session_state.df = dc
                    st.success(f"✅ Fixed {len(fixed)} columns!")
                    st.rerun()
        else:
            st.markdown('<div class="alert-success">✅ All data types look correct! No fixes needed.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: VISUALIZE
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 📈 Advanced Visualizations")

    viz_type = st.selectbox("Chart Type", ["Distribution Analysis", "Correlation Matrix", "Scatter Plot",
                                            "Box / Violin Plot", "3D Scatter", "Time Series / Line",
                                            "Categorical Analysis", "Pair Plot Heatmap"], key="viz_type")

    if viz_type == "Distribution Analysis":
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if nc:
            col = st.selectbox("Column", nc, key="dist_col")
            data = df[col].dropna()
            fig = make_subplots(rows=2, cols=2, subplot_titles=("Histogram + KDE", "Box Plot", "Q-Q Plot", "ECDF"))
            # Histogram
            fig.add_trace(go.Histogram(x=data, nbinsx=50, marker_color='#6C63FF', opacity=0.8, name='Hist'), row=1, col=1)
            # KDE overlay
            kde = gaussian_kde(data)
            xr = np.linspace(data.min(), data.max(), 200)
            hist_scale = len(data) * (data.max() - data.min()) / 50
            fig.add_trace(go.Scatter(x=xr, y=kde(xr) * hist_scale, mode='lines',
                                     line=dict(color='#43E97B', width=2.5), name='KDE'), row=1, col=1)
            # Box
            fig.add_trace(go.Box(y=data, marker_color='#FF6584', boxmean='sd', name='Box'), row=1, col=2)
            # Q-Q
            qq = stats.probplot(data, dist="norm")
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                     marker=dict(color='#38F9D7', size=4), name='Q-Q'), row=2, col=1)
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0],
                                     mode='lines', line=dict(color='#F9AB00', dash='dash'), name='Ref'), row=2, col=1)
            # ECDF
            x_sorted = np.sort(data)
            ecdf = np.arange(1, len(x_sorted)+1) / len(x_sorted)
            fig.add_trace(go.Scatter(x=x_sorted, y=ecdf, mode='lines',
                                     line=dict(color='#6C63FF', width=2), name='ECDF'), row=2, col=2)
            fig.update_layout(**plotly_dark_layout(height=650, showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Mean", f"{data.mean():.3f}")
                st.metric("Median", f"{data.median():.3f}")
            with c2:
                st.metric("Std Dev", f"{data.std():.3f}")
                st.metric("IQR", f"{data.quantile(0.75)-data.quantile(0.25):.3f}")
            with c3:
                st.metric("Skewness", f"{data.skew():.3f}")
                st.metric("Kurtosis", f"{data.kurtosis():.3f}")
            with c4:
                try:
                    _, p = shapiro(data[:5000])
                    st.metric("Shapiro p-val", f"{p:.4f}")
                    if p > 0.05:
                        st.markdown('<div class="alert-success">✅ Likely Normal</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-warning">⚠️ Not Normal</div>', unsafe_allow_html=True)
                except:
                    st.info("Sample too large for Shapiro test")

    elif viz_type == "Correlation Matrix":
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if len(nc) > 1:
            c1, c2 = st.columns([2, 1])
            with c1:
                meth = st.selectbox("Method", ["Pearson", "Spearman", "Kendall"], key="corr_meth")
            with c2:
                show_all = st.checkbox("Show full matrix (not triangular)", False, key="corr_full")
            corr = df[nc].corr(method=meth.lower())
            z = corr.values if show_all else np.where(np.triu(np.ones_like(corr, dtype=bool)), np.nan, corr)
            fig = go.Figure(go.Heatmap(
                z=z, x=corr.columns, y=corr.columns, colorscale='RdBu_r', zmid=0,
                text=np.round(corr.values, 2), texttemplate='%{text:.2f}', textfont=dict(size=9),
                colorbar=dict(title="r")
            ))
            fig.update_layout(**plotly_dark_layout(height=600, title=f"{meth} Correlation Matrix"))
            st.plotly_chart(fig, use_container_width=True)
            # Top correlations
            pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i,j])
                     for i in range(len(nc)) for j in range(i+1, len(nc))]
            top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:8]
            st.markdown("### 🔗 Strongest Correlations")
            st.dataframe(pd.DataFrame(top_pairs, columns=['Feature A', 'Feature B', 'Correlation']).round(4),
                        use_container_width=True)

    elif viz_type == "Scatter Plot":
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if len(nc) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1: xc = st.selectbox("X-axis", nc, key="scat_x")
            with c2: yc = st.selectbox("Y-axis", [c for c in nc if c != xc], key="scat_y")
            with c3: color_c = st.selectbox("Color by", ["None"] + df.columns.tolist(), key="scat_color")
            fig = px.scatter(df, x=xc, y=yc, color=color_c if color_c != "None" else None,
                            trendline="ols", opacity=0.65, marginal_x="histogram", marginal_y="violin",
                            title=f"{xc} vs {yc}")
            fig.update_layout(**plotly_dark_layout(height=600))
            st.plotly_chart(fig, use_container_width=True)
            corr_val = df[[xc, yc]].corr().iloc[0, 1]
            st.metric("Pearson Correlation", f"{corr_val:.4f}", f"{'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.4 else 'Weak'} correlation")

    elif viz_type == "Box / Violin Plot":
        nc = df.select_dtypes(include=np.number).columns.tolist()
        cc = df.select_dtypes(include='object').columns.tolist()
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
                groups = df[gc].unique() if gc else [None]
                colors = px.colors.qualitative.Plotly
                for i, g in enumerate(groups):
                    gdata = df[df[gc] == g][yc].dropna() if gc else df[yc].dropna()
                    name = str(g) if g is not None else yc
                    fig.add_trace(go.Violin(y=gdata, name=name, box_visible=True,
                                           fillcolor=colors[i % len(colors)], opacity=0.75, line_color='white'))
            fig.update_layout(**plotly_dark_layout(height=500))
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "3D Scatter":
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if len(nc) >= 3:
            c1, c2, c3 = st.columns(3)
            with c1: xc = st.selectbox("X", nc, key="3d_x")
            with c2: yc = st.selectbox("Y", [c for c in nc if c != xc], key="3d_y")
            with c3: zc = st.selectbox("Z", [c for c in nc if c not in [xc, yc]], key="3d_z")
            color_c = st.selectbox("Color by", ["None"] + df.columns.tolist(), key="3d_color")
            fig = px.scatter_3d(df.dropna(subset=[xc, yc, zc]), x=xc, y=yc, z=zc,
                               color=color_c if color_c != "None" else None, opacity=0.65,
                               title=f"3D: {xc} × {yc} × {zc}", height=650)
            fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)')),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E8E9F0'))
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Categorical Analysis":
        cc = df.select_dtypes(include='object').columns.tolist()
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if cc:
            c1, c2 = st.columns(2)
            with c1: cat_col = st.selectbox("Category", cc, key="cat_col")
            with c2: num_col = st.selectbox("Metric (optional)", ["Count"] + nc, key="cat_num")
            vc = df[cat_col].value_counts().head(25)
            if num_col == "Count":
                fig = px.bar(x=vc.index, y=vc.values, color=vc.values, color_continuous_scale='Viridis',
                            labels={'x': cat_col, 'y': 'Count'}, title=f"Distribution of {cat_col}")
            else:
                agg = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(25)
                fig = px.bar(x=agg.index, y=agg.values, color=agg.values, color_continuous_scale='Plasma',
                            labels={'x': cat_col, 'y': f'Mean {num_col}'}, title=f"Mean {num_col} by {cat_col}")
            fig.update_layout(**plotly_dark_layout(height=450, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Time Series / Line":
        date_cols = [c for c in df.columns if df[c].dtype in ['datetime64[ns]'] or 'date' in c.lower() or 'time' in c.lower()]
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if nc:
            c1, c2 = st.columns(2)
            with c1: x_c = st.selectbox("X axis (time/index)", ["Index"] + date_cols + df.columns.tolist(), key="ts_x")
            with c2: y_cols = st.multiselect("Y columns", nc, default=nc[:min(3, len(nc))], key="ts_y")
            if y_cols:
                xdata = df.index if x_c == "Index" else df[x_c]
                fig = go.Figure()
                clrs = ['#6C63FF', '#FF6584', '#43E97B', '#38F9D7', '#F9AB00']
                for i, yc in enumerate(y_cols):
                    fig.add_trace(go.Scatter(x=xdata, y=df[yc], mode='lines',
                                            name=yc, line=dict(color=clrs[i % len(clrs)], width=2)))
                fig.update_layout(**plotly_dark_layout(height=500, title="Time Series"))
                st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pair Plot Heatmap":
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if len(nc) >= 2:
            sel = st.multiselect("Select columns (max 6)", nc, default=nc[:min(5, len(nc))], key="pair_cols")
            if len(sel) >= 2:
                fig = make_subplots(rows=len(sel), cols=len(sel))
                for i, r in enumerate(sel):
                    for j, c in enumerate(sel):
                        if i == j:
                            kd = gaussian_kde(df[r].dropna())
                            xr = np.linspace(df[r].min(), df[r].max(), 100)
                            fig.add_trace(go.Scatter(x=xr, y=kd(xr), mode='lines',
                                                    line=dict(color='#6C63FF', width=2), showlegend=False), row=i+1, col=j+1)
                        else:
                            fig.add_trace(go.Scatter(x=df[c], y=df[r], mode='markers',
                                                    marker=dict(size=2, color='#FF6584', opacity=0.4),
                                                    showlegend=False), row=i+1, col=j+1)
                fig.update_layout(**plotly_dark_layout(height=min(800, 200 * len(sel)),
                                                       title="Pair Plot Matrix"))
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4: ML MODELS
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🤖 Machine Learning")

    all_cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("🎯 Target Variable", all_cols, key="ml_tgt")
    with c2:
        feats = st.multiselect("📊 Features", [c for c in all_cols if c != target],
                              default=[c for c in all_cols if c != target][:min(15, len(all_cols)-1)], key="ml_feats")

    if not feats:
        st.warning("⚠️ Select at least one feature column.")
        st.stop()

    with st.expander("⚙️ Advanced Training Options"):
        c1, c2, c3 = st.columns(3)
        with c1:
            use_knn_imp = st.checkbox("KNN Imputation", False, key="knn_imp")
            use_feat_sel = st.checkbox("Auto Feature Selection (SelectKBest)", False, key="feat_sel")
        with c2:
            use_cv = st.checkbox("Cross-validation (5-fold)", True, key="use_cv")
            test_sz = st.slider("Test Size %", 10, 40, 20, key="test_sz") / 100
        with c3:
            scale_before = st.selectbox("Pre-scale Features", ["None", "StandardScaler", "RobustScaler"], key="pre_scale")

    try:
        X, y, enc, t_enc, ptype = prepare_ml_data(df, target, feats, use_knn=use_knn_imp)
    except Exception as e:
        st.error(f"Data preparation error: {e}")
        st.stop()

    st.markdown(f"""
    <div class="alert-info">
        <strong>🔍 Problem Type: {ptype.upper()}</strong> · Target: <code>{target}</code> · 
        Features: {len(feats)} · Samples: {len(X):,}
        {f"· Classes: {len(np.unique(y))}" if ptype == 'classification' else f"· Target range: [{float(y.min()):.2f}, {float(y.max()):.2f}]"}
    </div>
    """, unsafe_allow_html=True)

    # Model catalog
    if ptype == 'classification':
        model_catalog = {
            "🌲 Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "⚡ XGBoost": xgb.XGBClassifier(n_estimators=150, random_state=42, n_jobs=-1, eval_metric='logloss'),
            "💡 LightGBM": lgb.LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1),
            "🌳 Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "📈 Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "🔗 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            "🧠 Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
            "📍 KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
            "🔵 SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
            "📊 Naive Bayes": GaussianNB()
        }
    else:
        model_catalog = {
            "🌲 Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "⚡ XGBoost": xgb.XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "💡 LightGBM": lgb.LGBMRegressor(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1),
            "🌳 Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "📈 Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "📏 Linear Regression": LinearRegression(n_jobs=-1),
            "🔷 Ridge": Ridge(alpha=1.0),
            "🔹 Lasso": Lasso(alpha=0.1, max_iter=2000),
            "⚖️ ElasticNet": ElasticNet(alpha=0.1, max_iter=2000),
            "🧠 Neural Network": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
            "🔵 SVR": SVR(kernel='rbf')
        }

    sel_models = st.multiselect("📋 Select Models to Train", list(model_catalog.keys()),
                                default=list(model_catalog.keys())[:5], key="sel_models")

    if st.button("🚀 Train Selected Models", key="train_btn", use_container_width=True, type="primary"):
        if not sel_models:
            st.warning("Select at least one model.")
            st.stop()

        X_arr = X.values
        if scale_before != "None":
            sc = StandardScaler() if scale_before == "StandardScaler" else RobustScaler()
            X_arr = sc.fit_transform(X_arr)

        if use_feat_sel:
            k = min(15, X_arr.shape[1])
            fs = SelectKBest(f_classif if ptype == 'classification' else f_regression, k=k)
            X_arr = fs.fit_transform(X_arr, y)
            sel_feat_names = [feats[i] for i in fs.get_support(indices=True)]
            st.markdown(f'<div class="alert-success">✅ Feature selection: kept {k} of {len(feats)} features</div>', unsafe_allow_html=True)
        else:
            sel_feat_names = feats

        strat_arg = y if ptype == 'classification' else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y, test_size=test_sz, random_state=42, stratify=strat_arg)
        except:
            X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y, test_size=test_sz, random_state=42)

        results = []
        prog = st.progress(0)
        status = st.empty()
        total = len(sel_models)

        for idx, mname in enumerate(sel_models):
            status.markdown(f'<div class="alert-info">⚙️ Training <b>{mname}</b> ({idx+1}/{total})...</div>', unsafe_allow_html=True)
            try:
                mdl = model_catalog[mname]
                mdl.fit(X_tr, y_tr)
                y_pred = mdl.predict(X_te)

                cv_score = None
                if use_cv:
                    cv = StratifiedKFold(5, shuffle=True, random_state=42) if ptype == 'classification' else KFold(5, shuffle=True, random_state=42)
                    scoring = 'accuracy' if ptype == 'classification' else 'r2'
                    cvs = cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
                    cv_score = f"{cvs.mean():.4f} ± {cvs.std():.4f}"

                if ptype == 'classification':
                    acc = accuracy_score(y_te, y_pred)
                    avg = 'binary' if len(np.unique(y)) == 2 else 'weighted'
                    prec = precision_score(y_te, y_pred, average=avg, zero_division=0)
                    rec = recall_score(y_te, y_pred, average=avg, zero_division=0)
                    f1 = f1_score(y_te, y_pred, average=avg, zero_division=0)
                    try:
                        pp = mdl.predict_proba(X_te) if hasattr(mdl, 'predict_proba') else None
                        auc = roc_auc_score(y_te, pp[:, 1] if len(np.unique(y)) == 2 else pp, multi_class='ovr', average='weighted') if pp is not None else None
                    except:
                        auc = None
                    results.append({'Model': mname, 'Accuracy': acc, 'Precision': prec,
                                    'Recall': rec, 'F1': f1, 'AUC': auc, 'CV Score': cv_score, 'Score': acc})
                else:
                    mse = mean_squared_error(y_te, y_pred)
                    r2 = r2_score(y_te, y_pred)
                    mae = mean_absolute_error(y_te, y_pred)
                    try: mape = mean_absolute_percentage_error(y_te, y_pred)
                    except: mape = None
                    results.append({'Model': mname, 'R²': r2, 'RMSE': np.sqrt(mse), 'MAE': mae,
                                    'MAPE': mape, 'CV Score': cv_score, 'Score': r2})

                st.session_state.trained_models[mname] = {
                    'model': mdl, 'features': sel_feat_names, 'target': target,
                    'type': ptype, 'encoders': enc, 't_enc': t_enc,
                    'X_test': X_te, 'y_test': y_te, 'y_pred': y_pred
                }
                if hasattr(mdl, 'feature_importances_'):
                    st.session_state.feature_importance = {'features': sel_feat_names, 'importance': mdl.feature_importances_}

            except Exception as e:
                st.error(f"❌ {mname}: {e}")
            prog.progress((idx + 1) / total)

        status.empty(); prog.empty()

        if results:
            rdf = pd.DataFrame(results).sort_values('Score', ascending=False).reset_index(drop=True)
            best_m = rdf.iloc[0]['Model']
            st.session_state.best_model = best_m

            st.markdown("### 🏆 Model Comparison")

            # ── Bar Chart ──
            score_col = 'Accuracy' if ptype == 'classification' else 'R²'
            fig = px.bar(rdf, x='Model', y=score_col,
                        color=score_col, color_continuous_scale='Viridis',
                        text=rdf[score_col].apply(lambda x: f'{float(x):.4f}' if x else 'N/A'),
                        title=f"Model Performance — {score_col}")
            fig.update_traces(textposition='outside', textfont_size=11)
            fig.update_layout(**plotly_dark_layout(height=420, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

            # ── Beautiful HTML Comparison Table ──
            st.markdown("### 📊 Detailed Metrics Comparison Table")

            if ptype == 'classification':
                headers    = ['#', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Score (5-fold)']
                data_keys  = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'CV Score']
                higher_better = {'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'}
            else:
                headers    = ['#', 'Model', 'R²', 'RMSE', 'MAE', 'MAPE', 'CV Score (5-fold)']
                data_keys  = ['R²', 'RMSE', 'MAE', 'MAPE', 'CV Score']
                higher_better = {'R²'}

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
                    if v is not None and v != 'N/A':
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
                            f'<div style="font-weight:700;font-size:14px;color:#E8E9F0">{row["Model"]}</div>'
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

            st.markdown(f'<div class="alert-success">🏆 Best Model: <b>{best_m}</b> | {score_col}: {rdf.iloc[0]["Score"]:.4f}</div>', unsafe_allow_html=True)

            # Detailed best model analysis
            best_info = st.session_state.trained_models[best_m]
            if ptype == 'classification':
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🎯 Confusion Matrix")
                    cm = confusion_matrix(best_info['y_test'], best_info['y_pred'])
                    cnames = t_enc.classes_ if t_enc else [str(i) for i in range(len(cm))]
                    fig2 = go.Figure(go.Heatmap(z=cm, x=cnames, y=cnames, colorscale='Blues',
                                               text=cm, texttemplate='%{text}', textfont=dict(size=16)))
                    fig2.update_layout(**plotly_dark_layout(height=380, xaxis_title="Predicted", yaxis_title="Actual"))
                    st.plotly_chart(fig2, use_container_width=True)
                with c2:
                    st.markdown("### 📋 Classification Report")
                    tnames = t_enc.classes_ if t_enc else [str(i) for i in np.unique(y)]
                    rep = classification_report(best_info['y_test'], best_info['y_pred'], target_names=tnames, output_dict=True)
                    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True, height=380)
            else:
                st.markdown("### 📈 Actual vs Predicted")
                pred_df = pd.DataFrame({'Actual': best_info['y_test'], 'Predicted': best_info['y_pred']})
                fig2 = px.scatter(pred_df, x='Actual', y='Predicted', opacity=0.5, trendline="ols")
                mn = min(pred_df.min().min(), pred_df['Actual'].min())
                mx = max(pred_df.max().max(), pred_df['Actual'].max())
                fig2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Perfect',
                                         line=dict(color='#43E97B', dash='dash', width=2)))
                fig2.update_layout(**plotly_dark_layout(height=450))
                st.plotly_chart(fig2, use_container_width=True)

            if st.session_state.feature_importance:
                st.markdown("### 🎯 Feature Importance")
                fi = st.session_state.feature_importance
                fi_df = pd.DataFrame({'Feature': fi['features'], 'Importance': fi['importance']}).sort_values('Importance', ascending=True).tail(15)
                fig3 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                             color='Importance', color_continuous_scale='Viridis', title="Top Features")
                fig3.update_layout(**plotly_dark_layout(height=450, coloraxis_showscale=False))
                st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 5: PREDICT
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🎯 Prediction Engine")

    if not st.session_state.trained_models:
        st.markdown('<div class="alert-warning">⚠️ No trained models found. Go to the <b>ML Models</b> tab first.</div>', unsafe_allow_html=True)
        st.stop()

    mname = st.selectbox("Select Model", list(st.session_state.trained_models.keys()), key="pred_m")
    minfo = st.session_state.trained_models[mname]
    mdl = minfo['model']
    f_cols = minfo['features']
    p_type = minfo['type']
    p_encs = minfo['encoders']
    p_tenc = minfo['t_enc']

    st.markdown(f"""
    <div class="alert-info">🤖 <b>{mname}</b> · Type: {p_type.title()} · Features: {len(f_cols)}</div>
    """, unsafe_allow_html=True)

    mode = st.radio("Mode", ["Single Prediction", "Batch Prediction (CSV)"], horizontal=True, key="pred_mode")

    if mode == "Single Prediction":
        st.markdown("### 📝 Enter Feature Values")
        input_data = {}
        cols4 = st.columns(4)
        for i, col in enumerate(f_cols):
            if col not in df.columns:
                continue
            with cols4[i % 4]:
                if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    mn, mx, mv = float(df[col].min()), float(df[col].max()), float(df[col].mean())
                    input_data[col] = st.number_input(col, mn, mx, mv, step=(mx-mn)/200, key=f"inp_{col}")
                else:
                    uv = sorted(df[col].dropna().unique().tolist())
                    input_data[col] = st.selectbox(col, uv, key=f"inp_{col}")

        if st.button("🔮 Predict", key="pred_single", use_container_width=True, type="primary"):
            try:
                inp_df = pd.DataFrame([input_data])
                for col in inp_df.columns:
                    if col in p_encs:
                        try: inp_df[col] = p_encs[col].transform(inp_df[col].astype(str))
                        except: inp_df[col] = 0
                inp_df = inp_df.reindex(columns=f_cols, fill_value=0)
                pred = mdl.predict(inp_df)[0]
                if p_type == 'classification' and p_tenc:
                    pred_show = p_tenc.inverse_transform([int(pred)])[0]
                else:
                    pred_show = f"{pred:.4f}" if p_type == 'regression' else pred

                if p_type == 'classification' and hasattr(mdl, 'predict_proba'):
                    proba = mdl.predict_proba(inp_df)[0]
                    conf = max(proba) * 100
                    st.markdown(f"""
                    <div class="pred-result">
                        <div style="font-size:16px;color:var(--text-muted);margin-bottom:12px;">PREDICTION RESULT</div>
                        <div class="pred-value">{pred_show}</div>
                        <div class="pred-conf">⚡ Confidence: {conf:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    classes = p_tenc.classes_ if p_tenc else [str(i) for i in range(len(proba))]
                    pf = pd.DataFrame({'Class': classes, 'Probability': proba * 100}).sort_values('Probability', ascending=False)
                    fig = px.bar(pf, x='Class', y='Probability', color='Probability',
                                color_continuous_scale='Viridis', text=pf['Probability'].apply(lambda x: f'{x:.2f}%'))
                    fig.update_traces(textposition='outside')
                    fig.update_layout(**plotly_dark_layout(height=380, coloraxis_showscale=False))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="pred-result">
                        <div style="font-size:16px;color:var(--text-muted);margin-bottom:12px;">PREDICTION RESULT</div>
                        <div class="pred-value">{pred_show}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    else:
        st.markdown("### 📊 Batch Prediction from CSV")
        up_pred = st.file_uploader("Upload CSV", type=['csv'], key="batch_pred")
        if up_pred:
            bdf = pd.read_csv(up_pred)
            st.caption(f"Loaded: {len(bdf):,} rows")
            st.dataframe(bdf.head(10), use_container_width=True)
            if st.button("🔮 Predict All", use_container_width=True, type="primary", key="batch_btn"):
                missing = set(f_cols) - set(bdf.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    try:
                        Xb = bdf[f_cols].copy()
                        for col in Xb.columns:
                            if col in p_encs:
                                try: Xb[col] = p_encs[col].transform(Xb[col].astype(str))
                                except: Xb[col] = 0
                        preds = mdl.predict(Xb)
                        if p_type == 'classification' and p_tenc:
                            preds = p_tenc.inverse_transform(preds.astype(int))
                        bdf['Prediction'] = preds
                        if p_type == 'classification' and hasattr(mdl, 'predict_proba'):
                            pp = mdl.predict_proba(Xb)
                            bdf['Confidence_%'] = (pp.max(axis=1) * 100).round(2)
                        st.success(f"✅ Predicted {len(bdf):,} samples!")
                        st.dataframe(bdf, use_container_width=True)
                        download_button(bdf, "csv", "📥 Download Predictions", "dl_pred")
                    except Exception as e:
                        st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════
# TAB 6: FEATURES
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 🧬 Feature Engineering & Analysis")

    with st.expander("🤖 Auto Feature Engineering", expanded=True):
        tgt_fe = st.selectbox("Target (for correlation-based selection)", ["None"] + df.columns.tolist(), key="fe_tgt")
        if st.button("🧬 Generate Features", key="gen_fe", use_container_width=True):
            dfc = df.copy()
            nc = dfc.select_dtypes(include=np.number).columns.tolist()
            tgt = tgt_fe if tgt_fe != "None" else None
            if tgt in nc: nc.remove(tgt)
            new_feats = []
            if len(nc) >= 2 and tgt and tgt in dfc.columns:
                corrs = dfc[nc].corrwith(dfc[tgt]).abs().sort_values(ascending=False)
                top = corrs.head(4).index.tolist()
            else:
                top = nc[:4]
            # Interaction features
            for i in range(len(top)):
                for j in range(i+1, len(top)):
                    c1n, c2n = top[i], top[j]
                    dfc[f'{c1n}×{c2n}'] = dfc[c1n] * dfc[c2n]
                    new_feats.append(f'{c1n}×{c2n}')
                    if (dfc[c2n] != 0).all():
                        dfc[f'{c1n}÷{c2n}'] = dfc[c1n] / dfc[c2n].replace(0, 1e-8)
                        new_feats.append(f'{c1n}÷{c2n}')
            # Aggregation features
            if len(nc) >= 3:
                dfc['row_mean'] = dfc[nc].mean(axis=1)
                dfc['row_std'] = dfc[nc].std(axis=1)
                dfc['row_max'] = dfc[nc].max(axis=1)
                dfc['row_min'] = dfc[nc].min(axis=1)
                new_feats.extend(['row_mean', 'row_std', 'row_max', 'row_min'])
            st.success(f"✅ Generated {len(new_feats)} new features!")
            st.write(", ".join(new_feats))
            if st.button("➕ Add to Dataset", key="add_fe", use_container_width=True):
                push_history(dfc, f"🧬 Added {len(new_feats)} engineered features")
                st.session_state.df = dfc
                st.rerun()

    with st.expander("🎯 Clustering Analysis", expanded=True):
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if len(nc) >= 2:
            cl_feats = st.multiselect("Features", nc, default=nc[:min(5, len(nc))], key="cl_feats")
            c1, c2, c3 = st.columns(3)
            with c1: cl_meth = st.selectbox("Method", ["K-Means", "DBSCAN", "Agglomerative"], key="cl_meth")
            with c2:
                if cl_meth != "DBSCAN":
                    n_cl = st.slider("Clusters", 2, 12, 4, key="n_cl")
                else:
                    eps_v = st.slider("DBSCAN eps", 0.1, 5.0, 0.5, key="dbscan_eps")
            with c3: use_pca_cl = st.checkbox("PCA reduction", True, key="pca_cl")

            if cl_feats and st.button("🎯 Run Clustering", key="cl_btn", use_container_width=True):
                try:
                    Xc = df[cl_feats].dropna()
                    Xs = StandardScaler().fit_transform(Xc)
                    if use_pca_cl and len(cl_feats) > 2:
                        Xs = PCA(n_components=min(3, len(cl_feats)), random_state=42).fit_transform(Xs)

                    if cl_meth == "K-Means":
                        cl = KMeans(n_clusters=n_cl, random_state=42, n_init=10).fit_predict(Xs)
                    elif cl_meth == "DBSCAN":
                        cl = DBSCAN(eps=eps_v, min_samples=5).fit_predict(Xs)
                    else:
                        cl = AgglomerativeClustering(n_clusters=n_cl).fit_predict(Xs)

                    try:
                        sil = silhouette_score(Xs, cl)
                        st.metric("Silhouette Score", f"{sil:.4f}", f"{'Excellent' if sil > 0.7 else 'Good' if sil > 0.5 else 'Fair'}")
                    except: pass

                    fig = px.scatter(x=Xs[:, 0], y=Xs[:, 1], color=cl.astype(str),
                                    title=f"{cl_meth} Clustering", labels={'x': 'PC1' if use_pca_cl else cl_feats[0], 'y': 'PC2' if use_pca_cl else cl_feats[1]})
                    fig.update_layout(**plotly_dark_layout(height=450))
                    st.plotly_chart(fig, use_container_width=True)

                    cs = pd.DataFrame({'Cluster': np.unique(cl), 'Size': [(cl == c).sum() for c in np.unique(cl)],
                                       '%': [(cl == c).sum() / len(cl) * 100 for c in np.unique(cl)]}).round(1)
                    st.dataframe(cs, use_container_width=True)

                    if st.button("➕ Add Clusters to Dataset", key="add_cl", use_container_width=True):
                        dfc = df.copy(); dfc['Cluster'] = -1
                        dfc.loc[Xc.index, 'Cluster'] = cl
                        push_history(dfc, f"🎯 {cl_meth} clustering")
                        st.session_state.df = dfc; st.rerun()
                except Exception as e:
                    st.error(f"Clustering error: {e}")

    with st.expander("📉 Dimensionality Reduction", expanded=False):
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if len(nc) >= 3:
            c1, c2 = st.columns(2)
            with c1: dr_meth = st.selectbox("Method", ["PCA", "t-SNE", "ICA"], key="dr_meth")
            with c2: dr_n = st.slider("Components", 2, min(10, len(nc)), 3, key="dr_n")

            if st.button("📉 Apply Reduction", key="dr_btn", use_container_width=True):
                try:
                    Xd = df[nc].fillna(df[nc].median())
                    Xds = StandardScaler().fit_transform(Xd)
                    if dr_meth == "PCA":
                        r = PCA(n_components=dr_n, random_state=42)
                        Xr = r.fit_transform(Xds)
                        ve = r.explained_variance_ratio_
                        fig2 = px.bar(x=[f'PC{i+1}' for i in range(dr_n)], y=ve*100,
                                     title="Explained Variance per Component", labels={'x': 'Component', 'y': 'Variance %'})
                        fig2.update_layout(**plotly_dark_layout(height=300))
                        st.plotly_chart(fig2, use_container_width=True)
                        st.metric("Total Variance Explained", f"{ve.sum()*100:.1f}%")
                    elif dr_meth == "t-SNE":
                        r = TSNE(n_components=min(dr_n, 3), random_state=42, perplexity=min(30, len(df)-1))
                        Xr = r.fit_transform(Xds)
                    else:
                        r = FastICA(n_components=dr_n, random_state=42)
                        Xr = r.fit_transform(Xds)

                    fig = px.scatter(x=Xr[:, 0], y=Xr[:, 1], opacity=0.6,
                                    title=f"{dr_meth} — 2D Projection",
                                    labels={'x': f'{dr_meth} 1', 'y': f'{dr_meth} 2'})
                    fig.update_layout(**plotly_dark_layout(height=500))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")


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
            if samp_t == "Random %":
                samp_pct = st.slider("Percentage", 1, 99, 30, key="samp_pct")
            elif samp_t == "Fixed N":
                samp_n = st.number_input("N rows", 1, len(df), min(1000, len(df)), key="samp_n")
            elif samp_t == "Stratified":
                strat_c = st.selectbox("Stratify by", df.columns.tolist(), key="strat_c")
                samp_pct2 = st.slider("%", 1, 99, 30, key="samp_pct2")
            elif samp_t == "Bootstrap":
                boot_n = st.number_input("N samples", 1, len(df)*3, min(1000, len(df)), key="boot_n")
            else:
                step_v = st.number_input("Step size", 1, 100, 5, key="step_v")

        if st.button("🎲 Apply Sampling", key="samp_btn", use_container_width=True):
            try:
                if samp_t == "Random %": ds = df.sample(frac=samp_pct/100, random_state=42)
                elif samp_t == "Fixed N": ds = df.sample(n=min(samp_n, len(df)), random_state=42)
                elif samp_t == "Stratified":
                    ds = df.groupby(strat_c, group_keys=False).apply(lambda x: x.sample(frac=samp_pct2/100, random_state=42))
                elif samp_t == "Bootstrap": ds = df.sample(n=boot_n, replace=True, random_state=42)
                else: ds = df.iloc[::step_v]
                push_history(ds, f"🎲 {samp_t} sampling")
                st.session_state.df = ds
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("🔍 Advanced Filtering", expanded=True):
        fc = st.selectbox("Column to filter", df.columns.tolist(), key="flt_col")
        if pd.api.types.is_numeric_dtype(df[fc]):
            c1, c2 = st.columns(2)
            with c1: ft = st.selectbox("Filter", ["Range", "Greater than", "Less than", "Between percentiles"], key="flt_t")
            with c2:
                mn, mx = float(df[fc].min()), float(df[fc].max())
                if ft == "Range": fv = st.slider("Range", mn, mx, (mn, mx), key="flt_rng")
                elif ft == "Between percentiles": fv = st.slider("Percentiles", 0, 100, (10, 90), key="flt_pct")
                else: fv = st.number_input("Threshold", mn, mx, float(df[fc].median()), key="flt_val")

            if st.button("🔍 Apply Filter", key="flt_btn", use_container_width=True):
                if ft == "Range": dff = df[(df[fc] >= fv[0]) & (df[fc] <= fv[1])]
                elif ft == "Greater than": dff = df[df[fc] > fv]
                elif ft == "Less than": dff = df[df[fc] < fv]
                else:
                    lo, hi = df[fc].quantile(fv[0]/100), df[fc].quantile(fv[1]/100)
                    dff = df[(df[fc] >= lo) & (df[fc] <= hi)]
                push_history(dff, f"🔍 Filter {fc}")
                st.session_state.df = dff; st.rerun()
        else:
            uv = sorted(df[fc].dropna().unique().tolist())
            sv = st.multiselect("Select values", uv, default=uv[:min(5, len(uv))], key="flt_sel")
            if sv and st.button("🔍 Apply Filter", key="flt_cat_btn", use_container_width=True):
                dff = df[df[fc].isin(sv)]
                push_history(dff, f"🔍 Filter {fc}")
                st.session_state.df = dff; st.rerun()

    with st.expander("🔗 Merge Datasets", expanded=False):
        if st.session_state.df2 is not None:
            df2 = st.session_state.df2
            st.caption(f"Second dataset: {df2.shape[0]:,} × {df2.shape[1]:,}")
            c1, c2, c3 = st.columns(3)
            with c1: jt = st.selectbox("Join type", ["inner", "left", "right", "outer"], key="jt")
            with c2: lk = st.selectbox("Left key", df.columns.tolist(), key="lk")
            with c3: rk = st.selectbox("Right key", df2.columns.tolist(), key="rk")
            if st.button("🔗 Merge", key="mrg_btn", use_container_width=True):
                try:
                    merged = pd.merge(df, df2, left_on=lk, right_on=rk, how=jt, suffixes=('', '_2'))
                    push_history(merged, f"🔗 {jt.title()} merge on {lk}")
                    st.session_state.df = merged; st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Upload a second dataset in the sidebar to enable merging.")

    with st.expander("✏️ Column Operations", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            op_type = st.selectbox("Operation", ["Rename Column", "Drop Column", "Change Dtype", "Create from Formula"], key="col_op")
        with c2:
            op_col = st.selectbox("Column", df.columns.tolist(), key="op_col") if op_type != "Create from Formula" else None

        if op_type == "Rename Column":
            new_name = st.text_input("New name", op_col or "", key="new_name")
            if st.button("✅ Rename", use_container_width=True, key="rename_btn"):
                dfc = df.rename(columns={op_col: new_name})
                push_history(dfc, f"✏️ Renamed {op_col} → {new_name}")
                st.session_state.df = dfc; st.rerun()

        elif op_type == "Drop Column":
            drop_multi = st.multiselect("Columns to drop", df.columns.tolist(), key="drop_cols")
            if drop_multi and st.button("🗑️ Drop Columns", use_container_width=True, key="drop_btn"):
                dfc = df.drop(columns=drop_multi)
                push_history(dfc, f"🗑️ Dropped {len(drop_multi)} columns")
                st.session_state.df = dfc; st.rerun()

        elif op_type == "Change Dtype":
            new_type = st.selectbox("New type", ["int64", "float64", "str", "category", "datetime64[ns]"], key="new_type")
            if st.button("🔄 Convert", use_container_width=True, key="conv_btn"):
                try:
                    dfc = df.copy()
                    dfc[op_col] = dfc[op_col].astype(new_type)
                    push_history(dfc, f"🔄 {op_col}: → {new_type}")
                    st.session_state.df = dfc; st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        elif op_type == "Create from Formula":
            new_col = st.text_input("New column name", "new_feature", key="formula_name")
            formula = st.text_input("Formula (use column names as variables, e.g. Age * Income)", key="formula")
            st.caption("Available columns: " + ", ".join(df.columns.tolist()))
            if formula and st.button("✅ Create Column", use_container_width=True, key="formula_btn"):
                try:
                    dfc = df.copy()
                    local_vars = {c: dfc[c] for c in dfc.columns}
                    local_vars['np'] = np
                    dfc[new_col] = eval(formula, {"__builtins__": {}}, local_vars)
                    push_history(dfc, f"➕ Created {new_col}")
                    st.session_state.df = dfc; st.rerun()
                except Exception as e:
                    st.error(f"Formula error: {e}")


# ═══════════════════════════════════════════════════════════
# TAB 8: AUTOML
# ═══════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## 🏆 AutoML — Automated Machine Learning")

    st.markdown("""
    <div class="alert-info">
        <strong>🤖 AutoML Pipeline:</strong> Auto feature selection → Multi-model training (XGBoost, LightGBM, RF, ET) → Ensemble → Best model saved
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        aml_tgt = st.selectbox("🎯 Target Variable", df.columns.tolist(), key="aml_tgt")
    with c2:
        aml_feats_k = st.slider("Max features to select", 5, min(50, len(df.columns)-1), min(20, len(df.columns)-1), key="aml_k")
    with c3:
        speed_mode = st.selectbox("⚡ Speed Mode", ["⚡ Fast (recommended)", "⚖️ Balanced", "🎯 Accurate (slow)"], key="aml_speed")

    _speed_cfg = {
        "⚡ Fast (recommended)": dict(n_est=50,  cv_folds=3, max_rows=5000,  use_ensemble=False, use_knn=False),
        "⚖️ Balanced":           dict(n_est=100, cv_folds=3, max_rows=10000, use_ensemble=True,  use_knn=False),
        "🎯 Accurate (slow)":    dict(n_est=200, cv_folds=5, max_rows=None,  use_ensemble=True,  use_knn=True),
    }
    cfg = _speed_cfg[speed_mode]

    c1, c2 = st.columns(2)
    with c1:
        use_aml_cv = st.checkbox("Cross-validation in AutoML", True, key="aml_cv")
    with c2:
        st.markdown(f"""<div style="padding:8px 12px;background:rgba(108,99,255,0.08);border-radius:8px;font-size:12px;color:rgba(232,233,240,0.7);margin-top:4px">
            🌲 Trees: <b>{cfg['n_est']}</b> &nbsp;|&nbsp; CV folds: <b>{cfg['cv_folds']}</b> &nbsp;|&nbsp;
            Ensemble: <b>{'Yes' if cfg['use_ensemble'] else 'No'}</b> &nbsp;|&nbsp;
            Max rows: <b>{cfg['max_rows'] or 'All'}</b>
        </div>""", unsafe_allow_html=True)

    if st.button("🚀 Launch AutoML", use_container_width=True, type="primary", key="aml_btn"):
        avail = [c for c in df.columns if c != aml_tgt]
        with st.spinner("🤖 AutoML running..."):
            try:
                X, y, enc, t_enc, ptype = prepare_ml_data(df, aml_tgt, avail, use_knn=cfg['use_knn'])
                st.markdown(f'<div class="alert-info">Problem Type: <b>{ptype.upper()}</b> | Samples: {len(X):,}</div>', unsafe_allow_html=True)

                # ── Smart row sampling for large datasets ──
                if cfg['max_rows'] and len(X) > cfg['max_rows']:
                    sample_idx = np.random.RandomState(42).choice(len(X), cfg['max_rows'], replace=False)
                    X = X.iloc[sample_idx].reset_index(drop=True)
                    y = y[sample_idx] if isinstance(y, np.ndarray) else y.iloc[sample_idx].reset_index(drop=True)
                    st.markdown(f'<div class="alert-warning">⚡ Sampled {cfg["max_rows"]:,} rows for speed. Use Accurate mode for full data.</div>', unsafe_allow_html=True)

                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                                               stratify=y if ptype == 'classification' else None)
                except:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

                k = min(aml_feats_k, X_tr.shape[1])
                fs = SelectKBest(f_classif if ptype == 'classification' else f_regression, k=k)
                Xtr_s = fs.fit_transform(X_tr, y_tr)
                Xte_s = fs.transform(X_te)
                sel_f = [avail[i] for i in fs.get_support(indices=True)]
                st.markdown(f'<div class="alert-success">✅ Feature selection: {k} features selected from {len(avail)}</div>', unsafe_allow_html=True)

                n_est = cfg['n_est']
                if ptype == 'classification':
                    aml_models = {
                        "XGBoost":      xgb.XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
                        "LightGBM":     lgb.LGBMClassifier(n_estimators=n_est, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),
                        "Random Forest":RandomForestClassifier(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1),
                        "Extra Trees":  ExtraTreesClassifier(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1),
                    }
                else:
                    aml_models = {
                        "XGBoost":      xgb.XGBRegressor(n_estimators=n_est, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1),
                        "LightGBM":     lgb.LGBMRegressor(n_estimators=n_est, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),
                        "Random Forest":RandomForestRegressor(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1),
                        "Extra Trees":  ExtraTreesRegressor(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1),
                    }

                results = []; trained_list = []
                prog = st.progress(0)
                status_txt = st.empty()
                total_steps = len(aml_models) + (1 if cfg['use_ensemble'] else 0)
                for i, (nm, md) in enumerate(aml_models.items()):
                    status_txt.markdown(f'<div class="alert-info">⏳ Training <b>{nm}</b> ({i+1}/{len(aml_models)})...</div>', unsafe_allow_html=True)
                    md.fit(Xtr_s, y_tr)
                    yp = md.predict(Xte_s)
                    trained_list.append((nm, md))
                    sc = accuracy_score(y_te, yp) if ptype == 'classification' else r2_score(y_te, yp)
                    cv_s = None
                    if use_aml_cv:
                        try:
                            cv_scores = cross_val_score(md, Xtr_s, y_tr, cv=cfg['cv_folds'], n_jobs=-1,
                                                        scoring='accuracy' if ptype == 'classification' else 'r2')
                            cv_s = f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                        except: pass
                    results.append({'Model': nm, 'Score': sc, 'CV Score': cv_s})
                    prog.progress((i+1) / total_steps)
                status_txt.empty()

                # Ensemble — only in Balanced / Accurate mode
                if cfg['use_ensemble']:
                    try:
                        status_txt.markdown('<div class="alert-info">⏳ Building <b>Ensemble</b>...</div>', unsafe_allow_html=True)
                        if ptype == 'classification':
                            ens = VotingClassifier(estimators=trained_list, voting='soft')
                        else:
                            ens = VotingRegressor(estimators=trained_list)
                        ens.fit(Xtr_s, y_tr)
                        yp_ens = ens.predict(Xte_s)
                        ens_sc = accuracy_score(y_te, yp_ens) if ptype == 'classification' else r2_score(y_te, yp_ens)
                        results.append({'Model': '🎯 Ensemble (Voting)', 'Score': ens_sc, 'CV Score': None})
                        trained_list.append(('🎯 Ensemble (Voting)', ens))
                        status_txt.empty()
                    except Exception as e:
                        st.warning(f"Ensemble failed: {e}")
                prog.progress(1.0); prog.empty()

                rdf = pd.DataFrame(results).sort_values('Score', ascending=False)
                best_name = rdf.iloc[0]['Model']
                best_sc = rdf.iloc[0]['Score']

                st.markdown("### 🏆 AutoML Results")
                score_label = 'Accuracy' if ptype == 'classification' else 'R²'
                fig = px.bar(rdf, x='Model', y='Score', color='Score',
                            color_continuous_scale='Viridis',
                            text=rdf['Score'].apply(lambda x: f'{x:.4f}'),
                            title=f"AutoML — {score_label} Comparison")
                fig.update_traces(textposition='outside')
                fig.update_layout(**plotly_dark_layout(height=420, coloraxis_showscale=False))
                st.plotly_chart(fig, use_container_width=True)

                # Rich AutoML Comparison Table — relative min-max normalization
                rdf2 = rdf.reset_index(drop=True)
                medals2 = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']

                # Pre-compute min/max of Score column for relative bars
                all_scores = rdf2['Score'].astype(float).tolist()
                sc_min, sc_max = min(all_scores), max(all_scores)
                sc_range = sc_max - sc_min if sc_max != sc_min else 1.0

                # CV scores min/max
                cv_vals_aml = [float(r['CV Score']) for _, r in rdf2.iterrows()
                               if r.get('CV Score') is not None]
                cv_min_aml = min(cv_vals_aml) if cv_vals_aml else 0
                cv_max_aml = max(cv_vals_aml) if cv_vals_aml else 1
                cv_range_aml = cv_max_aml - cv_min_aml if cv_max_aml != cv_min_aml else 1.0

                rows2_html = ''
                for i, row in rdf2.iterrows():
                    is_best2 = (i == 0)
                    medal2   = medals2[i] if i < len(medals2) else f'<span style="color:#666">#{i+1}</span>'
                    row_bg2  = 'rgba(67,233,123,0.05)' if is_best2 else ('rgba(255,255,255,0.025)' if i % 2 == 0 else 'rgba(0,0,0,0)')
                    border2  = 'border-left:3px solid #43E97B;' if is_best2 else 'border-left:3px solid transparent;'
                    sc       = float(row['Score'])

                    # Relative bar: best model = 100%, worst = 4%
                    bar_w2   = max(4, int(((sc - sc_min) / sc_range) * 100))

                    # Color based on relative rank position
                    rank_pct = (sc - sc_min) / sc_range  # 0=worst, 1=best
                    if rank_pct >= 0.75:   bar_c2 = '#43E97B'; bg_c2 = 'rgba(67,233,123,0.12)'
                    elif rank_pct >= 0.50: bar_c2 = '#38F9D7'; bg_c2 = 'rgba(56,249,215,0.10)'
                    elif rank_pct >= 0.25: bar_c2 = '#F9AB00'; bg_c2 = 'rgba(249,171,0,0.10)'
                    else:                  bar_c2 = '#FF4757'; bg_c2 = 'rgba(255,71,87,0.10)'

                    # Score display — smart format
                    sc_disp = f'{sc:.4f}'
                    # Quality label
                    qlabel   = 'Best' if is_best2 else ('Good' if rank_pct >= 0.50 else ('Fair' if rank_pct >= 0.25 else 'Weak'))
                    qlabel_c = bar_c2

                    # CV Score cell
                    cv_val = row.get('CV Score', None)
                    if cv_val is not None:
                        cv_f = float(cv_val)
                        cv_bar_w = max(4, int(((cv_f - cv_min_aml) / cv_range_aml) * 100))
                        cv_c = '#43E97B' if cv_f >= 0.80 else '#38F9D7' if cv_f >= 0.65 else '#F9AB00' if cv_f >= 0.50 else '#FF4757'
                        cv_html = f'''<div style="display:flex;flex-direction:column;gap:4px;align-items:center">
                            <span style="font-size:14px;font-weight:700;color:{cv_c}">{cv_f:.4f}</span>
                            <div style="background:rgba(255,255,255,0.07);border-radius:3px;height:4px;width:80px">
                                <div style="width:{cv_bar_w}%;background:{cv_c};height:4px;border-radius:3px"></div>
                            </div>
                        </div>'''
                    else:
                        cv_html = '<span style="color:#444;font-size:13px">—</span>'

                    rows2_html += f'''
                    <tr style="background:{row_bg2};{border2}">
                        <td style="padding:16px;text-align:center;font-size:20px">{medal2}</td>
                        <td style="padding:14px 20px;min-width:180px">
                            <div style="font-weight:700;font-size:15px;color:#E8E9F0">{row["Model"]}</div>
                            {"<div style='margin-top:5px'><span style='background:rgba(67,233,123,0.18);color:#43E97B;font-size:10px;padding:2px 9px;border-radius:20px;font-weight:700;letter-spacing:0.5px'>★ BEST MODEL</span></div>" if is_best2 else ""}
                        </td>
                        <td style="padding:14px 20px;background:{bg_c2};min-width:240px">
                            <div style="display:flex;flex-direction:column;gap:6px">
                                <div style="display:flex;align-items:center;justify-content:space-between">
                                    <span style="font-size:20px;font-weight:900;color:#E8E9F0;letter-spacing:-0.5px">{sc_disp}</span>
                                    <span style="font-size:10px;font-weight:700;color:{qlabel_c};background:rgba(255,255,255,0.07);padding:2px 7px;border-radius:4px">{qlabel}</span>
                                </div>
                                <div style="background:rgba(255,255,255,0.08);border-radius:5px;height:8px;width:100%">
                                    <div style="width:{bar_w2}%;background:{bar_c2};height:8px;border-radius:5px"></div>
                                </div>
                                <div style="font-size:10px;color:rgba(232,233,240,0.4)">Rank {i+1} of {len(rdf2)} · bar = relative performance</div>
                            </div>
                        </td>
                        <td style="padding:14px 20px;text-align:center;vertical-align:middle">{cv_html}</td>
                    </tr>'''

                _automl_html = (
                    f'<div style="border-radius:16px;overflow:hidden;border:1px solid rgba(108,99,255,0.2);margin:16px 0;">'
                    f'<div style="background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(255,101,132,0.06));padding:16px 20px;border-bottom:1px solid rgba(255,255,255,0.06)">'
                    f'<div style="font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:700;color:#E8E9F0">AutoML Algorithm Leaderboard</div>'
                    f'<div style="font-size:12px;color:rgba(232,233,240,0.5);margin-top:2px">Sorted by {score_label} · Best model highlighted in green</div>'
                    f'</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
                    f'<thead style="background:rgba(0,0,0,0.3)"><tr>'
                    f'<th style="padding:12px 16px;text-align:center;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#6C63FF;border-bottom:2px solid rgba(108,99,255,0.3)">Rank</th>'
                    f'<th style="padding:12px 20px;text-align:left;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#6C63FF;border-bottom:2px solid rgba(108,99,255,0.3)">Model</th>'
                    f'<th style="padding:12px 20px;text-align:left;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#6C63FF;border-bottom:2px solid rgba(108,99,255,0.3)">{score_label} Score</th>'
                    f'<th style="padding:12px 20px;text-align:center;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#6C63FF;border-bottom:2px solid rgba(108,99,255,0.3)">CV Mean</th>'
                    f'</tr></thead>'
                    f'<tbody>{rows2_html}</tbody>'
                    f'</table>'
                    f'<div style="background:rgba(0,0,0,0.2);padding:14px 20px;border-top:1px solid rgba(255,255,255,0.04);display:flex;gap:24px;flex-wrap:wrap;align-items:center">'
                    f'<span style="font-size:11px;color:rgba(232,233,240,0.45)">💡 Bars show <b style="color:rgba(232,233,240,0.7)">relative performance</b> — best model always gets full bar regardless of absolute score</span>'
                    f'<span style="font-size:11px;color:rgba(232,233,240,0.45)">Score range: <b style="color:rgba(232,233,240,0.7)">{sc_min:.4f}</b> → <b style="color:#43E97B">{sc_max:.4f}</b></span>'
                    f'<div style="display:flex;gap:10px;font-size:11px;color:rgba(232,233,240,0.5)"><span>🟢 Best</span><span>🩵 Good</span><span>🟡 Fair</span><span>🔴 Weak</span></div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(_automl_html, unsafe_allow_html=True)
                st.markdown(f'<div class="alert-success">🏆 Best: <b>{best_name}</b> | {score_label}: {best_sc:.4f}</div>', unsafe_allow_html=True)

                # Feature importance
                best_obj = next(m for n, m in trained_list if n == best_name)
                fi_src = best_obj if hasattr(best_obj, 'feature_importances_') else None
                if fi_src is None and hasattr(best_obj, 'estimators_'):
                    try: fi_src = best_obj.estimators_[0][1]
                    except: pass
                if fi_src and hasattr(fi_src, 'feature_importances_'):
                    fi_df = pd.DataFrame({'Feature': sel_f, 'Importance': fi_src.feature_importances_})
                    fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)
                    fig2 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                                 color='Importance', color_continuous_scale='Viridis', title="Top Features")
                    fig2.update_layout(**plotly_dark_layout(height=420, coloraxis_showscale=False))
                    st.plotly_chart(fig2, use_container_width=True)

                # Save
                st.session_state.trained_models[f"AutoML_{best_name}"] = {
                    'model': best_obj, 'features': sel_f, 'target': aml_tgt,
                    'type': ptype, 'encoders': enc, 't_enc': t_enc,
                    'X_test': Xte_s, 'y_test': y_te, 'y_pred': yp_ens if '🎯' in best_name else yp
                }
                st.success("✅ AutoML complete! Best model saved to Predict tab.")

            except Exception as e:
                st.error(f"AutoML error: {e}")
                import traceback
                st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════
# TAB 9: EXPORT
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# TAB 9: STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("## 🔬 Statistical Tests & Hypothesis Testing")
    nc = df.select_dtypes(include=np.number).columns.tolist()
    cc = df.select_dtypes(include='object').columns.tolist()

    test_type = st.selectbox("Select Test", [
        "📊 T-Test (2 group means comparison)",
        "📊 Mann-Whitney U (non-parametric T-Test)",
        "📊 ANOVA (3+ group means comparison)",
        "📊 Chi-Square (categorical independence)",
        "📊 Shapiro-Wilk (normality check)",
        "📊 Correlation Matrix with P-values",
    ], key="stat_test_type")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if "T-Test" in test_type or "Mann-Whitney" in test_type:
        c1, c2, c3 = st.columns(3)
        with c1: num_col = st.selectbox("Numeric Column", nc, key="tt_num")
        with c2: grp_col = st.selectbox("Group Column", cc + [c for c in nc if df[c].nunique() <= 10], key="tt_grp")
        with c3: alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01, key="tt_alpha")

        if st.button("▶ Run Test", use_container_width=True, key="tt_run"):
            try:
                groups = df.groupby(grp_col)[num_col].apply(lambda x: x.dropna().values)
                if len(groups) < 2:
                    st.error("Need at least 2 groups!")
                else:
                    g1_name, g1 = list(groups.items())[0]
                    g2_name, g2 = list(groups.items())[1]
                    if "Mann" in test_type:
                        stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                        test_name = "Mann-Whitney U"
                    else:
                        stat, p = ttest_ind(g1, g2)
                        test_name = "Independent T-Test"

                    result_color = "#43E97B" if p > alpha else "#FF4757"
                    conclusion = "✅ Fail to reject H₀ — No significant difference" if p > alpha else "❌ Reject H₀ — Significant difference exists"
                    st.markdown(f'<div style="background:rgba(0,0,0,0.3);border:1px solid {result_color};border-radius:16px;padding:28px;margin:16px 0;text-align:center">'
                                f'<div style="font-size:13px;color:rgba(232,233,240,0.6);margin-bottom:8px">{test_name} Result</div>'
                                f'<div style="display:flex;justify-content:center;gap:48px;margin:16px 0">'
                                f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">TEST STATISTIC</div><div style="font-size:32px;font-weight:800;color:#E8E9F0">{stat:.4f}</div></div>'
                                f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">P-VALUE</div><div style="font-size:32px;font-weight:800;color:{result_color}">{p:.4f}</div></div>'
                                f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">ALPHA</div><div style="font-size:32px;font-weight:800;color:#E8E9F0">{alpha}</div></div>'
                                f'</div>'
                                f'<div style="font-size:15px;font-weight:600;color:{result_color}">{conclusion}</div>'
                                f'</div>', unsafe_allow_html=True)

                    fig = go.Figure()
                    for nm, g in groups.items():
                        fig.add_trace(go.Box(y=g, name=str(nm), boxpoints='outliers'))
                    fig.update_layout(**plotly_dark_layout(title=f"{num_col} by {grp_col}", height=380))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Test error: {e}")

    elif "ANOVA" in test_type:
        c1, c2, c3 = st.columns(3)
        with c1: num_col = st.selectbox("Numeric Column", nc, key="anova_num")
        with c2: grp_col = st.selectbox("Group Column", cc + [c for c in nc if df[c].nunique() <= 10], key="anova_grp")
        with c3: alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="anova_alpha")
        if st.button("▶ Run ANOVA", use_container_width=True, key="anova_run"):
            try:
                groups = [g.dropna().values for _, g in df.groupby(grp_col)[num_col]]
                f_stat, p = f_oneway(*groups)
                result_color = "#43E97B" if p > alpha else "#FF4757"
                conclusion = "✅ Fail to reject H₀ — Group means are equal" if p > alpha else "❌ Reject H₀ — At least one group mean differs"
                st.markdown(f'<div style="background:rgba(0,0,0,0.3);border:1px solid {result_color};border-radius:16px;padding:28px;margin:16px 0;text-align:center">'
                            f'<div style="font-size:13px;color:rgba(232,233,240,0.6)">One-Way ANOVA · {df[grp_col].nunique()} groups</div>'
                            f'<div style="display:flex;justify-content:center;gap:48px;margin:16px 0">'
                            f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">F-STATISTIC</div><div style="font-size:32px;font-weight:800;color:#E8E9F0">{f_stat:.4f}</div></div>'
                            f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">P-VALUE</div><div style="font-size:32px;font-weight:800;color:{result_color}">{p:.4f}</div></div>'
                            f'</div><div style="font-size:15px;font-weight:600;color:{result_color}">{conclusion}</div></div>', unsafe_allow_html=True)
                fig = px.violin(df, x=grp_col, y=num_col, box=True, color=grp_col)
                fig.update_layout(**plotly_dark_layout(height=400))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")

    elif "Chi-Square" in test_type:
        c1, c2, c3 = st.columns(3)
        with c1: col1 = st.selectbox("Column 1", cc, key="chi_c1")
        with c2: col2 = st.selectbox("Column 2", cc, key="chi_c2")
        with c3: alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="chi_alpha")
        if st.button("▶ Run Chi-Square", use_container_width=True, key="chi_run"):
            try:
                ct = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, expected = chi2_contingency(ct)
                result_color = "#43E97B" if p > alpha else "#FF4757"
                conclusion = "✅ Variables are INDEPENDENT" if p > alpha else "❌ Variables are DEPENDENT (significant association)"
                st.markdown(f'<div style="background:rgba(0,0,0,0.3);border:1px solid {result_color};border-radius:16px;padding:28px;margin:16px 0;text-align:center">'
                            f'<div style="display:flex;justify-content:center;gap:48px;margin:16px 0">'
                            f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">CHI² STATISTIC</div><div style="font-size:32px;font-weight:800;color:#E8E9F0">{chi2:.4f}</div></div>'
                            f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">P-VALUE</div><div style="font-size:32px;font-weight:800;color:{result_color}">{p:.4f}</div></div>'
                            f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">DOF</div><div style="font-size:32px;font-weight:800;color:#E8E9F0">{dof}</div></div>'
                            f'</div><div style="font-size:15px;font-weight:600;color:{result_color}">{conclusion}</div></div>', unsafe_allow_html=True)
                fig = px.imshow(ct, text_auto=True, color_continuous_scale='Viridis', title="Contingency Table")
                fig.update_layout(**plotly_dark_layout(height=400))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")

    elif "Shapiro" in test_type:
        col = st.selectbox("Select Column", nc, key="sw_col")
        if st.button("▶ Run Shapiro-Wilk", use_container_width=True, key="sw_run"):
            try:
                data = df[col].dropna()
                if len(data) > 5000: data = data.sample(5000, random_state=42)
                stat, p = shapiro(data)
                is_normal = p > 0.05
                result_color = "#43E97B" if is_normal else "#F9AB00"
                conclusion = "✅ Data is NORMAL (Gaussian distribution)" if is_normal else "⚠️ Data is NOT normal — consider log/sqrt transform"
                st.markdown(f'<div style="background:rgba(0,0,0,0.3);border:1px solid {result_color};border-radius:16px;padding:28px;margin:16px 0;text-align:center">'
                            f'<div style="display:flex;justify-content:center;gap:48px;margin:16px 0">'
                            f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">W STATISTIC</div><div style="font-size:32px;font-weight:800;color:#E8E9F0">{stat:.4f}</div></div>'
                            f'<div><div style="font-size:11px;color:rgba(232,233,240,0.5)">P-VALUE</div><div style="font-size:32px;font-weight:800;color:{result_color}">{p:.4f}</div></div>'
                            f'</div><div style="font-size:15px;font-weight:600;color:{result_color}">{conclusion}</div></div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(data, title=f"{col} Distribution", color_discrete_sequence=['#6C63FF'])
                    fig.update_layout(**plotly_dark_layout(height=350))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    from scipy.stats import probplot
                    qq = probplot(data)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='#6C63FF', size=4)))
                    fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*np.array(qq[0][0])+qq[1][1], mode='lines', line=dict(color='#43E97B', width=2), name='Normal line'))
                    fig2.update_layout(**plotly_dark_layout(title="Q-Q Plot", height=350))
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")

    elif "Correlation" in test_type:
        sel_cols = st.multiselect("Select Numeric Columns", nc, default=nc[:min(8, len(nc))], key="corr_cols")
        if len(sel_cols) >= 2:
            corr = df[sel_cols].corr()
            p_matrix = pd.DataFrame(np.ones_like(corr), columns=sel_cols, index=sel_cols)
            for i in range(len(sel_cols)):
                for j in range(i+1, len(sel_cols)):
                    _, p_val = stats.pearsonr(df[sel_cols[i]].dropna(), df[sel_cols[j]].dropna())
                    p_matrix.iloc[i,j] = p_val
                    p_matrix.iloc[j,i] = p_val
            fig = px.imshow(corr.round(3), text_auto=True, color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1, title="Correlation Matrix (with p-values)")
            fig.update_layout(**plotly_dark_layout(height=500))
            st.plotly_chart(fig, use_container_width=True)
            sig_pairs = []
            for i in range(len(sel_cols)):
                for j in range(i+1, len(sel_cols)):
                    if p_matrix.iloc[i,j] < 0.05:
                        sig_pairs.append(f"**{sel_cols[i]}** ↔ **{sel_cols[j]}**: r={corr.iloc[i,j]:.3f}, p={p_matrix.iloc[i,j]:.4f}")
            if sig_pairs:
                st.markdown("**📌 Significant correlations (p < 0.05):**")
                for pair in sig_pairs: st.markdown(f"- {pair}")


# ═══════════════════════════════════════════════════════════
# TAB 10: SQL QUERY
# ═══════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown("## 🗄️ SQL Query on DataFrame")

    if not SQL_AVAILABLE:
        st.markdown('<div class="alert-warning">⚠️ <b>pandasql</b> not installed. Run: <code>pip install pandasql</code></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">💡 Use <b>df</b> as your table name. Full SQL syntax supported.</div>', unsafe_allow_html=True)

        example_queries = [
            "SELECT * FROM df LIMIT 10",
            f"SELECT {', '.join(df.columns[:3].tolist())} FROM df LIMIT 20",
            f"SELECT COUNT(*) as total, AVG({df.select_dtypes(include=np.number).columns[0] if len(df.select_dtypes(include=np.number).columns) > 0 else 'rowid'}) as avg_val FROM df",
        ]
        ex_query = st.selectbox("📋 Example Queries", ["Custom"] + example_queries, key="sql_example")
        default_q = "" if ex_query == "Custom" else ex_query

        sql_query = st.text_area("Write your SQL query:", value=default_q, height=120, key="sql_input",
                                 placeholder="SELECT * FROM df WHERE column > 100 LIMIT 50")
        c1, c2 = st.columns([3,1])
        with c1:
            run_sql = st.button("▶ Execute Query", use_container_width=True, type="primary", key="sql_run")
        with c2:
            save_result = st.checkbox("Save result as new dataset", key="sql_save")

        if run_sql and sql_query.strip():
            try:
                with st.spinner("Running query..."):
                    result = pdsql.sqldf(sql_query, {'df': df})
                st.markdown(f'<div class="alert-success">✅ Query returned <b>{len(result):,} rows</b> × <b>{len(result.columns)} columns</b></div>', unsafe_allow_html=True)
                st.dataframe(result, use_container_width=True, height=400)

                if save_result:
                    push_history(result, f"🗄️ SQL Query result")
                    st.session_state.df = result
                    st.success("✅ Result saved as current dataset!")
                    st.rerun()

                st.session_state.sql_history.append({'query': sql_query, 'rows': len(result), 'time': datetime.now()})
                download_button(result, "csv", "📥 Download Result", "sql_dl")
            except Exception as e:
                st.error(f"SQL Error: {e}")

        if st.session_state.sql_history:
            st.markdown("### 📜 Query History")
            for i, h in enumerate(reversed(st.session_state.sql_history[-10:])):
                with st.expander(f"{h['time'].strftime('%H:%M:%S')} · {h['rows']} rows · {h['query'][:60]}...", expanded=False):
                    st.code(h['query'], language='sql')
                    if st.button("↩️ Re-run", key=f"sql_rerun_{i}"):
                        st.session_state['sql_input'] = h['query']
                        st.rerun()


# ═══════════════════════════════════════════════════════════
# TAB 11: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
with tabs[10]:
    st.markdown("## 🧠 SHAP — Model Explainability")

    if not SHAP_AVAILABLE:
        st.markdown('<div class="alert-warning">⚠️ <b>SHAP</b> not installed. Run: <code>pip install shap</code></div>', unsafe_allow_html=True)
    elif not st.session_state.trained_models:
        st.markdown('<div class="alert-warning">⚠️ No trained models found. Train a model in the ML Models or AutoML tab first.</div>', unsafe_allow_html=True)
    else:
        model_names = list(st.session_state.trained_models.keys())
        sel_model_name = st.selectbox("🤖 Select Model to Explain", model_names, key="shap_model")
        model_info = st.session_state.trained_models[sel_model_name]

        if st.button("🧠 Generate SHAP Explanation", use_container_width=True, type="primary", key="shap_run"):
            try:
                with st.spinner("Computing SHAP values... (may take 30-60 seconds)"):
                    mdl = model_info['model']
                    X_test = model_info['X_test']
                    features = model_info['features']

                    # Choose explainer based on model type
                    try:
                        explainer = shap.TreeExplainer(mdl)
                        shap_vals = explainer.shap_values(X_test)
                    except:
                        explainer = shap.KernelExplainer(mdl.predict, shap.sample(X_test, 50))
                        shap_vals = explainer.shap_values(shap.sample(X_test, 50))

                    # For multiclass, take first class or average
                    if isinstance(shap_vals, list):
                        sv = np.abs(shap_vals[0])
                    else:
                        sv = shap_vals

                    st.session_state.shap_values = {'vals': sv, 'data': X_test, 'features': features, 'model': sel_model_name}
                    st.success("✅ SHAP values computed!")

            except Exception as e:
                st.error(f"SHAP error: {e}")

        if st.session_state.shap_values and st.session_state.shap_values.get('model') == sel_model_name:
            sv = st.session_state.shap_values['vals']
            features = st.session_state.shap_values['features']
            X_test = st.session_state.shap_values['data']

            shap_tab1, shap_tab2, shap_tab3 = st.tabs(["📊 Feature Importance", "🔍 Sample Explanation", "🌡️ Heatmap"])

            # Normalize X_test to always be a DataFrame
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=features)

            # Normalize sv to always be 2D numpy array
            sv_arr = np.array(sv)
            if isinstance(sv_arr, list):
                sv_arr = np.array(sv_arr[0])
            if sv_arr.ndim == 1:
                sv_arr = sv_arr.reshape(1, -1)
            # For multiclass (3D), take mean across classes
            if sv_arr.ndim == 3:
                sv_arr = sv_arr.mean(axis=0)

            with shap_tab1:
                mean_abs_shap = np.abs(sv_arr).mean(axis=0)
                fi_df = pd.DataFrame({'Feature': features, 'SHAP Importance': mean_abs_shap})
                fi_df = fi_df.sort_values('SHAP Importance', ascending=True).tail(20)
                fig = px.bar(fi_df, x='SHAP Importance', y='Feature', orientation='h',
                            color='SHAP Importance', color_continuous_scale='Viridis',
                            title="Mean |SHAP| — Global Feature Importance")
                fig.update_layout(**plotly_dark_layout(height=500, coloraxis_showscale=False))
                st.plotly_chart(fig, use_container_width=True)

            with shap_tab2:
                max_idx = min(99, len(X_test) - 1)
                sample_idx = st.slider("Select Sample Index", 0, max_idx, 0, key="shap_sample")
                sample_sv = sv_arr[sample_idx] if sample_idx < len(sv_arr) else sv_arr[0]
                feat_vals = X_test.iloc[sample_idx].values if sample_idx < len(X_test) else X_test.iloc[0].values
                sample_df = pd.DataFrame({
                    'Feature': features,
                    'SHAP Value': sample_sv,
                    'Feature Value': feat_vals
                })
                sample_df['Direction'] = sample_df['SHAP Value'].apply(
                    lambda x: '▲ Increases prediction' if x > 0 else '▼ Decreases prediction')
                sample_df = sample_df.iloc[sample_df['SHAP Value'].abs().argsort()[::-1]]
                colors = ['#43E97B' if v > 0 else '#FF4757' for v in sample_df['SHAP Value']]
                fig2 = go.Figure(go.Bar(x=sample_df['SHAP Value'], y=sample_df['Feature'],
                                       orientation='h', marker_color=colors))
                fig2.update_layout(**plotly_dark_layout(title=f"Sample #{sample_idx} — Feature Contributions", height=500))
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(sample_df[['Feature', 'Feature Value', 'SHAP Value', 'Direction']].reset_index(drop=True),
                            use_container_width=True, hide_index=True)

            with shap_tab3:
                top_n = min(10, len(features))
                mean_shap_per_feat = np.abs(sv_arr).mean(axis=0)
                top_feats_idx = np.argsort(mean_shap_per_feat)[-top_n:][::-1]
                heat_data = sv_arr[:, top_feats_idx]
                top_feat_names = [features[i] for i in top_feats_idx]
                fig3 = px.imshow(heat_data.T, x=list(range(len(heat_data))), y=top_feat_names,
                                color_continuous_scale='RdBu_r', title="SHAP Values Heatmap (samples × features)",
                                labels={'x': 'Sample Index', 'y': 'Feature', 'color': 'SHAP'})
                fig3.update_layout(**plotly_dark_layout(height=500))
                st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 12: AI ASSISTANT  (Google Gemini — FREE)
# ═══════════════════════════════════════════════════════════
with tabs[11]:
    st.markdown("## 💬 AI Data Assistant")

    # ── Provider Selection & API Key Setup ──
    with st.expander("🔑 Setup — Choose AI Provider (Free Options Available)", expanded=not st.session_state.get('ai_api_key','')):
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
            <div style="background:rgba(67,233,123,0.08);border:1px solid rgba(67,233,123,0.25);border-radius:12px;padding:14px">
                <div style="font-weight:700;color:#43E97B;margin-bottom:6px">🆓 Google Gemini — RECOMMENDED</div>
                <div style="font-size:12px;color:rgba(232,233,240,0.7)">✅ Completely FREE · 1500 requests/day<br>✅ Very accurate (Gemini 1.5 Flash)<br>✅ Key milti hai: <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#43E97B">aistudio.google.com</a></div>
            </div>
            <div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.25);border-radius:12px;padding:14px">
                <div style="font-weight:700;color:#a8a4ff;margin-bottom:6px">🆓 Groq — FASTEST FREE</div>
                <div style="font-size:12px;color:rgba(232,233,240,0.7)">✅ Free tier · Llama 3.1 70B model<br>✅ Super fast responses<br>✅ Key milti hai: <a href="https://console.groq.com" target="_blank" style="color:#a8a4ff">console.groq.com</a></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        provider = st.selectbox("🤖 Select Provider", ["🆓 Google Gemini (Free)", "🆓 Groq (Free)", "💳 Anthropic Claude (Paid)"],
                                key="ai_provider", index=0)
        st.session_state['ai_provider'] = provider

        if "Gemini" in provider:
            placeholder, prefix, label = "AIza...", "AIza", "Google Gemini API Key"
            help_url = "https://aistudio.google.com/app/apikey"
        elif "Groq" in provider:
            placeholder, prefix, label = "gsk_...", "gsk_", "Groq API Key"
            help_url = "https://console.groq.com/keys"
        else:
            placeholder, prefix, label = "sk-ant-...", "sk-ant-", "Anthropic API Key"
            help_url = "https://console.anthropic.com"

        st.markdown(f'<div class="alert-info">🔗 Get your free key here: <a href="{help_url}" target="_blank" style="color:#a8a4ff"><b>{help_url}</b></a></div>', unsafe_allow_html=True)

        key_input = st.text_input(label, type="password",
                                  value=st.session_state.get('ai_api_key', ''),
                                  placeholder=placeholder, key="api_key_field")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Key", use_container_width=True, key="save_key_btn"):
                if key_input and len(key_input) > 10:
                    st.session_state['ai_api_key'] = key_input
                    st.success("✅ API key saved! You can now start chatting.")
                    st.rerun()
                else:
                    st.error("❌ Please enter a valid API key")
        with c2:
            if st.button("🗑️ Clear Key", use_container_width=True, key="clear_key_btn"):
                st.session_state['ai_api_key'] = ''
                st.rerun()

    api_key    = st.session_state.get('ai_api_key', '')
    provider   = st.session_state.get('ai_provider', '🆓 Google Gemini (Free)')

    if not api_key:
        st.markdown('<div class="alert-warning">⚠️ <b>API key nahi mili.</b> Upar "Setup" section mein apni free key enter karo.<br><br>👉 Google Gemini ke liye: <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#F9AB00">aistudio.google.com/app/apikey</a> — bilkul free hai!</div>', unsafe_allow_html=True)
    else:
        # ── Build rich data context ──
        nc_cols = df.select_dtypes(include=np.number).columns.tolist()
        cc_cols = df.select_dtypes(include='object').columns.tolist()
        miss_cols = df.isnull().sum()
        miss_cols = miss_cols[miss_cols > 0].to_dict()
        try:    desc_stats = df.describe().round(3).to_dict()
        except: desc_stats = {}
        data_ctx = (
            f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.\n"
            f"Numeric columns ({len(nc_cols)}): {', '.join(nc_cols[:15])}.\n"
            f"Categorical columns ({len(cc_cols)}): {', '.join(cc_cols[:15])}.\n"
            f"Missing values: {df.isnull().sum().sum()} ({'cols: ' + str(miss_cols) if miss_cols else 'none'}).\n"
            f"Dtypes: {dict(df.dtypes.astype(str))}.\n"
            f"Stats: {desc_stats}.\n"
            f"Duplicates: {df.duplicated().sum()}.\n"
            f"Trained models: {list(st.session_state.trained_models.keys()) or 'None'}.\n"
        )
        system_prompt = (
            "You are an expert data scientist embedded in an ML Analytics app. "
            "Dataset context:\n\n" + data_ctx + "\n\n"
            "Instructions: Be concise and actionable. Reference actual column names. "
            "Use markdown formatting, bullet points, and code blocks where helpful. "
            "Suggest Python/pandas code when relevant."
        )

        # ── Quick prompts ──
        st.markdown("**💡 Quick questions:**")
        qcols = st.columns(3)
        quick_prompts = [
            "What are the main patterns in this data?",
            "Which columns have data quality issues?",
            "What ML model would you recommend?",
            "Which features are most important?",
            "Are there outliers I should handle?",
            "What transforms to apply before modeling?",
            "Give me a full EDA summary.",
            "How to handle the missing values?",
            "Classification or regression problem?"
        ]
        for i, qp in enumerate(quick_prompts):
            with qcols[i % 3]:
                if st.button(qp[:36] + ("…" if len(qp) > 36 else ""), key=f"qp_{i}", use_container_width=True):
                    st.session_state['ai_prefill'] = qp

        # ── Chat history ──
        if st.session_state.chat_history:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(
                        f'<div style="display:flex;justify-content:flex-end;margin:10px 0">'
                        f'<div style="background:rgba(108,99,255,0.12);border:1px solid rgba(108,99,255,0.3);'
                        f'border-radius:16px 16px 4px 16px;padding:12px 18px;max-width:80%;font-size:14px">'
                        f'👤 <b>You</b><br><span style="color:#E8E9F0">{msg["content"]}</span>'
                        f'</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="background:rgba(67,233,123,0.06);border:1px solid rgba(67,233,123,0.2);'
                        f'border-radius:4px 16px 16px 16px;padding:12px 18px;margin:10px 0;font-size:14px">'
                        f'🤖 <b>AI Assistant</b></div>', unsafe_allow_html=True)
                    st.markdown(msg["content"])
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Input ──
        prefill = st.session_state.pop('ai_prefill', '')
        user_input = st.text_area("Ask anything about your data:", value=prefill, height=100,
                                  placeholder="e.g. What patterns exist? Which model is best?",
                                  key="ai_chat_input")
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1: send_btn   = st.button("📤 Send", use_container_width=True, type="primary", key="ai_send")
        with c2: clear_btn  = st.button("🗑️ Clear", use_container_width=True, key="ai_clear")
        with c3: export_btn = st.button("📥 Export", use_container_width=True, key="ai_export")

        if clear_btn:
            st.session_state.chat_history = []
            st.rerun()

        if export_btn and st.session_state.chat_history:
            chat_txt = "\n\n".join([f"{'YOU' if m['role']=='user' else 'AI'}: {m['content']}" for m in st.session_state.chat_history])
            st.download_button("💾 Download", chat_txt, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain", key="chat_dl")

        if send_btn and user_input.strip():
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            ai_reply = ""

            try:
                with st.spinner("🤖 AI is thinking..."):

                    # ── Google Gemini (FREE) ──
                    if "Gemini" in provider:
                        history_for_gemini = []
                        for m in st.session_state.chat_history[-12:]:
                            role = "user" if m['role'] == 'user' else "model"
                            history_for_gemini.append({"role": role, "parts": [{"text": m['content']}]})
                        # Inject system context into first user message
                        if history_for_gemini and history_for_gemini[0]['role'] == 'user':
                            history_for_gemini[0]['parts'][0]['text'] = system_prompt + "\n\nUser question: " + history_for_gemini[0]['parts'][0]['text']

                        resp = requests.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                            headers={"Content-Type": "application/json"},
                            json={"contents": history_for_gemini,
                                  "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.7}},
                            timeout=60
                        )
                        if resp.status_code == 200:
                            ai_reply = resp.json()['candidates'][0]['content']['parts'][0]['text']
                        elif resp.status_code == 400:
                            ai_reply = f"❌ **Bad Request:** {resp.json().get('error', {}).get('message', resp.text[:200])}"
                        elif resp.status_code == 403:
                            ai_reply = "❌ **Invalid API Key.** Check your Gemini key at aistudio.google.com/app/apikey"
                        elif resp.status_code == 429:
                            ai_reply = "⚠️ **Rate limit hit.** Free tier: 15 requests/min. Wait a moment and retry."
                        else:
                            ai_reply = f"❌ **Gemini Error {resp.status_code}:** {resp.text[:300]}"

                    # ── Groq (FREE) ──
                    elif "Groq" in provider:
                        groq_msgs = [{"role": "system", "content": system_prompt}]
                        for m in st.session_state.chat_history[-12:]:
                            groq_msgs.append({"role": m['role'], "content": m['content']})
                        resp = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                            json={"model": "llama-3.1-70b-versatile", "messages": groq_msgs, "max_tokens": 1500},
                            timeout=60
                        )
                        if resp.status_code == 200:
                            ai_reply = resp.json()['choices'][0]['message']['content']
                        elif resp.status_code == 401:
                            ai_reply = "❌ **Invalid Groq API Key.** Check at console.groq.com/keys"
                        elif resp.status_code == 429:
                            ai_reply = "⚠️ **Rate limit hit.** Wait a moment and retry."
                        else:
                            ai_reply = f"❌ **Groq Error {resp.status_code}:** {resp.text[:300]}"

                    # ── Anthropic (Paid) ──
                    else:
                        anth_msgs = [{"role": m['role'], "content": m['content']} for m in st.session_state.chat_history[-12:]]
                        resp = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
                            json={"model": "claude-sonnet-4-20250514", "max_tokens": 1500, "system": system_prompt, "messages": anth_msgs},
                            timeout=60
                        )
                        if resp.status_code == 200:
                            ai_reply = resp.json()['content'][0]['text']
                        elif resp.status_code == 401:
                            ai_reply = "❌ **Invalid Anthropic API Key.**"
                        elif resp.status_code == 429:
                            ai_reply = "⚠️ **Rate limit hit.** Wait and retry."
                        else:
                            ai_reply = f"❌ **Anthropic Error {resp.status_code}:** {resp.text[:300]}"

            except requests.exceptions.Timeout:
                ai_reply = "⏱️ **Request timed out.** Try again."
            except requests.exceptions.ConnectionError:
                ai_reply = "🌐 **Connection error.** Check your internet."
            except Exception as e:
                ai_reply = f"❌ **Error:** {str(e)}"

            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_reply})
            st.rerun()




# ═══════════════════════════════════════════════════════════
# TAB 13: EXPORT
# ═══════════════════════════════════════════════════════════
with tabs[12]:
    st.markdown("## 💾 Export Data & Models")

    mem = df.memory_usage(deep=True).sum() / 1024**2
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;">
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;">
            <div><div class="metric-label">ROWS</div><div class="metric-value">{df.shape[0]:,}</div></div>
            <div><div class="metric-label">COLUMNS</div><div class="metric-value">{df.shape[1]:,}</div></div>
            <div><div class="metric-label">MEMORY</div><div class="metric-value">{mem:.1f} MB</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📄 Export Formats")
    c1, c2, c3, c4 = st.columns(4)
    with c1: download_button(df, "csv", "📄 CSV", "exp_csv")
    with c2: download_button(df, "excel", "📊 Excel", "exp_excel")
    with c3: download_button(df, "json", "📋 JSON", "exp_json")
    with c4:
        try:
            pb = io.BytesIO()
            df.to_parquet(pb, index=False)
            st.download_button("📦 Parquet", pb.getvalue(),
                              f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                              "application/octet-stream", key="exp_parquet", use_container_width=True)
        except:
            st.button("📦 Parquet (unavailable)", disabled=True, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.session_state.trained_models:
        st.markdown(f"### 🤖 Trained Models ({len(st.session_state.trained_models)})")
        for mn in st.session_state.trained_models:
            mi = st.session_state.trained_models[mn]
            is_best = mn == st.session_state.best_model or f"AutoML_{st.session_state.best_model}" == mn
            st.markdown(f"""
            <div class="model-row {'best' if is_best else ''}">
                <span>{'🏆 ' if is_best else '✅ '}<b>{mn}</b></span>
                <span style="color:var(--text-muted);font-size:12px;">Type: {mi.get('type','?')} · Target: {mi.get('target','?')} · Features: {len(mi.get('features',[]))}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("### 📜 Operation History")
        for i, h in enumerate(reversed(st.session_state.history[-20:])):
            with st.expander(f"{h['time'].strftime('%H:%M:%S')} · {h['action']}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Rows", f"{h['shape'][0]:,}")
                with c2: st.metric("Cols", f"{h['shape'][1]:,}")
                with c3: st.metric("Memory", f"{h['df'].memory_usage(deep=True).sum()/1024**2:.1f} MB")
                with c4:
                    if st.button("↩️ Restore", key=f"rst_{i}", use_container_width=True):
                        st.session_state.df = h['df'].copy()
                        st.success("✅ Restored!")
                        st.rerun()


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-title">🚀 ML Analytics Pro v4.0 ULTRA</div>
    <div class="footer-sub">
        XGBoost · LightGBM · Scikit-learn · SHAP · Plotly · Streamlit · Claude AI<br>
        Statistical Tests · SQL Query · Auto Dtype Fixer · AI Assistant · AutoML
    </div>
</div>
""", unsafe_allow_html=True)
