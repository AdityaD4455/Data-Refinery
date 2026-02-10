import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                   LabelEncoder, QuantileTransformer, PowerTransformer)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               AdaBoostClassifier, AdaBoostRegressor, IsolationForest,
                               ExtraTreesClassifier, ExtraTreesRegressor,
                               VotingClassifier, VotingRegressor, StackingClassifier)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, 
                                  ElasticNet, BayesianRidge, SGDClassifier, SGDRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                     RandomizedSearchCV, StratifiedKFold, KFold)
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              mean_squared_error, r2_score, mean_absolute_error,
                              mean_absolute_percentage_error, silhouette_score)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression, 
                                       mutual_info_classif, mutual_info_regression,
                                       RFE, SelectFromModel, VarianceThreshold)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import json
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera
import seaborn as sns

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    layout="wide",
    page_title="🚀 ML Analytics Pro - Advanced AI",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Initialize session state
for key in ['df', 'df2', 'history', 'trained_models', 'best_model', 'last_change', 
            'auto_insights', 'feature_importance', 'model_explanations', 'data_quality_score']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'history' and key != 'trained_models' else ({} if key == 'trained_models' else [])

# Enhanced CSS with advanced animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&family=Orbitron:wght@400;700;900&display=swap');

    * { font-family: 'Poppins', sans-serif; }
    
    .main {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        padding: 2rem;
        position: relative;
        overflow-x: hidden;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Particle effect overlay */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: particleFloat 30s linear infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes particleFloat {
        0% { transform: translateY(0); }
        100% { transform: translateY(-100px); }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(30px) saturate(180%);
        border-radius: 24px;
        padding: 35px;
        margin: 25px 0;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 15px 60px rgba(0, 0, 0, 0.3), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.7s;
    }

    .glass-card:hover::before {
        left: 100%;
    }

    .glass-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 25px 80px rgba(102, 126, 234, 0.4),
                    0 10px 40px rgba(118, 75, 162, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }

    .change-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 25px 0;
        color: white;
        animation: slideInScale 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }

    .change-summary::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: rotateGlow 10s linear infinite;
    }

    @keyframes slideInScale {
        0% { 
            opacity: 0; 
            transform: translateX(-50px) scale(0.8);
        }
        100% { 
            opacity: 1; 
            transform: translateX(0) scale(1);
        }
    }

    @keyframes rotateGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .metric-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px 25px;
        margin: 15px 0;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }

    .metric-container::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.6s ease;
    }

    .metric-container:hover {
        transform: translateY(-15px) scale(1.03);
        box-shadow: 0 30px 70px rgba(102, 126, 234, 0.5),
                    0 15px 35px rgba(118, 75, 162, 0.4);
        border-color: rgba(102, 126, 234, 0.6);
    }

    .metric-container:hover::after {
        transform: scaleX(1);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 16px 32px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        width: 100%;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }

    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }

    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.7),
                    0 10px 25px rgba(118, 75, 162, 0.5);
    }

    .stButton > button:active {
        transform: translateY(-2px) scale(0.98);
    }

    h1, h2, h3 { 
        color: white !important; 
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5),
                     0 0 40px rgba(118, 75, 162, 0.3);
        font-family: 'Orbitron', sans-serif;
        animation: titleGlow 3s ease-in-out infinite;
    }

    @keyframes titleGlow {
        0%, 100% { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5), 0 0 40px rgba(118, 75, 162, 0.3); }
        50% { text-shadow: 0 0 30px rgba(102, 126, 234, 0.8), 0 0 60px rgba(118, 75, 162, 0.6); }
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.98) 0%, rgba(48, 43, 99, 0.98) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.5);
    }

    section[data-testid="stSidebar"] * { color: white !important; }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-left: 5px solid #667eea;
        border-radius: 12px;
        padding: 20px;
        margin: 18px 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        animation: pulseGlow 2s ease-in-out infinite;
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2); }
        50% { box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4); }
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        color: white;
        transition: all 0.3s ease;
        border-radius: 12px;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.5);
        animation: tabSelect 0.3s ease;
    }

    @keyframes tabSelect {
        0% { transform: scale(0.95); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .stDataFrame {
        animation: fadeInUp 0.5s ease;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Progress bar enhancement */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: progressFlow 2s linear infinite;
    }

    @keyframes progressFlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* AI Badge */
    .ai-badge {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 1px;
        animation: badgePulse 2s ease-in-out infinite;
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
    }

    @keyframes badgePulse {
        0%, 100% { transform: scale(1); box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4); }
        50% { transform: scale(1.05); box-shadow: 0 8px 25px rgba(0, 210, 255, 0.6); }
    }

    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Tooltip enhancement */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 8px 12px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s, transform 0.3s;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# Advanced utility functions
def calculate_data_quality_score(df):
    """AI-powered data quality assessment"""
    score = 100
    issues = []
    
    # Missing values penalty
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    if missing_pct > 0:
        penalty = min(30, missing_pct * 3)
        score -= penalty
        issues.append(f"Missing data: {missing_pct:.1f}% ({penalty:.0f} points)")
    
    # Duplicate penalty
    dup_pct = (df.duplicated().sum() / len(df) * 100)
    if dup_pct > 0:
        penalty = min(20, dup_pct * 5)
        score -= penalty
        issues.append(f"Duplicates: {dup_pct:.1f}% ({penalty:.0f} points)")
    
    # Data type consistency
    type_issues = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col])
                type_issues += 1
            except:
                pass
    if type_issues > 0:
        penalty = min(15, type_issues * 3)
        score -= penalty
        issues.append(f"Type inconsistency: {type_issues} columns ({penalty:.0f} points)")
    
    # Outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_count += outliers
    
    outlier_pct = (outlier_count / (len(df) * len(numeric_cols)) * 100) if len(numeric_cols) > 0 else 0
    if outlier_pct > 5:
        penalty = min(15, (outlier_pct - 5) * 2)
        score -= penalty
        issues.append(f"Outliers: {outlier_pct:.1f}% ({penalty:.0f} points)")
    
    # Class imbalance for categorical
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].nunique() < 20:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                if imbalance_ratio > 10:
                    penalty = min(10, imbalance_ratio / 5)
                    score -= penalty
                    issues.append(f"Class imbalance in {col}: {imbalance_ratio:.1f}x ({penalty:.0f} points)")
                    break
    
    return max(0, score), issues

def auto_generate_insights(df):
    """AI-powered automatic insights generation"""
    insights = []
    
    # Statistical insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Top 5 numeric columns
        skewness = df[col].skew()
        if abs(skewness) > 1:
            insights.append(f"📊 '{col}' is {'highly right-skewed' if skewness > 1 else 'highly left-skewed'} (skewness: {skewness:.2f})")
        
        # Detect potential target variable
        if df[col].nunique() < 10 and df[col].nunique() > 1:
            insights.append(f"🎯 '{col}' could be a good target variable ({df[col].nunique()} unique values)")
    
    # Correlation insights
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            for col1, col2, corr_val in high_corr[:3]:
                insights.append(f"🔗 Strong correlation: '{col1}' ↔ '{col2}' (r={corr_val:.2f})")
    
    # Categorical insights
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:3]:
        unique_count = df[col].nunique()
        if unique_count < 20:
            dominant = df[col].value_counts().iloc[0]
            dominant_pct = dominant / len(df) * 100
            if dominant_pct > 50:
                insights.append(f"📈 '{col}' dominated by one value ({dominant_pct:.1f}%)")
    
    # Missing pattern insights
    missing_cols = df.columns[df.isnull().sum() > 0]
    if len(missing_cols) > 0:
        total_missing = df[missing_cols].isnull().sum().sum()
        insights.append(f"❓ {len(missing_cols)} columns with missing data ({total_missing:,} total)")
    
    return insights

def advanced_feature_engineering(df, target_col=None):
    """AI-powered automatic feature engineering"""
    df_new = df.copy()
    new_features = []
    
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Polynomial features for top correlated pairs
    if len(numeric_cols) >= 2:
        if target_col and target_col in df_new.columns:
            correlations = df_new[numeric_cols].corrwith(df_new[target_col]).abs().sort_values(ascending=False)
            top_features = correlations.head(3).index.tolist()
        else:
            top_features = numeric_cols[:3]
        
        # Interactions
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                col1, col2 = top_features[i], top_features[j]
                new_col = f"{col1}_x_{col2}"
                df_new[new_col] = df_new[col1] * df_new[col2]
                new_features.append(new_col)
        
        # Ratios
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                col1, col2 = top_features[i], top_features[j]
                if (df_new[col2] != 0).all():
                    new_col = f"{col1}_div_{col2}"
                    df_new[new_col] = df_new[col1] / df_new[col2].replace(0, 1)
                    new_features.append(new_col)
    
    # Aggregations
    if len(numeric_cols) >= 3:
        df_new['mean_all'] = df_new[numeric_cols].mean(axis=1)
        df_new['std_all'] = df_new[numeric_cols].std(axis=1)
        df_new['max_all'] = df_new[numeric_cols].max(axis=1)
        df_new['min_all'] = df_new[numeric_cols].min(axis=1)
        new_features.extend(['mean_all', 'std_all', 'max_all', 'min_all'])
    
    return df_new, new_features

def hyperparameter_tuning(model, X, y, problem_type, method='random'):
    """Advanced hyperparameter optimization"""
    param_grids = {
        'RandomForestClassifier': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForestRegressor': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoostingClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7]
        },
        'XGBClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }
    
    model_name = type(model).__name__
    if model_name in param_grids:
        if method == 'random':
            search = RandomizedSearchCV(model, param_grids[model_name], 
                                       n_iter=10, cv=3, random_state=42, n_jobs=-1)
        else:
            search = GridSearchCV(model, param_grids[model_name], cv=3, n_jobs=-1)
        
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    
    return model, {}

def push_history(df, action):
    """Save operation with change tracking"""
    changes = {
        "action": action,
        "timestamp": datetime.now(),
        "rows_before": len(st.session_state.df) if st.session_state.df is not None else 0,
        "rows_after": len(df),
        "cols_before": len(st.session_state.df.columns) if st.session_state.df is not None else 0,
        "cols_after": len(df.columns)
    }
    
    st.session_state.history.append({
        "time": datetime.now(),
        "action": action,
        "df": df.copy(),
        "shape": df.shape,
        "changes": changes
    })
    
    st.session_state.last_change = changes
    
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]

def show_change_summary():
    """Display what changed in last operation"""
    if st.session_state.last_change:
        ch = st.session_state.last_change
        rows_diff = ch['rows_after'] - ch['rows_before']
        cols_diff = ch['cols_after'] - ch['cols_before']
        
        st.markdown(f"""
        <div class="change-summary">
            <div style="position: relative; z-index: 1;">
                <h4>📊 Recent Changes: {ch['action']}</h4>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                    <div>
                        <div style="font-size: 12px; opacity: 0.9;">ROWS</div>
                        <div style="font-size: 20px; font-weight: 700;">
                            {ch['rows_before']:,} → {ch['rows_after']:,}
                        </div>
                        <div style="font-size: 14px; opacity: 0.8;">
                            {'+' if rows_diff >= 0 else ''}{rows_diff:,}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 12px; opacity: 0.9;">COLUMNS</div>
                        <div style="font-size: 20px; font-weight: 700;">
                            {ch['cols_before']:,} → {ch['cols_after']:,}
                        </div>
                        <div style="font-size: 14px; opacity: 0.8;">
                            {'+' if cols_diff >= 0 else ''}{cols_diff:,}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 12px; opacity: 0.9;">TIME</div>
                        <div style="font-size: 20px; font-weight: 700;">
                            {ch['timestamp'].strftime('%H:%M:%S')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def download_button(df, format_type, label, key):
    """Download with error handling"""
    try:
        if format_type == "csv":
            csv = df.to_csv(index=False)
            st.download_button(label, csv, f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                             "text/csv", key=key, use_container_width=True)
        elif format_type == "excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            st.download_button(label, buffer.getvalue(), 
                             f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                             "application/vnd.ms-excel", key=key, use_container_width=True)
        elif format_type == "json":
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(label, json_str, 
                             f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                             "application/json", key=key, use_container_width=True)
    except Exception as e:
        st.error(f"Download error: {str(e)}")

def detect_problem_type(df, target_col):
    """Advanced problem type detection"""
    if target_col not in df.columns:
        return 'classification'
    
    if df[target_col].dtype == 'object':
        return 'classification'
    
    unique_ratio = df[target_col].nunique() / len(df[target_col].dropna())
    unique_count = df[target_col].nunique()
    
    # Check if values are integers
    is_integer = (df[target_col].dropna() % 1 == 0).all()
    
    if unique_count < 20 and (is_integer or unique_ratio < 0.05):
        return 'classification'
    
    return 'regression'

def prepare_ml_data(df, target_col, feature_cols, use_advanced=False):
    """Advanced data preparation with multiple imputation strategies"""
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Advanced imputation
    if use_advanced:
        numeric_cols = [col for col in feature_cols if df_clean[col].dtype in [np.float64, np.int64]]
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    else:
        for col in feature_cols:
            if col in df_clean.columns:
                if df_clean[col].dtype in [np.float64, np.int64]:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna('missing', inplace=True)
    
    X = df_clean[feature_cols].copy()
    feature_encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            feature_encoders[col] = le
    
    y = df_clean[target_col]
    target_encoder = None
    problem_type = detect_problem_type(df_clean, target_col)
    
    if problem_type == 'classification' and y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    return X, y, feature_encoders, target_encoder, problem_type

# Header with advanced animation
st.markdown("""
<div style="text-align: center; padding: 40px; position: relative;">
    <div style="position: absolute; top: 20px; right: 20px;">
        <span class="ai-badge">🤖 AI-POWERED</span>
    </div>
    <h1 style="font-size: 58px; font-weight: 900; margin-bottom: 10px;">
        🚀 ML ANALYTICS PRO
    </h1>
    <div style="font-size: 20px; color: rgba(255, 255, 255, 0.95); font-weight: 500; letter-spacing: 1px;">
        Advanced AI • Real-time Insights • Production Ready
    </div>
    <div style="margin-top: 20px; font-size: 14px; opacity: 0.8;">
        Powered by XGBoost, LightGBM & Advanced ML
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader("Upload dataset", type=['csv', 'xlsx', 'json', 'parquet'], key="main_upload")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            if st.session_state.df is None or len(st.session_state.history) == 0:
                st.session_state.df = df
                push_history(df, "📁 Dataset uploaded")
                
                # Auto-generate insights
                st.session_state.auto_insights = auto_generate_insights(df)
                score, issues = calculate_data_quality_score(df)
                st.session_state.data_quality_score = {'score': score, 'issues': issues}
                
                st.success(f"✅ {df.shape[0]:,} × {df.shape[1]:,}")
                st.rerun()
            elif st.session_state.df is not None:
                if st.button("Replace current data?", use_container_width=True):
                    st.session_state.df = df
                    push_history(df, "📁 Dataset replaced")
                    st.session_state.auto_insights = auto_generate_insights(df)
                    score, issues = calculate_data_quality_score(df)
                    st.session_state.data_quality_score = {'score': score, 'issues': issues}
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Second dataset
    st.markdown("### 📊 Second Dataset")
    uploaded_file2 = st.file_uploader("For merging", type=['csv', 'xlsx', 'json'], key="second_upload")
    
    if uploaded_file2:
        try:
            if uploaded_file2.name.endswith('.csv'):
                df2 = pd.read_csv(uploaded_file2)
            elif uploaded_file2.name.endswith('.xlsx'):
                df2 = pd.read_excel(uploaded_file2)
            else:
                df2 = pd.read_json(uploaded_file2)
            
            st.session_state.df2 = df2
            st.success(f"✅ {df2.shape[0]:,} rows")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Sample data
    if st.button("🎲 Load Sample Data", use_container_width=True):
        np.random.seed(42)
        sample_df = pd.DataFrame({
            'Age': np.random.randint(18, 70, 2000),
            'Income': np.random.randint(20000, 150000, 2000),
            'Score': np.random.randint(50, 100, 2000),
            'Experience': np.random.randint(0, 30, 2000),
            'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 2000, p=[0.3, 0.4, 0.2, 0.1]),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 2000),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 2000),
            'Satisfaction': np.random.randint(1, 11, 2000),
            'Target': np.random.choice([0, 1], 2000, p=[0.65, 0.35])
        })
        st.session_state.df = sample_df
        push_history(sample_df, "🎲 Sample data loaded")
        st.session_state.auto_insights = auto_generate_insights(sample_df)
        score, issues = calculate_data_quality_score(sample_df)
        st.session_state.data_quality_score = {'score': score, 'issues': issues}
        st.rerun()
    
    st.markdown("---")
    
    # Data quality score
    if st.session_state.data_quality_score:
        score = st.session_state.data_quality_score['score']
        color = '#00ff88' if score >= 80 else '#ffaa00' if score >= 60 else '#ff4444'
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); border-radius: 15px; margin: 15px 0;">
            <div style="font-size: 14px; opacity: 0.9;">DATA QUALITY</div>
            <div style="font-size: 48px; font-weight: 900; color: {color}; text-shadow: 0 0 20px {color};">
                {score:.0f}
            </div>
            <div style="font-size: 12px; opacity: 0.8;">/ 100</div>
        </div>
        """, unsafe_allow_html=True)

# Main content
if st.session_state.df is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 80px;">
        <div style="font-size: 80px; margin-bottom: 20px;">🤖</div>
        <h2>👈 Upload Your Dataset to Begin</h2>
        <p style="font-size: 18px; opacity: 0.9; margin: 25px 0; line-height: 1.8;">
            🚀 Advanced ML Algorithms<br>
            🧠 AI-Powered Insights<br>
            ⚡ Real-time Processing<br>
            🎯 Production-Ready Models
        </p>
        <div style="margin-top: 30px;">
            <span class="ai-badge">POWERED BY AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# Show recent changes
show_change_summary()

# AI Insights Banner
if st.session_state.auto_insights:
    with st.expander("🧠 AI-Generated Insights", expanded=True):
        for insight in st.session_state.auto_insights[:5]:
            st.markdown(f"""
            <div class="insight-box">
                {insight}
            </div>
            """, unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["📊 Overview", "🔧 Cleaning", "📈 Viz", "🤖 ML", "🎯 Predict", "🧬 Features", "⚙️ Advanced", "🏆 AutoML", "💾 Export"])

# TAB 1: Overview (Enhanced)
with tabs[0]:
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 45px; margin-bottom: 10px;">📝</div>
            <div style="font-size: 13px; opacity: 0.9; letter-spacing: 1px;">ROWS</div>
            <div style="font-size: 36px; font-weight: 800; margin-top: 8px;">{df.shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 45px; margin-bottom: 10px;">🔢</div>
            <div style="font-size: 13px; opacity: 0.9; letter-spacing: 1px;">COLUMNS</div>
            <div style="font-size: 36px; font-weight: 800; margin-top: 8px;">{df.shape[1]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        color = '#00ff88' if missing_pct < 5 else '#ffaa00' if missing_pct < 15 else '#ff4444'
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 45px; margin-bottom: 10px;">❓</div>
            <div style="font-size: 13px; opacity: 0.9; letter-spacing: 1px;">MISSING</div>
            <div style="font-size: 36px; font-weight: 800; margin-top: 8px; color: {color};">{missing_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 45px; margin-bottom: 10px;">💾</div>
            <div style="font-size: 13px; opacity: 0.9; letter-spacing: 1px;">MEMORY</div>
            <div style="font-size: 36px; font-weight: 800; margin-top: 8px;">{memory_mb:.1f} MB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        dup_pct = (df.duplicated().sum() / len(df) * 100)
        color = '#00ff88' if dup_pct == 0 else '#ffaa00' if dup_pct < 5 else '#ff4444'
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 45px; margin-bottom: 10px;">🔄</div>
            <div style="font-size: 13px; opacity: 0.9; letter-spacing: 1px;">DUPLICATES</div>
            <div style="font-size: 36px; font-weight: 800; margin-top: 8px; color: {color};">{dup_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 👀 Data Preview")
        n_rows = st.slider("Rows to display", 5, 100, 15, key="preview_rows")
        st.dataframe(df.head(n_rows), use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📊 Type Distribution")
        type_counts = df.dtypes.value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index.astype(str),
            values=type_counts.values,
            hole=0.5,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe']),
            textfont=dict(size=14, color='white')
        )])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Column Analysis")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notnull().sum(),
            'Null': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique': df.nunique(),
            'Unique %': (df.nunique() / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📈 Statistics Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe().T
            stats_df['cv'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
            stats_df['range'] = stats_df['max'] - stats_df['min']
            st.dataframe(stats_df, use_container_width=True, height=400)

# TAB 2: Advanced Cleaning
with tabs[1]:
    st.markdown("## 🔧 Advanced Data Cleaning")
    
    with st.expander("❓ Missing Values Analysis", expanded=True):
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            # Visualize missing data pattern
            fig = px.bar(missing_df, x='Column', y='Percentage', 
                        title='Missing Data Distribution',
                        color='Percentage',
                        color_continuous_scale='Reds')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                target_col = st.selectbox("Column", missing_df['Column'].tolist(), key="missing_col")
            with col2:
                strategy = st.selectbox("Strategy", 
                    ["Drop rows", "Fill mean", "Fill median", "Fill mode", 
                     "Forward fill", "Backward fill", "KNN Imputer", "Fill value"],
                    key="missing_strategy")
            with col3:
                if strategy == "Fill value":
                    fill_val = st.text_input("Value", "", key="fill_val")
            
            if st.button("🔧 Apply Fix", key="missing_fix", use_container_width=True):
                df_clean = df.copy()
                try:
                    if strategy == "Drop rows":
                        df_clean = df_clean.dropna(subset=[target_col])
                    elif strategy == "Fill mean":
                        df_clean[target_col].fillna(df_clean[target_col].mean(), inplace=True)
                    elif strategy == "Fill median":
                        df_clean[target_col].fillna(df_clean[target_col].median(), inplace=True)
                    elif strategy == "Fill mode":
                        mode = df_clean[target_col].mode()
                        if len(mode) > 0:
                            df_clean[target_col].fillna(mode[0], inplace=True)
                    elif strategy == "Forward fill":
                        df_clean[target_col].fillna(method='ffill', inplace=True)
                    elif strategy == "Backward fill":
                        df_clean[target_col].fillna(method='bfill', inplace=True)
                    elif strategy == "KNN Imputer":
                        if df_clean[target_col].dtype in [np.float64, np.int64]:
                            imputer = KNNImputer(n_neighbors=5)
                            df_clean[target_col] = imputer.fit_transform(df_clean[[target_col]])
                        else:
                            st.warning("KNN Imputer only works with numeric columns")
                            st.stop()
                    elif strategy == "Fill value":
                        df_clean[target_col].fillna(fill_val, inplace=True)
                    
                    push_history(df_clean, f"🔧 {strategy} - {target_col}")
                    st.session_state.df = df_clean
                    st.success(f"✅ Applied {strategy}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.success("✅ No missing values detected!")
    
    with st.expander("🔄 Duplicates Management", expanded=True):
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠️ Found {dup_count:,} duplicates ({dup_count/len(df)*100:.2f}%)")
            
            # Show sample duplicates
            dup_df = df[df.duplicated(keep=False)].head(10)
            st.dataframe(dup_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Remove All Duplicates", key="remove_all_dups", use_container_width=True):
                    df_clean = df.drop_duplicates()
                    push_history(df_clean, f"🗑️ Removed {dup_count} duplicates")
                    st.session_state.df = df_clean
                    st.success(f"✅ Removed {dup_count} duplicates!")
                    st.rerun()
            
            with col2:
                subset_cols = st.multiselect("Or remove based on columns", df.columns.tolist(), key="dup_subset")
                if subset_cols and st.button("🗑️ Remove Selected", key="remove_subset_dups", use_container_width=True):
                    df_clean = df.drop_duplicates(subset=subset_cols)
                    removed = len(df) - len(df_clean)
                    push_history(df_clean, f"🗑️ Removed {removed} duplicates (subset)")
                    st.session_state.df = df_clean
                    st.success(f"✅ Removed {removed} duplicates!")
                    st.rerun()
        else:
            st.success("✅ No duplicates found!")
    
    with st.expander("🎯 Advanced Outlier Detection", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col1, col2, col3 = st.columns(3)
            with col1:
                outlier_col = st.selectbox("Column", numeric_cols, key="outlier_col")
            with col2:
                method = st.selectbox("Method", 
                    ["IQR (1.5x)", "IQR (3x)", "Z-Score (2σ)", "Z-Score (3σ)", 
                     "Isolation Forest", "Modified Z-Score"],
                    key="outlier_method")
            with col3:
                visualize = st.checkbox("Show visualization", value=True, key="outlier_viz")
            
            if outlier_col and visualize:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Box Plot", "Distribution"))
                
                fig.add_trace(go.Box(y=df[outlier_col], name=outlier_col, 
                                    marker_color='#667eea'), row=1, col=1)
                fig.add_trace(go.Histogram(x=df[outlier_col], name=outlier_col,
                                          marker_color='#764ba2', nbinsx=50), row=1, col=2)
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='white'), height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min", f"{df[outlier_col].min():.2f}")
                with col2:
                    st.metric("Max", f"{df[outlier_col].max():.2f}")
                with col3:
                    st.metric("Mean", f"{df[outlier_col].mean():.2f}")
                with col4:
                    st.metric("Std", f"{df[outlier_col].std():.2f}")
            
            if st.button("🗑️ Remove Outliers", key="remove_outliers", use_container_width=True):
                df_clean = df.copy()
                try:
                    if "IQR (1.5x)" in method:
                        Q1 = df_clean[outlier_col].quantile(0.25)
                        Q3 = df_clean[outlier_col].quantile(0.75)
                        IQR = Q3 - Q1
                        df_clean = df_clean[
                            (df_clean[outlier_col] >= Q1 - 1.5 * IQR) &
                            (df_clean[outlier_col] <= Q3 + 1.5 * IQR)
                        ]
                    elif "IQR (3x)" in method:
                        Q1 = df_clean[outlier_col].quantile(0.25)
                        Q3 = df_clean[outlier_col].quantile(0.75)
                        IQR = Q3 - Q1
                        df_clean = df_clean[
                            (df_clean[outlier_col] >= Q1 - 3 * IQR) &
                            (df_clean[outlier_col] <= Q3 + 3 * IQR)
                        ]
                    elif "Z-Score (2σ)" in method:
                        z = np.abs((df_clean[outlier_col] - df_clean[outlier_col].mean()) / df_clean[outlier_col].std())
                        df_clean = df_clean[z < 2]
                    elif "Z-Score (3σ)" in method:
                        z = np.abs((df_clean[outlier_col] - df_clean[outlier_col].mean()) / df_clean[outlier_col].std())
                        df_clean = df_clean[z < 3]
                    elif "Isolation Forest" in method:
                        iso = IsolationForest(contamination=0.1, random_state=42)
                        outliers = iso.fit_predict(df_clean[[outlier_col]])
                        df_clean = df_clean[outliers == 1]
                    elif "Modified Z-Score" in method:
                        median = df_clean[outlier_col].median()
                        mad = np.median(np.abs(df_clean[outlier_col] - median))
                        modified_z = 0.6745 * (df_clean[outlier_col] - median) / mad
                        df_clean = df_clean[np.abs(modified_z) < 3.5]
                    
                    removed = len(df) - len(df_clean)
                    push_history(df_clean, f"🎯 Removed {removed} outliers ({method})")
                    st.session_state.df = df_clean
                    st.success(f"✅ Removed {removed} outliers using {method}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with st.expander("🏷️ Smart Encoding", expanded=False):
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            col1, col2 = st.columns(2)
            with col1:
                encode_col = st.selectbox("Column", cat_cols, key="encode_col")
            with col2:
                encode_type = st.selectbox("Type", 
                    ["Label Encoding", "One-Hot Encoding", "Frequency Encoding", "Target Encoding"],
                    key="encode_type")
            
            if encode_col:
                unique_count = df[encode_col].nunique()
                st.write(f"**Unique values: {unique_count}**")
                
                if unique_count <= 20:
                    value_counts = df[encode_col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Distribution of {encode_col}",
                               labels={'x': encode_col, 'y': 'Count'})
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                    font=dict(color='white'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df[encode_col].value_counts().head(20), use_container_width=True)
            
            if st.button("🏷️ Encode", key="encode_btn", use_container_width=True):
                df_encode = df.copy()
                try:
                    if encode_type == "Label Encoding":
                        le = LabelEncoder()
                        df_encode[encode_col] = le.fit_transform(df_encode[encode_col].astype(str))
                    elif encode_type == "One-Hot Encoding":
                        df_encode = pd.get_dummies(df_encode, columns=[encode_col], prefix=encode_col)
                    elif encode_type == "Frequency Encoding":
                        freq = df_encode[encode_col].value_counts(normalize=True)
                        df_encode[f'{encode_col}_freq'] = df_encode[encode_col].map(freq)
                    elif encode_type == "Target Encoding":
                        st.info("Target encoding requires a target variable. Select one:")
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        target = st.selectbox("Target", numeric_cols, key="target_encode")
                        if target:
                            target_mean = df_encode.groupby(encode_col)[target].mean()
                            df_encode[f'{encode_col}_target'] = df_encode[encode_col].map(target_mean)
                    
                    push_history(df_encode, f"🏷️ {encode_type} - {encode_col}")
                    st.session_state.df = df_encode
                    st.success(f"✅ Encoded {encode_col}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("No categorical columns found")
    
    with st.expander("⚖️ Data Scaling & Transformation", expanded=False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                scale_cols = st.multiselect("Columns to scale", numeric_cols, 
                                           default=numeric_cols[:3], key="scale_cols")
            with col2:
                scaler_type = st.selectbox("Scaler", 
                    ["StandardScaler", "MinMaxScaler", "RobustScaler", 
                     "QuantileTransformer", "PowerTransformer"],
                    key="scaler_type")
            
            if scale_cols and st.button("⚖️ Apply Scaling", key="scale_btn", use_container_width=True):
                df_scaled = df.copy()
                try:
                    if scaler_type == "StandardScaler":
                        scaler = StandardScaler()
                    elif scaler_type == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    elif scaler_type == "RobustScaler":
                        scaler = RobustScaler()
                    elif scaler_type == "QuantileTransformer":
                        scaler = QuantileTransformer()
                    else:
                        scaler = PowerTransformer()
                    
                    df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
                    
                    push_history(df_scaled, f"⚖️ {scaler_type} applied")
                    st.session_state.df = df_scaled
                    st.success(f"✅ Applied {scaler_type}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Continue in next part due to length...
# TAB 3: Advanced Visualization
with tabs[2]:
    st.markdown("## 📈 Advanced Visualizations")
    
    viz_type = st.selectbox("Visualization Type", 
        ["Distribution Analysis", "Correlation Matrix", "Scatter Plot", "Box Plot", 
         "Violin Plot", "Pair Plot", "3D Scatter", "Heatmap", "Time Series"],
        key="viz_type")
    
    if viz_type == "Distribution Analysis":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Column", numeric_cols, key="dist_col")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Histogram", "Box Plot", "Q-Q Plot", "KDE"),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Histogram
            fig.add_trace(go.Histogram(x=df[col], marker_color='#667eea', name='Histogram',
                                      nbinsx=50), row=1, col=1)
            
            # Box Plot
            fig.add_trace(go.Box(y=df[col], marker_color='#764ba2', name='Box Plot'), 
                         row=1, col=2)
            
            # Q-Q Plot
            from scipy import stats as sp_stats
            qq = sp_stats.probplot(df[col].dropna(), dist="norm")
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                    marker=dict(color='#f093fb'), name='Q-Q Plot'), 
                         row=2, col=1)
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0],
                                    mode='lines', line=dict(color='white', dash='dash'),
                                    name='Reference'), row=2, col=1)
            
            # KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(df[col].dropna())
            x_range = np.linspace(df[col].min(), df[col].max(), 100)
            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines',
                                    fill='tozeroy', line=dict(color='#4facfe'),
                                    name='KDE'), row=2, col=2)
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white'),
                height=700,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[col].mean():.2f}")
                st.metric("Median", f"{df[col].median():.2f}")
            with col2:
                st.metric("Std Dev", f"{df[col].std():.2f}")
                st.metric("Variance", f"{df[col].var():.2f}")
            with col3:
                st.metric("Skewness", f"{df[col].skew():.2f}")
                st.metric("Kurtosis", f"{df[col].kurtosis():.2f}")
            with col4:
                # Normality tests
                try:
                    _, p_shapiro = shapiro(df[col].dropna()[:5000])  # Shapiro-Wilk
                    st.metric("Shapiro p-value", f"{p_shapiro:.4f}")
                    if p_shapiro > 0.05:
                        st.success("✅ Likely Normal")
                    else:
                        st.warning("⚠️ Not Normal")
                except:
                    st.info("Sample too large")
    
    elif viz_type == "Correlation Matrix":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_method = st.selectbox("Method", ["Pearson", "Spearman", "Kendall"], key="corr_method")
            
            if corr_method == "Pearson":
                corr = df[numeric_cols].corr(method='pearson')
            elif corr_method == "Spearman":
                corr = df[numeric_cols].corr(method='spearman')
            else:
                corr = df[numeric_cols].corr(method='kendall')
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            corr_masked = corr.mask(mask)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_masked.values,
                x=corr_masked.columns,
                y=corr_masked.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_masked.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title=f"{corr_method} Correlation Matrix",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Find strongest correlations
            st.markdown("### 🔗 Strongest Correlations")
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append({
                        'Feature 1': corr.columns[i],
                        'Feature 2': corr.columns[j],
                        'Correlation': corr.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', 
                                                          key=abs, ascending=False).head(10)
            st.dataframe(corr_df, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="scatter_y")
            with col3:
                color_col = st.selectbox("Color by", ["None"] + df.columns.tolist(), key="scatter_color")
            
            if color_col == "None":
                fig = px.scatter(df, x=x_col, y=y_col, 
                               title=f"{x_col} vs {y_col}",
                               trendline="ols",
                               opacity=0.7,
                               marginal_x="histogram",
                               marginal_y="histogram")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f"{x_col} vs {y_col}",
                               trendline="ols",
                               opacity=0.7,
                               marginal_x="violin",
                               marginal_y="violin")
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            st.metric("Correlation", f"{corr:.4f}")
    
    elif viz_type == "3D Scatter":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
            with col2:
                y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="3d_y")
            with col3:
                z_col = st.selectbox("Z-axis", [c for c in numeric_cols if c not in [x_col, y_col]], key="3d_z")
            
            color_col = st.selectbox("Color by", ["None"] + df.columns.tolist(), key="3d_color")
            
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                              color=color_col if color_col != "None" else None,
                              title=f"3D: {x_col} vs {y_col} vs {z_col}",
                              opacity=0.7)
            
            fig.update_layout(
                scene=dict(
                    bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 4: Advanced ML Models
with tabs[3]:
    st.markdown("## 🤖 Advanced Machine Learning")
    
    all_cols = df.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Need at least 2 columns")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("🎯 Target Variable", all_cols, key="ml_target")
    with col2:
        available_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect("📊 Features", available_features, 
                                     default=available_features[:min(15, len(available_features))],
                                     key="ml_features")
    
    if not feature_cols:
        st.warning("Select at least one feature")
        st.stop()
    
    # Advanced options
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_advanced_imputation = st.checkbox("Advanced Imputation (KNN)", value=False, key="adv_impute")
            use_feature_selection = st.checkbox("Auto Feature Selection", value=False, key="feat_select")
        with col2:
            use_hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False, key="hyperparam")
            use_cross_validation = st.checkbox("Cross Validation (5-fold)", value=True, key="cv")
        with col3:
            use_ensemble = st.checkbox("Ensemble Models", value=False, key="ensemble")
            balance_classes = st.checkbox("Balance Classes", value=False, key="balance")
    
    try:
        X, y, feature_encoders, target_encoder, problem_type = prepare_ml_data(
            df, target_col, feature_cols, use_advanced=use_advanced_imputation
        )
        
        st.markdown(f"""
        <div class="insight-box">
            <strong>🤖 Problem Type: {problem_type.upper()}</strong><br>
            Target: {target_col} | Features: {len(feature_cols)} | Samples: {len(X):,} | 
            Classes: {len(np.unique(y)) if problem_type == 'classification' else 'N/A'}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Data prep error: {str(e)}")
        st.stop()
    
    # Enhanced model selection
    if problem_type == 'classification':
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
            "Neural Network": MLPClassifier(hidden_layers_sizes=(100, 50), max_iter=500, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "Naive Bayes": GaussianNB()
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "LightGBM": lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "Linear Regression": LinearRegression(n_jobs=-1),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=1.0),
            "ElasticNet": ElasticNet(alpha=1.0),
            "Neural Network": MLPRegressor(hidden_layers=(100, 50), max_iter=500, random_state=42),
            "SVR": SVR(kernel='rbf')
        }
    
    selected_models = st.multiselect("📊 Select Models", list(models.keys()), 
                                    default=list(models.keys())[:5], key="selected_models")
    
    test_size = st.slider("Test Size %", 10, 40, 25, key="test_size") / 100
    
    if st.button("🚀 Train Models", key="train_models", use_container_width=True, type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=42, stratify=y if problem_type == 'classification' else None)
        
        # Feature selection
        if use_feature_selection:
            st.info("🔍 Performing feature selection...")
            if problem_type == 'classification':
                selector = SelectKBest(f_classif, k=min(10, len(feature_cols)))
            else:
                selector = SelectKBest(f_regression, k=min(10, len(feature_cols)))
            
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            st.success(f"✅ Selected {len(selected_features)} features: {', '.join(selected_features)}")
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for idx, model_name in enumerate(selected_models):
            status.markdown(f"<div class='loading-spinner'></div> Training {model_name}...", unsafe_allow_html=True)
            
            try:
                model = models[model_name]
                
                # Hyperparameter tuning
                if use_hyperparameter_tuning and model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
                    status.text(f"🔧 Tuning {model_name} hyperparameters...")
                    model, best_params = hyperparameter_tuning(model, X_train, y_train, problem_type)
                    st.info(f"Best params for {model_name}: {best_params}")
                
                # Train
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Cross validation
                if use_cross_validation:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                               scoring='accuracy' if problem_type == 'classification' else 'r2')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean, cv_std = None, None
                
                # Metrics
                if problem_type == 'classification':
                    acc = accuracy_score(y_test, y_pred)
                    avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
                    prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                    rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                    
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test)
                            if len(np.unique(y)) == 2:
                                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                            else:
                                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                        else:
                            auc = None
                    except:
                        auc = None
                    
                    results.append({
                        'Model': model_name,
                        'Accuracy': f"{acc:.4f}",
                        'Precision': f"{prec:.4f}",
                        'Recall': f"{rec:.4f}",
                        'F1-Score': f"{f1:.4f}",
                        'AUC': f"{auc:.4f}" if auc else "N/A",
                        'CV Score': f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_mean else "N/A",
                        'Score': acc
                    })
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    try:
                        mape = mean_absolute_percentage_error(y_test, y_pred)
                    except:
                        mape = None
                    
                    results.append({
                        'Model': model_name,
                        'RMSE': f"{rmse:.4f}",
                        'MAE': f"{mae:.4f}",
                        'MAPE': f"{mape:.4f}" if mape else "N/A",
                        'R²': f"{r2:.4f}",
                        'CV Score': f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_mean else "N/A",
                        'Score': r2
                    })
                
                # Store model
                st.session_state.trained_models[model_name] = {
                    'model': model,
                    'features': feature_cols,
                    'target': target_col,
                    'type': problem_type,
                    'feature_encoders': feature_encoders,
                    'target_encoder': target_encoder,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'cv_scores': cv_scores if use_cross_validation else None
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    st.session_state.feature_importance = {
                        'features': feature_cols if not use_feature_selection else selected_features,
                        'importance': model.feature_importances_
                    }
            
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
            
            progress.progress((idx + 1) / len(selected_models))
        
        status.empty()
        progress.empty()
        
        if results:
            results_df = pd.DataFrame(results)
            best_idx = results_df['Score'].astype(float).idxmax()
            best_model_name = results_df.loc[best_idx, 'Model']
            st.session_state.best_model = best_model_name
            
            st.markdown("### 🏆 Model Performance")
            
            # Visualize results
            score_col = 'Accuracy' if problem_type == 'classification' else 'R²'
            fig = px.bar(results_df, x='Model', y=results_df[score_col].astype(float),
                        title=f"Model Comparison - {score_col}",
                        color=results_df[score_col].astype(float),
                        color_continuous_scale='viridis',
                        text=results_df[score_col])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white'), height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(results_df.drop('Score', axis=1), use_container_width=True)
            
            st.success(f"🏆 Best Model: **{best_model_name}** (Score: {results_df.loc[best_idx, 'Score']:.4f})")
            
            # Detailed metrics for best model
            best_model_info = st.session_state.trained_models[best_model_name]
            
            if problem_type == 'classification':
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Confusion Matrix")
                    cm = confusion_matrix(best_model_info['y_test'], best_model_info['y_pred'])
                    
                    # Get class names
                    if target_encoder:
                        class_names = target_encoder.classes_
                    else:
                        class_names = [str(i) for i in range(len(cm))]
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=class_names,
                        y=class_names,
                        colorscale='Blues',
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 16}
                    ))
                    fig.update_layout(
                        title=f"{best_model_name} - Confusion Matrix",
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Classification Report")
                    if target_encoder:
                        target_names = target_encoder.classes_
                    else:
                        target_names = [str(i) for i in np.unique(y)]
                    
                    report = classification_report(best_model_info['y_test'], 
                                                  best_model_info['y_pred'],
                                                  target_names=target_names,
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).T
                    st.dataframe(report_df.round(3), use_container_width=True, height=400)
            
            else:
                st.markdown("### 📈 Prediction vs Actual")
                
                pred_df = pd.DataFrame({
                    'Actual': best_model_info['y_test'],
                    'Predicted': best_model_info['y_pred']
                })
                
                fig = px.scatter(pred_df, x='Actual', y='Predicted',
                               title=f"{best_model_name} - Predictions",
                               trendline="ols",
                               opacity=0.6)
                
                # Add perfect prediction line
                min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect Prediction',
                                        line=dict(color='red', dash='dash')))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='white'), height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals
                residuals = best_model_info['y_test'] - best_model_info['y_pred']
                fig2 = px.histogram(residuals, nbins=50, title="Residuals Distribution")
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                 font=dict(color='white'))
                st.plotly_chart(fig2, use_container_width=True)
            
            # Feature importance
            if st.session_state.feature_importance:
                st.markdown("### 🎯 Feature Importance")
                imp_df = pd.DataFrame({
                    'Feature': st.session_state.feature_importance['features'],
                    'Importance': st.session_state.feature_importance['importance']
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                           title="Top 15 Most Important Features",
                           color='Importance',
                           color_continuous_scale='viridis')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='white'), height=500)
                st.plotly_chart(fig, use_container_width=True)

# TAB 5: Advanced Predictions
with tabs[4]:
    st.markdown("## 🎯 Advanced Predictions")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ Train models first in the ML tab")
        st.stop()
    
    model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()), 
                             key="pred_model")
    
    model_info = st.session_state.trained_models[model_name]
    model = model_info['model']
    feature_cols = model_info['features']
    problem_type = model_info['type']
    feature_encoders = model_info['feature_encoders']
    target_encoder = model_info['target_encoder']
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>🤖 Model: {model_name}</strong><br>
        Type: {problem_type.title()} | Features: {len(feature_cols)} | 
        Test Score: {model_info.get('Score', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    pred_mode = st.radio("Prediction Mode", ["Single Prediction", "Batch Prediction", "Interactive Prediction"], 
                        key="pred_mode", horizontal=True)
    
    if pred_mode == "Single Prediction":
        st.markdown("### 📝 Enter Feature Values")
        
        input_data = {}
        cols = st.columns(4)
        
        for idx, col in enumerate(feature_cols):
            with cols[idx % 4]:
                if col not in df.columns:
                    continue
                
                if df[col].dtype in [np.float64, np.int64]:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    step = (max_val - min_val) / 100
                    input_data[col] = st.number_input(col, min_val, max_val, mean_val,
                                                     step=step, key=f"input_{col}")
                else:
                    unique_vals = df[col].unique().tolist()
                    input_data[col] = st.selectbox(col, unique_vals, key=f"input_{col}")
        
        if st.button("🔮 Predict", key="predict_single", use_container_width=True, type="primary"):
            try:
                input_df = pd.DataFrame([input_data])
                
                # Encode features
                for col in input_df.columns:
                    if col in feature_encoders:
                        le = feature_encoders[col]
                        try:
                            input_df[col] = le.transform(input_df[col].astype(str))
                        except:
                            input_df[col] = 0
                
                input_df = input_df[feature_cols]
                prediction = model.predict(input_df)[0]
                
                # Decode if classification
                if problem_type == 'classification' and target_encoder:
                    prediction_raw = int(prediction)
                    prediction = target_encoder.inverse_transform([prediction_raw])[0]
                
                if problem_type == 'classification':
                    proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
                    
                    st.markdown(f"""
                    <div class="change-summary" style="text-align: center; padding: 50px;">
                        <div style="position: relative; z-index: 1;">
                            <h2>🎯 Prediction Result</h2>
                            <h1 style="font-size: 72px; margin: 30px 0; font-family: 'Orbitron', sans-serif;">
                                {prediction}
                            </h1>
                            {f'<p style="font-size: 24px;">Confidence: {max(proba)*100:.2f}%</p>' if proba is not None else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if proba is not None and len(proba) > 1:
                        classes = target_encoder.classes_ if target_encoder else [str(i) for i in range(len(proba))]
                        prob_df = pd.DataFrame({'Class': classes, 'Probability': proba * 100})
                        prob_df = prob_df.sort_values('Probability', ascending=False)
                        
                        fig = px.bar(prob_df, x='Class', y='Probability',
                                   title="Class Probabilities",
                                   color='Probability',
                                   color_continuous_scale='viridis',
                                   text='Probability')
                        fig.update_traces(texttemplate='%{text:.2f}%')
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                        font=dict(color='white'), height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="change-summary" style="text-align: center; padding: 50px;">
                        <div style="position: relative; z-index: 1;">
                            <h2>🎯 Prediction Result</h2>
                            <h1 style="font-size: 72px; margin: 30px 0; font-family: 'Orbitron', sans-serif;">
                                {prediction:.4f}
                            </h1>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    elif pred_mode == "Batch Prediction":
        st.markdown("### 📊 Batch Prediction from File")
        upload_pred = st.file_uploader("Upload CSV file", type=['csv'], key="batch_file")
        
        if upload_pred:
            pred_df = pd.read_csv(upload_pred)
            st.markdown(f"**Loaded: {len(pred_df):,} rows**")
            st.dataframe(pred_df.head(15), use_container_width=True)
            
            if st.button("🔮 Predict All", key="predict_batch", use_container_width=True, type="primary"):
                missing = set(feature_cols) - set(pred_df.columns)
                if missing:
                    st.error(f"❌ Missing columns: {missing}")
                else:
                    try:
                        with st.spinner("Making predictions..."):
                            X_pred = pred_df[feature_cols].copy()
                            
                            for col in X_pred.columns:
                                if col in feature_encoders:
                                    le = feature_encoders[col]
                                    try:
                                        X_pred[col] = le.transform(X_pred[col].astype(str))
                                    except:
                                        X_pred[col] = 0
                            
                            predictions = model.predict(X_pred)
                            
                            if problem_type == 'classification' and target_encoder:
                                predictions = target_encoder.inverse_transform(predictions.astype(int))
                            
                            pred_df['Prediction'] = predictions
                            
                            if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                                probas = model.predict_proba(X_pred)
                                pred_df['Confidence'] = (probas.max(axis=1) * 100).round(2)
                                
                                # Add probabilities for each class
                                if target_encoder:
                                    for idx, class_name in enumerate(target_encoder.classes_):
                                        pred_df[f'Prob_{class_name}'] = (probas[:, idx] * 100).round(2)
                        
                        st.success(f"✅ Successfully predicted {len(pred_df):,} samples!")
                        
                        # Summary stats
                        if problem_type == 'classification':
                            st.markdown("### 📊 Prediction Summary")
                            value_counts = pred_df['Prediction'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = px.pie(values=value_counts.values, names=value_counts.index,
                                           title="Prediction Distribution", hole=0.4)
                                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                                font=dict(color='white'))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                avg_conf = pred_df['Confidence'].mean()
                                st.metric("Average Confidence", f"{avg_conf:.2f}%")
                                st.dataframe(value_counts.reset_index(), use_container_width=True)
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        download_button(pred_df, "csv", "📥 Download Predictions", "download_pred")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# TAB 6: Feature Engineering
with tabs[5]:
    st.markdown("## 🧬 Advanced Feature Engineering")
    
    with st.expander("🤖 Auto Feature Engineering", expanded=True):
        target_for_fe = st.selectbox("Target (optional)", ["None"] + df.columns.tolist(), key="fe_target")
        
        if st.button("🧬 Generate Features", key="auto_fe", use_container_width=True):
            target = target_for_fe if target_for_fe != "None" else None
            
            with st.spinner("Generating features using AI..."):
                df_engineered, new_features = advanced_feature_engineering(df, target)
            
            st.success(f"✅ Created {len(new_features)} new features!")
            st.write("New features:", ', '.join(new_features))
            
            if st.button("➕ Add to Dataset", key="add_engineered", use_container_width=True):
                push_history(df_engineered, f"🧬 Added {len(new_features)} engineered features")
                st.session_state.df = df_engineered
                st.success("✅ Features added!")
                st.rerun()
    
    with st.expander("🎯 Clustering Analysis", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            cols = st.multiselect("Features for clustering", numeric_cols, 
                                 default=numeric_cols[:min(5, len(numeric_cols))], key="cluster_cols")
            
            if cols:
                col1, col2, col3 = st.columns(3)
                with col1:
                    method = st.selectbox("Method", ["K-Means", "DBSCAN", "Agglomerative", "Spectral"], 
                                        key="cluster_method")
                with col2:
                    if method == "K-Means" or method == "Agglomerative" or method == "Spectral":
                        n_clusters = st.slider("Clusters", 2, 15, 4, key="n_clusters")
                with col3:
                    use_pca = st.checkbox("Apply PCA first", value=True, key="cluster_pca")
                
                if st.button("🎯 Cluster", key="cluster_btn", use_container_width=True):
                    try:
                        X = df[cols].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Optional PCA
                        if use_pca and len(cols) > 2:
                            pca = PCA(n_components=min(3, len(cols)))
                            X_scaled = pca.fit_transform(X_scaled)
                            st.info(f"PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
                        
                        # Clustering
                        if method == "K-Means":
                            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        elif method == "DBSCAN":
                            clusterer = DBSCAN(eps=0.5, min_samples=5)
                        elif method == "Agglomerative":
                            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                        else:
                            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
                        
                        clusters = clusterer.fit_predict(X_scaled)
                        
                        # Silhouette score
                        try:
                            sil_score = silhouette_score(X_scaled, clusters)
                            st.metric("Silhouette Score", f"{sil_score:.4f}")
                        except:
                            pass
                        
                        # Visualize
                        if X_scaled.shape[1] >= 2:
                            viz_df = pd.DataFrame({
                                'PC1': X_scaled[:, 0],
                                'PC2': X_scaled[:, 1],
                                'Cluster': clusters
                            })
                            
                            fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                                           title=f"{method} Clustering Results",
                                           color_continuous_scale='viridis')
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                            font=dict(color='white'), height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Cluster statistics
                        cluster_stats = pd.DataFrame({
                            'Cluster': np.unique(clusters),
                            'Size': [np.sum(clusters == c) for c in np.unique(clusters)],
                            'Percentage': [np.sum(clusters == c) / len(clusters) * 100 for c in np.unique(clusters)]
                        })
                        st.dataframe(cluster_stats, use_container_width=True)
                        
                        if st.button("➕ Add Clusters to Dataset", key="add_clusters", use_container_width=True):
                            df_clustered = df.copy()
                            df_clustered['Cluster'] = -1
                            df_clustered.loc[X.index, 'Cluster'] = clusters
                            push_history(df_clustered, f"🎯 {method} clustering")
                            st.session_state.df = df_clustered
                            st.success("✅ Clusters added!")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with st.expander("📉 Dimensionality Reduction", expanded=False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox("Method", ["PCA", "t-SNE", "ICA", "NMF"], key="dim_method")
            with col2:
                n_components = st.slider("Components", 2, min(15, len(numeric_cols)), 3, key="dim_components")
            
            if st.button("📉 Apply", key="dim_reduce", use_container_width=True):
                try:
                    X = df[numeric_cols].fillna(df[numeric_cols].median())
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    if method == "PCA":
                        reducer = PCA(n_components=n_components, random_state=42)
                    elif method == "t-SNE":
                        reducer = TSNE(n_components=min(n_components, 3), random_state=42, perplexity=30)
                    elif method == "ICA":
                        reducer = FastICA(n_components=n_components, random_state=42)
                    else:
                        reducer = NMF(n_components=n_components, random_state=42)
                    
                    X_reduced = reducer.fit_transform(X_scaled if method != "NMF" else X)
                    
                    # Visualize
                    if n_components >= 2:
                        viz_df = pd.DataFrame(X_reduced[:, :min(3, n_components)])
                        
                        if n_components >= 3:
                            fig = px.scatter_3d(viz_df, x=0, y=1, z=2,
                                              title=f"{method} - 3D Visualization",
                                              opacity=0.7)
                            fig.update_layout(
                                scene=dict(bgcolor='rgba(0,0,0,0)'),
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                height=600
                            )
                        else:
                            fig = px.scatter(viz_df, x=0, y=1,
                                           title=f"{method} - 2D Visualization",
                                           opacity=0.7)
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                            font=dict(color='white'), height=500)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Variance explained (for PCA)
                    if method == "PCA":
                        var_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(n_components)],
                            'Variance Explained (%)': reducer.explained_variance_ratio_ * 100,
                            'Cumulative Variance (%)': np.cumsum(reducer.explained_variance_ratio_) * 100
                        })
                        
                        fig2 = px.bar(var_df, x='Component', y='Variance Explained (%)',
                                    title="Variance Explained by Component",
                                    text='Variance Explained (%)')
                        fig2.update_traces(texttemplate='%{text:.2f}%')
                        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                         font=dict(color='white'))
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        st.dataframe(var_df, use_container_width=True)
                        st.metric("Total Variance Explained", 
                                f"{reducer.explained_variance_ratio_.sum()*100:.2f}%")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# TAB 7: Advanced Operations
with tabs[6]:
    st.markdown("## ⚙️ Advanced Operations")
    
    with st.expander("🎲 Smart Sampling", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            sample_type = st.selectbox("Sampling Type", 
                ["Random", "Stratified", "Systematic", "Bootstrap"], key="sample_type")
        with col2:
            if sample_type == "Random":
                sample_size = st.slider("Percentage", 1, 100, 30, key="sample_pct")
            elif sample_type == "Stratified":
                strat_col = st.selectbox("Stratify by", df.columns.tolist(), key="strat_col")
                sample_size = st.slider("Percentage", 1, 100, 30, key="strat_pct")
            elif sample_type == "Bootstrap":
                n_samples = st.number_input("Number of samples", 1, len(df), 
                                          min(1000, len(df)), key="boot_n")
            else:
                step = st.number_input("Step size", 1, 100, 5, key="sys_step")
        
        if st.button("🎲 Apply Sampling", key="sample_btn", use_container_width=True):
            try:
                if sample_type == "Random":
                    df_sample = df.sample(frac=sample_size/100, random_state=42)
                elif sample_type == "Stratified":
                    df_sample = df.groupby(strat_col, group_keys=False).apply(
                        lambda x: x.sample(frac=sample_size/100, random_state=42)
                    )
                elif sample_type == "Bootstrap":
                    df_sample = df.sample(n=n_samples, replace=True, random_state=42)
                else:
                    df_sample = df.iloc[::step]
                
                push_history(df_sample, f"🎲 {sample_type} sampling")
                st.session_state.df = df_sample
                st.success(f"✅ Sampled {len(df_sample):,} rows!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with st.expander("🔍 Advanced Filtering", expanded=True):
        filter_col = st.selectbox("Column", df.columns.tolist(), key="filter_col")
        
        if df[filter_col].dtype in [np.float64, np.int64]:
            col1, col2 = st.columns(2)
            with col1:
                filter_type = st.selectbox("Filter type", 
                    ["Range", "Greater than", "Less than", "Between percentiles"],
                    key="filter_type")
            with col2:
                if filter_type == "Range":
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    filter_range = st.slider("Range", min_val, max_val, (min_val, max_val), key="filter_range")
                elif filter_type == "Between percentiles":
                    pct_range = st.slider("Percentile range", 0, 100, (25, 75), key="pct_range")
                else:
                    threshold = st.number_input("Threshold", 
                                              float(df[filter_col].min()),
                                              float(df[filter_col].max()),
                                              float(df[filter_col].median()),
                                              key="filter_threshold")
            
            if st.button("🔍 Filter", key="filter_num_btn", use_container_width=True):
                if filter_type == "Range":
                    df_filtered = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
                elif filter_type == "Greater than":
                    df_filtered = df[df[filter_col] > threshold]
                elif filter_type == "Less than":
                    df_filtered = df[df[filter_col] < threshold]
                else:
                    lower = df[filter_col].quantile(pct_range[0]/100)
                    upper = df[filter_col].quantile(pct_range[1]/100)
                    df_filtered = df[(df[filter_col] >= lower) & (df[filter_col] <= upper)]
                
                push_history(df_filtered, f"🔍 Filtered {filter_col}")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered):,} rows!")
                st.rerun()
        else:
            unique_vals = df[filter_col].unique().tolist()
            selected = st.multiselect("Select values", unique_vals, 
                                     default=unique_vals[:min(5, len(unique_vals))], key="filter_vals")
            
            if selected and st.button("🔍 Filter", key="filter_cat_btn", use_container_width=True):
                df_filtered = df[df[filter_col].isin(selected)]
                push_history(df_filtered, f"🔍 Filtered {filter_col}")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered):,} rows!")
                st.rerun()
    
    with st.expander("🔗 Merge Datasets", expanded=False):
        if st.session_state.df2 is not None:
            df2 = st.session_state.df2
            st.info(f"Second dataset: {df2.shape[0]:,} × {df2.shape[1]:,}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                join_type = st.selectbox("Join type", ["inner", "left", "right", "outer", "cross"], 
                                        key="join_type")
            with col2:
                left_key = st.selectbox("Left key", df.columns.tolist(), key="left_key")
            with col3:
                right_key = st.selectbox("Right key", df2.columns.tolist(), key="right_key")
            
            if st.button("🔗 Merge", key="merge_btn", use_container_width=True):
                try:
                    if join_type == "cross":
                        merged = df.merge(df2, how='cross')
                    else:
                        merged = pd.merge(df, df2, left_on=left_key, right_on=right_key, 
                                        how=join_type, suffixes=("", "_2"))
                    
                    push_history(merged, f"🔗 {join_type.title()} merge on {left_key}")
                    st.session_state.df = merged
                    st.success(f"✅ Merged! {merged.shape[0]:,} × {merged.shape[1]:,}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("Upload second dataset in sidebar")

# TAB 8: AutoML
with tabs[7]:
    st.markdown("## 🏆 AutoML - Automated Machine Learning")
    
    st.markdown("""
    <div class="insight-box">
        <strong>🤖 AutoML automatically:</strong><br>
        • Selects the best features<br>
        • Tries multiple algorithms<br>
        • Optimizes hyperparameters<br>
        • Creates ensemble models<br>
        • Provides detailed reports
    </div>
    """, unsafe_allow_html=True)
    
    all_cols = df.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Need at least 2 columns")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        automl_target = st.selectbox("🎯 Target Variable", all_cols, key="automl_target")
    with col2:
        automl_max_time = st.slider("Max time (minutes)", 1, 30, 5, key="automl_time")
    
    if st.button("🚀 Run AutoML", key="run_automl", use_container_width=True, type="primary"):
        available_features = [c for c in all_cols if c != automl_target]
        
        with st.spinner("🤖 AutoML is running..."):
            try:
                # Prepare data
                X, y, feature_encoders, target_encoder, problem_type = prepare_ml_data(
                    df, automl_target, available_features, use_advanced=True
                )
                
                st.info(f"Problem type: {problem_type.upper()}")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y if problem_type == 'classification' else None
                )
                
                # Feature selection
                st.info("🔍 Selecting best features...")
                if problem_type == 'classification':
                    selector = SelectKBest(f_classif, k=min(20, len(available_features)))
                else:
                    selector = SelectKBest(f_regression, k=min(20, len(available_features)))
                
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                selected_features = [available_features[i] for i in selector.get_support(indices=True)]
                
                st.success(f"✅ Selected {len(selected_features)} features")
                
                # Train multiple models
                st.info("🤖 Training multiple models...")
                
                if problem_type == 'classification':
                    models = {
                        "XGBoost": xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                        "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
                        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                    }
                else:
                    models = {
                        "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                        "LightGBM": lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
                        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                    }
                
                results = []
                trained_models_automl = []
                
                progress = st.progress(0)
                for idx, (name, model) in enumerate(models.items()):
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)
                    trained_models_automl.append((name, model))
                    
                    if problem_type == 'classification':
                        score = accuracy_score(y_test, y_pred)
                        results.append({'Model': name, 'Score': score})
                    else:
                        score = r2_score(y_test, y_pred)
                        results.append({'Model': name, 'Score': score})
                    
                    progress.progress((idx + 1) / len(models))
                
                progress.empty()
                
                # Create ensemble
                st.info("🎯 Creating ensemble model...")
                
                if problem_type == 'classification':
                    ensemble = VotingClassifier(
                        estimators=trained_models_automl,
                        voting='soft'
                    )
                else:
                    ensemble = VotingRegressor(estimators=trained_models_automl)
                
                ensemble.fit(X_train_selected, y_train)
                y_pred_ensemble = ensemble.predict(X_test_selected)
                
                if problem_type == 'classification':
                    ensemble_score = accuracy_score(y_test, y_pred_ensemble)
                else:
                    ensemble_score = r2_score(y_test, y_pred_ensemble)
                
                results.append({'Model': 'Ensemble (Voting)', 'Score': ensemble_score})
                
                # Display results
                results_df = pd.DataFrame(results).sort_values('Score', ascending=False)
                
                st.markdown("### 🏆 AutoML Results")
                
                fig = px.bar(results_df, x='Model', y='Score',
                           title="Model Comparison",
                           color='Score',
                           color_continuous_scale='viridis',
                           text='Score')
                fig.update_traces(texttemplate='%{text:.4f}')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='white'), height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(results_df, use_container_width=True)
                
                best_model = results_df.iloc[0]['Model']
                best_score = results_df.iloc[0]['Score']
                
                st.success(f"🏆 Best Model: **{best_model}** (Score: {best_score:.4f})")
                
                # Feature importance
                best_model_obj = next(m for n, m in trained_models_automl if n == best_model) if best_model != 'Ensemble (Voting)' else ensemble
                
                if hasattr(best_model_obj, 'feature_importances_'):
                    st.markdown("### 🎯 Top Features")
                    imp_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': best_model_obj.feature_importances_ if best_model != 'Ensemble (Voting)' else ensemble.estimators_[0][1].feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                               title="Top 10 Most Important Features",
                               color='Importance',
                               color_continuous_scale='viridis')
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                    font=dict(color='white'), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save best model
                st.session_state.trained_models[f"AutoML_{best_model}"] = {
                    'model': best_model_obj if best_model != 'Ensemble (Voting)' else ensemble,
                    'features': selected_features,
                    'target': automl_target,
                    'type': problem_type,
                    'feature_encoders': feature_encoders,
                    'target_encoder': target_encoder,
                    'X_test': X_test_selected,
                    'y_test': y_test,
                    'y_pred': y_pred_ensemble if best_model == 'Ensemble (Voting)' else y_pred
                }
                
                st.success("✅ AutoML completed! Best model saved.")
                
            except Exception as e:
                st.error(f"AutoML error: {str(e)}")

# TAB 9: Export
with tabs[8]:
    st.markdown("## 💾 Export Data & Models")
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <h3>📊 Current Dataset</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 25px 0;">
            <div>
                <div style="font-size: 14px; opacity: 0.9;">ROWS</div>
                <div style="font-size: 32px; font-weight: 800; margin-top: 8px;">{df.shape[0]:,}</div>
            </div>
            <div>
                <div style="font-size: 14px; opacity: 0.9;">COLUMNS</div>
                <div style="font-size: 32px; font-weight: 800; margin-top: 8px;">{df.shape[1]:,}</div>
            </div>
            <div>
                <div style="font-size: 14px; opacity: 0.9;">MEMORY</div>
                <div style="font-size: 32px; font-weight: 800; margin-top: 8px;">{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📄 Export Formats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        download_button(df, "csv", "📄 CSV", "export_csv")
    with col2:
        download_button(df, "excel", "📊 Excel", "export_excel")
    with col3:
        download_button(df, "json", "📋 JSON", "export_json")
    with col4:
        try:
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            st.download_button("📦 Parquet", parquet_buffer.getvalue(),
                             f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                             "application/octet-stream", key="export_parquet", use_container_width=True)
        except:
            st.info("Parquet export unavailable")
    
    st.markdown("---")
    
    # Export models
    if st.session_state.trained_models:
        st.markdown("### 🤖 Export Trained Models")
        st.info(f"Available models: {len(st.session_state.trained_models)}")
        
        for model_name in st.session_state.trained_models.keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"✓ {model_name}")
            with col2:
                # You can add pickle export here
                st.button("📥 Export", key=f"export_{model_name}", disabled=True, use_container_width=True)
    
    st.markdown("---")
    
    # History
    if st.session_state.history:
        st.markdown("### 📜 Operation History")
        
        for i, h in enumerate(reversed(st.session_state.history[-15:])):
            with st.expander(f"{h['time'].strftime('%H:%M:%S')} - {h['action']}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{h['shape'][0]:,}")
                with col2:
                    st.metric("Columns", f"{h['shape'][1]:,}")
                with col3:
                    memory = h['df'].memory_usage(deep=True).sum() / 1024**2
                    st.metric("Memory", f"{memory:.1f} MB")
                with col4:
                    if st.button("↩️ Restore", key=f"restore_{i}", use_container_width=True):
                        st.session_state.df = h['df'].copy()
                        st.success("✅ Restored!")
                        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 40px;">
    <h3 style="font-family: 'Orbitron', sans-serif; font-size: 28px;">🚀 ML ANALYTICS PRO</h3>
    <p style="opacity: 0.9; font-size: 16px; margin-top: 15px;">
        Advanced AI • Production-Ready • Real-time Insights
    </p>
    <p style="opacity: 0.7; font-size: 13px; margin-top: 10px;">
        Powered by XGBoost, LightGBM, Scikit-learn & Plotly
    </p>
    <div style="margin-top: 20px;">
        <span class="ai-badge">v2.0 - AI EDITION</span>
    </div>
</div>
""", unsafe_allow_html=True)

