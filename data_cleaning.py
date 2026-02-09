import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               AdaBoostClassifier, AdaBoostRegressor, IsolationForest)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              mean_squared_error, r2_score, mean_absolute_error)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    layout="wide",
    page_title="🚀 Advanced AI-ML Analytics Platform",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

# =====================================================
# ENHANCED CSS STYLING WITH ANIMATIONS
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 2rem;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .glass-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }

    .metric-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 25px 20px;
        margin: 12px 0;
        border: 2px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        transform: rotate(45deg);
        transition: all 0.6s;
    }

    .metric-container:hover::before {
        left: 100%;
    }

    .metric-container:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
        border-color: rgba(255, 255, 255, 0.4);
    }

    .metric-icon {
        font-size: 48px;
        margin-bottom: 10px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .metric-label {
        font-size: 13px;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
        width: 100%;
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
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        background: transparent;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        font-size: 14px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.35);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
        transform: scale(1.05);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(15px);
        border-right: 2px solid rgba(255, 255, 255, 0.2);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        margin: 15px 0;
        animation: slideIn 0.5s ease;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .model-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }

    .model-card:hover {
        transform: translateX(10px);
        border-color: rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 18px;
        margin: 15px 0;
        color: white;
        animation: fadeIn 0.8s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    h1, h2, h3 {
        color: white !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def push_history(df, action):
    """Save operation to history"""
    st.session_state.history.append({
        "time": datetime.now(),
        "action": action,
        "df": df.copy(),
        "shape": df.shape
    })

def show_history():
    """Display operation history"""
    if st.session_state.history:
        st.markdown("### 📜 Operation History")
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"🕐 {h['time'].strftime('%H:%M:%S')} - {h['action']}", expanded=False):
                st.write(f"**Rows:** {h['shape'][0]:,} | **Columns:** {h['shape'][1]:,}")
                if st.button("↩️ Restore", key=f"restore_{i}"):
                    st.session_state.df = h['df'].copy()
                    st.success("✅ Dataset restored!")
                    st.rerun()

def download_button(df, format_type, label, key):
    """Create download button for different formats"""
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
        st.download_button(label, json_str, f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                          "application/json", key=key, use_container_width=True)

def detect_problem_type(df, target_col):
    """Automatically detect if problem is classification or regression"""
    unique_values = df[target_col].nunique()
    total_values = len(df[target_col])
    
    if df[target_col].dtype == 'object' or unique_values < 20:
        return 'classification'
    elif unique_values / total_values > 0.05:
        return 'regression'
    else:
        return 'classification'

def get_feature_importance(model, feature_names, top_n=10):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
    return None

def create_animated_metric(icon, label, value, delta=None):
    """Create beautiful animated metric card"""
    delta_html = f"<p style='font-size: 14px; color: {'#00ff88' if delta and delta > 0 else '#ff4444'}; margin-top: 5px;'>{'↑' if delta and delta > 0 else '↓'} {abs(delta) if delta else 0}%</p>" if delta else ""
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="text-align: center; padding: 20px; margin-bottom: 30px;">
    <h1 style="font-size: 48px; font-weight: 800; margin-bottom: 10px; 
                background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: float 3s ease-in-out infinite;">
        🚀 Advanced AI-ML Analytics Platform
    </h1>
    <p style="font-size: 18px; color: rgba(255, 255, 255, 0.9); margin-top: 10px;">
        Transform Data into Intelligence with Next-Gen AI
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'xlsx', 'json'],
        help="Supported formats: CSV, Excel, JSON",
        key="main_upload"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            push_history(df, "📁 Dataset uploaded")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Second dataset for merging
    st.markdown("### 📊 Second Dataset (Optional)")
    uploaded_file2 = st.file_uploader(
        "Upload for merging",
        type=['csv', 'xlsx', 'json'],
        key="second_upload"
    )
    
    if uploaded_file2:
        try:
            if uploaded_file2.name.endswith('.csv'):
                df2 = pd.read_csv(uploaded_file2)
            elif uploaded_file2.name.endswith('.xlsx'):
                df2 = pd.read_excel(uploaded_file2)
            else:
                df2 = pd.read_json(uploaded_file2)
            
            st.session_state.df2 = df2
            st.success(f"✅ Loaded: {df2.shape[0]:,} rows")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Sample datasets
    st.markdown("### 🎲 Try Sample Data")
    if st.button("Load Sample Dataset", use_container_width=True):
        sample_df = pd.DataFrame({
            'Age': np.random.randint(18, 70, 1000),
            'Income': np.random.randint(20000, 150000, 1000),
            'Score': np.random.randint(50, 100, 1000),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'Target': np.random.choice([0, 1], 1000)
        })
        st.session_state.df = sample_df
        st.success("✅ Sample data loaded!")
        st.rerun()

# =====================================================
# MAIN CONTENT
# =====================================================

if st.session_state.df is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 60px 40px;">
        <h2 style="font-size: 36px; margin-bottom: 20px;">👈 Upload Your Dataset to Begin</h2>
        <p style="font-size: 18px; opacity: 0.9; margin-bottom: 30px;">
            Unlock powerful AI-driven insights, predictions, and visualizations
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 30px;">
            <div style="text-align: center;">
                <div style="font-size: 48px; margin-bottom: 10px;">🤖</div>
                <p style="font-size: 14px;">Advanced ML Models</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
                <p style="font-size: 14px;">Smart Visualizations</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 48px; margin-bottom: 10px;">⚡</div>
                <p style="font-size: 14px;">Real-time Predictions</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 48px; margin-bottom: 10px;">🎯</div>
                <p style="font-size: 14px;">Auto-ML Features</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# =====================================================
# TABS
# =====================================================

tabs = st.tabs([
    "📊 Overview",
    "🔧 Data Cleaning",
    "📈 Visualization",
    "🤖 ML Models",
    "🎯 Predictions",
    "🧬 Feature Engineering",
    "⚙️ Advanced",
    "💾 Export"
])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================

with tabs[0]:
    st.markdown("## 📊 Dataset Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_animated_metric("📝", "Total Rows", f"{df.shape[0]:,}")
    
    with col2:
        create_animated_metric("🔢", "Total Columns", f"{df.shape[1]:,}")
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        create_animated_metric("❓", "Missing Data", f"{missing_pct:.1f}%")
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        create_animated_metric("💾", "Memory", f"{memory_mb:.1f} MB")
    
    st.markdown("---")
    
    # Data Preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(20), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Column Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Column Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notnull().sum(),
            'Null': df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📊 Data Type Distribution")
        type_counts = df.dtypes.value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index.astype(str),
            title="Column Types",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistical Summary
    st.markdown("### 📈 Statistical Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    else:
        st.info("No numeric columns found")

# =====================================================
# TAB 2: DATA CLEANING
# =====================================================

with tabs[1]:
    st.markdown("## 🔧 Data Cleaning & Transformation")
    
    # Missing Values
    with st.expander("❓ Handle Missing Values", expanded=True):
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                strategy = st.selectbox(
                    "Strategy",
                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Forward fill", "Backward fill"]
                )
            
            with col2:
                target_col = st.selectbox("Column", missing_df['Column'].tolist())
            
            if st.button("🔧 Apply", key="missing_apply"):
                df_clean = df.copy()
                
                if strategy == "Drop rows":
                    df_clean = df_clean.dropna(subset=[target_col])
                elif strategy == "Fill with mean":
                    df_clean[target_col].fillna(df_clean[target_col].mean(), inplace=True)
                elif strategy == "Fill with median":
                    df_clean[target_col].fillna(df_clean[target_col].median(), inplace=True)
                elif strategy == "Fill with mode":
                    df_clean[target_col].fillna(df_clean[target_col].mode()[0], inplace=True)
                elif strategy == "Forward fill":
                    df_clean[target_col].fillna(method='ffill', inplace=True)
                else:
                    df_clean[target_col].fillna(method='bfill', inplace=True)
                
                push_history(df, f"🔧 {strategy} - {target_col}")
                st.session_state.df = df_clean
                st.success(f"✅ Applied {strategy} to {target_col}")
                st.rerun()
        else:
            st.success("✅ No missing values found!")
    
    # Outlier Detection
    with st.expander("🎯 Detect & Remove Outliers", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                outlier_col = st.selectbox("Select column", numeric_cols, key="outlier_col")
            
            with col2:
                method = st.selectbox("Method", ["IQR", "Z-Score", "Isolation Forest"])
            
            if outlier_col:
                # Visualize outliers
                fig = px.box(df, y=outlier_col, title=f"Outliers in {outlier_col}")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🗑️ Remove Outliers", key="remove_outliers"):
                    df_clean = df.copy()
                    
                    if method == "IQR":
                        Q1 = df_clean[outlier_col].quantile(0.25)
                        Q3 = df_clean[outlier_col].quantile(0.75)
                        IQR = Q3 - Q1
                        df_clean = df_clean[
                            (df_clean[outlier_col] >= Q1 - 1.5 * IQR) &
                            (df_clean[outlier_col] <= Q3 + 1.5 * IQR)
                        ]
                    elif method == "Z-Score":
                        z_scores = np.abs((df_clean[outlier_col] - df_clean[outlier_col].mean()) / df_clean[outlier_col].std())
                        df_clean = df_clean[z_scores < 3]
                    else:
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers = iso_forest.fit_predict(df_clean[[outlier_col]])
                        df_clean = df_clean[outliers == 1]
                    
                    removed = len(df) - len(df_clean)
                    push_history(df, f"🎯 Removed {removed} outliers from {outlier_col}")
                    st.session_state.df = df_clean
                    st.success(f"✅ Removed {removed} outliers!")
                    st.rerun()
    
    # Data Transformation
    with st.expander("🔄 Data Transformation", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                transform_col = st.selectbox("Select column", numeric_cols, key="transform_col")
            
            with col2:
                transform_type = st.selectbox(
                    "Transformation",
                    ["Standard Scaler", "Min-Max Scaler", "Robust Scaler", "Log Transform", "Square Root", "Box-Cox"]
                )
            
            if st.button("🔄 Transform", key="transform_btn"):
                df_transform = df.copy()
                
                if transform_type == "Standard Scaler":
                    scaler = StandardScaler()
                    df_transform[transform_col] = scaler.fit_transform(df_transform[[transform_col]])
                elif transform_type == "Min-Max Scaler":
                    scaler = MinMaxScaler()
                    df_transform[transform_col] = scaler.fit_transform(df_transform[[transform_col]])
                elif transform_type == "Robust Scaler":
                    scaler = RobustScaler()
                    df_transform[transform_col] = scaler.fit_transform(df_transform[[transform_col]])
                elif transform_type == "Log Transform":
                    df_transform[transform_col] = np.log1p(df_transform[transform_col])
                elif transform_type == "Square Root":
                    df_transform[transform_col] = np.sqrt(df_transform[transform_col])
                else:
                    from scipy import stats
                    df_transform[transform_col], _ = stats.boxcox(df_transform[transform_col] + 1)
                
                push_history(df, f"🔄 {transform_type} - {transform_col}")
                st.session_state.df = df_transform
                st.success(f"✅ Applied {transform_type}!")
                st.rerun()
    
    # Encode Categorical
    with st.expander("🏷️ Encode Categorical Variables", expanded=False):
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if cat_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                encode_col = st.selectbox("Select column", cat_cols, key="encode_col")
            
            with col2:
                encode_type = st.selectbox("Encoding", ["Label Encoding", "One-Hot Encoding"])
            
            if st.button("🏷️ Encode", key="encode_btn"):
                df_encode = df.copy()
                
                if encode_type == "Label Encoding":
                    le = LabelEncoder()
                    df_encode[encode_col] = le.fit_transform(df_encode[encode_col].astype(str))
                else:
                    df_encode = pd.get_dummies(df_encode, columns=[encode_col], prefix=encode_col)
                
                push_history(df, f"🏷️ {encode_type} - {encode_col}")
                st.session_state.df = df_encode
                st.success(f"✅ Encoded {encode_col}!")
                st.rerun()

# =====================================================
# TAB 3: VISUALIZATION
# =====================================================

with tabs[2]:
    st.markdown("## 📈 Interactive Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Plot", "Correlation Heatmap", "Scatter Plot", "Box Plot", 
         "Violin Plot", "3D Scatter", "Pair Plot", "Time Series"]
    )
    
    if viz_type == "Distribution Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="dist_col")
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
            
            fig.add_trace(
                go.Histogram(x=df[col], name="Distribution", marker_color='#667eea'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(y=df[col], name="Box Plot", marker_color='#764ba2'),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            st.markdown("### 🤖 AI Insights")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr:
                for col1, col2, corr in high_corr[:5]:
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>Strong Correlation Detected!</strong><br>
                        {col1} ↔️ {col2}: {corr:.3f}<br>
                        💡 Consider removing one feature to avoid multicollinearity
                    </div>
                    """, unsafe_allow_html=True)
    
    elif viz_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            
            color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist(), key="scatter_color")
            
            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=color_col if color_col else None,
                title=f"{x_col} vs {y_col}",
                trendline="ols",
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Numeric column", numeric_cols, key="box_y")
            with col2:
                x_col = st.selectbox("Group by (optional)", [None] + cat_cols, key="box_x")
            
            fig = px.box(
                df, x=x_col if x_col else None, y=y_col,
                title=f"Distribution of {y_col}",
                color=x_col if x_col else None,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Violin Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Numeric column", numeric_cols, key="violin_y")
            with col2:
                x_col = st.selectbox("Group by (optional)", [None] + cat_cols, key="violin_x")
            
            fig = px.violin(
                df, x=x_col if x_col else None, y=y_col,
                title=f"Violin Plot - {y_col}",
                box=True,
                color=x_col if x_col else None
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D Scatter":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, key="3d_y")
            with col3:
                z_col = st.selectbox("Z-axis", numeric_cols, key="3d_z")
            
            color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist(), key="3d_color")
            
            fig = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                color=color_col if color_col else None,
                title="3D Scatter Plot",
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                scene=dict(
                    bgcolor='rgba(0,0,0,0)',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 4: ML MODELS
# =====================================================

with tabs[3]:
    st.markdown("## 🤖 Machine Learning Models")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for ML modeling")
        st.stop()
    
    # Model Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox("🎯 Target Variable", all_cols, key="ml_target")
    
    with col2:
        feature_cols = st.multiselect(
            "📊 Feature Columns",
            [col for col in all_cols if col != target_col],
            default=[col for col in numeric_cols if col != target_col][:5]
        )
    
    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column")
        st.stop()
    
    # Auto-detect problem type
    problem_type = detect_problem_type(df, target_col)
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>🤖 Auto-Detection</strong><br>
        Problem Type: <strong>{problem_type.upper()}</strong><br>
        Target: {target_col} | Features: {len(feature_cols)}
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection
    if problem_type == 'classification':
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Lasso Regression": Lasso(alpha=1.0, random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
            "SVR": SVR(),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42)
        }
    
    selected_models = st.multiselect(
        "🎯 Select Models to Train",
        list(models.keys()),
        default=list(models.keys())[:3]
    )
    
    # Train-Test Split
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    
    if st.button("🚀 Train Models", key="train_models", use_container_width=True):
        # Prepare data
        df_ml = df.dropna(subset=[target_col] + feature_cols)
        
        # Encode categorical features
        X = df_ml[feature_cols].copy()
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if classification
        y = df_ml[target_col]
        if problem_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train models
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            
            model = models[model_name]
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results.append({
                    'Model': model_name,
                    'Accuracy': f"{accuracy:.4f}",
                    'Precision': f"{precision:.4f}",
                    'Recall': f"{recall:.4f}",
                    'F1-Score': f"{f1:.4f}",
                    'CV Score': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
                    'Score': accuracy
                })
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                           scoring='r2')
                
                results.append({
                    'Model': model_name,
                    'RMSE': f"{rmse:.4f}",
                    'MAE': f"{mae:.4f}",
                    'R² Score': f"{r2:.4f}",
                    'CV R² Score': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
                    'Score': r2
                })
            
            # Save model
            st.session_state.trained_models[model_name] = {
                'model': model,
                'features': feature_cols,
                'target': target_col,
                'type': problem_type
            }
            
            progress_bar.progress((idx + 1) / len(selected_models))
        
        status_text.empty()
        progress_bar.empty()
        
        # Display results
        results_df = pd.DataFrame(results)
        
        # Find best model
        best_idx = results_df['Score'].astype(float).idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        st.session_state.best_model = best_model_name
        
        st.markdown("### 🏆 Model Performance")
        st.dataframe(
            results_df.drop('Score', axis=1).style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Best Model: {best_model_name}</h3>
            <p>This model has been saved and can be used for predictions!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Importance (if available)
        best_model_obj = st.session_state.trained_models[best_model_name]['model']
        importance_df = get_feature_importance(best_model_obj, feature_cols)
        
        if importance_df is not None:
            st.markdown("### 📊 Feature Importance")
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top Features - {best_model_name}",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix for Classification
        if problem_type == 'classification':
            st.markdown("### 🎯 Confusion Matrix")
            
            model = st.session_state.trained_models[best_model_name]['model']
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                title=f"Confusion Matrix - {best_model_name}"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 5: PREDICTIONS
# =====================================================

with tabs[4]:
    st.markdown("## 🎯 Make Predictions")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ Please train models first in the ML Models tab")
        st.stop()
    
    # Select model
    model_name = st.selectbox(
        "Select Model",
        list(st.session_state.trained_models.keys()),
        index=list(st.session_state.trained_models.keys()).index(st.session_state.best_model) 
        if st.session_state.best_model else 0
    )
    
    model_info = st.session_state.trained_models[model_name]
    model = model_info['model']
    feature_cols = model_info['features']
    problem_type = model_info['type']
    
    st.markdown(f"""
    <div class="model-card">
        <h3>📋 Model: {model_name}</h3>
        <p><strong>Type:</strong> {problem_type.title()}</p>
        <p><strong>Features:</strong> {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}</p>
        <p><strong>Total Features:</strong> {len(feature_cols)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction mode
    pred_mode = st.radio("Prediction Mode", ["Single Prediction", "Batch Prediction"])
    
    if pred_mode == "Single Prediction":
        st.markdown("### 📝 Enter Feature Values")
        
        input_data = {}
        cols = st.columns(3)
        
        for idx, col in enumerate(feature_cols):
            with cols[idx % 3]:
                if df[col].dtype in [np.float64, np.int64]:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    input_data[col] = st.number_input(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{col}"
                    )
                else:
                    unique_vals = df[col].unique().tolist()
                    input_data[col] = st.selectbox(col, unique_vals, key=f"input_{col}")
        
        if st.button("🔮 Predict", key="predict_single", use_container_width=True):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical
            for col in input_df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                input_df[col] = le.transform(input_df[col].astype(str))
            
            # Predict
            prediction = model.predict(input_df)[0]
            
            # Display result
            if problem_type == 'classification':
                proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
                
                st.markdown(f"""
                <div class="success-box" style="text-align: center; padding: 40px;">
                    <h2 style="font-size: 48px; margin-bottom: 20px;">🎯 Prediction</h2>
                    <h1 style="font-size: 72px; font-weight: 800;">{prediction}</h1>
                    {f'<p style="font-size: 18px; margin-top: 20px;">Confidence: {max(proba)*100:.1f}%</p>' if proba is not None else ''}
                </div>
                """, unsafe_allow_html=True)
                
                if proba is not None:
                    # Probability chart
                    classes = model.classes_ if hasattr(model, 'classes_') else range(len(proba))
                    fig = px.bar(
                        x=classes,
                        y=proba,
                        labels={'x': 'Class', 'y': 'Probability'},
                        title="Prediction Probabilities",
                        color=proba,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="success-box" style="text-align: center; padding: 40px;">
                    <h2 style="font-size: 48px; margin-bottom: 20px;">🎯 Prediction</h2>
                    <h1 style="font-size: 72px; font-weight: 800;">{prediction:.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.markdown("### 📊 Batch Prediction")
        
        upload_pred = st.file_uploader("Upload CSV for predictions", type=['csv'])
        
        if upload_pred:
            pred_df = pd.read_csv(upload_pred)
            st.dataframe(pred_df.head(), use_container_width=True)
            
            if st.button("🔮 Predict All", key="predict_batch", use_container_width=True):
                # Check columns
                missing_cols = set(feature_cols) - set(pred_df.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    # Prepare data
                    X_pred = pred_df[feature_cols].copy()
                    
                    # Encode categorical
                    for col in X_pred.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        le.fit(df[col].astype(str))
                        X_pred[col] = le.transform(X_pred[col].astype(str))
                    
                    # Predict
                    predictions = model.predict(X_pred)
                    
                    # Add to dataframe
                    pred_df['Prediction'] = predictions
                    
                    if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(X_pred)
                        pred_df['Confidence'] = probas.max(axis=1)
                    
                    st.success(f"✅ Predictions complete for {len(pred_df)} rows!")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download results
                    download_button(pred_df, "csv", "📥 Download Predictions", "download_predictions")

# =====================================================
# TAB 6: FEATURE ENGINEERING
# =====================================================

with tabs[5]:
    st.markdown("## 🧬 Advanced Feature Engineering")
    
    # Clustering
    with st.expander("🎯 Clustering Analysis", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            cluster_cols = st.multiselect(
                "Select features for clustering",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if cluster_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    method = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])
                
                with col2:
                    if method == "K-Means":
                        n_clusters = st.slider("Number of clusters", 2, 10, 3)
                    elif method == "DBSCAN":
                        eps = st.slider("Epsilon", 0.1, 5.0, 0.5)
                        min_samples = st.slider("Min samples", 2, 10, 3)
                    else:
                        n_clusters = st.slider("Number of clusters", 2, 10, 3)
                
                if st.button("🎯 Apply Clustering", key="clustering_btn"):
                    X = df[cluster_cols].dropna()
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Clustering
                    if method == "K-Means":
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                    elif method == "DBSCAN":
                        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                    else:
                        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                    
                    clusters = clusterer.fit_predict(X_scaled)
                    
                    # Visualize
                    if len(cluster_cols) >= 2:
                        viz_df = df.copy()
                        viz_df['Cluster'] = -1
                        viz_df.loc[X.index, 'Cluster'] = clusters
                        
                        fig = px.scatter(
                            viz_df,
                            x=cluster_cols[0],
                            y=cluster_cols[1],
                            color='Cluster',
                            title=f"{method} Clustering Results",
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add to dataset
                    if st.button("➕ Add to Dataset", key="add_clusters"):
                        df_clustered = df.copy()
                        df_clustered['Cluster'] = -1
                        df_clustered.loc[X.index, 'Cluster'] = clusters
                        
                        push_history(df, f"🎯 {method} clustering added")
                        st.session_state.df = df_clustered
                        st.success("✅ Cluster column added!")
                        st.rerun()
    
    # Dimensionality Reduction
    with st.expander("📉 Dimensionality Reduction", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.selectbox("Method", ["PCA", "Truncated SVD"])
            
            with col2:
                n_components = st.slider("Components", 2, min(10, len(numeric_cols)), 2)
            
            if st.button("📉 Apply Reduction", key="reduction_btn"):
                X = df[numeric_cols].dropna()
                
                if method == "PCA":
                    reducer = PCA(n_components=n_components, random_state=42)
                else:
                    reducer = TruncatedSVD(n_components=n_components, random_state=42)
                
                X_reduced = reducer.fit_transform(X)
                
                # Visualize first 2 components
                fig = px.scatter(
                    x=X_reduced[:, 0],
                    y=X_reduced[:, 1],
                    title=f"{method} - First 2 Components",
                    labels={'x': 'Component 1', 'y': 'Component 2'},
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Explained variance
                if hasattr(reducer, 'explained_variance_ratio_'):
                    var_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(n_components)],
                        'Variance': reducer.explained_variance_ratio_
                    })
                    
                    fig = px.bar(
                        var_df,
                        x='Component',
                        y='Variance',
                        title="Explained Variance by Component",
                        color='Variance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>📊 Total Variance Explained</strong><br>
                        {reducer.explained_variance_ratio_.sum()*100:.2f}% with {n_components} components
                    </div>
                    """, unsafe_allow_html=True)
    
    # Feature Selection
    with st.expander("🎯 Feature Selection", expanded=False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            target_col = st.selectbox("Target variable", numeric_cols, key="fs_target")
            feature_cols = [c for c in numeric_cols if c != target_col]
            
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.selectbox("Method", ["F-statistic", "Mutual Information"])
            
            with col2:
                k = st.slider("Number of features", 1, len(feature_cols), min(5, len(feature_cols)))
            
            if st.button("🎯 Select Features", key="feature_select_btn"):
                X = df[feature_cols].fillna(0)
                y = df[target_col].fillna(0)
                
                problem_type = detect_problem_type(df, target_col)
                
                if method == "F-statistic":
                    if problem_type == 'classification':
                        selector = SelectKBest(f_classif, k=k)
                    else:
                        selector = SelectKBest(f_regression, k=k)
                else:
                    selector = SelectKBest(mutual_info_classif, k=k)
                
                selector.fit(X, y)
                
                scores = pd.DataFrame({
                    'Feature': feature_cols,
                    'Score': selector.scores_
                }).sort_values('Score', ascending=False)
                
                st.markdown("### 📊 Feature Scores")
                fig = px.bar(
                    scores.head(k),
                    x='Score',
                    y='Feature',
                    orientation='h',
                    title=f"Top {k} Features",
                    color='Score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                selected_features = scores.head(k)['Feature'].tolist()
                
                st.markdown(f"""
                <div class="insight-box">
                    <strong>🎯 Selected Features</strong><br>
                    {', '.join(selected_features)}
                </div>
                """, unsafe_allow_html=True)

# =====================================================
# TAB 7: ADVANCED OPERATIONS
# =====================================================

with tabs[6]:
    st.markdown("## ⚙️ Advanced Operations")
    
    # Sampling
    with st.expander("🎲 Data Sampling", expanded=True):
        sample_type = st.selectbox(
            "Sampling method",
            ["Random fraction", "Random n rows", "Stratified sampling"]
        )
        
        if sample_type == "Random fraction":
            frac = st.slider("Sample fraction", 0.01, 1.0, 0.2)
            
            if st.button("🎲 Sample", key="sample_frac"):
                df_sample = df.sample(frac=frac, random_state=42)
                push_history(df, f"🎲 Sampled {frac*100:.1f}%")
                st.session_state.df = df_sample
                st.success(f"✅ Sampled {len(df_sample)} rows!")
                st.rerun()
        
        elif sample_type == "Random n rows":
            n = st.number_input("Number of rows", 1, len(df), min(1000, len(df)))
            
            if st.button("🎲 Sample", key="sample_n"):
                df_sample = df.sample(n=int(n), random_state=42)
                push_history(df, f"🎲 Sampled {n} rows")
                st.session_state.df = df_sample
                st.success(f"✅ Sampled {n} rows!")
                st.rerun()
        
        else:
            strat_col = st.selectbox("Stratify by", df.columns.tolist())
            frac = st.slider("Sample fraction", 0.01, 1.0, 0.2, key="strat_frac")
            
            if st.button("🎲 Sample", key="sample_strat"):
                try:
                    df_sample = df.groupby(strat_col, group_keys=False).apply(
                        lambda x: x.sample(frac=min(frac, 1.0), random_state=42)
                    ).reset_index(drop=True)
                    push_history(df, f"🎲 Stratified by {strat_col}")
                    st.session_state.df = df_sample
                    st.success(f"✅ Sampled {len(df_sample)} rows!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Filtering
    with st.expander("🔍 Filter Data", expanded=True):
        filter_col = st.selectbox("Column to filter", df.columns.tolist())
        
        if df[filter_col].dtype in [np.float64, np.int64]:
            min_val = float(df[filter_col].min())
            max_val = float(df[filter_col].max())
            
            filter_range = st.slider(
                "Value range",
                min_val, max_val,
                (min_val, max_val)
            )
            
            if st.button("🔍 Filter", key="filter_numeric"):
                df_filtered = df[
                    (df[filter_col] >= filter_range[0]) &
                    (df[filter_col] <= filter_range[1])
                ]
                push_history(df, f"🔍 Filtered {filter_col}")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered)} rows!")
                st.rerun()
        else:
            unique_vals = df[filter_col].unique().tolist()
            selected_vals = st.multiselect(
                "Select values",
                unique_vals,
                default=unique_vals[:min(5, len(unique_vals))]
            )
            
            if selected_vals and st.button("🔍 Filter", key="filter_cat"):
                df_filtered = df[df[filter_col].isin(selected_vals)]
                push_history(df, f"🔍 Filtered {filter_col}")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered)} rows!")
                st.rerun()
    
    # Merging
    with st.expander("🔗 Merge Datasets", expanded=True):
        if st.session_state.df2 is not None:
            df2 = st.session_state.df2
            
            st.info(f"Second dataset: {df2.shape[0]} rows × {df2.shape[1]} columns")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"])
            
            with col2:
                left_key = st.selectbox("Left key", df.columns.tolist())
            
            with col3:
                right_key = st.selectbox("Right key", df2.columns.tolist())
            
            if st.button("🔗 Merge", key="merge_btn"):
                try:
                    merged = pd.merge(
                        df, df2,
                        left_on=left_key,
                        right_on=right_key,
                        how=join_type,
                        suffixes=("", "_2")
                    )
                    push_history(df, f"🔗 Merged on {left_key}")
                    st.session_state.df = merged
                    st.success(f"✅ Merged! {merged.shape[0]} rows × {merged.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("👈 Upload second dataset in sidebar")
    
    # Sorting
    with st.expander("↕️ Sort Data", expanded=False):
        sort_cols = st.multiselect("Columns to sort by", df.columns.tolist())
        
        if sort_cols:
            ascending = st.checkbox("Ascending order", value=True)
            
            if st.button("↕️ Sort", key="sort_btn"):
                df_sorted = df.sort_values(by=sort_cols, ascending=ascending)
                push_history(df, f"↕️ Sorted by {', '.join(sort_cols)}")
                st.session_state.df = df_sorted
                st.success("✅ Data sorted!")
                st.rerun()

# =====================================================
# TAB 8: EXPORT
# =====================================================

with tabs[7]:
    st.markdown("## 💾 Export Your Data")
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <h3>📊 Current Dataset</h3>
        <p style="font-size: 18px; margin: 15px 0;">
            <strong>{df.shape[0]:,}</strong> rows × <strong>{df.shape[1]:,}</strong> columns
        </p>
        <p style="opacity: 0.9;">
            Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        download_button(df, "csv", "📄 CSV", "export_csv")
    
    with col2:
        download_button(df, "excel", "📊 Excel", "export_excel")
    
    with col3:
        download_button(df, "json", "📋 JSON", "export_json")
    
    st.markdown("---")
    
    # History
    show_history()

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 40px 20px;">
    <h3 style="margin-bottom: 15px;">🚀 Advanced AI-ML Analytics Platform</h3>
    <p style="font-size: 16px; opacity: 0.9; margin-bottom: 10px;">
        Transform • Analyze • Visualize • Predict • Optimize
    </p>
    <p style="font-size: 14px; opacity: 0.75;">
        Built with ❤️ using Streamlit, Scikit-learn & Plotly
    </p>
    <p style="font-size: 12px; opacity: 0.6; margin-top: 10px;">
        Powered by Advanced Machine Learning Algorithms
    </p>
</div>
""", unsafe_allow_html=True)
