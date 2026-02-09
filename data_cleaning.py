import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    layout="wide",
    page_title="AI-Powered Data Analytics Platform",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# =====================================================
# ENHANCED CSS STYLING
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 2.5rem;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 35px;
        margin: 25px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }

    .metric-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 30px 25px;
        margin: 15px 0;
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
        font-size: 52px;
        margin-bottom: 12px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .metric-label {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
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

    .stButton > button:active {
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-bottom: 25px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 14px 28px;
        background: transparent;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        font-size: 15px;
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

    .dataframe {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(15px);
        border-right: 2px solid rgba(255, 255, 255, 0.2);
        padding: 2.5rem 1.5rem;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .stFileUploader {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 25px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: rgba(255, 255, 255, 0.5);
        background: rgba(255, 255, 255, 0.2);
    }

    .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 8px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 20px;
    }

    .stExpander {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 15px 0;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: white;
        font-size: 15px;
        line-height: 1.6;
    }

    .download-section {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        border: 2px solid rgba(255, 255, 255, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def create_metric_card(icon, label, value):
    """Create beautiful animated metric cards"""
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def download_button(df, format_type, label, key):
    """Enhanced download button with proper formatting"""
    try:
        if format_type == "csv":
            data = df.to_csv(index=False).encode('utf-8')
            mime = "text/csv"
            filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif format_type == "excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            data = buffer.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        elif format_type == "json":
            data = df.to_json(orient='records', indent=2).encode('utf-8')
            mime = "application/json"
            filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime,
            key=key,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def push_history(df, action):
    """Save operation to history"""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        "time": datetime.now(),
        "action": action,
        "df": df.copy()
    })
    
    # Keep only last 10 operations
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]

def show_history():
    """Display operation history"""
    if "history" in st.session_state and st.session_state.history:
        st.markdown("### 📜 Operation History")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #667eea;">
                <strong>{h['time'].strftime('%H:%M:%S')}</strong> - {h['action']}<br>
                <small>Shape: {h['df'].shape[0]} rows × {h['df'].shape[1]} cols</small>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔄 Undo Last Operation", use_container_width=True):
            if len(st.session_state.history) > 0:
                last = st.session_state.history.pop()
                st.session_state.df = last["df"]
                st.success("✅ Undone!")
                st.rerun()

def detect_outliers(df):
    """Detect outliers using IsolationForest"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return None
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    iso = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso.fit_predict(X)
    
    return outliers

def perform_ml(df, target_col, task_type, test_size=0.2):
    """Perform ML classification or regression"""
    try:
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            return {
                "model": model,
                "predictions": y_pred,
                "actual": y_test,
                "report": classification_report(y_test, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "feature_importance": dict(zip(X.columns, model.feature_importances_))
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            return {
                "model": model,
                "predictions": y_pred,
                "actual": y_test,
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mse": mean_squared_error(y_test, y_pred),
                "feature_importance": dict(zip(X.columns, model.feature_importances_))
            }
    except Exception as e:
        st.error(f"ML Error: {str(e)}")
        return None

def perform_clustering(df, n_clusters=3, method="kmeans"):
    """Perform clustering analysis"""
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found for clustering")
            return None, None
        
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        
        clusters = clusterer.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        return clusters, pca_result
    except Exception as e:
        st.error(f"Clustering Error: {str(e)}")
        return None, None

# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "df2" not in st.session_state:
    st.session_state.df2 = None

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 42px; margin: 0;">🤖</h1>
        <h2 style="margin: 10px 0;">Data Refinery</h2>
        <p style="opacity: 0.9; font-size: 14px;">AI-Powered Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main dataset upload
    st.markdown("### 📂 Upload Main Dataset")
    uploaded_file = st.file_uploader(
        "Choose CSV/Excel file",
        type=["csv", "xlsx", "xls"],
        key="main_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Second dataset for merging
    st.markdown("### 📂 Upload Second Dataset (Optional)")
    uploaded_file2 = st.file_uploader(
        "For merging operations",
        type=["csv", "xlsx", "xls"],
        key="second_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_file2:
        try:
            if uploaded_file2.name.endswith('.csv'):
                df2 = pd.read_csv(uploaded_file2)
            else:
                df2 = pd.read_excel(uploaded_file2)
            
            st.session_state.df2 = df2
            st.success(f"✅ Loaded: {df2.shape[0]} rows × {df2.shape[1]} cols")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.df is not None:
        st.markdown("### 📊 Quick Stats")
        df = st.session_state.df
        
        st.metric("Total Rows", f"{df.shape[0]:,}")
        st.metric("Total Columns", f"{df.shape[1]:,}")
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

# =====================================================
# MAIN CONTENT
# =====================================================

# Header
st.markdown("""
<div style="text-align: center; padding: 30px 0 20px 0;">
    <h1 style="font-size: 52px; margin: 0; background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        🤖 AI-Powered Data Analytics Platform
    </h1>
    <p style="font-size: 18px; opacity: 0.95; margin-top: 15px;">Transform, Analyze, and Visualize Your Data with AI</p>
</div>
""", unsafe_allow_html=True)

# Check if data is loaded
if st.session_state.df is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 60px 40px;">
        <h2 style="font-size: 48px; margin-bottom: 20px;">📊</h2>
        <h3 style="margin-bottom: 15px;">Welcome to Data Refinery!</h3>
        <p style="font-size: 16px; opacity: 0.9; line-height: 1.8;">
            Upload your dataset using the sidebar to get started.<br>
            Supported formats: CSV, Excel (.xlsx, .xls)
        </p>
        <br>
        <p style="font-size: 14px; opacity: 0.8;">
            ✨ Advanced transformations • 🤖 ML predictions • 📈 Beautiful visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Get current dataframe
df = st.session_state.df
df2 = st.session_state.df2

# =====================================================
# TABS
# =====================================================

tabs = st.tabs([
    "📊 Overview",
    "🔍 Data Cleaning",
    "⚡ Transformations",
    "📈 Visualizations",
    "🤖 ML Predictions",
    "🎯 Clustering",
    "⚙️ Advanced Ops",
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
        create_metric_card("📏", "Total Rows", f"{df.shape[0]:,}")
    with col2:
        create_metric_card("📊", "Total Columns", f"{df.shape[1]:,}")
    with col3:
        create_metric_card("💾", "Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        create_metric_card("❌", "Missing Values", f"{df.isnull().sum().sum():,}")
    
    st.markdown("---")
    
    # Data Preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📋 Column Info")
        
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null': df.isnull().sum()
        })
        
        st.dataframe(info_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Statistical Summary
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# =====================================================
# TAB 2: DATA CLEANING
# =====================================================

with tabs[1]:
    st.markdown("## 🔍 Data Cleaning Operations")
    
    col1, col2 = st.columns(2)
    
    # Missing Values Handling
    with col1:
        st.markdown("### 🧹 Handle Missing Values")
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            st.info(f"Found {len(missing_cols)} columns with missing values")
            
            selected_cols = st.multiselect(
                "Select columns to clean",
                missing_cols,
                default=missing_cols[:3] if len(missing_cols) >= 3 else missing_cols
            )
            
            if selected_cols:
                method = st.selectbox(
                    "Cleaning method",
                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with 0", "Forward fill", "Backward fill"]
                )
                
                if st.button("🧹 Clean Data", key="clean_missing", use_container_width=True):
                    with st.spinner("Cleaning..."):
                        df_clean = df.copy()
                        
                        if method == "Drop rows":
                            df_clean = df_clean.dropna(subset=selected_cols)
                        elif method == "Fill with mean":
                            for col in selected_cols:
                                if df_clean[col].dtype in [np.float64, np.int64]:
                                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                        elif method == "Fill with median":
                            for col in selected_cols:
                                if df_clean[col].dtype in [np.float64, np.int64]:
                                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        elif method == "Fill with mode":
                            for col in selected_cols:
                                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                        elif method == "Fill with 0":
                            df_clean[selected_cols] = df_clean[selected_cols].fillna(0)
                        elif method == "Forward fill":
                            df_clean[selected_cols] = df_clean[selected_cols].fillna(method='ffill')
                        elif method == "Backward fill":
                            df_clean[selected_cols] = df_clean[selected_cols].fillna(method='bfill')
                        
                        push_history(df, f"🧹 Cleaned {len(selected_cols)} columns - {method}")
                        st.session_state.df = df_clean
                        st.success(f"✅ Cleaned! Removed/filled {df.isnull().sum().sum() - df_clean.isnull().sum().sum()} missing values")
                        st.rerun()
        else:
            st.success("✅ No missing values found!")
    
    # Duplicates Handling
    with col2:
        st.markdown("### 🔄 Handle Duplicates")
        
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate rows")
            
            subset_cols = st.multiselect(
                "Check duplicates based on columns (leave empty for all)",
                df.columns.tolist(),
                key="dup_cols"
            )
            
            keep = st.radio("Keep", ["first", "last", "none"])
            
            if st.button("🗑️ Remove Duplicates", key="remove_dup", use_container_width=True):
                with st.spinner("Removing..."):
                    df_clean = df.drop_duplicates(
                        subset=subset_cols if subset_cols else None,
                        keep=False if keep == "none" else keep
                    )
                    
                    push_history(df, f"🔄 Removed {len(df) - len(df_clean)} duplicates")
                    st.session_state.df = df_clean
                    st.success(f"✅ Removed {len(df) - len(df_clean)} duplicates!")
                    st.rerun()
        else:
            st.success("✅ No duplicates found!")
    
    st.markdown("---")
    
    # Outlier Detection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Outlier Detection")
        
        if st.button("🔍 Detect Outliers", key="detect_outliers", use_container_width=True):
            with st.spinner("Analyzing..."):
                outliers = detect_outliers(df)
                
                if outliers is not None:
                    outlier_count = (outliers == -1).sum()
                    st.info(f"Found {outlier_count} potential outliers ({outlier_count/len(df)*100:.2f}%)")
                    
                    if outlier_count > 0 and st.button("🗑️ Remove Outliers", key="remove_outliers", use_container_width=True):
                        df_clean = df[outliers != -1]
                        push_history(df, f"🎯 Removed {outlier_count} outliers")
                        st.session_state.df = df_clean
                        st.success("✅ Outliers removed!")
                        st.rerun()
    
    # Column Operations
    with col2:
        st.markdown("### ✂️ Column Operations")
        
        operation = st.selectbox(
            "Operation",
            ["Drop columns", "Rename column", "Change data type"]
        )
        
        if operation == "Drop columns":
            cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            
            if cols_to_drop and st.button("✂️ Drop", key="drop_cols", use_container_width=True):
                df_new = df.drop(columns=cols_to_drop)
                push_history(df, f"✂️ Dropped {len(cols_to_drop)} columns")
                st.session_state.df = df_new
                st.success("✅ Columns dropped!")
                st.rerun()
        
        elif operation == "Rename column":
            col_to_rename = st.selectbox("Select column", df.columns.tolist())
            new_name = st.text_input("New name")
            
            if new_name and st.button("✏️ Rename", key="rename_col", use_container_width=True):
                df_new = df.rename(columns={col_to_rename: new_name})
                push_history(df, f"✏️ Renamed {col_to_rename} → {new_name}")
                st.session_state.df = df_new
                st.success("✅ Column renamed!")
                st.rerun()
        
        else:
            col_to_convert = st.selectbox("Select column", df.columns.tolist())
            new_type = st.selectbox("New type", ["int", "float", "str", "category"])
            
            if st.button("🔄 Convert", key="convert_type", use_container_width=True):
                try:
                    df_new = df.copy()
                    df_new[col_to_convert] = df_new[col_to_convert].astype(new_type)
                    push_history(df, f"🔄 Converted {col_to_convert} to {new_type}")
                    st.session_state.df = df_new
                    st.success("✅ Type converted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# =====================================================
# TAB 3: TRANSFORMATIONS
# =====================================================

with tabs[2]:
    st.markdown("## ⚡ Data Transformations")
    
    col1, col2 = st.columns(2)
    
    # Scaling
    with col1:
        st.markdown("### 📏 Feature Scaling")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            cols_to_scale = st.multiselect(
                "Select columns to scale",
                numeric_cols,
                key="scale_cols"
            )
            
            scaler_type = st.selectbox(
                "Scaler type",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            )
            
            if cols_to_scale and st.button("📏 Scale", key="scale_btn", use_container_width=True):
                with st.spinner("Scaling..."):
                    df_scaled = df.copy()
                    
                    if scaler_type == "StandardScaler":
                        scaler = StandardScaler()
                    elif scaler_type == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    
                    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
                    
                    push_history(df, f"📏 Scaled {len(cols_to_scale)} columns with {scaler_type}")
                    st.session_state.df = df_scaled
                    st.success("✅ Scaling complete!")
                    st.rerun()
        else:
            st.info("No numeric columns found")
    
    # Encoding
    with col2:
        st.markdown("### 🔤 Encoding Categorical Variables")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            cols_to_encode = st.multiselect(
                "Select columns to encode",
                categorical_cols,
                key="encode_cols"
            )
            
            encoding_type = st.selectbox(
                "Encoding type",
                ["Label Encoding", "One-Hot Encoding"]
            )
            
            if cols_to_encode and st.button("🔤 Encode", key="encode_btn", use_container_width=True):
                with st.spinner("Encoding..."):
                    df_encoded = df.copy()
                    
                    if encoding_type == "Label Encoding":
                        for col in cols_to_encode:
                            le = LabelEncoder()
                            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    else:
                        df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, prefix=cols_to_encode)
                    
                    push_history(df, f"🔤 Encoded {len(cols_to_encode)} columns - {encoding_type}")
                    st.session_state.df = df_encoded
                    st.success("✅ Encoding complete!")
                    st.rerun()
        else:
            st.info("No categorical columns found")
    
    st.markdown("---")
    
    # Feature Engineering
    st.markdown("### 🔧 Feature Engineering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        operation = st.selectbox(
            "Operation",
            ["Add", "Subtract", "Multiply", "Divide", "Power", "Log", "Sqrt"]
        )
    
    with col2:
        if operation in ["Add", "Subtract", "Multiply", "Divide"]:
            col1_name = st.selectbox("Column 1", numeric_cols, key="col1_eng")
    
    with col3:
        if operation in ["Add", "Subtract", "Multiply", "Divide"]:
            col2_name = st.selectbox("Column 2", numeric_cols, key="col2_eng")
        else:
            col_name = st.selectbox("Column", numeric_cols, key="col_eng")
    
    new_col_name = st.text_input("New column name", f"new_feature_{len(df.columns)}")
    
    if st.button("🔧 Create Feature", key="create_feature", use_container_width=True):
        try:
            df_new = df.copy()
            
            if operation == "Add":
                df_new[new_col_name] = df_new[col1_name] + df_new[col2_name]
            elif operation == "Subtract":
                df_new[new_col_name] = df_new[col1_name] - df_new[col2_name]
            elif operation == "Multiply":
                df_new[new_col_name] = df_new[col1_name] * df_new[col2_name]
            elif operation == "Divide":
                df_new[new_col_name] = df_new[col1_name] / df_new[col2_name].replace(0, np.nan)
            elif operation == "Power":
                power = st.number_input("Power", value=2.0)
                df_new[new_col_name] = df_new[col_name] ** power
            elif operation == "Log":
                df_new[new_col_name] = np.log(df_new[col_name].replace(0, np.nan))
            elif operation == "Sqrt":
                df_new[new_col_name] = np.sqrt(df_new[col_name])
            
            push_history(df, f"🔧 Created feature: {new_col_name}")
            st.session_state.df = df_new
            st.success("✅ Feature created!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# =====================================================
# TAB 4: VISUALIZATIONS
# =====================================================

with tabs[3]:
    st.markdown("## 📈 Data Visualizations")
    
    viz_type = st.selectbox(
        "Select visualization type",
        ["Distribution Plot", "Correlation Heatmap", "Scatter Plot", "Box Plot", "Bar Chart", "Line Chart", "Pie Chart"]
    )
    
    if viz_type == "Distribution Plot":
        col = st.selectbox("Select column", df.select_dtypes(include=[np.number]).columns.tolist())
        
        fig = px.histogram(
            df, x=col,
            title=f"Distribution of {col}",
            marginal="box",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            
            fig = px.imshow(
                corr,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation")
    
    elif viz_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        
        color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
        
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=None if color_col == "None" else color_col,
            title=f"{x_col} vs {y_col}",
            trendline="ols"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:5])
        
        if selected_cols:
            fig = px.box(
                df[selected_cols],
                title="Box Plot Comparison",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Category column", df.columns.tolist(), key="bar_x")
        with col2:
            y_col = st.selectbox("Value column", df.select_dtypes(include=[np.number]).columns.tolist(), key="bar_y")
        
        agg_func = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"])
        
        if agg_func == "count":
            grouped = df.groupby(x_col).size().reset_index(name=y_col)
        else:
            grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
        
        fig = px.bar(
            grouped.head(20), x=x_col, y=y_col,
            title=f"{agg_func.title()} of {y_col} by {x_col}",
            color=y_col,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        y_cols = st.multiselect("Select Y columns", numeric_cols, default=numeric_cols[:2])
        x_col = st.selectbox("X-axis (index if none)", ["Index"] + numeric_cols)
        
        if y_cols:
            if x_col == "Index":
                fig = px.line(df, y=y_cols, title="Line Chart")
            else:
                fig = px.line(df, x=x_col, y=y_cols, title=f"{', '.join(y_cols)} over {x_col}")
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pie Chart":
        col = st.selectbox("Select column", df.columns.tolist())
        
        value_counts = df[col].value_counts().head(10).reset_index()
        value_counts.columns = [col, 'count']
        
        fig = px.pie(
            value_counts,
            values='count',
            names=col,
            title=f"Distribution of {col}"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 5: ML PREDICTIONS
# =====================================================

with tabs[4]:
    st.markdown("## 🤖 Machine Learning Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_col = st.selectbox("Target column", df.columns.tolist())
    
    with col2:
        task_type = st.selectbox("Task type", ["classification", "regression"])
    
    with col3:
        test_size = st.slider("Test size %", 10, 50, 20) / 100
    
    if st.button("🚀 Train Model", key="train_ml", use_container_width=True):
        with st.spinner("Training model... This may take a moment..."):
            results = perform_ml(df, target_col, task_type, test_size)
            
            if results:
                st.success("✅ Model trained successfully!")
                
                if task_type == "classification":
                    # Classification Metrics
                    st.markdown("### 📊 Classification Report")
                    
                    report_df = pd.DataFrame(results['report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Confusion Matrix
                    st.markdown("### 🎯 Confusion Matrix")
                    
                    fig = px.imshow(
                        results['confusion_matrix'],
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Blues',
                        title="Confusion Matrix"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Regression Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        create_metric_card("📊", "R² Score", f"{results['r2']:.4f}")
                    with col2:
                        create_metric_card("📉", "RMSE", f"{results['rmse']:.4f}")
                    with col3:
                        create_metric_card("📈", "MSE", f"{results['mse']:.4f}")
                    
                    # Actual vs Predicted
                    st.markdown("### 🎯 Actual vs Predicted")
                    
                    plot_df = pd.DataFrame({
                        'Actual': results['actual'],
                        'Predicted': results['predictions']
                    })
                    
                    fig = px.scatter(
                        plot_df, x='Actual', y='Predicted',
                        title="Actual vs Predicted Values",
                        trendline="ols"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                st.markdown("### 🎯 Feature Importance")
                
                fi_df = pd.DataFrame(
                    list(results['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(
                    fi_df, x='Importance', y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title="Top 15 Most Important Features"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 6: CLUSTERING
# =====================================================

with tabs[5]:
    st.markdown("## 🎯 Clustering Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox("Clustering algorithm", ["K-Means", "DBSCAN"])
    
    with col2:
        if method == "K-Means":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
        else:
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
            n_clusters = 3
    
    if st.button("🎯 Perform Clustering", key="cluster_btn", use_container_width=True):
        with st.spinner("Finding patterns..."):
            clusters, pca_result = perform_clustering(
                df,
                n_clusters,
                method.lower().replace("-", "")
            )
            
            if clusters is not None:
                st.success("✅ Clustering complete!")
                
                # Cluster Visualization
                st.markdown("### 📊 Cluster Visualization (PCA)")
                
                plot_df = pd.DataFrame({
                    'PC1': pca_result[:, 0],
                    'PC2': pca_result[:, 1],
                    'Cluster': clusters.astype(str)
                })
                
                fig = px.scatter(
                    plot_df, x='PC1', y='PC2',
                    color='Cluster',
                    title=f"{method} Clustering Results",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster Distribution
                st.markdown("### 📈 Cluster Distribution")
                
                cluster_counts = pd.Series(clusters).value_counts().sort_index().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                cluster_counts['Cluster'] = cluster_counts['Cluster'].astype(str)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        cluster_counts,
                        values='Count',
                        names='Cluster',
                        title="Cluster Size Distribution"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        cluster_counts,
                        x='Cluster',
                        y='Count',
                        title="Cluster Counts",
                        color='Count',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add to dataset option
                st.markdown("---")
                if st.button("➕ Add Clusters to Dataset", key="add_cluster", use_container_width=True):
                    push_history(df, f"🎯 Added {method} clusters")
                    st.session_state.df['Cluster'] = clusters
                    st.success("✅ Cluster column added to dataset!")
                    st.rerun()

# =====================================================
# TAB 7: ADVANCED OPERATIONS
# =====================================================

with tabs[6]:
    st.markdown("## ⚙️ Advanced Operations")
    
    # Data Sampling
    with st.expander("🎲 Data Sampling", expanded=True):
        sample_type = st.selectbox(
            "Sampling method",
            ["Random fraction", "Random n rows", "Stratified sampling"]
        )
        
        if sample_type == "Random fraction":
            frac = st.slider("Sample fraction", 0.01, 1.0, 0.2)
            
            if st.button("🎲 Sample Data", key="sample_frac", use_container_width=True):
                df_sample = df.sample(frac=frac, random_state=42)
                push_history(df, f"🎲 Sampled {frac*100:.1f}% of data")
                st.session_state.df = df_sample
                st.success(f"✅ Sampled {len(df_sample)} rows!")
                st.rerun()
        
        elif sample_type == "Random n rows":
            n = st.number_input("Number of rows", 1, len(df), min(1000, len(df)))
            
            if st.button("🎲 Sample Data", key="sample_n", use_container_width=True):
                df_sample = df.sample(n=int(n), random_state=42)
                push_history(df, f"🎲 Sampled {n} rows")
                st.session_state.df = df_sample
                st.success(f"✅ Sampled {n} rows!")
                st.rerun()
        
        else:
            strat_col = st.selectbox("Stratify by column", df.columns.tolist())
            frac = st.slider("Sample fraction", 0.01, 1.0, 0.2, key="strat_frac")
            
            if st.button("🎲 Sample Data", key="sample_strat", use_container_width=True):
                try:
                    df_sample = df.groupby(strat_col, group_keys=False).apply(
                        lambda x: x.sample(frac=min(frac, 1.0), random_state=42)
                    ).reset_index(drop=True)
                    push_history(df, f"🎲 Stratified sampling by {strat_col}")
                    st.session_state.df = df_sample
                    st.success(f"✅ Sampled {len(df_sample)} rows with stratification!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Data Filtering
    with st.expander("🔍 Filter Data", expanded=True):
        filter_col = st.selectbox("Select column to filter", df.columns.tolist())
        
        if df[filter_col].dtype in [np.float64, np.int64]:
            min_val = float(df[filter_col].min())
            max_val = float(df[filter_col].max())
            
            filter_range = st.slider(
                "Value range",
                min_val, max_val,
                (min_val, max_val)
            )
            
            if st.button("🔍 Apply Filter", key="filter_numeric", use_container_width=True):
                df_filtered = df[
                    (df[filter_col] >= filter_range[0]) &
                    (df[filter_col] <= filter_range[1])
                ]
                push_history(df, f"🔍 Filtered {filter_col}: {filter_range[0]} to {filter_range[1]}")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered)} rows!")
                st.rerun()
        else:
            unique_vals = df[filter_col].unique().tolist()
            selected_vals = st.multiselect(
                "Select values to keep",
                unique_vals,
                default=unique_vals[:min(5, len(unique_vals))]
            )
            
            if selected_vals and st.button("🔍 Apply Filter", key="filter_cat", use_container_width=True):
                df_filtered = df[df[filter_col].isin(selected_vals)]
                push_history(df, f"🔍 Filtered {filter_col}: {len(selected_vals)} values")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered)} rows!")
                st.rerun()
    
    # Merge Datasets
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
            
            if st.button("🔗 Merge", key="merge_btn", use_container_width=True):
                try:
                    merged = pd.merge(
                        df, df2,
                        left_on=left_key,
                        right_on=right_key,
                        how=join_type,
                        suffixes=("", "_2")
                    )
                    push_history(df, f"🔗 Merged datasets on {left_key}={right_key} ({join_type})")
                    st.session_state.df = merged
                    st.success(f"✅ Merged! New shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"Merge error: {str(e)}")
        else:
            st.info("👈 Upload a second dataset in the sidebar to enable merging")
    
    # Sort Data
    with st.expander("↕️ Sort Data", expanded=False):
        sort_cols = st.multiselect("Select columns to sort by", df.columns.tolist())
        
        if sort_cols:
            ascending = st.checkbox("Ascending order", value=True)
            
            if st.button("↕️ Sort", key="sort_btn", use_container_width=True):
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
    <div class="download-section">
        <h3 style="margin-bottom: 20px;">📊 Current Dataset</h3>
        <p style="font-size: 16px; margin-bottom: 15px;">
            <strong>{df.shape[0]:,}</strong> rows × <strong>{df.shape[1]:,}</strong> columns
        </p>
        <p style="font-size: 14px; opacity: 0.9;">
            Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        download_button(df, "csv", "📄 Download CSV", key="export_csv")
    
    with col2:
        download_button(df, "excel", "📊 Download Excel", key="export_excel")
    
    with col3:
        download_button(df, "json", "📋 Download JSON", key="export_json")
    
    st.markdown("---")
    
    # Operation History
    show_history()
    
    st.markdown("---")
    
    # Export Log
    if "history" in st.session_state and st.session_state.history:
        st.markdown("### 📜 Export Operation Log")
        
        if st.button("📥 Download Operation Log", key="export_log", use_container_width=True):
            log_df = pd.DataFrame([{
                "Timestamp": h["time"].strftime('%Y-%m-%d %H:%M:%S'),
                "Operation": h["action"],
                "Rows": h["df"].shape[0],
                "Columns": h["df"].shape[1]
            } for h in st.session_state.history])
            
            download_button(log_df, "csv", "Download Log", key="log_csv")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 40px 20px;">
    <h3 style="margin-bottom: 15px;">🤖 AI-Powered Data Analytics Platform</h3>
    <p style="font-size: 16px; opacity: 0.9; margin-bottom: 10px;">
        Transform • Analyze • Visualize • Predict
    </p>
    <p style="font-size: 14px; opacity: 0.75;">
        Built with ❤️ using Streamlit, Scikit-learn & Plotly
    </p>
</div>
""", unsafe_allow_html=True)
