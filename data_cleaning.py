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
                              precision_score, recall_score, f1_score,
                              mean_squared_error, r2_score, mean_absolute_error)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    layout="wide",
    page_title="🚀 ML Analytics Pro",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Initialize session state
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
if 'last_change' not in st.session_state:
    st.session_state.last_change = None

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    * { font-family: 'Poppins', sans-serif; }
    
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
        transition: all 0.4s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.25);
    }

    .change-summary {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        color: white;
        animation: slideIn 0.5s ease;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
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
    }

    .metric-container:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
    }

    h1, h2, h3 { color: white !important; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2); }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(15px);
    }

    section[data-testid="stSidebar"] * { color: white !important; }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 18px;
        margin: 15px 0;
        color: white;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 10px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
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
            <h4>📊 Recent Changes: {ch['action']}</h4>
            <p><strong>Rows:</strong> {ch['rows_before']:,} → {ch['rows_after']:,} 
               ({'+' if rows_diff >= 0 else ''}{rows_diff:,})</p>
            <p><strong>Columns:</strong> {ch['cols_before']:,} → {ch['cols_after']:,} 
               ({'+' if cols_diff >= 0 else ''}{cols_diff:,})</p>
            <p><strong>Time:</strong> {ch['timestamp'].strftime('%H:%M:%S')}</p>
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
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                st.download_button(label, buffer.getvalue(), 
                                 f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                 "application/vnd.ms-excel", key=key, use_container_width=True)
            except:
                csv = df.to_csv(index=False)
                st.download_button(label + " (CSV)", csv, 
                                 f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                                 "text/csv", key=key+"_csv", use_container_width=True)
        elif format_type == "json":
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(label, json_str, 
                             f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                             "application/json", key=key, use_container_width=True)
    except Exception as e:
        st.error(f"Download error: {str(e)}")

def detect_problem_type(df, target_col):
    """Detect classification vs regression"""
    if target_col not in df.columns:
        return 'classification'
    
    if df[target_col].dtype == 'object':
        return 'classification'
    
    unique_ratio = df[target_col].nunique() / len(df[target_col].dropna())
    
    if df[target_col].nunique() < 15 or unique_ratio < 0.05:
        return 'classification'
    return 'regression'

def prepare_ml_data(df, target_col, feature_cols):
    """Prepare data with proper encoding"""
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Fill missing features
    for col in feature_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.int64]:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna('missing', inplace=True)
    
    # Encode features
    X = df_clean[feature_cols].copy()
    feature_encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            feature_encoders[col] = le
    
    # Encode target
    y = df_clean[target_col]
    target_encoder = None
    problem_type = detect_problem_type(df_clean, target_col)
    
    if problem_type == 'classification' and y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    return X, y, feature_encoders, target_encoder, problem_type

# Header
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="font-size: 48px; font-weight: 800;">🚀 ML Analytics Pro</h1>
    <p style="font-size: 18px; color: rgba(255, 255, 255, 0.9);">
        Production-Ready Machine Learning Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader("Upload dataset", type=['csv', 'xlsx', 'json'], key="main_upload")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            if st.session_state.df is None or len(st.session_state.history) == 0:
                st.session_state.df = df
                push_history(df, "📁 Dataset uploaded")
                st.success(f"✅ {df.shape[0]:,} × {df.shape[1]:,}")
            elif st.session_state.df is not None:
                if st.button("Replace current data?", use_container_width=True):
                    st.session_state.df = df
                    push_history(df, "📁 Dataset replaced")
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
            'Age': np.random.randint(18, 70, 1000),
            'Income': np.random.randint(20000, 150000, 1000),
            'Score': np.random.randint(50, 100, 1000),
            'Experience': np.random.randint(0, 30, 1000),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'Target': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
        })
        st.session_state.df = sample_df
        push_history(sample_df, "🎲 Sample data loaded")
        st.rerun()

# Main content
if st.session_state.df is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 60px;">
        <h2>👈 Upload Your Dataset to Begin</h2>
        <p style="font-size: 18px; opacity: 0.9; margin: 20px 0;">
            Powerful ML • Real-time Updates • Accurate Predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# Show recent changes
show_change_summary()

# Tabs
tabs = st.tabs(["📊 Overview", "🔧 Cleaning", "📈 Viz", "🤖 ML", "🎯 Predict", "🧬 Features", "⚙️ Advanced", "💾 Export"])

# TAB 1: Overview
with tabs[0]:
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 40px;">📝</div>
            <div style="font-size: 14px; opacity: 0.9;">ROWS</div>
            <div style="font-size: 32px; font-weight: 700;">{df.shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 40px;">🔢</div>
            <div style="font-size: 14px; opacity: 0.9;">COLUMNS</div>
            <div style="font-size: 32px; font-weight: 700;">{df.shape[1]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 40px;">❓</div>
            <div style="font-size: 14px; opacity: 0.9;">MISSING</div>
            <div style="font-size: 32px; font-weight: 700;">{missing_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 40px;">💾</div>
            <div style="font-size: 14px; opacity: 0.9;">MEMORY</div>
            <div style="font-size: 32px; font-weight: 700;">{memory_mb:.1f} MB</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 👀 Data Preview")
    n_rows = st.slider("Rows to display", 5, 50, 10, key="preview_rows")
    st.dataframe(df.head(n_rows), use_container_width=True, height=400)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Column Info")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notnull().sum(),
            'Null': df.isnull().sum(),
            'Unique': df.nunique()
        })
        st.dataframe(info_df, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Type Distribution")
        type_counts = df.dtypes.value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index.astype(str), 
                    title="Column Types", hole=0.4)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📈 Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

# TAB 2: Cleaning
with tabs[1]:
    st.markdown("## 🔧 Data Cleaning")
    
    # Missing values
    with st.expander("❓ Missing Values", expanded=True):
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
                target_col = st.selectbox("Column", missing_df['Column'].tolist(), key="missing_col")
            with col2:
                strategy = st.selectbox("Strategy", 
                    ["Drop rows", "Fill mean", "Fill median", "Fill mode", "Forward fill", "Fill value"],
                    key="missing_strategy")
            
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
                    elif strategy == "Fill value":
                        df_clean[target_col].fillna(fill_val, inplace=True)
                    
                    push_history(df_clean, f"🔧 {strategy} - {target_col}")
                    st.session_state.df = df_clean
                    st.success(f"✅ Applied {strategy}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.success("✅ No missing values!")
    
    # Duplicates
    with st.expander("🔄 Duplicates", expanded=True):
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠️ {dup_count:,} duplicates ({dup_count/len(df)*100:.2f}%)")
            if st.button("🗑️ Remove Duplicates", key="remove_dups", use_container_width=True):
                df_clean = df.drop_duplicates()
                push_history(df_clean, f"🗑️ Removed {dup_count} duplicates")
                st.session_state.df = df_clean
                st.success(f"✅ Removed {dup_count} duplicates!")
                st.rerun()
        else:
            st.success("✅ No duplicates!")
    
    # Outliers
    with st.expander("🎯 Outliers", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                outlier_col = st.selectbox("Column", numeric_cols, key="outlier_col")
            with col2:
                method = st.selectbox("Method", ["IQR", "Z-Score", "Isolation Forest"], key="outlier_method")
            
            if outlier_col:
                fig = px.box(df, y=outlier_col, title=f"Outliers in {outlier_col}")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🗑️ Remove Outliers", key="remove_outliers", use_container_width=True):
                    df_clean = df.copy()
                    try:
                        if method == "IQR":
                            Q1 = df_clean[outlier_col].quantile(0.25)
                            Q3 = df_clean[outlier_col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_clean = df_clean[
                                (df_clean[outlier_col] >= Q1 - 1.5 * IQR) &
                                (df_clean[outlier_col] <= Q3 + 1.5 * IQR)
                            ]
                        elif method == "Z-Score":
                            z = np.abs((df_clean[outlier_col] - df_clean[outlier_col].mean()) / df_clean[outlier_col].std())
                            df_clean = df_clean[z < 3]
                        else:
                            iso = IsolationForest(contamination=0.1, random_state=42)
                            outliers = iso.fit_predict(df_clean[[outlier_col]])
                            df_clean = df_clean[outliers == 1]
                        
                        removed = len(df) - len(df_clean)
                        push_history(df_clean, f"🎯 Removed {removed} outliers ({method})")
                        st.session_state.df = df_clean
                        st.success(f"✅ Removed {removed} outliers!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Encoding
    with st.expander("🏷️ Encode Categorical", expanded=False):
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            col1, col2 = st.columns(2)
            with col1:
                encode_col = st.selectbox("Column", cat_cols, key="encode_col")
            with col2:
                encode_type = st.selectbox("Type", ["Label Encoding", "One-Hot"], key="encode_type")
            
            if encode_col:
                st.write(f"**Unique: {df[encode_col].nunique()}**")
                st.write(df[encode_col].value_counts().head(10))
            
            if st.button("🏷️ Encode", key="encode_btn", use_container_width=True):
                df_encode = df.copy()
                try:
                    if encode_type == "Label Encoding":
                        le = LabelEncoder()
                        df_encode[encode_col] = le.fit_transform(df_encode[encode_col].astype(str))
                    else:
                        df_encode = pd.get_dummies(df_encode, columns=[encode_col], prefix=encode_col)
                    
                    push_history(df_encode, f"🏷️ {encode_type} - {encode_col}")
                    st.session_state.df = df_encode
                    st.success(f"✅ Encoded {encode_col}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# TAB 3: Visualization
with tabs[2]:
    st.markdown("## 📈 Visualizations")
    
    viz_type = st.selectbox("Type", 
        ["Distribution", "Correlation", "Scatter", "Box Plot", "Histogram"],
        key="viz_type")
    
    if viz_type == "Distribution":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Column", numeric_cols, key="dist_col")
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
            fig.add_trace(go.Histogram(x=df[col], marker_color='#667eea'), row=1, col=1)
            fig.add_trace(go.Box(y=df[col], marker_color='#764ba2'), row=1, col=2)
            fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{df[col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[col].median():.2f}")
            with col3:
                st.metric("Std", f"{df[col].std():.2f}")
    
    elif viz_type == "Correlation":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="Correlation Matrix")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white'), height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="scatter_y")
            
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}", 
                           trendline="ols", opacity=0.7)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white'), height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            y_col = st.selectbox("Column", numeric_cols, key="box_y")
            fig = px.box(df, y=y_col, title=f"Distribution of {y_col}", points="outliers")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white'), height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Column", numeric_cols, key="hist_col")
            bins = st.slider("Bins", 10, 100, 30, key="hist_bins")
            fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram - {col}")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white'), height=500)
            st.plotly_chart(fig, use_container_width=True)

# TAB 4: ML Models
with tabs[3]:
    st.markdown("## 🤖 Machine Learning")
    
    all_cols = df.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Need at least 2 columns")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("🎯 Target", all_cols, key="ml_target")
    with col2:
        available_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect("📊 Features", available_features, 
                                     default=available_features[:min(10, len(available_features))],
                                     key="ml_features")
    
    if not feature_cols:
        st.warning("Select features")
        st.stop()
    
    try:
        X, y, feature_encoders, target_encoder, problem_type = prepare_ml_data(df, target_col, feature_cols)
        
        st.markdown(f"""
        <div class="insight-box">
            <strong>🤖 Problem Type: {problem_type.upper()}</strong><br>
            Target: {target_col} | Features: {len(feature_cols)} | Samples: {len(X):,}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Data prep error: {str(e)}")
        st.stop()
    
    # Model selection
    if problem_type == 'classification':
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }
    
    selected_models = st.multiselect("Models", list(models.keys()), 
                                    default=list(models.keys())[:3], key="selected_models")
    
    test_size = st.slider("Test Size %", 10, 40, 20, key="test_size") / 100
    
    if st.button("🚀 Train Models", key="train_models", use_container_width=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for idx, model_name in enumerate(selected_models):
            status.text(f"Training {model_name}...")
            
            try:
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if problem_type == 'classification':
                    acc = accuracy_score(y_test, y_pred)
                    avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
                    prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                    rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                    
                    results.append({
                        'Model': model_name,
                        'Accuracy': f"{acc:.4f}",
                        'Precision': f"{prec:.4f}",
                        'Recall': f"{rec:.4f}",
                        'F1': f"{f1:.4f}",
                        'Score': acc
                    })
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results.append({
                        'Model': model_name,
                        'RMSE': f"{rmse:.4f}",
                        'MAE': f"{mae:.4f}",
                        'R²': f"{r2:.4f}",
                        'Score': r2
                    })
                
                st.session_state.trained_models[model_name] = {
                    'model': model,
                    'features': feature_cols,
                    'target': target_col,
                    'type': problem_type,
                    'feature_encoders': feature_encoders,
                    'target_encoder': target_encoder,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred
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
            
            st.markdown("### 🏆 Performance")
            st.dataframe(results_df.drop('Score', axis=1), use_container_width=True)
            
            st.success(f"🏆 Best: {best_model_name} (Score: {results_df.loc[best_idx, 'Score']:.4f})")
            
            # Confusion matrix for classification
            if problem_type == 'classification':
                st.markdown("### 🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, aspect="auto", title=f"{best_model_name}")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)

# TAB 5: Predictions
with tabs[4]:
    st.markdown("## 🎯 Predictions")
    
    if not st.session_state.trained_models:
        st.warning("Train models first")
        st.stop()
    
    model_name = st.selectbox("Model", list(st.session_state.trained_models.keys()), 
                             key="pred_model")
    
    model_info = st.session_state.trained_models[model_name]
    model = model_info['model']
    feature_cols = model_info['features']
    problem_type = model_info['type']
    feature_encoders = model_info['feature_encoders']
    target_encoder = model_info['target_encoder']
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>Model: {model_name}</strong><br>
        Type: {problem_type.title()} | Features: {len(feature_cols)}
    </div>
    """, unsafe_allow_html=True)
    
    pred_mode = st.radio("Mode", ["Single", "Batch"], key="pred_mode", horizontal=True)
    
    if pred_mode == "Single":
        st.markdown("### 📝 Input Values")
        
        input_data = {}
        cols = st.columns(3)
        
        for idx, col in enumerate(feature_cols):
            with cols[idx % 3]:
                if col not in df.columns:
                    continue
                
                if df[col].dtype in [np.float64, np.int64]:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    input_data[col] = st.number_input(col, min_val, max_val, mean_val, 
                                                     key=f"input_{col}")
                else:
                    unique_vals = df[col].unique().tolist()
                    input_data[col] = st.selectbox(col, unique_vals, key=f"input_{col}")
        
        if st.button("🔮 Predict", key="predict_single", use_container_width=True):
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
                    prediction = target_encoder.inverse_transform([int(prediction)])[0]
                
                if problem_type == 'classification':
                    proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
                    
                    st.markdown(f"""
                    <div class="change-summary" style="text-align: center; padding: 40px;">
                        <h2>🎯 Prediction</h2>
                        <h1 style="font-size: 60px; margin: 20px 0;">{prediction}</h1>
                        {f'<p>Confidence: {max(proba)*100:.1f}%</p>' if proba is not None else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if proba is not None and len(proba) > 1:
                        classes = target_encoder.classes_ if target_encoder else [str(i) for i in range(len(proba))]
                        prob_df = pd.DataFrame({'Class': classes, 'Probability': proba})
                        fig = px.bar(prob_df, x='Class', y='Probability', title="Probabilities")
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                        font=dict(color='white'))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="change-summary" style="text-align: center; padding: 40px;">
                        <h2>🎯 Prediction</h2>
                        <h1 style="font-size: 60px; margin: 20px 0;">{prediction:.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    else:
        st.markdown("### 📊 Batch Prediction")
        upload_pred = st.file_uploader("Upload CSV", type=['csv'], key="batch_file")
        
        if upload_pred:
            pred_df = pd.read_csv(upload_pred)
            st.dataframe(pred_df.head(10), use_container_width=True)
            
            if st.button("🔮 Predict All", key="predict_batch", use_container_width=True):
                missing = set(feature_cols) - set(pred_df.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    try:
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
                        
                        st.success(f"✅ Predictions: {len(pred_df):,} rows")
                        st.dataframe(pred_df, use_container_width=True)
                        
                        download_button(pred_df, "csv", "📥 Download", "download_pred")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# TAB 6: Feature Engineering
with tabs[5]:
    st.markdown("## 🧬 Feature Engineering")
    
    with st.expander("🎯 Clustering", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            cols = st.multiselect("Features", numeric_cols, default=numeric_cols[:3], key="cluster_cols")
            
            if cols:
                method = st.selectbox("Method", ["K-Means", "DBSCAN"], key="cluster_method")
                n_clusters = st.slider("Clusters", 2, 10, 3, key="n_clusters")
                
                if st.button("🎯 Cluster", key="cluster_btn", use_container_width=True):
                    try:
                        X = df[cols].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        if method == "K-Means":
                            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                        else:
                            clusterer = DBSCAN(eps=0.5, min_samples=3)
                        
                        clusters = clusterer.fit_predict(X_scaled)
                        
                        if len(cols) >= 2:
                            viz_df = df.copy()
                            viz_df['Cluster'] = -1
                            viz_df.loc[X.index, 'Cluster'] = clusters
                            
                            fig = px.scatter(viz_df, x=cols[0], y=cols[1], color='Cluster', 
                                           title=f"{method} Results")
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                            font=dict(color='white'))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if st.button("➕ Add to Dataset", key="add_cluster", use_container_width=True):
                            df_clustered = df.copy()
                            df_clustered['Cluster'] = -1
                            df_clustered.loc[X.index, 'Cluster'] = clusters
                            push_history(df_clustered, f"🎯 {method} clustering")
                            st.session_state.df = df_clustered
                            st.success("✅ Added!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with st.expander("📉 PCA", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            n_components = st.slider("Components", 2, min(10, len(numeric_cols)), 2, key="pca_n")
            
            if st.button("📉 Apply PCA", key="pca_btn", use_container_width=True):
                try:
                    X = df[numeric_cols].fillna(df[numeric_cols].median())
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    pca = PCA(n_components=n_components, random_state=42)
                    X_reduced = pca.fit_transform(X_scaled)
                    
                    fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], 
                                   title="PCA - First 2 Components")
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                    font=dict(color='white'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    var_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(n_components)],
                        'Variance': pca.explained_variance_ratio_
                    })
                    
                    fig2 = px.bar(var_df, x='Component', y='Variance', title="Explained Variance")
                    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                     font=dict(color='white'))
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    st.metric("Total Variance", f"{pca.explained_variance_ratio_.sum()*100:.2f}%")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# TAB 7: Advanced
with tabs[6]:
    st.markdown("## ⚙️ Advanced Operations")
    
    with st.expander("🎲 Sampling", expanded=True):
        sample_type = st.selectbox("Type", ["Random %", "Random N"], key="sample_type")
        
        if sample_type == "Random %":
            frac = st.slider("Percentage", 1, 100, 20, key="sample_pct") / 100
            if st.button("🎲 Sample", key="sample_pct_btn", use_container_width=True):
                df_sample = df.sample(frac=frac, random_state=42)
                push_history(df_sample, f"🎲 Sampled {frac*100:.0f}%")
                st.session_state.df = df_sample
                st.success(f"✅ Sampled {len(df_sample):,} rows!")
                st.rerun()
        else:
            n = st.number_input("Rows", 1, len(df), min(1000, len(df)), key="sample_n")
            if st.button("🎲 Sample", key="sample_n_btn", use_container_width=True):
                df_sample = df.sample(n=int(n), random_state=42)
                push_history(df_sample, f"🎲 Sampled {n} rows")
                st.session_state.df = df_sample
                st.success(f"✅ Sampled {n} rows!")
                st.rerun()
    
    with st.expander("🔍 Filtering", expanded=True):
        filter_col = st.selectbox("Column", df.columns.tolist(), key="filter_col")
        
        if df[filter_col].dtype in [np.float64, np.int64]:
            min_val = float(df[filter_col].min())
            max_val = float(df[filter_col].max())
            filter_range = st.slider("Range", min_val, max_val, (min_val, max_val), key="filter_range")
            
            if st.button("🔍 Filter", key="filter_btn", use_container_width=True):
                df_filtered = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
                push_history(df_filtered, f"🔍 Filtered {filter_col}")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered):,} rows!")
                st.rerun()
        else:
            unique_vals = df[filter_col].unique().tolist()
            selected = st.multiselect("Values", unique_vals, default=unique_vals[:5], key="filter_vals")
            
            if selected and st.button("🔍 Filter", key="filter_cat_btn", use_container_width=True):
                df_filtered = df[df[filter_col].isin(selected)]
                push_history(df_filtered, f"🔍 Filtered {filter_col}")
                st.session_state.df = df_filtered
                st.success(f"✅ Filtered to {len(df_filtered):,} rows!")
                st.rerun()
    
    with st.expander("🔗 Merge Datasets", expanded=True):
        if st.session_state.df2 is not None:
            df2 = st.session_state.df2
            st.info(f"Second dataset: {df2.shape[0]:,} × {df2.shape[1]:,}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                join_type = st.selectbox("Type", ["inner", "left", "right", "outer"], key="join_type")
            with col2:
                left_key = st.selectbox("Left key", df.columns.tolist(), key="left_key")
            with col3:
                right_key = st.selectbox("Right key", df2.columns.tolist(), key="right_key")
            
            if st.button("🔗 Merge", key="merge_btn", use_container_width=True):
                try:
                    merged = pd.merge(df, df2, left_on=left_key, right_on=right_key, 
                                    how=join_type, suffixes=("", "_2"))
                    push_history(merged, f"🔗 Merged on {left_key}")
                    st.session_state.df = merged
                    st.success(f"✅ Merged! {merged.shape[0]:,} × {merged.shape[1]:,}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("Upload second dataset in sidebar")

# TAB 8: Export
with tabs[7]:
    st.markdown("## 💾 Export Data")
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <h3>📊 Current Dataset</h3>
        <p><strong>{df.shape[0]:,}</strong> rows × <strong>{df.shape[1]:,}</strong> columns</p>
        <p>Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        download_button(df, "csv", "📄 CSV", "export_csv")
    with col2:
        download_button(df, "excel", "📊 Excel", "export_excel")
    with col3:
        download_button(df, "json", "📋 JSON", "export_json")
    
    st.markdown("---")
    
    if st.session_state.history:
        st.markdown("### 📜 History")
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"{h['time'].strftime('%H:%M:%S')} - {h['action']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{h['shape'][0]:,}")
                with col2:
                    st.metric("Cols", f"{h['shape'][1]:,}")
                with col3:
                    if st.button("↩️ Restore", key=f"restore_{i}"):
                        st.session_state.df = h['df'].copy()
                        st.success("✅ Restored!")
                        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 30px;">
    <h3>🚀 ML Analytics Pro</h3>
    <p style="opacity: 0.9;">Production-Ready • Real-time Updates • Accurate Predictions</p>
    <p style="opacity: 0.7; font-size: 14px;">Built with Streamlit, Scikit-learn & Plotly</p>
</div>
""", unsafe_allow_html=True)
