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
# ADVANCED CSS STYLING WITH PERFECT ANIMATIONS
# =====================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Container with Animated Gradient */
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

    /* Glassmorphic Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }

    /* Enhanced Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
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
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
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
        font-size: 14px;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    /* Beautiful Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
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

    /* Animated Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        background: transparent;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
        transform: scale(1.05);
    }

    /* Enhanced DataFrame Styling */
    .dataframe {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(15px);
        border-right: 2px solid rgba(255, 255, 255, 0.2);
        padding: 2rem 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* File Uploader Styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.4);
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: rgba(255, 255, 255, 0.7);
        background: rgba(255, 255, 255, 0.15);
    }

    /* Animated Title */
    h1 {
        animation: fadeInDown 1s ease-out;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        text-align: center;
        padding: 20px 0;
        text-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        letter-spacing: 2px;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Section Headers */
    h2, h3 {
        color: white;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        padding: 15px 0;
        border-bottom: 3px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 20px;
    }

    /* Alert Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px;
        animation: slideInRight 0.5s ease-out;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 18px;
        margin: 15px 0;
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 18px;
        font-weight: 600;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(5px);
    }

    /* Select Box Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
    }

    /* Multi-Select Styling */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 12px;
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }

    /* Slider Styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent !important;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Metric Cards in Grid */
    .css-1r6slb0 {
        gap: 20px;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }

    /* Tooltip Styling */
    .stTooltipIcon {
        color: white;
    }

    /* Code Block Styling */
    code {
        background: rgba(0, 0, 0, 0.3);
        padding: 4px 8px;
        border-radius: 6px;
        color: #f0f0f0;
        font-family: 'Courier New', monospace;
    }

    /* Download Button Special Styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def to_bytes_download(df, fmt="csv"):
    """Convert dataframe to downloadable bytes"""
    buf = io.BytesIO()
    fname = f"data_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        if fmt == "csv":
            s = df.to_csv(index=False).encode("utf-8")
            return s, fname + ".csv", "text/csv"
        elif fmt == "excel":
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
            return buf.getvalue(), fname + ".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif fmt == "json":
            s = df.to_json(orient="records", indent=2).encode("utf-8")
            return s, fname + ".json", "application/json"
        elif fmt == "parquet":
            df.to_parquet(buf, index=False)
            return buf.getvalue(), fname + ".parquet", "application/octet-stream"
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        return None, None, None


def download_button(df, fmt="csv", label=None, key=None):
    """Create styled download button"""
    b, filename, mime = to_bytes_download(df, fmt)
    if b:
        st.download_button(
            label=label or f"📥 Download {fmt.upper()}",
            data=b,
            file_name=filename,
            mime=mime,
            key=key,
            use_container_width=True
        )


def push_history(df, action_desc):
    """Save state to history"""
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "df": df.copy(),
        "action": action_desc,
        "time": datetime.now()
    })


def undo():
    """Undo last operation"""
    if "history" in st.session_state and len(st.session_state.history) > 1:
        st.session_state.history.pop()
        last = st.session_state.history[-1]
        st.session_state.df = last["df"].copy()
        st.success(f"✅ Undone: {last['action']}")
        st.rerun()
    else:
        st.warning("⚠️ No more steps to undo")


def show_history():
    """Display operation history"""
    if "history" in st.session_state and st.session_state.history:
        st.markdown("### 📜 Operation History")
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Step {len(st.session_state.history) - i + 1}: {item['action']}", expanded=(i == 1)):
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{item['df'].shape[0]:,}")
                col2.metric("Columns", item['df'].shape[1])
                col3.write(f"⏰ {item['time'].strftime('%H:%M:%S')}")


def load_file(file):
    """Load uploaded file with error handling"""
    if not file:
        return None

    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file, encoding="utf-8")
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        elif name.endswith(".json"):
            return pd.read_json(file)
        else:
            return pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file, encoding="latin1")
        except:
            st.error("❌ Failed to read file")
            return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def create_metric_card(icon, label, value, delta=None):
    """Create beautiful animated metric card"""
    delta_html = f'<div style="color: #10b981; font-size: 16px; margin-top: 10px;">↗ {delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# AI/ML FUNCTIONS
# =====================================================

def detect_outliers_ai(df, contamination=0.1):
    """AI-powered outlier detection"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns for outlier detection")
        return df, []

    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    outliers = iso_forest.fit_predict(X)

    df['is_outlier'] = outliers == -1
    return df, df[df['is_outlier']].index.tolist()


def auto_data_profiling(df):
    """Generate comprehensive data insights"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())

    insights = {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'missing_cells': missing_cells,
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_columns': len(df.select_dtypes(include=np.number).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
    }

    # Quality score calculation
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    duplicate_ratio = insights['duplicate_rows'] / insights['total_rows'] if insights['total_rows'] > 0 else 0
    quality_score = max(0, 100 - (missing_ratio * 50) - (duplicate_ratio * 50))
    insights['quality_score'] = round(quality_score, 2)

    return insights


def smart_recommendations(df):
    """AI-powered column type recommendations"""
    recommendations = []

    for col in df.columns:
        dtype = df[col].dtype
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0

        if dtype == 'object':
            # Check for datetime
            try:
                pd.to_datetime(df[col].dropna().head(100), errors='raise')
                recommendations.append({
                    'column': col,
                    'current': str(dtype),
                    'recommended': 'datetime',
                    'confidence': 0.9,
                    'reason': 'Contains date/time patterns'
                })
            except:
                # Check for category
                if unique_ratio < 0.05:
                    recommendations.append({
                        'column': col,
                        'current': str(dtype),
                        'recommended': 'category',
                        'confidence': 0.85,
                        'reason': f'Only {df[col].nunique()} unique values ({unique_ratio * 100:.1f}%)'
                    })

    return recommendations


def build_ml_model(df, target_col, model_type='classification'):
    """Train ML model with comprehensive metrics"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocessing
    X_processed = pd.get_dummies(X, drop_first=True)
    X_processed = X_processed.fillna(X_processed.mean())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    if model_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return {
            'model': model,
            'accuracy': model.score(X_test, y_test),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(X_processed.columns, model.feature_importances_))
        }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return {
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(X_processed.columns, model.feature_importances_))
        }


def perform_clustering(df, n_clusters=3, method='kmeans'):
    """Perform clustering with PCA visualization"""
    numeric_df = df.select_dtypes(include=np.number).fillna(0)

    if numeric_df.shape[1] < 2:
        st.warning("⚠️ Need at least 2 numeric columns")
        return None, None

    # Clustering
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = DBSCAN(eps=0.5, min_samples=5)

    clusters = model.fit_predict(numeric_df)

    # PCA for visualization
    if numeric_df.shape[1] > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_df)
    else:
        pca_result = numeric_df.values

    return clusters, pca_result


# =====================================================
# MAIN APPLICATION
# =====================================================

# Animated Header
st.markdown("""
<div style="text-align: center; padding: 30px 0;">
    <h1 style="font-size: 56px; margin: 0; animation: fadeInDown 1s ease;">
        🤖 AI-Powered Data Analytics Platform
    </h1>
    <p style="font-size: 20px; color: white; margin-top: 15px; font-weight: 300; letter-spacing: 1px;">
        Advanced Data Cleaning • Machine Learning • Beautiful Visualizations
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 📁 Data Management")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Primary Dataset",
        type=["csv", "xlsx", "json"],
        help="Supports CSV, Excel, and JSON formats"
    )

    st.markdown("---")

    allow_merge = st.checkbox("🔗 Merge with Second Dataset", value=False)
    uploaded_file2 = None

    if allow_merge:
        uploaded_file2 = st.file_uploader(
            "📂 Upload Second Dataset",
            type=["csv", "xlsx", "json"],
            key="file2"
        )

    st.markdown("---")

    # Quick actions
    st.markdown("### ⚡ Quick Actions")

    if st.button("🔄 Reset to Original", use_container_width=True):
        if 'df_original' in st.session_state:
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.history = [{
                "df": st.session_state.df.copy(),
                "action": "🔄 Reset to original",
                "time": datetime.now()
            }]
            st.success("✅ Reset successful!")
            st.rerun()

    if st.button("↩️ Undo Last Action", use_container_width=True):
        undo()

# =====================================================
# LOAD DATA
# =====================================================

if uploaded_file:
    with st.spinner("🔄 Loading your dataset..."):
        df = load_file(uploaded_file)

        if df is None:
            st.stop()

        if 'df' not in st.session_state:
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            st.session_state.history = [{
                "df": df.copy(),
                "action": "📂 Loaded dataset",
                "time": datetime.now()
            }]
            st.success(f"✅ Successfully loaded {df.shape[0]:,} rows and {df.shape[1]} columns!")
else:
    # Welcome Screen
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 50px;">
        <h2 style="color: white; border: none;">👋 Welcome to AI-Powered Analytics!</h2>
        <p style="color: white; font-size: 18px; margin: 20px 0;">
            Upload your dataset to get started with powerful AI-driven insights
        </p>

        <div style="margin-top: 40px; text-align: left; color: white;">
            <h3 style="border: none;">✨ Key Features:</h3>
            <ul style="font-size: 16px; line-height: 2;">
                <li>🤖 AI-Powered Outlier Detection</li>
                <li>📊 Interactive Visualizations</li>
                <li>🧹 Smart Data Cleaning</li>
                <li>🎯 Machine Learning Models</li>
                <li>📈 Advanced Analytics</li>
                <li>💾 Multiple Export Formats</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if uploaded_file2:
    df2 = load_file(uploaded_file2)
else:
    df2 = None

# =====================================================
# SIDEBAR STATS
# =====================================================

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", f"{st.session_state.df.shape[0]:,}")
        st.metric("Columns", st.session_state.df.shape[1])
    with col2:
        missing_pct = (st.session_state.df.isnull().sum().sum() /
                       (st.session_state.df.shape[0] * st.session_state.df.shape[1]) * 100)
        st.metric("Missing %", f"{missing_pct:.1f}%")
        st.metric("Duplicates", st.session_state.df.duplicated().sum())

# =====================================================
# TABS
# =====================================================

tabs = st.tabs([
    "📋 Overview",
    "🧹 Data Cleaning",
    "📊 EDA & Visualization",
    "🔄 Transformations",
    "🤖 AI/ML Models",
    "🎯 Clustering",
    "⚙️ Advanced Ops",
    "💾 Export"
])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================

with tabs[0]:
    st.markdown("## 📋 Dataset Overview")

    with st.spinner("🔍 Analyzing your data..."):
        insights = auto_data_profiling(st.session_state.df)

    # Metrics Grid
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("📊", "Total Rows", f"{insights['total_rows']:,}")
    with col2:
        create_metric_card("📑", "Total Columns", f"{insights['total_columns']}")
    with col3:
        create_metric_card("⭐", "Quality Score", f"{insights['quality_score']:.1f}%")
    with col4:
        create_metric_card("💾", "Memory", insights['memory_usage'])

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        create_metric_card("❓", "Missing Cells", f"{insights['missing_cells']:,}")
    with col6:
        create_metric_card("🔄", "Duplicates", f"{insights['duplicate_rows']:,}")
    with col7:
        create_metric_card("🔢", "Numeric", f"{insights['numeric_columns']}")
    with col8:
        create_metric_card("🏷️", "Categorical", f"{insights['categorical_columns']}")

    # Data Preview
    st.markdown("---")
    st.markdown("### 🔍 Data Preview")

    with st.container():
        st.dataframe(
            st.session_state.df.head(100),
            use_container_width=True,
            height=400
        )

    # Column Information
    st.markdown("---")
    st.markdown("### 📋 Column Details")

    col_info = pd.DataFrame({
        'Column': st.session_state.df.columns,
        'Type': st.session_state.df.dtypes.astype(str),
        'Non-Null': st.session_state.df.count().values,
        'Null Count': st.session_state.df.isnull().sum().values,
        'Unique': st.session_state.df.nunique().values,
        '% Missing': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2).values
    })

    st.dataframe(col_info, use_container_width=True)

    # AI Recommendations
    st.markdown("---")
    st.markdown("### 🤖 AI Recommendations")

    recommendations = smart_recommendations(st.session_state.df)

    if recommendations:
        st.info(f"💡 Found {len(recommendations)} optimization suggestions")
        for rec in recommendations:
            st.markdown(f"""
            <div class="glass-card">
                <strong>📌 {rec['column']}</strong><br>
                Convert from <code>{rec['current']}</code> to <code>{rec['recommended']}</code><br>
                <small>Confidence: {rec['confidence'] * 100:.0f}% • {rec['reason']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ All column types are optimal!")

# =====================================================
# TAB 2: DATA CLEANING
# =====================================================

with tabs[1]:
    st.markdown("## 🧹 Data Cleaning Operations")
    df = st.session_state.df

    # Column Operations
    with st.expander("🗑️ Remove Columns", expanded=True):
        cols_to_drop = st.multiselect(
            "Select columns to remove",
            df.columns.tolist(),
            help="Choose columns you want to delete"
        )

        if st.button("🗑️ Drop Selected Columns", key="drop_cols"):
            if cols_to_drop:
                push_history(df, f"🗑️ Dropped: {', '.join(cols_to_drop)}")
                df = df.drop(columns=cols_to_drop)
                st.session_state.df = df
                st.success(f"✅ Removed {len(cols_to_drop)} column(s)")
                st.rerun()
            else:
                st.warning("⚠️ No columns selected")

    # Missing Values
    with st.expander("❓ Handle Missing Values", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            strategy = st.selectbox(
                "Choose strategy",
                [
                    "Drop rows with nulls",
                    "Fill with mean",
                    "Fill with median",
                    "Fill with mode",
                    "Forward fill",
                    "Backward fill",
                    "Custom value",
                    "🤖 Smart Fill (AI)"
                ]
            )

        with col2:
            if strategy == "Custom value":
                custom_val = st.text_input("Value", "0")

        if st.button("✨ Apply Strategy", key="missing"):
            push_history(df, f"❓ {strategy}")

            if strategy == "Drop rows with nulls":
                df = df.dropna()
            elif strategy == "Fill with mean":
                for c in df.select_dtypes(include=np.number).columns:
                    df[c].fillna(df[c].mean(), inplace=True)
            elif strategy == "Fill with median":
                for c in df.select_dtypes(include=np.number).columns:
                    df[c].fillna(df[c].median(), inplace=True)
            elif strategy == "Fill with mode":
                for c in df.columns:
                    mode_val = df[c].mode()
                    if not mode_val.empty:
                        df[c].fillna(mode_val.iloc[0], inplace=True)
            elif strategy == "Forward fill":
                df.ffill(inplace=True)
            elif strategy == "Backward fill":
                df.bfill(inplace=True)
            elif strategy == "Custom value":
                df.fillna(custom_val, inplace=True)
            elif strategy == "🤖 Smart Fill (AI)":
                for c in df.columns:
                    if df[c].dtype in [np.float64, np.int64]:
                        df[c].fillna(df[c].median(), inplace=True)
                    else:
                        mode_val = df[c].mode()
                        if not mode_val.empty:
                            df[c].fillna(mode_val.iloc[0], inplace=True)

            st.session_state.df = df
            st.success("✅ Strategy applied successfully!")
            st.rerun()

    # Text Cleaning
    with st.expander("✨ Clean Text & Columns", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔤 Clean Column Names", use_container_width=True):
                push_history(df, "✨ Cleaned column names")
                df.columns = [
                    str(c).strip().lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("(", "").replace(")", "")
                    .replace("[", "").replace("]", "")
                    for c in df.columns
                ]
                st.session_state.df = df
                st.success("✅ Column names cleaned!")
                st.rerun()

        with col2:
            if st.button("✂️ Trim Whitespace", use_container_width=True):
                push_history(df, "✨ Trimmed whitespace")
                for c in df.select_dtypes(include="object").columns:
                    df[c] = df[c].astype(str).str.strip()
                st.session_state.df = df
                st.success("✅ Whitespace removed!")
                st.rerun()

    # Duplicates
    with st.expander("🔄 Handle Duplicates", expanded=False):
        dupe_cols = st.multiselect(
            "Check duplicates based on columns (empty = all)",
            df.columns.tolist()
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👁️ Show Duplicates", use_container_width=True):
                if dupe_cols:
                    dupes = df[df.duplicated(subset=dupe_cols, keep=False)]
                else:
                    dupes = df[df.duplicated(keep=False)]

                st.write(f"Found **{len(dupes)}** duplicate rows")
                if len(dupes) > 0:
                    st.dataframe(dupes.head(50), use_container_width=True)

        with col2:
            if st.button("🗑️ Remove Duplicates", use_container_width=True):
                push_history(df, "🔄 Removed duplicates")
                if dupe_cols:
                    df = df.drop_duplicates(subset=dupe_cols)
                else:
                    df = df.drop_duplicates()
                st.session_state.df = df
                st.success("✅ Duplicates removed!")
                st.rerun()

    # AI Outlier Detection
    with st.expander("🤖 AI-Powered Outlier Detection", expanded=False):
        st.markdown("Uses **Isolation Forest** algorithm for intelligent anomaly detection")

        contamination = st.slider(
            "Expected outlier percentage",
            0.01, 0.5, 0.1,
            help="Proportion of data expected to be outliers"
        )

        if st.button("🔍 Detect Outliers", use_container_width=True):
            with st.spinner("🤖 AI is analyzing patterns..."):
                df_outliers, outlier_idx = detect_outliers_ai(df, contamination)

                st.success(f"🎯 Found **{len(outlier_idx)}** potential outliers")

                if len(outlier_idx) > 0:
                    st.dataframe(
                        df_outliers.loc[outlier_idx].head(50),
                        use_container_width=True
                    )

                    if st.button("❌ Remove Outliers", use_container_width=True):
                        push_history(df, f"🤖 Removed {len(outlier_idx)} outliers")
                        df = df_outliers[~df_outliers['is_outlier']].drop(columns=['is_outlier'])
                        st.session_state.df = df
                        st.success("✅ Outliers removed!")
                        st.rerun()

# =====================================================
# TAB 3: EDA & VISUALIZATION
# =====================================================

with tabs[2]:
    st.markdown("## 📊 Exploratory Data Analysis")
    df = st.session_state.df

    # Statistical Summary
    with st.expander("📈 Statistical Summary", expanded=True):
        st.dataframe(
            df.describe(include='all').transpose(),
            use_container_width=True
        )

    # Column Analysis
    with st.expander("🔍 Deep Dive Analysis", expanded=True):
        selected_col = st.selectbox("Select column to analyze", df.columns.tolist())

        if selected_col:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"### Statistics for `{selected_col}`")
                st.write(df[selected_col].describe())

                missing_pct = (df[selected_col].isnull().sum() / len(df)) * 100
                st.metric("Missing Data", f"{missing_pct:.2f}%")
                st.progress(min(missing_pct / 100, 1.0))

            with col2:
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    # Histogram
                    fig = px.histogram(
                        df, x=selected_col,
                        nbins=50,
                        title=f"Distribution of {selected_col}",
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Box Plot
                    fig2 = px.box(
                        df, y=selected_col,
                        title=f"Box Plot: {selected_col}",
                        color_discrete_sequence=['#764ba2']
                    )
                    fig2.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    # Bar Chart for categorical
                    vc = df[selected_col].value_counts().head(20).reset_index()
                    vc.columns = [selected_col, 'count']

                    fig = px.bar(
                        vc, x=selected_col, y='count',
                        title=f"Top 20 Values: {selected_col}",
                        color='count',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Correlation Analysis
    with st.expander("🔗 Correlation Matrix", expanded=False):
        num_df = df.select_dtypes(include=np.number)

        if not num_df.empty and num_df.shape[1] > 1:
            corr = num_df.corr()

            fig = px.imshow(
                corr,
                text_auto='.2f',
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top correlations
            st.markdown("#### 🔝 Strongest Correlations")
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    corr_pairs.append({
                        'Column 1': corr.columns[i],
                        'Column 2': corr.columns[j],
                        'Correlation': corr.iloc[i, j]
                    })

            corr_df = pd.DataFrame(corr_pairs).sort_values(
                'Correlation', key=abs, ascending=False
            ).head(10)

            st.dataframe(corr_df, use_container_width=True)
        else:
            st.info("ℹ️ Need at least 2 numeric columns")

    # Scatter Plot
    with st.expander("📍 Scatter Plot Analysis", expanded=False):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:
            col1, col2, col3 = st.columns(3)

            with col1:
                x_col = st.selectbox("X-axis", num_cols, key='scatter_x')
            with col2:
                y_col = st.selectbox("Y-axis", num_cols, index=1, key='scatter_y')
            with col3:
                color_by = st.selectbox("Color by", [None] + df.columns.tolist(), key='scatter_color')

            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=color_by,
                trendline="ols",
                title=f"{x_col} vs {y_col}",
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Need at least 2 numeric columns")

# =====================================================
# TAB 4: TRANSFORMATIONS
# =====================================================

with tabs[3]:
    st.markdown("## 🔄 Data Transformations")
    df = st.session_state.df

    # Encoding
    with st.expander("🏷️ Categorical Encoding", expanded=True):
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### One-Hot Encoding")
            one_hot_cols = st.multiselect("Select columns", cat_cols, key='onehot')
            if st.button("Apply One-Hot", use_container_width=True):
                if one_hot_cols:
                    push_history(df, f"🏷️ One-hot: {', '.join(one_hot_cols)}")
                    df = pd.get_dummies(df, columns=one_hot_cols)
                    st.session_state.df = df
                    st.success("✅ Applied!")
                    st.rerun()

        with col2:
            st.markdown("#### Label Encoding")
            label_cols = st.multiselect("Select columns", cat_cols, key='label')
            if st.button("Apply Label Encoding", use_container_width=True):
                if label_cols:
                    push_history(df, f"🏷️ Label: {', '.join(label_cols)}")
                    le = LabelEncoder()
                    for c in label_cols:
                        df[c] = le.fit_transform(df[c].astype(str))
                    st.session_state.df = df
                    st.success("✅ Applied!")
                    st.rerun()

    # Scaling
    with st.expander("⚖️ Feature Scaling", expanded=True):
        scalers = {
            "StandardScaler (z-score)": StandardScaler,
            "MinMaxScaler (0-1)": MinMaxScaler,
            "RobustScaler (outlier-resistant)": RobustScaler
        }

        col1, col2 = st.columns([1, 2])

        with col1:
            scaler_choice = st.selectbox("Scaler", list(scalers.keys()))

        with col2:
            scale_cols = st.multiselect(
                "Columns to scale",
                df.select_dtypes(include=np.number).columns.tolist()
            )

        if st.button("⚖️ Apply Scaling", use_container_width=True):
            if scale_cols:
                push_history(df, f"⚖️ {scaler_choice}")
                scaler = scalers[scaler_choice]()
                df[scale_cols] = scaler.fit_transform(df[scale_cols])
                st.session_state.df = df
                st.success("✅ Scaling applied!")
                st.rerun()

    # DateTime Features
    with st.expander("📅 Extract DateTime Features", expanded=False):
        possible_date_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ['date', 'time', 'year', 'month'])
        ]

        date_col = st.selectbox(
            "Select datetime column",
            [None] + possible_date_cols + df.columns.tolist()
        )

        if st.button("📅 Extract Features", use_container_width=True):
            if date_col:
                push_history(df, f"📅 Extracted from {date_col}")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])

                df[f"{date_col}_year"] = df[date_col].dt.year
                df[f"{date_col}_month"] = df[date_col].dt.month
                df[f"{date_col}_day"] = df[date_col].dt.day
                df[f"{date_col}_hour"] = df[date_col].dt.hour
                df[f"{date_col}_weekday"] = df[date_col].dt.weekday
                df[f"{date_col}_is_weekend"] = df[date_col].dt.weekday >= 5

                st.session_state.df = df
                st.success("✅ Features extracted!")
                st.rerun()
            else:
                st.warning("⚠️ Select a column first")

    # Custom Column
    with st.expander("➕ Create Custom Column", expanded=False):
        st.markdown("Create columns using Python expressions")

        new_col = st.text_input("New column name")
        expr = st.text_area(
            "Expression",
            placeholder="df['col1'] + df['col2']",
            help="Use df['column_name'] syntax"
        )

        if st.button("➕ Create Column", use_container_width=True):
            if new_col and expr:
                push_history(df, f"➕ Created {new_col}")
                try:
                    df[new_col] = eval(expr, {"np": np, "pd": pd, "df": df.copy()})
                    st.session_state.df = df
                    st.success(f"✅ Created '{new_col}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# =====================================================
# TAB 5: ML MODELS
# =====================================================

with tabs[4]:
    st.markdown("## 🤖 Machine Learning Models")
    df = st.session_state.df

    st.info("💡 Train ML models with one click!")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox("Model Type", ["Classification", "Regression"])

    with col2:
        target = st.selectbox("Target Column", df.columns.tolist())

    if st.button("🚀 Train Model", use_container_width=True):
        if target:
            with st.spinner("🤖 Training model..."):
                try:
                    results = build_ml_model(df, target, model_type.lower())

                    st.success("✅ Model trained successfully!")

                    if model_type == "Classification":
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            create_metric_card("🎯", "Accuracy", f"{results['accuracy'] * 100:.2f}%")
                        with col2:
                            create_metric_card("✨", "Precision",
                                               f"{results['report']['weighted avg']['precision'] * 100:.2f}%")
                        with col3:
                            create_metric_card("🔍", "Recall",
                                               f"{results['report']['weighted avg']['recall'] * 100:.2f}%")

                        # Confusion Matrix
                        st.markdown("### Confusion Matrix")
                        fig = px.imshow(
                            results['confusion_matrix'],
                            text_auto=True,
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            create_metric_card("📊", "R² Score", f"{results['r2']:.4f}")
                        with col2:
                            create_metric_card("📉", "RMSE", f"{results['rmse']:.4f}")
                        with col3:
                            create_metric_card("📈", "MSE", f"{results['mse']:.4f}")

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
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# =====================================================
# TAB 6: CLUSTERING
# =====================================================

with tabs[5]:
    st.markdown("## 🎯 Clustering Analysis")
    df = st.session_state.df

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox("Algorithm", ["K-Means", "DBSCAN"])

    with col2:
        if method == "K-Means":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
        else:
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5)

    if st.button("🎯 Perform Clustering", use_container_width=True):
        with st.spinner("🤖 Finding patterns..."):
            clusters, pca_result = perform_clustering(
                df,
                n_clusters if method == "K-Means" else 3,
                method.lower().replace("-", "")
            )

            if clusters is not None:
                st.success("✅ Clustering complete!")

                # Visualization
                plot_df = pd.DataFrame({
                    'PC1': pca_result[:, 0],
                    'PC2': pca_result[:, 1],
                    'Cluster': clusters.astype(str)
                })

                fig = px.scatter(
                    plot_df, x='PC1', y='PC2',
                    color='Cluster',
                    title=f"{method} Clustering Results"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

                # Distribution
                cluster_counts = pd.Series(clusters).value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']

                fig2 = px.pie(
                    cluster_counts,
                    values='Count',
                    names='Cluster',
                    title="Cluster Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)

                if st.button("➕ Add to Dataset", use_container_width=True):
                    push_history(df, f"🎯 Added {method} clusters")
                    st.session_state.df['Cluster'] = clusters
                    st.success("✅ Added!")
                    st.rerun()

# =====================================================
# TAB 7: ADVANCED OPERATIONS
# =====================================================

with tabs[6]:
    st.markdown("## ⚙️ Advanced Operations")
    df = st.session_state.df

    # Sampling
    with st.expander("🎲 Data Sampling", expanded=True):
        sample_type = st.selectbox(
            "Sampling method",
            ["Random fraction", "Random n rows", "Stratified"]
        )

        if sample_type == "Random fraction":
            frac = st.slider("Fraction", 0.01, 1.0, 0.2)
            if st.button("Apply", use_container_width=True):
                push_history(df, f"🎲 Sample {frac * 100}%")
                df = df.sample(frac=frac, random_state=42)
                st.session_state.df = df
                st.success("✅ Sampled!")
                st.rerun()

        elif sample_type == "Random n rows":
            n = st.number_input("Rows", 1, len(df), min(100, len(df)))
            if st.button("Apply", use_container_width=True):
                push_history(df, f"🎲 Sample {n} rows")
                df = df.sample(n=int(n), random_state=42)
                st.session_state.df = df
                st.success("✅ Sampled!")
                st.rerun()

        else:
            strat_col = st.selectbox("Stratify by", df.columns.tolist())
            frac = st.slider("Fraction", 0.01, 1.0, 0.2)
            if st.button("Apply", use_container_width=True):
                push_history(df, f"🎲 Stratified by {strat_col}")
                df = df.groupby(strat_col, group_keys=False).apply(
                    lambda x: x.sample(frac=min(frac, 1.0), random_state=42)
                )
                st.session_state.df = df
                st.success("✅ Sampled!")
                st.rerun()

    # Merge
    with st.expander("🔗 Merge Datasets", expanded=False):
        if df2 is not None:
            st.info(f"Second dataset: {df2.shape[0]} rows × {df2.shape[1]} cols")

            col1, col2, col3 = st.columns(3)

            with col1:
                join_type = st.selectbox("Join", ["inner", "left", "right", "outer"])
            with col2:
                left_key = st.selectbox("Left key", df.columns.tolist())
            with col3:
                right_key = st.selectbox("Right key", df2.columns.tolist())

            if st.button("🔗 Merge", use_container_width=True):
                push_history(df, f"🔗 Merged on {left_key}={right_key}")
                try:
                    merged = pd.merge(
                        df, df2,
                        left_on=left_key,
                        right_on=right_key,
                        how=join_type,
                        suffixes=("", "_2")
                    )
                    st.session_state.df = merged
                    st.success(f"✅ Merged! New shape: {merged.shape}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")
        else:
            st.info("👈 Upload second dataset in sidebar")

# =====================================================
# TAB 8: EXPORT
# =====================================================

with tabs[7]:
    st.markdown("## 💾 Export Your Data")
    df = st.session_state.df

    st.markdown("### 📥 Download Options")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        download_button(df, "csv", "📄 CSV", key="csv")
    with col2:
        download_button(df, "excel", "📊 Excel", key="excel")
    with col3:
        download_button(df, "json", "📋 JSON", key="json")
    with col4:
        download_button(df, "parquet", "📦 Parquet", key="parquet")

    st.markdown("---")

    # History
    show_history()

    st.markdown("---")

    # Export log
    st.markdown("### 📜 Export Operation Log")

    if st.button("📥 Download Log", use_container_width=True):
        if "history" in st.session_state:
            log = pd.DataFrame([{
                "Time": h["time"].strftime('%Y-%m-%d %H:%M:%S'),
                "Action": h["action"],
                "Rows": h["df"].shape[0],
                "Columns": h["df"].shape[1]
            } for h in st.session_state.history])

            download_button(log, "csv", "Download Log CSV", key="log")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 30px;">
    <h3>🎨 AI-Powered Data Analytics Platform v2.0</h3>
    <p>Built with Streamlit • Scikit-learn • Plotly</p>
    <p style="font-size: 14px; opacity: 0.8;">Made with ❤️ for data enthusiasts worldwide</p>
</div>
""", unsafe_allow_html=True)
