# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Advanced Data Cleaning App")

########################
# Utility functions
########################

def to_bytes_download(df, fmt="csv"):
    """Return a tuple (bytes, filename) for download"""
    buf = io.BytesIO()
    fname = f"data_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if fmt == "csv":
        s = df.to_csv(index=False).encode("utf-8")
        return s, fname + ".csv", "text/csv"
    elif fmt == "excel":
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        return buf.getvalue(), fname + ".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif fmt == "json":
        s = df.to_json(orient="records").encode("utf-8")
        return s, fname + ".json", "application/json"
    elif fmt == "parquet":
        try:
            df.to_parquet(buf, index=False)
            return buf.getvalue(), fname + ".parquet", "application/octet-stream"
        except Exception as e:
            st.error("Parquet export requires 'pyarrow' or 'fastparquet'. If not installed, export to CSV/Excel/json instead.")
            return None, None, None
    else:
        return None, None, None

def download_button(df, fmt="csv", label=None):
    b, filename, mime = to_bytes_download(df, fmt)
    if b is None:
        return
    st.download_button(label=label or f"Download ({fmt})", data=b, file_name=filename, mime=mime)

def push_history(df, action_desc):
    """Push a copy of dataframe and action description into session_state history."""
    if "history" not in st.session_state:
        st.session_state.history = []
    # store small sample for display? we'll store full df (be mindful for very big data)
    st.session_state.history.append({"df": df.copy(), "action": action_desc, "time": datetime.now()})

def undo():
    """Undo last operation - pop history and set data to previous snapshot."""
    if "history" in st.session_state and len(st.session_state.history) > 1:
        # drop last
        st.session_state.history.pop()
        last = st.session_state.history[-1]
        st.session_state.df = last["df"].copy()
        st.success(f"Undid last action. Current state: {last['action']}")
    else:
        st.warning("No more steps to undo.")

def redo():
    # Simple redo would require a separate redo stack. For simplicity, not implemented here.
    st.info("Redo not implemented in this version. Use undo reset to original and reapply steps if needed.")

def show_history():
    if "history" in st.session_state:
        for i, item in enumerate(st.session_state.history[::-1], 1):
            st.write(f"Step {-i}: {item['action']} — {item['time'].strftime('%Y-%m-%d %H:%M:%S')} (rows: {item['df'].shape[0]}, cols: {item['df'].shape[1]})")

########################
# App layout
########################

st.title("Advanced Data Cleaning & EDA App")

# --- Uploads / Merge ---
st.sidebar.header("1) Upload & Manage Datasets")
uploaded_file = st.sidebar.file_uploader("Upload primary CSV", type=["csv", "xlsx", "json"], key="u1")
allow_merge = st.sidebar.checkbox("Upload second file to MERGE (optional)", value=False)
uploaded_file2 = None
if allow_merge:
    uploaded_file2 = st.sidebar.file_uploader("Upload second CSV/XLSX/JSON", type=["csv", "xlsx", "json"], key="u2")

# Load datasets
def load_file(file):
    if file is None:
        return None
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file, encoding="latin1")
        elif name.endswith(".xlsx"):
            return pd.read_excel(file)
        elif name.endswith(".json"):
            return pd.read_json(file)
        else:
            return pd.read_csv(file, encoding="latin1")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

if uploaded_file:
    df = load_file(uploaded_file)
    if df is None:
        st.stop()
    # initialize session_state
    st.session_state.df_original = df.copy()
    st.session_state.df = df.copy()
    st.session_state.history = [{"df": df.copy(), "action": "Loaded original dataset", "time": datetime.now()}]
else:
    st.info("Upload a CSV/XLSX/JSON to start.")
    st.stop()

if uploaded_file2:
    df2 = load_file(uploaded_file2)
else:
    df2 = None

########################
# Multi-page via tabs
########################
tabs = st.tabs(["Preview", "Cleaning", "EDA & Viz", "Transformations", "Sampling & Merge", "Export & History"])
########################
# PREVIEW TAB
########################
with tabs[0]:
    st.subheader("Data Preview")
    st.write("Rows, Columns:", st.session_state.df.shape)
    st.dataframe(st.session_state.df.head(100), use_container_width=True)
    if st.button("Reset to original"):
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.history = [{"df": st.session_state.df.copy(), "action": "Reset to original", "time": datetime.now()}]
        st.experimental_rerun()

########################
# CLEANING TAB
########################
with tabs[1]:
    st.subheader("Cleaning Options")

    df = st.session_state.df

    # Column operations
    st.markdown("**Column Operations**")
    cols = df.columns.tolist()
    remove_cols = st.multiselect("Drop columns", cols)
    if st.button("Drop selected columns"):
        if remove_cols:
            push_history(df, f"Dropped columns: {remove_cols}")
            df = df.drop(columns=remove_cols)
            st.session_state.df = df
            st.success(f"Dropped: {remove_cols}")
        else:
            st.warning("No columns selected")

    # Null handling
    st.markdown("**Missing Value Handling**")
    mv_strategy = st.selectbox("Choose missing value strategy", ["None", "Drop rows with any nulls", "Fill with mean (numeric)", "Fill with median (numeric)", "Fill with mode", "Forward fill (ffill)", "Backward fill (bfill)", "Custom value"])
    if mv_strategy == "Custom value":
        custom_value = st.text_input("Enter custom replacement value (applied to all nulls):")
    if st.button("Apply missing value strategy"):
        push_history(df, f"Missing strategy: {mv_strategy}")
        if mv_strategy == "Drop rows with any nulls":
            df = df.dropna()
        elif mv_strategy == "Fill with mean (numeric)":
            for c in df.select_dtypes(include=np.number).columns:
                df[c] = df[c].fillna(df[c].mean())
        elif mv_strategy == "Fill with median (numeric)":
            for c in df.select_dtypes(include=np.number).columns:
                df[c] = df[c].fillna(df[c].median())
        elif mv_strategy == "Fill with mode":
            for c in df.columns:
                df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else np.nan)
        elif mv_strategy == "Forward fill (ffill)":
            df = df.fillna(method="ffill")
        elif mv_strategy == "Backward fill (bfill)":
            df = df.fillna(method="bfill")
        elif mv_strategy == "Custom value":
            if custom_value == "":
                st.warning("Enter a replacement value.")
            else:
                df = df.fillna(custom_value)
        st.session_state.df = df
        st.success("Missing value strategy applied.")

    # String trimming and column name cleaning
    st.markdown("**Column name & string cleaning**")
    if st.button("Clean column names (strip, lower, replace spaces -> _ )"):
        push_history(df, "Cleaned column names")
        df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
        st.session_state.df = df
        st.success("Column names cleaned.")

    if st.button("Trim whitespace in all string columns"):
        push_history(df, "Trim whitespaces")
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
        st.session_state.df = df
        st.success("Trimmed string columns.")

    # Duplicates
    st.markdown("**Duplicates**")
    dupe_cols = st.multiselect("Columns to consider for duplicate detection (leave empty for full row)", cols)
    if st.button("Show duplicate rows"):
        if dupe_cols:
            dupes = df[df.duplicated(subset=dupe_cols, keep=False)]
        else:
            dupes = df[df.duplicated(keep=False)]
        st.write(f"Found {len(dupes)} duplicate rows")
        st.dataframe(dupes.head(50))
    if st.button("Remove duplicates"):
        push_history(df, f"Removed duplicates subset={dupe_cols if dupe_cols else 'all'}")
        if dupe_cols:
            df = df.drop_duplicates(subset=dupe_cols)
        else:
            df = df.drop_duplicates()
        st.session_state.df = df
        st.success("Duplicates removed.")

    # Outlier removal with z-score threshold
    st.markdown("**Outlier Removal (Z-score)**")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    chosen_num_cols = st.multiselect("Numeric columns to use for outlier detection", num_cols)
    z_thresh = st.number_input("Z-score threshold (rows with any selected column |z| > thresh will be removed)", value=3.0)
    if st.button("Remove outliers by z-score"):
        if not chosen_num_cols:
            st.warning("Select numeric columns first.")
        else:
            push_history(df, f"Outlier removal using cols:{chosen_num_cols} thresh:{z_thresh}")
            from scipy.stats import zscore
            z = np.abs(zscore(df[chosen_num_cols].dropna()))
            if z.ndim == 1:
                mask = z < z_thresh
            else:
                mask = (z < z_thresh).all(axis=1)
            df = df.loc[df[chosen_num_cols].dropna().index[mask]]
            st.session_state.df = df
            st.success("Outliers removed (rows dropped).")

    # Undo button
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Undo last step"):
            undo()
    with col2:
        if st.button("Reset to original (clear history)"):
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.history = [{"df": st.session_state.df.copy(), "action": "Reset to original", "time": datetime.now()}]
            st.experimental_rerun()

########################
# EDA & VIZ TAB
########################
with tabs[2]:
    st.subheader("Exploratory Data Analysis & Visualizations")
    df = st.session_state.df

    st.markdown("**Quick statistics**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing cells", int(df.isnull().sum().sum()))

    st.markdown("**Column summary (select column to inspect)**")
    col = st.selectbox("Choose column", df.columns.tolist())
    if col:
        st.write(df[col].describe(include="all"))
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, nbins=40, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.box(df, y=col, title=f"Boxplot of {col}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x=col, y="count", title=f"Counts of {col}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Correlation matrix (numeric columns)**")
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        corr = num.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation heatmap")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns to show correlation.")

    st.markdown("**Scatter plot (choose two numeric columns)**")
    c1, c2 = st.columns(2)
    xcol = c1.selectbox("X axis", num.columns.tolist() if not num.empty else [])
    ycol = c2.selectbox("Y axis", num.columns.tolist() if not num.empty else [])
    if xcol and ycol:
        fig = px.scatter(df, x=xcol, y=ycol, trendline="ols", title=f"Scatter: {xcol} vs {ycol}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pairplot sample (first 6 numeric cols)**")
    if st.button("Show pairplot (may be slow)"):
        to_plot = num.iloc[:, :6]
        if to_plot.shape[1] >= 2:
            fig = px.scatter_matrix(to_plot)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least two numeric columns for pairplot.")

########################
# TRANSFORMATIONS TAB
########################
with tabs[3]:
    st.subheader("Transformations: Encoding, Scaling, Date/Time features, Custom column")

    df = st.session_state.df

    # Encoding
    st.markdown("**Encoding categorical columns**")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    one_hot = st.multiselect("One-hot encode (select columns)", cat_cols)
    label_enc = st.multiselect("Label encode (select columns)", cat_cols)
    if st.button("Apply encodings"):
        push_history(df, f"One-hot: {one_hot} ; Label: {label_enc}")
        if one_hot:
            df = pd.get_dummies(df, columns=one_hot, dummy_na=False)
        if label_enc:
            le = LabelEncoder()
            for c in label_enc:
                try:
                    df[c] = le.fit_transform(df[c].astype(str))
                except Exception as e:
                    st.error(f"Label encoding {c} failed: {e}")
        st.session_state.df = df
        st.success("Encodings applied.")

    # Scaling
    st.markdown("**Scaling numeric columns**")
    scalers = {"None": None, "StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}
    chosen_scaler = st.selectbox("Choose scaler", list(scalers.keys()))
    scale_cols = st.multiselect("Columns to scale", df.select_dtypes(include=np.number).columns.tolist())
    if st.button("Apply scaling"):
        if chosen_scaler == "None" or not scale_cols:
            st.warning("Select a scaler and at least one numeric column.")
        else:
            push_history(df, f"Scaling: {chosen_scaler} on {scale_cols}")
            scaler = scalers[chosen_scaler]()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            st.session_state.df = df
            st.success("Scaling applied.")

    # Date/time features
    st.markdown("**Date/time features**")
    possible_ts = [c for c in df.columns if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()]
    ts_col = st.selectbox("Select timestamp column (auto-detected suggestions shown)", options=[None] + possible_ts)
    if st.button("Extract date/time features"):
        if not ts_col:
            st.warning("Choose a timestamp column first.")
        else:
            push_history(df, f"Extracted date/time features from {ts_col}")
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col])
            df[f"{ts_col}_year"] = df[ts_col].dt.year
            df[f"{ts_col}_month"] = df[ts_col].dt.month
            df[f"{ts_col}_day"] = df[ts_col].dt.day
            df[f"{ts_col}_hour"] = df[ts_col].dt.hour
            df[f"{ts_col}_weekday"] = df[ts_col].dt.weekday
            df[f"{ts_col}_is_weekend"] = df[ts_col].dt.weekday >= 5
            st.session_state.df = df
            st.success("Date/time features extracted.")

    # Custom formula column
    st.markdown("**Create new column from expression**")
    st.write("You can use pandas expressions with column names. Example: (colA + colB)/2 or df['colA'].str[:3]")
    new_col_name = st.text_input("New column name")
    expr = st.text_area("Expression (use 'df' to reference dataframe or column names directly). Example: (df['col1'] + df['col2'])/2")
    if st.button("Create column from expression"):
        if not new_col_name or not expr:
            st.warning("Provide new column name and expression.")
        else:
            push_history(df, f"Created column {new_col_name} from expression")
            try:
                # make df available in eval context
                local_df = df.copy()
                # evaluate safely with pandas available
                result = eval(expr, {"np": np, "pd": pd, "df": local_df})
                df[new_col_name] = result
                st.session_state.df = df
                st.success(f"Column '{new_col_name}' created.")
            except Exception as e:
                st.error(f"Failed to create column: {e}")

########################
# SAMPLING & MERGE TAB
########################
with tabs[4]:
    st.subheader("Sampling, Merge & Advanced ops")
    df = st.session_state.df

    # Sampling
    st.markdown("**Sampling**")
    samp_type = st.selectbox("Sampling type", ["None", "Random sample by fraction", "Random sample by n rows", "Stratified sample by column"])
    if samp_type == "Random sample by fraction":
        frac = st.slider("Fraction", 0.01, 1.0, 0.2)
        if st.button("Apply random fraction sample"):
            push_history(df, f"Random sample fraction={frac}")
            df = df.sample(frac=frac, random_state=42)
            st.session_state.df = df
            st.success("Random fraction sampling applied.")
    elif samp_type == "Random sample by n rows":
        n = st.number_input("Number of rows", min_value=1, max_value=int(df.shape[0]), value=100)
        if st.button("Apply random n sample"):
            push_history(df, f"Random sample n={n}")
            df = df.sample(n=int(n), random_state=42)
            st.session_state.df = df
            st.success("Random n sampling applied.")
    elif samp_type == "Stratified sample by column":
        strat_col = st.selectbox("Choose stratify column", df.columns.tolist())
        frac = st.slider("Fraction per group", 0.01, 1.0, 0.2)
        if st.button("Apply stratified sampling"):
            push_history(df, f"Stratified sample by {strat_col} frac={frac}")
            df = df.groupby(strat_col, group_keys=False).apply(lambda x: x.sample(frac=frac if frac < 1 else 1.0, random_state=42))
            st.session_state.df = df
            st.success("Stratified sampling applied.")

    # Merge (join two uploaded files)
    st.markdown("**Merge two datasets (if second uploaded)**")
    if df2 is not None:
        st.write("Second file uploaded with shape:", df2.shape)
        join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"])
        common_cols = list(set(df.columns).intersection(set(df2.columns)))
        if common_cols:
            left_on = st.selectbox("Select key column from first dataset", common_cols)
            right_on = st.selectbox("Select key column from second dataset", common_cols, index=0)
        else:
            left_on = st.selectbox("Select key column from first dataset", df.columns.tolist())
            right_on = st.selectbox("Select key column from second dataset", df2.columns.tolist())
        if st.button("Apply merge"):
            push_history(df, f"Merged with second dataset on {left_on}=={right_on}, how={join_type}")
            try:
                merged = pd.merge(df, df2, left_on=left_on, right_on=right_on, how=join_type, suffixes=("", "_2"))
                st.session_state.df = merged
                st.success("Merge completed.")
            except Exception as e:
                st.error(f"Merge failed: {e}")
    else:
        st.info("Upload second file in the sidebar to enable merging.")

########################
# EXPORT & HISTORY TAB
########################
with tabs[5]:
    st.subheader("Export cleaned data & Operation history")
    df = st.session_state.df

    st.markdown("**Export options**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        download_button(df, "csv", label="Download CSV")
    with col_b:
        download_button(df, "excel", label="Download Excel (.xlsx)")
    with col_c:
        download_button(df, "json", label="Download JSON")

    if st.button("Download Parquet"):
        b, fname, mime = to_bytes_download(df, "parquet")
        if b:
            st.download_button("Download Parquet", data=b, file_name=fname, mime=mime)

    st.markdown("---")
    st.markdown("**Operation History (undo available)**")
    show_history()
    if st.button("Show last 5 rows of action log"):
        for i, item in enumerate(st.session_state.history[-5:]):
            st.write(f"{i+1}: {item['action']} at {item['time'].strftime('%Y-%m-%d %H:%M:%S')} (shape {item['df'].shape})")

    if st.button("Export operation log as CSV"):
        log_df = pd.DataFrame([{"time": h["time"], "action": h["action"], "rows": h["df"].shape[0], "cols": h["df"].shape[1]} for h in st.session_state.history])
        download_button(log_df, "csv", label="Download log as CSV")

st.sidebar.markdown("---")
st.sidebar.write("App version: 1.0 — built for easy extension")
st.sidebar.write("Tip: inspect the code to add more features like profiling (ydata-profiling), advanced outlier detection (IsolationForest), and model pipeline export.")
