# 🚀 ML Analytics Pro v3.0 — Elite Edition

> **Professional • Fast • Accurate • Beautiful**  
> *Upload your data. Clean it. Visualize it. Train ML models. Get AI insights — all in one place.*

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Features](#-live-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [All 13 Tabs Explained](#-all-13-tabs-explained)
- [Supported File Formats](#-supported-file-formats)
- [AI Assistant Setup](#-ai-assistant-setup)
- [Optional Dependencies](#-optional-dependencies)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)

---

## 🌟 Overview

**ML Analytics Pro** is a fully interactive, no-code machine learning web application built with **Streamlit**. It lets anyone — from data analysts to ML engineers — upload a dataset and go from raw data to trained models in minutes, all inside a beautiful dark-themed UI.

No Jupyter notebooks. No manual code. Just upload and explore.

---

## ✨ Live Features

| Feature | Details |
|---|---|
| 📁 **Multi-format Upload** | CSV, Excel, JSON, Parquet |
| 🧹 **Smart Data Cleaning** | Missing values, duplicates, outliers, encoding |
| 📊 **10+ Chart Types** | Histogram, scatter, heatmap, box plot, violin & more |
| 🤖 **10+ ML Models** | Random Forest, XGBoost, LightGBM, SVM, Neural Net... |
| 🏆 **AutoML** | Auto-trains all models, picks the best one |
| 🎯 **Live Prediction** | Enter values → get prediction instantly |
| 🧠 **SHAP Explainability** | Understand *why* a model made a prediction |
| 🔬 **Statistical Tests** | t-test, chi-square, ANOVA, Shapiro-Wilk |
| 🗄️ **SQL on DataFrames** | Write SQL queries directly on your data |
| 💬 **AI Chat Assistant** | Ask questions about your data using LLaMA 3.3 (Groq) |
| 📜 **Operation History** | Undo any change, restore any previous state |
| 💾 **Export Anywhere** | CSV, Excel, JSON, Parquet |

---

## 🛠 Tech Stack

### Core
| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `plotly` | Interactive charts & visualizations |

### Machine Learning
| Library | Purpose |
|---|---|
| `scikit-learn` | Core ML models, preprocessing, metrics |
| `xgboost` | XGBoost classifier & regressor |
| `lightgbm` | LightGBM classifier & regressor |

### Statistics & Analysis
| Library | Purpose |
|---|---|
| `scipy` | t-test, ANOVA, chi-square, normality tests |
| `shap` *(optional)* | Model explainability (SHAP values) |

### AI & Automation
| Library | Purpose |
|---|---|
| `requests` | Groq API calls for AI chat (LLaMA 3.3) |
| `optuna` *(optional)* | Hyperparameter auto-tuning |
| `pandasql` *(optional)* | SQL queries on DataFrames |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ml-analytics-pro.git
cd ml-analytics-pro
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### `requirements.txt`
```
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
lightgbm
scipy
requests
shap
optuna
pandasql
openpyxl
pyarrow
```

---

## ▶️ How to Run

```bash
streamlit run data_cleaning.py
```

Then open your browser at: **`http://localhost:8501`**

---

## 🗂 All 13 Tabs Explained

### 1. 📊 Overview
Get a bird's-eye view of your dataset.
- Total rows, columns, missing %, memory usage, duplicates
- Column-wise summary: dtype, non-null count, unique values
- Descriptive statistics with coefficient of variation
- Data types distribution — Pie chart

---

### 2. 🔧 Clean
Fix your messy data with one click.
- **Missing Values** — Fill with mean, median, mode, forward/backward fill, KNN Imputer, or custom value
- **Duplicates** — Detect and remove (full or subset-based)
- **Outliers** — Remove using IQR (1.5×/3×), Z-Score (2σ/3σ), Isolation Forest, or Modified Z-Score
- **Encoding** — Label Encoding, One-Hot Encoding, Frequency Encoding for categorical columns

---

### 3. 📈 Visualize
Explore your data visually.
- Histogram, Scatter Plot, Box Plot, Violin Plot
- Heatmap (Correlation Matrix)
- Bar Chart, Pie Chart, Line Chart
- KDE distribution, Pair plots
- All charts are interactive (zoom, hover, download)

---

### 4. 🤖 ML Models
Train and compare multiple models side-by-side.

**Classification Models:** Random Forest, XGBoost, LightGBM, Extra Trees, Gradient Boosting, Logistic Regression, Neural Network, KNN, SVM, Naive Bayes

**Regression Models:** Random Forest, XGBoost, LightGBM, Extra Trees, Gradient Boosting, Linear Regression, Ridge, Lasso, ElasticNet, Neural Network, SVR

**Features:**
- Auto-detect problem type (Classification vs Regression)
- Cross-validation (5-fold Stratified/KFold)
- Metrics: Accuracy, F1, AUC, Precision, Recall / R², RMSE, MAE, MAPE
- Confusion matrix, ROC curve, Residual plots

---

### 5. 🎯 Predict
Use your trained model for real-time predictions.
- Select a trained model
- Fill in feature values using interactive input fields
- Get prediction result with confidence score instantly

---

### 6. 🧬 Features
Understand and select the most important features.
- **Feature Importance** — Bar chart from tree-based models
- **SelectKBest** — Auto-select top K features
- **PCA** — Principal Component Analysis (dimensionality reduction)
- **t-SNE** — 2D visual clustering of data
- **ICA** — Independent Component Analysis

---

### 7. ⚙️ Advanced
Fine-grained data transformations.
- **Scaling** — StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
- **Dtype Auto-Fix** — Automatically correct column data types
- **Column Operations** — Rename, drop, reorder columns
- **Custom Transformations** — Apply math operations on columns

---

### 8. 🏆 AutoML
Let the app find the best model for you.
- Trains all available models automatically
- Ranks models by performance score
- Optuna-powered hyperparameter tuning *(if installed)*
- Creates Voting Ensemble from top models
- One-click best model selection

---

### 9. 🔬 Stats Tests
Run proper statistical hypothesis tests.
| Test | Use Case |
|---|---|
| **t-test** | Compare means of 2 groups |
| **Chi-Square** | Test independence between categorical variables |
| **ANOVA** | Compare means across 3+ groups |
| **Mann-Whitney U** | Non-parametric alternative to t-test |
| **Shapiro-Wilk** | Check if data is normally distributed |

---

### 10. 🗄️ SQL Query
Write SQL directly on your DataFrame.
- Uses `pandasql` under the hood
- Full SELECT, WHERE, GROUP BY, ORDER BY support
- Query history saved
- Results displayed as interactive table
- Example: `SELECT age, AVG(salary) FROM df GROUP BY age`

---

### 11. 🧠 SHAP
Understand why your model made a prediction.
- Works with tree-based models (Random Forest, XGBoost, LightGBM)
- SHAP Summary Plot — which features matter most overall
- SHAP Waterfall Plot — why a single prediction was made
- Force Plot — visual breakdown per prediction

---

### 12. 💬 AI Assistant
Chat with an AI about your data.
- Powered by **Groq API** → **LLaMA 3.3 70B** model
- Context-aware: AI knows your column names, shapes, stats
- Quick prompts: "What patterns exist?", "Which model is best?", "Any data quality issues?"
- Chat history export as `.txt`
- Requires a free [Groq API key](https://console.groq.com/keys)

---

### 13. 💾 Export
Save your cleaned data and models.
- **CSV** — Universal format
- **Excel (.xlsx)** — For business reports
- **JSON** — For web/API use
- **Parquet** — For big data / cloud pipelines
- **Operation History** — See every change made, restore any previous version

---

## 📁 Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | Most common, recommended |
| Excel | `.xlsx`, `.xls` | Multi-sheet supported |
| JSON | `.json` | Records or columns orientation |
| Parquet | `.parquet` | Fast, compressed format |

---

## 🤖 AI Assistant Setup

1. Go to [console.groq.com/keys](https://console.groq.com/keys) — it's **free**
2. Create an API key
3. In the app sidebar → paste your Groq API key
4. Go to **💬 AI Assistant** tab and start chatting!

> Model used: `llama-3.3-70b-versatile` via Groq

---

## 🔧 Optional Dependencies

These features activate automatically if the library is installed:

| Library | Feature Unlocked |
|---|---|
| `shap` | 🧠 SHAP Tab — model explainability |
| `optuna` | 🏆 AutoML — hyperparameter tuning |
| `pandasql` | 🗄️ SQL Query tab |

Install all at once:
```bash
pip install shap optuna pandasql
```

---

## 📂 Project Structure

```
ml-analytics-pro/
│
├── data_cleaning.py        # Main application file (all 13 tabs)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 💡 Tips for Best Results

- **For Classification** — Make sure target column has categorical/binary values
- **For Regression** — Target column should be numeric and continuous
- **Large datasets** — Use Parquet format for faster loading
- **SHAP** — Works best with tree-based models (Random Forest, XGBoost)
- **AutoML** — Start here if you're unsure which model to use

---

## 🙌 Built With

```
Streamlit · Pandas · NumPy · Scikit-learn
XGBoost · LightGBM · Plotly · SciPy
SHAP · Optuna · PandasSQL · Groq API
```

---

<div align="center">

**⭐ If this project helped you, give it a star!**

Made with ❤️ using Python & Streamlit

</div>
