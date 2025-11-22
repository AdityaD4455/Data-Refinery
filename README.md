🚀 Data-Refinery
A Streamlit-powered application for automated data cleaning, preprocessing, EDA, and visualization.
⭐ Overview

Data-Refinery is a lightweight yet powerful data-cleaning web application built using Streamlit.
It enables users to upload datasets, clean them using multiple preprocessing techniques, explore data visually, and download the refined output — all through an intuitive interface.

🔧 Core Features
🧹 Data Cleaning

Drop selected columns
Handle missing values (drop / fill)
Convert data types
Remove duplicates (full or column-based)
Z-score based outlier detection
Replace categorical values with user-defined numeric mapping
Label Encoding & One-Hot Encoding
Column substring modification with new column creation

📊 Exploratory Data Analysis (EDA)
Dataset shape
Missing value summary
Basic statistics
Correlation matrix
Data-type overview

📈 Visualizations
Interactive charts:
Histogram
Scatter Plot
Bar Chart
Line Plot
Box Plot

Users can choose axes and chart types dynamically.

💾 Output
Download cleaned dataset as a CSV
Real-time preview before export

🛠 Tech Stack
Python 3.x
Pandas, NumPy
Scikit-learn
SciPy
Streamlit
Plotly / Matplotlib


📂 Project Structure
Data-Refinery
│── data_cleaning.py
│── requirements.txt
│── README.md


▶️ Run Locally
1. Install dependencies:
pip install -r requirements.txt

2. Start the app:
streamlit run data_cleaning.py

3. Access in browser:

http://localhost:8501

🌐 Deploying

You can deploy on Streamlit Cloud, Render, or HuggingFace Spaces:

Push project to GitHub

Select platform → Import repo

Choose data_cleaning.py as entry file

Deploy

You will get a public URL for sharing.
