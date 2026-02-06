🚀 Data-Refinery
Data-Refinery is a lightweight yet powerful web application built with Streamlit for automated data cleaning, preprocessing, and exploratory data analysis (EDA). It bridges the gap between raw data and actionable insights by providing a no-code interface for complex data manipulations.

⭐ Overview
Data cleaning accounts for 80% of a data scientist's time. Data-Refinery streamlines this process. Users can upload raw datasets (CSV/Excel), apply advanced cleaning techniques (like Z-score outlier detection and encoding), visualize distributions, and download the refined output instantly.

🔧 Key Features
🧹 Advanced Data Cleaning
Smart Handling: Drop specific columns or handle missing values (Drop rows/Fill with Mean, Median, Mode).

Outlier Detection: Automatically detect and filter outliers using Z-Score analysis.

Duplicate Management: Remove duplicates based on all columns or specific subsets.

String Manipulation: Modify column substrings and generate new feature columns dynamically.

⚙️ Preprocessing & Encoding
Type Conversion: Instantly convert column data types (e.g., String to Float).

Categorical Encoding:

Label Encoding: For ordinal data.

One-Hot Encoding: For nominal data.

Mapping: Replace categorical values with user-defined numeric maps.

📊 Exploratory Data Analysis (EDA)
Health Check: View dataset shape, data types, and missing value summaries.

Statistics: Generate descriptive statistics (mean, std, min, max, quartiles).

Correlation: Interactive heatmap to visualize relationships between variables.

📈 Interactive Visualization
Dynamic plotting using Plotly and Matplotlib:

📊 Distribution: Histograms and Box Plots.

📉 Relationships: Scatter Plots and Line Charts.

📊 Comparison: Bar Charts.

Users can dynamically select X and Y axes for all charts.


🛠 Tech StackCategoryTechnologiesFrameworkStreamlitLanguagePython 3.xData ManipulationPandas, NumPyMachine LearningScikit-learn, SciPyVisualizationPlotly, Matplotlib📂 Project StructurePlaintextData-Refinery/
│
├── app.py                 # Main application entry point
├── requirements.txt       # List of python dependencies
└── README.md              # Project documentation






