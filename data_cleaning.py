import streamlit as st
import pandas as pd
import base64
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore


def clean_data(data, drop_columns, remove_null, fill_null, fill_value, data_types, column_filters,
               remove_duplicates, selected_columns_for_duplicates,
               remove_outliers, selected_columns_for_outliers, outlier_threshold,
               time_series_handling, timestamp_column, replace_categorical_column, replacement_mapping,
               modify_column):

    # Drop Columns
    if drop_columns:
        data = data.drop(columns=drop_columns)

    # Handle Nulls
    if remove_null:
        data = data.dropna()

    if fill_null and fill_value is not None:
        data = data.fillna(fill_value)

    # Change Data Types
    for column, dtype in data_types.items():
        if column in data.columns:
            try:
                data[column] = data[column].astype(dtype)
            except Exception as e:
                st.error(f"Error converting `{column}` to {dtype}: {e}")

    # Column Filters
    for column, values in column_filters.items():
        if column in data.columns and values:
            data = data[data[column].astype(str).isin(values)]

    # Remove Duplicates
    if remove_duplicates:
        if selected_columns_for_duplicates:
            data = data.drop_duplicates(subset=selected_columns_for_duplicates)
        else:
            data = data.drop_duplicates()

    # Remove Outliers using z-score
    if remove_outliers and selected_columns_for_outliers:
        z_scores = zscore(data[selected_columns_for_outliers])
        if len(selected_columns_for_outliers) == 1:
            z_scores = pd.DataFrame({selected_columns_for_outliers[0]: z_scores})
        data = data[(abs(z_scores) < outlier_threshold).all(axis=1)]

    # Time-Series Handling
    if time_series_handling and timestamp_column in data.columns:
        try:
            data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors='coerce')
            data = data.dropna(subset=[timestamp_column])
            data.sort_values(by=timestamp_column, inplace=True)

            if "value" in data.columns:
                for lag in range(1, 4):
                    data[f'value_lag_{lag}'] = data['value'].shift(lag)

            st.success("Time series features created successfully.")

        except Exception as e:
            st.error(f"Timestamp processing error: {e}")

    # Replace Categorical Values
    if replace_categorical_column and replacement_mapping:
        if replace_categorical_column in data.columns:
            data[replace_categorical_column] = data[replace_categorical_column].map(replacement_mapping)

    # Modify Column
    if modify_column and modify_column.get("column") in data.columns:
        col = modify_column['column']
        new_col = modify_column['new_column']
        s = modify_column['start_index']
        e = modify_column['end_index']

        data[new_col] = data[col].astype(str).apply(
            lambda x: x[:s] + x[e + 1:]
        )

        if modify_column['remove_original']:
            data = data.drop(columns=[col])

    return data


def create_download_link(df, filename="cleaned_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Cleaned Data</a>'
    return href


def main():

    st.title("Data Cleaning App")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    replace_categorical_column = None
    replacement_mapping = {}

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file, encoding="latin1")

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        st.subheader("Column Data Types")
        st.dataframe(data.dtypes)

        st.subheader("Summary")
        st.dataframe(data.describe(include="all"))

        # Drop columns
        selected_columns_to_remove = st.multiselect("Select columns to remove", data.columns)

        # Null Handling
        remove_null = st.checkbox("Remove rows with null values")

        fill_null = st.checkbox("Fill null values instead of removing")
        fill_value = None
        if fill_null:
            fill_value = st.text_input("Enter replacement value:")

        # Change Data Types
        st.subheader("Change Column Data Types")
        data_types = {}
        for column in data.columns:
            dt_change = st.selectbox(
                f"Select new datatype for '{column}' (optional):",
                ["None", "int", "float", "str"],
                key=f"type_{column}"
            )
            if dt_change != "None":
                data_types[column] = dt_change

        # Column Filters
        st.subheader("Column Filters")
        selected_filter_columns = st.multiselect("Select columns to filter", data.columns)
        column_filters = {}
        for col in selected_filter_columns:
            filter_values = st.text_input(f"Enter allowed values for '{col}' (comma-separated)")
            if filter_values.strip():
                column_filters[col] = [v.strip() for v in filter_values.split(",")]

        # Duplicate Removal
        remove_duplicates = st.checkbox("Remove duplicate rows")
        selected_columns_for_duplicates = st.multiselect(
            "Columns for duplicate check (optional)", data.columns
        )

        # Outlier removal
        remove_outliers = st.checkbox("Remove outliers using Z-score")
        selected_columns_for_outliers = \
            st.multiselect("Select numeric columns", data.select_dtypes("number").columns)

        outlier_threshold = st.number_input("Outlier threshold (Z-score)", value=3.0)

        # Replace Categorical Values
        replace_categorical_checkbox = st.checkbox("Replace categorical values with numeric mapping")

        if replace_categorical_checkbox:
            replace_categorical_column = st.selectbox(
                "Select column to replace", data.columns
            )
            if replace_categorical_column:
                st.write("Enter numeric replacement for each category:")
                unique_vals = data[replace_categorical_column].unique()
                for v in unique_vals:
                    replacement_mapping[v] = st.number_input(
                        f"Replace '{v}' with:", value=0, key=f"map_{v}"
                    )

        # Modify Column
        st.subheader("Modify Column Values")
        modify_column_checkbox = st.checkbox("Enable column modification")
        modify_column = {}

        if modify_column_checkbox:
            modify_column["column"] = st.selectbox("Column to modify", data.columns)
            modify_column["start_index"] = st.number_input("Start index", value=0)
            modify_column["end_index"] = st.number_input("End index", value=0)
            modify_column["new_column"] = st.text_input("New column name")
            modify_column["remove_original"] = st.checkbox("Remove original column")

        # BUTTON: CLEAN DATA
        if st.button("Clean Data"):

            data = clean_data(
                data,
                selected_columns_to_remove,
                remove_null,
                fill_null,
                fill_value,
                data_types,
                column_filters,
                remove_duplicates,
                selected_columns_for_duplicates,
                remove_outliers,
                selected_columns_for_outliers,
                outlier_threshold,
                time_series_handling=True,
                timestamp_column="timestamp",
                replace_categorical_column=replace_categorical_column,
                replacement_mapping=replacement_mapping,
                modify_column=modify_column,
            )

            st.subheader("Cleaned Data")
            st.dataframe(data)

            st.subheader("Null Values After Cleaning")
            st.write(data.isnull().sum())

            st.markdown(create_download_link(data), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
