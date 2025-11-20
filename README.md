import pandas as pd
import numpy as np

# --- 1. Dataset Loading and Initial Overview ---

def load_data():
    """
    Loads the Iris dataset from the UCI repository.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    print("--- 1. Loading Data (Iris Dataset) ---")
    
    # UCI Iris Dataset URL (standard library dataset)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    try:
        df = pd.read_csv(url, header=None, names=column_names)
        print("Iris dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def show_dataset_details(df):
    """
    Displays the initial details of the dataset, including shape, first rows,
    data types, missing values summary, and descriptive statistics.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    print("\n--- 2. Dataset Details and Initial Inspection ---")
    
    # Rows and Columns
    print(f"\nShape (Rows, Columns): {df.shape}")

    # First 5 Rows
    print("\nFirst 5 Rows:")
    print(df.head())

    # Column/Data Types and Non-Null Counts
    print("\nData Types and Non-Null Counts (df.info()):")
    df.info()

    # Missing Values
    print("\nMissing Values (Count and Percentage):")
    missing_data = df.isnull().sum().sort_values(ascending=False)
    total_rows = len(df)
    missing_percent = (missing_data[missing_data > 0] / total_rows) * 100
    missing_info = pd.concat([missing_data, missing_percent.round(2)], axis=1, keys=['Missing Count', 'Missing %'])
    # Since Iris is clean, this usually won't print anything, but we keep the logic
    if not missing_info[missing_info['Missing Count'] > 0].empty:
        print(missing_info[missing_info['Missing Count'] > 0])
    else:
        print("No missing values detected.")
    
    # Descriptive Statistics
    print("\nDescriptive Statistics (Numerical Columns):")
    print(df.describe().T)
    
    # Descriptive Statistics (Categorical Columns - Mode)
    print("\nMode for Key Categorical Columns ('species'):")
    categorical_modes = df[['species']].mode().iloc[0]
    print(categorical_modes.to_string())


# --- 3. Outlier Detection and Treatment (IQR Method) ---
# Note: Missing Value Handling is skipped as Iris is a clean dataset.

def detect_and_treat_outliers(df, column='petal_width'):
    """
    Detects and treats outliers in a specified numerical column using the IQR method.
    Outliers are identified and capped at the fence boundaries.

    Args:
        df (pd.DataFrame): The DataFrame.
        column (str): The name of the column to check for outliers.
        (Using 'petal_width' for Iris dataset analysis)

    Returns:
        pd.DataFrame: The DataFrame with outliers treated (capped).
    """
    print(f"\n--- 3. Outlier Detection and Treatment for '{column}' (IQR Method) ---")

    # Calculate IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Detect outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    print(f"Number of Outliers detected in '{column}': {len(outliers)}")

    # Treat (Cap) outliers
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    
    # Verify treatment
    outliers_after = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Number of Outliers after capping: {len(outliers_after)}")
    print(f"\nDescriptive Stats for '{column}' after treatment:")
    print(df[column].describe().to_string())
    
    return df


# --- Main Execution ---

if __name__ == '__main__':
    # 1. Load Data
    iris_df = load_data()

    if iris_df is not None:
        # 2. Show Dataset Details
        show_dataset_details(iris_df.copy()) # Use a copy for initial inspection

        # 3. Outlier Detection and Treatment
        # Note: Iris dataset is typically clean, so we go directly to outlier check
        iris_df_final = detect_and_treat_outliers(iris_df, column='petal_width')
        
        # 4. Final Dataset Summary and Saving
        print("\n--- 4. Final Dataset Summary ---")
        print(f"Final DataFrame Shape (Full): {iris_df_final.shape}")

        # Limit to the first 20 entries as requested
        iris_df_subset = iris_df_final.head(20)

        # Save the subset cleaned DataFrame to a new CSV file
        output_filepath = 'iris_cleaned.csv'
        iris_df_subset.to_csv(output_filepath, index=False)
        print(f"\n--- 5. Saving Cleaned Data Subset (First 20 Entries) ---")
        print(f"Subset DataFrame Shape: {iris_df_subset.shape}")
        print(f"Cleaned dataset subset (first 20 rows) successfully saved to: '{output_filepath}'")
        
        print("EDA Complete. Data is now ready for feature engineering and modeling.")
