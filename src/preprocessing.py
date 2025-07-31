import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from tkinter import Tk, filedialog

def select_csv_file():
    Tk().withdraw()  # Hide GUI root
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    return file_path

def preprocess_csv(csv_path=None, label_col=None, random_seed=42):
    if csv_path is None:
        csv_path = select_csv_file()
    if not csv_path:
        raise ValueError("No file selected!")

    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if label_col is None:
        print("Available columns:", list(df.columns))
        label_col = input("Enter the name of the label column: ")

    exclude_cols = [label_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Separate features
    X_df = df[feature_cols].copy()

    # Identify categorical and numerical columns
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Apply Label Encoding to each categorical column
    for col in categorical_cols:
        le_col = LabelEncoder()
        X_df[col] = le_col.fit_transform(X_df[col].astype(str))

    # Now scale only numerical + encoded categorical columns
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    # Encode the target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[label_col])

    # Remove constant features
    var_thresh = VarianceThreshold(threshold=0.0)
    X = var_thresh.fit_transform(X)

    retained_feature_indices = var_thresh.get_support(indices=True)
    retained_feature_names = [feature_cols[i] for i in retained_feature_indices]

    print(f"Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features retained.")
    return X, y, retained_feature_names, label_col

