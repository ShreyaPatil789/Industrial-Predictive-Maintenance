import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load dataset from given file path
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """
    Perform preprocessing:
    - Convert Failure Type to binary
    - Drop unnecessary columns
    - Encode categorical variables
    """
    
    # Convert to binary target
    df["Failure"] = df["Failure Type"].apply(
        lambda x: 0 if x == "No Failure" else 1
    )

    # Drop unnecessary columns
    drop_columns = ["UDI", "Product ID", "Failure Type", "Target"]   
    df.drop(columns=drop_columns, inplace=True)

    # One-hot encode machine type
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    return df


def split_data(df):
    """
    Split dataset using stratification
    """
    X = df.drop("Failure", axis=1)
    y = df["Failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def prepare_data(filepath):
    """
    Full pipeline:
    Load -> Preprocess -> Split
    """
    df = load_data(filepath)
    df = preprocess_data(df)
    return split_data(df)