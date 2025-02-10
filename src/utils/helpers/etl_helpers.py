"""Function that help the etl process."""
import pandas as pd
import csv

from sklearn.model_selection import train_test_split

def extract_csv(file: csv) -> pd.DataFrame:
    """Extract data from a csv file."""
    return pd.read_csv(file)

def extract_excel(file: str) -> pd.DataFrame:
    """Extract data from an excel file."""
    return pd.read_excel(file)

def extract_sql(query: str) -> pd.DataFrame:
    """Extract data from a SQL database."""
    return pd.read_sql(query)

def train_test_val_split(df: pd.DataFrame, target: str):
    """Split data into training, test, and validation sets."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=2024)

    return X_train, X_test, X_val, y_train, y_test, y_val