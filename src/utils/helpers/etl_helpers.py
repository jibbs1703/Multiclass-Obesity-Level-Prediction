"""Function that help the etl process."""
import pandas as pd
import csv

def extract_csv(file: csv) -> pd.DataFrame:
    """Extract data from a csv file."""
    return pd.read_csv(file)

def extract_excel(file: str) -> pd.DataFrame:
    """Extract data from an excel file."""
    return pd.read_excel(file)

def extract_sql(query: str) -> pd.DataFrame:
    """Extract data from a SQL database."""
    return pd.read_sql(query)
