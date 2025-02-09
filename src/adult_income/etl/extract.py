"""module for data extraction."""
import pandas as pd
from utils.aws.s3 import S3Buckets
from utils.helpers.etl_helpers import extract_csv

def run_data_extraction(bucket_name: str, file_name: str) -> pd.DataFrame:
    """function to run data extraction."""
    s3_conn = S3Buckets.credentials()
    csv_file = s3_conn.read_file(bucket_name, file_name)
    return extract_csv(csv_file)

if __name__ == "__main__":
    BUCKET_NAME = "jibbs-raw-datasets"
    FILE_NAME = "uncleanedAdultIncome.csv"
    df = run_data_extraction(BUCKET_NAME, FILE_NAME)
    print(df.head())