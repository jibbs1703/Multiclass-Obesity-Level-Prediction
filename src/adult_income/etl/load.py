from io import StringIO

import pandas as pd

from utils.aws.s3 import S3Buckets


def load_datasets(bucket_name: str, df: pd.DataFrame, filename: str) -> None:
    """
    Uploads pandas DataFrame to S3 as a CSV file.

    Parameters:
    ----------
    df : pd.DataFrame : The DataFrame to be uploaded.
    file_name : str : The name of the file to be saved in the S3 bucket.
    """
    s3_connection = S3Buckets.credentials()
    folder = filename.split("_")[-1]
    csv_file = StringIO()
    df.to_csv(csv_file, index=False)
    s3_connection.upload_file(bucket_name=bucket_name, file=csv_file,
                              filename=filename, folder=f"adult_income/{folder}/")
