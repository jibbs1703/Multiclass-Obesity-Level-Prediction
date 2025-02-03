"""Module for extracting the data needed for the ETL process."""
from utils.helpers.etl_helpers import extract_csv
from utils.aws.s3 import S3Buckets

s3_connection = S3Buckets.credentials()
print(s3_connection.list_buckets())
df = s3_connection.read_file_to_dataframe("jibbs-raw-datasets", "uncleaned_AdultData.csv")
print(df.head())