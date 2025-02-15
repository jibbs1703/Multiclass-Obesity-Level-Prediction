"""Main module for the adult income classification project."""
from adult_income.etl.extract import run_data_extraction

# Define the bucket name and file name (Put in Config file)
BUCKET_NAME = "jibbs-raw-datasets"
FILE_NAME = "uncleaned_AdultData.csv"

# Run the data extraction process
df = run_data_extraction(BUCKET_NAME, FILE_NAME)

# Print the first 5 rows of the dataframe
print(df.info())
