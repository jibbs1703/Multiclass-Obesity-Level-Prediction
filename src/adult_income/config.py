TARGET = "salary"
NUMERIC_COLUMNS = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
    ]
CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
    ]
RAW_BUCKET_NAME = "jibbs-raw-datasets"
CLEAN_BUCKET_NAME = "jibbs-cleaned-datasets"
RAW_FILE_NAME = "uncleaned_AdultData.csv"
