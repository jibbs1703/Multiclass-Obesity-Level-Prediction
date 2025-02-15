from adult_income.etl.extract import run_data_extraction
from adult_income.etl.transform import Preprocess
from utils.aws.s3 import S3Buckets
from utils.helpers.etl_helpers import resample_training, train_test_val_split

if __name__ == "__main__":
# Define the features (Put in Config file)
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
    FILE_NAME = "uncleaned_AdultData.csv"
    DF = run_data_extraction(RAW_BUCKET_NAME, FILE_NAME)

    preprocessor = Preprocess(df=DF, numeric_columns=NUMERIC_COLUMNS,
                              target=TARGET, categorical_columns=CATEGORICAL_COLUMNS)
    df = preprocessor.run_preprocessor()
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(df=df,
                                                                          target=TARGET,split1=0.2,split2=0.3)
    X_train, y_train = resample_training(X=X_train, y=y_train,
                                         categorical_features=CATEGORICAL_COLUMNS)
    s3_connection = S3Buckets.credentials()