"""Main module for the adult income classification project."""
from adult_income import config
from adult_income.etl.extract import run_data_extraction
from adult_income.etl.load import load_datasets
from adult_income.etl.transform import Preprocess
from utils.helpers.etl_helpers import resample_training, train_test_val_split

if __name__ == "__main__":
    df = run_data_extraction(config.RAW_BUCKET_NAME, config.RAW_FILE_NAME)

    preprocessor = Preprocess(df=df, numeric_columns=config.NUMERIC_COLUMNS,
                              target=config.TARGET, categorical_columns=config.CATEGORICAL_COLUMNS)
    df = preprocessor.run_preprocessor()
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(
                                                     df=df,target=config.TARGET,split1=0.2,split2=0.3
                                                     )
    X_train, y_train = resample_training(
        X=X_train, y=y_train,categorical_features=config.CATEGORICAL_COLUMNS)
    datasets = {'X_train': X_train,'X_test': X_test,'X_val': X_val,
                  'y_train': y_train,'y_test': y_test,'y_val': y_val}
    for filename,df in datasets.items():
        load_datasets(bucket_name=config.CLEAN_BUCKET_NAME,filename=filename, df=df)