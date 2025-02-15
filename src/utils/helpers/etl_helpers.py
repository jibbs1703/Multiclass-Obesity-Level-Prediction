"""Function that help the etl process."""

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split


def extract_csv(file: str) -> pd.DataFrame:
    """
    Extract data from a CSV file.

    This function reads data from a CSV file and returns it as a DataFrame.

    Parameters:
    ----------
    file : str
        The path to the CSV file to be read.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(file)


def extract_excel(file: str) -> pd.DataFrame:
    """
    Extract data from an Excel file.

    This function reads data from an Excel file and returns it as a DataFrame.

    Parameters:
    ----------
    file : str
        The path to the Excel file to be read.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the data from the Excel file.
    """
    return pd.read_excel(file)


def extract_sql(query: str, con) -> pd.DataFrame:
    """
    Extract data from a SQL database.

    This function executes a SQL query on a database connection 
    and returns the result as a DataFrame.

    Parameters:
    ----------
    query : str
        The SQL query to be executed.
    con : SQLAlchemy connection object or database connection URL
        The database connection to use for the query.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the data from the SQL query.
    """
    return pd.read_sql(query, con)


def train_test_val_split(df: pd.DataFrame, target: str,
                         split1: float = 0.2, split2: float = 0.3):
    """Split data into training, test, and validation sets.

    This function splits the input dataframe `df` into training, test, and validation
    sets. The target column specified by `target` is separated from the feature columns.
    Two split ratios, `split1` and `split2`, determine the proportion of data in the 
    test and validation sets, respectively. The splits are performed using the 
    `train_test_split` function from scikit-learn.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe containing the features and the target column.
    target : str
        The name of the target column to be predicted.
    split1 : float, optional (default=0.2)
        The proportion of the data to include in the test split.
    split2 : float, optional (default=0.3)
        The proportion of the training data to include in the validation split.
        
     Returns:
    -------
    X_train : pd.DataFrame
        The training set features.
    X_test : pd.DataFrame
        The test set features.
    X_val : pd.DataFrame
        The validation set features.
    y_train : pd.Series
        The training set target.
    y_test : pd.Series
        The test set target.
    y_val : pd.Series
        The validation set target.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=split1, random_state=2024)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=split2, random_state=2024)

    return X_train, X_test, X_val, y_train, y_test, y_val


def resample_training(X: pd.DataFrame, y: pd.Series, categorical_features: list[str]):
    """
    Resample the training dataset using SMOTENC to handle imbalanced classes.

    This function applies the SMOTENC (Synthetic Minority Over-sampling Technique
    for Nominal and Continuous) algorithm to resample the training dataset `X` 
    and corresponding labels `y`. It generates synthetic samples for the minority
    class while considering both continuous and categorical features.

    Parameters:
    ----------
    X : pd.DataFrame :The input features dataframe to be resampled.
    y : pd.Series : The target variable to be resampled.
    categorical_features : List[str] : list of are categorical features

    Returns:
    -------
    X_resampled : pd.DataFrame
        The resampled input features dataframe.
    y_resampled : pd.Series
        The resampled target variable.
    """
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=2024)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    return X_resampled, y_resampled