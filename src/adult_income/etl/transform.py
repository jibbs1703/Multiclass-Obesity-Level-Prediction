"""module for data transformation abd preprocessing."""
from dataclasses import dataclass, field

from category_encoders import TargetEncoder
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


@dataclass
class Preprocess:
    df: DataFrame = None
    numeric_columns: list[str] = field(default_factory=list)
    target: str = ''
    categorical_columns: list[str] = field(default_factory=list)

    def scale_numeric(self) -> None:
        # Scaling the numerical columns
        scaler = MinMaxScaler()
        self.df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns])
    
    def encode_target(self) -> None:
        # Label Encode the Target
        le = LabelEncoder()
        self.df[self.target] = le.fit_transform(self.df[self.target])

    def encode_categorical(self) -> None:
        # Target Encoding the Categorical Columns
        encoder = TargetEncoder()
        self.df[self.categorical_columns] = encoder.fit_transform(self.df[self.categorical_columns], self.df[self.target])
        
    def run_preprocessor(self) -> DataFrame:
        self.scale_numeric()
        self.encode_target()
        self.encode_categorical()
        return self.df
