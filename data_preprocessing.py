import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import config
from logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, config.numeric_features),
            ("cat", categorical_pipeline, config.categorical_features),
        ])

    def fit_transform(self, df: pd.DataFrame):
        logger.info("Fitting and transforming data...")
        X = df.drop(columns=[config.target_column])
        y = df[config.target_column]
        X_transformed = self.preprocessor.fit_transform(X)
        logger.info(f"Transformed shape: {X_transformed.shape}")
        return X_transformed, y, self.preprocessor

    def transform(self, df: pd.DataFrame):
        X = df.drop(columns=[config.target_column], errors="ignore")
        return self.preprocessor.transform(X)
