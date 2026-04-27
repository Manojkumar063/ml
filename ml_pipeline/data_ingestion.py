import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from config import config
from logger import get_logger

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        os.makedirs(os.path.dirname(config.raw_data_path), exist_ok=True)

    def collect_data(self) -> pd.DataFrame:
        """Simulate data collection (replace with DB/API/S3 source in production)."""
        logger.info("Collecting data...")
        X, y = make_classification(
            n_samples=1000, n_features=5, n_informative=3,
            random_state=config.random_state
        )
        df = pd.DataFrame(X, columns=["age", "income", "score", "feat4", "feat5"])
        df["age"] = (df["age"] * 10 + 40).astype(int).clip(18, 80)
        df["income"] = (df["income"] * 20000 + 60000).astype(int).clip(20000, 200000)
        df["score"] = (df["score"] * 100).round(2)
        df["category"] = np.random.choice(["A", "B", "C"], size=len(df))
        df["region"] = np.random.choice(["North", "South", "East", "West"], size=len(df))
        df.drop(columns=["feat4", "feat5"], inplace=True)
        df["target"] = y

        # Inject missing values to simulate real-world data
        for col in ["age", "income", "category"]:
            df.loc[df.sample(frac=0.05, random_state=1).index, col] = np.nan

        df.to_csv(config.raw_data_path, index=False)
        logger.info(f"Data saved to {config.raw_data_path} | Shape: {df.shape}")
        return df

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {config.raw_data_path}")
        return pd.read_csv(config.raw_data_path)
