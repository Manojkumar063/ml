import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    # Data
    raw_data_path: str = os.path.join(BASE_DIR, "artifacts", "raw_data.csv")
    processed_data_path: str = os.path.join(BASE_DIR, "artifacts", "processed_data.csv")

    # Model
    model_path: str = os.path.join(BASE_DIR, "artifacts", "model.pkl")
    preprocessor_path: str = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")

    # Training
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "target"
    numeric_features: List[str] = field(default_factory=lambda: ["age", "income", "score"])
    categorical_features: List[str] = field(default_factory=lambda: ["category", "region"])

    # Logging
    log_path: str = os.path.join(BASE_DIR, "logs", "pipeline.log")

config = Config()
