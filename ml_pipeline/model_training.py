import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from config import config
from logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=config.random_state,
            n_jobs=-1,
        )

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
        )
        logger.info(f"Training on {X_train.shape[0]} samples | Evaluating on {X_test.shape[0]} samples")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        logger.info(f"Metrics: {metrics}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        return self.model, metrics
