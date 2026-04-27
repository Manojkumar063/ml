import pandas as pd
from model_registry import ModelRegistry
from logger import get_logger

logger = get_logger("predictor")

class Predictor:
    def __init__(self):
        registry = ModelRegistry()
        self.model = registry.load_model()
        self.preprocessor = registry.load_preprocessor()
        logger.info("Predictor initialized with saved artifacts.")

    def predict(self, input_df: pd.DataFrame) -> dict:
        X = self.preprocessor.transform(input_df)
        prediction = self.model.predict(X).tolist()
        probability = self.model.predict_proba(X)[:, 1].tolist()
        return {"prediction": prediction, "probability": probability}


if __name__ == "__main__":
    sample = pd.DataFrame([{
        "age": 35, "income": 75000, "score": 0.82,
        "category": "A", "region": "North"
    }])
    predictor = Predictor()
    result = predictor.predict(sample)
    logger.info(f"Prediction result: {result}")
