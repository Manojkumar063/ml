import os
import pickle
from config import config
from logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    @staticmethod
    def save(obj, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved artifact -> {path}")

    @staticmethod
    def load(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded artifact <- {path}")
        return obj

    def save_model(self, model):
        self.save(model, config.model_path)

    def save_preprocessor(self, preprocessor):
        self.save(preprocessor, config.preprocessor_path)

    def load_model(self):
        return self.load(config.model_path)

    def load_preprocessor(self):
        return self.load(config.preprocessor_path)
