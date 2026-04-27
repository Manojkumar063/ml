import sys
from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_registry import ModelRegistry
from logger import get_logger

logger = get_logger("pipeline")

def run_pipeline():
    logger.info("=" * 50)
    logger.info("ML PIPELINE STARTED")
    logger.info("=" * 50)

    try:
        # Stage 1: Data Ingestion
        logger.info("[Stage 1/4] Data Ingestion")
        ingestion = DataIngestion()
        df = ingestion.collect_data()

        # Stage 2: Preprocessing
        logger.info("[Stage 2/4] Data Preprocessing")
        preprocessor = DataPreprocessor()
        X, y, fitted_preprocessor = preprocessor.fit_transform(df)

        # Stage 3: Model Training
        logger.info("[Stage 3/4] Model Training")
        trainer = ModelTrainer()
        model, metrics = trainer.train(X, y)

        # Stage 4: Save Artifacts
        logger.info("[Stage 4/4] Saving Artifacts")
        registry = ModelRegistry()
        registry.save_model(model)
        registry.save_preprocessor(fitted_preprocessor)

        logger.info("=" * 50)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY | {metrics}")
        logger.info("=" * 50)
        return metrics

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
