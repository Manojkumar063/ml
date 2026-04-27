import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

from predictor import Predictor
from pipeline import run_pipeline
from logger import get_logger

logger = get_logger("api")
app = FastAPI(title="ML Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor once at startup
predictor: Predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    predictor = Predictor()
    logger.info("Model loaded at startup.")


class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=80, example=35)
    income: int = Field(..., ge=20000, le=200000, example=75000)
    score: float = Field(..., example=0.82)
    category: Literal["A", "B", "C"] = Field(..., example="A")
    region: Literal["North", "South", "East", "West"] = Field(..., example="North")


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.model_dump()])
        result = predictor.predict(df)
        pred = result["prediction"][0]
        prob = round(result["probability"][0], 4)
        return PredictResponse(
            prediction=pred,
            probability=prob,
            label="Positive" if pred == 1 else "Negative",
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain():
    global predictor
    try:
        metrics = run_pipeline()
        predictor = Predictor()
        logger.info("Model retrained and reloaded.")
        return {"status": "retrained", "metrics": metrics}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
