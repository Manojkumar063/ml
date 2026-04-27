# ML Prediction Pipeline

A full-stack machine learning pipeline with a FastAPI backend, React frontend, and Docker support. It covers the complete ML lifecycle — data ingestion, preprocessing, model training, artifact management, prediction serving, and on-demand retraining.

---

## Project Structure

```
ml_pipeline/
├── artifacts/                  # Saved model and preprocessor artifacts
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── raw_data.csv
├── backend/
│   ├── main.py                 # FastAPI app (predict, retrain, health endpoints)
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── public/
│   └── src/
│       └── App.js              # React UI for prediction and retraining
├── logs/
│   └── pipeline.log            # Unified log file
├── config.py                   # Centralized configuration (paths, features, hyperparams)
├── logger.py                   # Shared logger (file + console)
├── data_ingestion.py           # Data generation / loading
├── data_preprocessing.py       # Sklearn ColumnTransformer pipeline
├── model_training.py           # RandomForest training + evaluation
├── model_registry.py           # Pickle-based artifact save/load
├── predictor.py                # Inference wrapper
├── pipeline.py                 # End-to-end training pipeline runner
├── docker-compose.yml          # Multi-service Docker setup
└── requirements.txt            # Core ML dependencies
```

---

## Architecture Overview

```
[React Frontend]
      |
      | HTTP (port 3000 → 8000)
      ↓
[FastAPI Backend]
      |
      ├── /predict  → Predictor → preprocessor.pkl + model.pkl
      ├── /retrain  → pipeline.py → DataIngestion → Preprocessor → Trainer → ModelRegistry
      └── /health   → status check
```

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_ingestion.py` | Generates synthetic classification data (1000 samples, 5 features) with injected missing values. Saves to `artifacts/raw_data.csv`. |
| 2 | `data_preprocessing.py` | Applies median imputation + standard scaling on numeric features; mode imputation + one-hot encoding on categorical features via `ColumnTransformer`. |
| 3 | `model_training.py` | Trains a `RandomForestClassifier` (100 trees, max depth 6). Evaluates on a stratified 80/20 split. Reports accuracy and ROC-AUC. |
| 4 | `model_registry.py` | Serializes model and preprocessor to `artifacts/` using pickle. Loads them back for inference. |

---

## Features

- **Input features:** `age`, `income`, `score` (numeric) + `category` (A/B/C), `region` (North/South/East/West) (categorical)
- **Target:** Binary classification (0 = Negative, 1 = Positive)
- **Missing value handling:** Median imputation for numeric, most-frequent for categorical
- **Metrics reported:** Accuracy, ROC-AUC, full classification report
- **On-demand retraining:** Trigger via UI or API without restarting the server
- **Structured logging:** Timestamped logs to both console and `logs/pipeline.log`

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- Docker & Docker Compose (optional)

---

### Option 1: Run Locally

**1. Train the model**

```bash
cd ml_pipeline
pip install -r requirements.txt
python pipeline.py
```

**2. Start the backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**3. Start the frontend**

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`, backend at `http://localhost:8000`.

---

### Option 2: Run with Docker

```bash
cd ml_pipeline
docker-compose up --build
```

| Service  | URL |
|----------|-----|
| Frontend | http://localhost:3000 |
| Backend  | http://localhost:8000 |

Artifacts and logs are mounted as volumes so they persist across container restarts.

---

## API Reference

### `GET /health`
Returns server status.
```json
{ "status": "ok" }
```

---

### `POST /predict`
Run inference on a single input.

**Request body:**
```json
{
  "age": 35,
  "income": 75000,
  "score": 0.82,
  "category": "A",
  "region": "North"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.87,
  "label": "Positive"
}
```

**Field constraints:**

| Field | Type | Range / Values |
|-------|------|----------------|
| age | int | 18 – 80 |
| income | int | 20,000 – 200,000 |
| score | float | any |
| category | string | "A", "B", "C" |
| region | string | "North", "South", "East", "West" |

---

### `POST /retrain`
Retriggers the full pipeline and reloads the model in-memory.

**Response:**
```json
{
  "status": "retrained",
  "metrics": {
    "accuracy": 0.895,
    "roc_auc": 0.951
  }
}
```

---

## Configuration

All paths, feature names, and training hyperparameters are centralized in `config.py`:

```python
target_column      = "target"
numeric_features   = ["age", "income", "score"]
categorical_features = ["category", "region"]
test_size          = 0.2
random_state       = 42
```

To swap in a real data source, modify the `collect_data()` method in `data_ingestion.py`.

---

## Frontend UI

The React app provides:
- A form to input all 5 features with validation
- A prediction result card showing label, raw prediction, probability, and a visual probability bar
- A "Retrain Model" button that triggers `/retrain` and displays the new metrics

---

## Logging

All pipeline stages, API calls, and errors are logged to:
- Console (stdout)
- `logs/pipeline.log`

Log format:
```
2024-01-15 10:23:45 | INFO | pipeline | [Stage 1/4] Data Ingestion
```

---

## Dependencies

**ML Core (`requirements.txt`)**
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

**Backend (`backend/requirements.txt`)**
```
fastapi
uvicorn
pydantic
pandas
scikit-learn
```

**Frontend**
- React 18
- Fetch API (no external HTTP client)

---

## Extending the Project

| Goal | Where to change |
|------|----------------|
| Use real data (DB/S3/API) | `data_ingestion.py` → `collect_data()` |
| Swap model algorithm | `model_training.py` → `ModelTrainer.__init__()` |
| Add new features | `config.py` → `numeric_features` / `categorical_features` |
| Add authentication | `backend/main.py` → FastAPI middleware |
| Persist model versions | `model_registry.py` → add versioned paths |
