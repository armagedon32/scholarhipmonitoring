import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "knsubic-scholarship-demo-secret-change-in-prod")
    # Railway persists the SQLite file on a mounted volume; override with DATABASE_PATH env var.
    DATABASE = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "scholarship.db"))
    MODELS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
    DATASET_PATH = os.path.join(BASE_DIR, "data", "scholar_data.csv")
    MODEL_FILENAME = "retention_model.joblib"
    SCALER_FILENAME = "feature_pipeline.joblib"
    FEATURES_FILENAME = "features.json"
    # Performance targets adopted from the study (Chapter 2 / 4).
    ACCURACY_TARGET = 0.85
    F1_TARGET = 0.80
    # Simulated MFA for coordinator/admin roles.
    MFA_ENABLED = True
