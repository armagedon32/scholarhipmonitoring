import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# When DATABASE_PATH points into a mounted volume (Railway), the live dataset and
# retrained model artifacts are stored beside the database so they survive redeploys.
_VOLUME_DIR = os.path.dirname(os.environ["DATABASE_PATH"]) if os.environ.get("DATABASE_PATH") else None


class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get("SECRET_KEY", "knsubic-scholarship-demo-secret-change-in-prod")
    # Railway persists the SQLite file on a mounted volume; override with DATABASE_PATH env var.
    DATABASE = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "scholarship.db"))
    # Uploaded documents are stored beside the database so they persist on the Railway volume.
    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER", os.path.join(os.path.dirname(DATABASE) or BASE_DIR, "uploads"))
    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "doc", "docx", "xls", "xlsx"}

    # Live dataset + retrained model live on the mounted volume when deployed (persistent),
    # otherwise in the repository (local development).
    if _VOLUME_DIR:
        DATA_DIR = os.path.join(_VOLUME_DIR, "data")
        MODELS_DIR = os.path.join(_VOLUME_DIR, "models")
    else:
        DATA_DIR = os.path.join(BASE_DIR, "data")
        MODELS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
    DATASET_PATH = os.path.join(DATA_DIR, "scholar_data.csv")
    # Committed model artifacts shipped with the repo (used as the first-run bootstrap).
    REPO_MODELS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")

    MODEL_FILENAME = "retention_model.joblib"
    SCALER_FILENAME = "feature_pipeline.joblib"
    FEATURES_FILENAME = "features.json"
    # Performance targets adopted from the study (Chapter 2 / 4).
    ACCURACY_TARGET = 0.85
    F1_TARGET = 0.80
    # Simulated MFA for coordinator/admin roles.
    MFA_ENABLED = True