"""Runtime wrapper for the trained retention model."""
import json
import os
import shutil

import joblib

from config import Config
from ml.prep import encode_row


class RetentionModel:
    def __init__(self):
        self.model = None
        self.artifact = None
        self._artifact_mtime = None
        self.load()

    def _ensure_artifacts(self):
        """Copy the committed artifacts into the live model dir on first boot.

        On Railway the model dir lives on the volume (persistent); on a fresh volume
        it starts empty, so we seed it from the repo's bundled model until the
        coordinator uploads a dataset and retrains.
        """
        src_dir = Config.REPO_MODELS_DIR
        if not os.path.isdir(src_dir):
            return
        if os.path.exists(os.path.join(Config.MODELS_DIR, Config.FEATURES_FILENAME)):
            return  # already bootstrapped or retrained
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        for fn in (Config.MODEL_FILENAME, Config.FEATURES_FILENAME,
                   Config.SCALER_FILENAME, "decision_rules.txt"):
            src = os.path.join(src_dir, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(Config.MODELS_DIR, fn))

    def _mtime(self):
        p = os.path.join(Config.MODELS_DIR, Config.FEATURES_FILENAME)
        return os.path.getmtime(p) if os.path.exists(p) else -1

    def load(self, force=False):
        """Load the model, lazily refreshing when the artifacts were retrained."""
        self._ensure_artifacts()
        current = self._mtime()
        if not force and self.artifact is not None and current == self._artifact_mtime:
            return
        model_path = os.path.join(Config.MODELS_DIR, Config.MODEL_FILENAME)
        meta_path = os.path.join(Config.MODELS_DIR, Config.FEATURES_FILENAME)
        if not os.path.exists(model_path):
            self.model = None
            self.artifact = None
            self.available = False
            return
        self.model = joblib.load(model_path)
        try:
            with open(meta_path, encoding="utf-8") as fh:
                self.artifact = json.load(fh)
        except OSError:
            self.artifact = None
        self._artifact_mtime = current
        self.available = True

    def predict(self, row: dict):
        """Returns (status, probability_of_retention)."""
        self.load()
        if not self.available:
            return "Retained", 0.5
        columns = self.artifact["columns"]
        vec = encode_row(dict(row), columns)
        prob = self.model.predict_proba(vec)[0]
        cls = int(self.model.predict(vec)[0])
        retained_prob = float(prob[1] if self.model.classes_[1] == 1 else prob[0])
        return ("Retained" if cls == 1 else "At-Risk"), retained_prob

    def selected_algorithm(self):
        if self.artifact:
            return self.artifact.get("selected_algorithm", "Decision Tree")
        return "Decision Tree"

    def metrics(self):
        if self.artifact:
            return self.artifact.get("metrics", {})
        return {}

    def confusion_matrix(self):
        """TP/TN/FP/FN counts of the selected model on the held-out test set."""
        m = self.metrics().get(self.selected_algorithm(), {})
        return m.get("confusion_matrix") or {}

    def feature_importance(self):
        if self.artifact:
            return self.artifact.get("feature_importance", [])
        return []

    def dataset_count(self):
        """Number of records the model was trained on (from the saved artifact)."""
        return self.artifact.get("dataset_count", 0) if self.artifact else 0

    def dataset_years(self):
        """Distinct school years in the training data (0 if the source had no year info)."""
        return self.artifact.get("dataset_years", 0) if self.artifact else 0


model = RetentionModel()