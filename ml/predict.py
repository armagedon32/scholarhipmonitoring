"""Runtime wrapper for the trained retention model."""
import json
import os

import joblib

from config import Config
from ml.prep import encode_row


class RetentionModel:
    def __init__(self):
        self.model = None
        self.artifact = None
        self.load()

    def load(self):
        model_path = os.path.join(Config.MODELS_DIR, Config.MODEL_FILENAME)
        meta_path = os.path.join(Config.MODELS_DIR, Config.FEATURES_FILENAME)
        if not os.path.exists(model_path):
            self.available = False
            return
        self.model = joblib.load(model_path)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                self.artifact = json.load(fh)
        else:
            self.artifact = None
        self.available = True

    def predict(self, row: dict):
        """Returns (status, probability_of_retention)."""
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

    def feature_importance(self):
        if self.artifact:
            return self.artifact.get("feature_importance", [])
        return []


model = RetentionModel()
