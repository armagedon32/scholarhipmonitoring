"""Train the retention classification models (KDD: modeling phase).

Generates a realistic synthetic dataset for scholarship recipients at
Kolehiyo ng Subic, runs 80:20 train/test split with 10-fold cross-validation,
trains Decision Tree and Logistic Regression, and saves the best model so the
Flask app can use it for live predictions.

Run:  python -m ml.train_model
"""
import json
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from ml.prep import CATEGORY_LEVELS, build_columns, df_to_matrix


def generate_dataset(n=1800, seed=42, label_noise=0.07):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        scholarship_type = rng.choice(CATEGORY_LEVELS["scholarship_type"])
        socio_status = rng.choice(CATEGORY_LEVELS["socio_status"])
        units_enrolled = rng.randint(12, 21)
        attendance_rate = round(rng.uniform(45, 100), 1)

        # Philippine grading: lower GWA is better (1.00 highest, 5.00 fail).
        base_ability = rng.uniform(0.0, 1.0)
        gwa = round(4.6 - 2.6 * base_ability + rng.uniform(-0.2, 0.2), 2)
        gwa = max(1.0, min(4.5, gwa))
        semester_performance = round(
            max(1.0, min(4.5, gwa + rng.uniform(-0.3, 0.3))), 2
        )

        fail_prob = max(0.0, min(1.0, 0.60 - base_ability - (attendance_rate - 75) / 200))
        failed_subjects = rng.choices([0, 1, 2, 3, 4], weights=[
            1 - fail_prob, fail_prob * 0.5, fail_prob * 0.28,
            fail_prob * 0.15, fail_prob * 0.07])[0]

        # Scholastic pressure by grant type and financial stress by socio status.
        pressure = {"Academic": 0.15, "DOST": 0.1, "CHED_GIA": 0.05, "Municipal": 0.02}[scholarship_type]
        fin_stress = {"Low": 0.15, "Lower-Middle": 0.08, "Middle": 0.02, "Upper-Middle": 0.0}[socio_status]

        # Retention score; retained when positive.
        score = (
            -1.5 * (gwa - 2.5)
            - 1.2 * failed_subjects
            + 1.5 * ((attendance_rate - 70) / 30.0)
            + 0.3 * ((units_enrolled - 15) / 6.0)
            - 0.8 * pressure
            - 1.0 * fin_stress
        )
        # 1 = Retained (majority), 0 = At-Risk; positive score keeps the grant.
        retained = 0 if score > 0 else 1
        # Simulate imperfect historical records (label noise).
        if rng.random() < label_noise:
            retained = 1 - retained

        rows.append({
            "gwa": gwa,
            "failed_subjects": failed_subjects,
            "units_enrolled": units_enrolled,
            "attendance_rate": attendance_rate,
            "scholarship_type": scholarship_type,
            "socio_status": socio_status,
            "semester_performance": semester_performance,
            "retained": retained,
        })
    return pd.DataFrame(rows)


def train_and_report():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.DATASET_PATH), exist_ok=True)

    df = generate_dataset()
    df.to_csv(Config.DATASET_PATH, index=False)

    columns = build_columns()
    X = df_to_matrix(df, columns)
    y = df["retained"].to_numpy()

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42),
    }

    results = {}
    best = None
    for name, model in models.items():
        cv = cross_val_score(model, X_tr, y_tr, cv=10, scoring="f1").mean()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred)
        rec = recall_score(y_te, pred)
        f1 = f1_score(y_te, pred)
        err = 1 - acc
        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "error_rate": round(err, 4),
            "cv_f1_10fold": round(float(cv), 4),
            "meets_85_acc": acc >= Config.ACCURACY_TARGET,
            "meets_80_f1": f1 >= Config.F1_TARGET,
        }
        print(f"\n=== {name} ===")
        print(f"10-fold CV F1      : {cv:.4f}")
        print(f"Accuracy  (test)   : {acc*100:.2f}%")
        print(f"Precision          : {prec:.4f}")
        print(f"Recall             : {rec:.4f}")
        print(f"F1-Score           : {f1:.4f}")
        print(f"Error rate         : {err*100:.2f}%")

    # Select by F1 (balanced metric), ties broken by accuracy.
    best_name = max(results, key=lambda n: (results[n]["f1_score"], results[n]["accuracy"]))
    best_model = models[best_name]

    if best_name == "Decision Tree":
        feats = best_model.feature_importances_.tolist()
    else:
        feats = np.abs(best_model.coef_[0]).tolist()
    importance = sorted(
        (f, v) for f, v in zip(columns, feats))[::-1]

    artifact = {
        "model": Config.MODEL_FILENAME,
        "columns": columns,
        "selected_algorithm": best_name,
        "metrics": results,
        "feature_importance": [[f, round(float(v), 4)] for f, v in importance],
    }
    joblib.dump(best_model, os.path.join(Config.MODELS_DIR, Config.MODEL_FILENAME))
    with open(os.path.join(Config.MODELS_DIR, Config.FEATURES_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    # Export human-readable decision rules for the interpretability view.
    if best_name == "Decision Tree":
        _export_rules(best_model, columns)

    print(f"\nBest model: {best_name}  -> saved to {Config.MODELS_DIR}")
    return artifact


def _export_rules(tree, columns):
    import textwrap

    def recurse(node, path):
        n = tree.tree_
        if n.children_left[node] == n.children_right[node]:
            return [("rule", " AND ".join(path), int(n.value[node].argmax()))]
        feat = n.feature[node]
        th = n.threshold[node]
        name = columns[feat]
        left = recurse(n.children_left[node], path + [f"{name} <= {th:.2f}"])
        right = recurse(n.children_right[node], path + [f"{name} > {th:.2f}"])
        return left + right

    rules = recurse(0, [])
    output = []
    for kind, cond, cls in rules:
        label = "Retained" if cls == 1 else "At-Risk"
        output.append(f"IF {cond} THEN {label}")
    with open(os.path.join(Config.MODELS_DIR, "decision_rules.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(output))


if __name__ == "__main__":
    train_and_report()
