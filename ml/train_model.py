"""Train the retention classification models (KDD: modeling phase).

Generates a realistic 3-school-year history of scholarship recipients at
Kolehiyo ng Subic (SY 2022-2023, 2023-2024, 2024-2025). Each cohort is tracked
from the year they entered until they graduate or lose the grant, so the
resulting dataset has genuine temporal structure (students who go at-risk/drop
stop appearing in later years). The Retained/At-Risk label is NOT produced by a
hard-coded if-else rule: it is a stochastic outcome of many interacting factors,
so the classifiers have to genuinely learn the decision boundary from data.

The script then runs an 80:20 train/test split with 10-fold cross-validation,
trains Decision Tree, Random Forest and Logistic Regression, selects the best
model by F1, and saves it for live prediction in the Flask app.

Run:  python -m ml.train_model
"""
import json
import math
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from ml.prep import CATEGORY_LEVELS, build_columns, df_to_matrix

ACADEMIC_YEARS = ["2022-2023", "2023-2024", "2024-2025"]

INCOME_BOUNDS = {
    "Low": (40000, 120000),
    "Lower-Middle": (120000, 250000),
    "Middle": (250000, 500000),
    "Upper-Middle": (500000, 1200000),
}

SOCIO_WEIGHTS = [0.42, 0.30, 0.20, 0.08]  # Low, Lower-Middle, Middle, Upper-Middle
SCHOLARSHIP_TYPES = ["Academic", "DOST", "CHED_GIA", "Municipal"]


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def generate_history(seed=42, grantees_per_year=140):
    """Simulate three academic years of scholar cohorts (stochastic outcomes)."""
    rng = random.Random(seed)
    records = []

    for entry_year in range(len(ACADEMIC_YEARS)):
        for _ in range(grantees_per_year):
            # Latent, time-invariant attributes of the student.
            ability = rng.gauss(0.0, 1.0)
            engagement = rng.gauss(0.0, 1.0)
            student_effect = rng.gauss(0.0, 0.35)
            socio_status = rng.choices(
                list(INCOME_BOUNDS), weights=SOCIO_WEIGHTS)[0]
            scholarship_type = rng.choice(SCHOLARSHIP_TYPES)
            lo, hi = INCOME_BOUNDS[socio_status]
            annual_income = rng.randint(lo, hi)
            fin_stress = {"Low": 0.20, "Lower-Middle": 0.10, "Middle": 0.04,
                          "Upper-Middle": 0.00}[socio_status]
            pressure = {"Academic": 0.16, "DOST": 0.12, "CHED_GIA": 0.06,
                        "Municipal": 0.03}[scholarship_type]

            # Track the student year by year until they lose the grant.
            for year_idx, year in enumerate(range(entry_year, len(ACADEMIC_YEARS))):
                level_year = year_idx  # 0 = 1st, 1 = 2nd, 2 = 3rd/4th
                year_drift = 0.12 * level_year  # risk grows in later years

                attendance_rate = round(min(100.0, max(40.0,
                    88.0 + 9.0 * engagement + 5.0 * ability + rng.gauss(0, 6))), 1)
                gwa = round(min(5.0, max(1.0,
                    4.55 - 2.45 * ability - 0.15 * engagement
                    + 1.2 * year_drift + rng.gauss(0, 0.22))), 2)
                units_enrolled = rng.randint(12, 21)
                semester_performance = round(min(5.0, max(1.0, gwa + rng.gauss(0, 0.25))), 2)

                fail_rate = max(0.0, 0.02 + (100.0 - attendance_rate) / 320.0
                                + max(0.0, gwa - 3.0) * 0.18 - 0.04 * ability)
                failed_subjects = rng.choices([0, 1, 2, 3], weights=[
                    1 - fail_rate, fail_rate * 0.6, fail_rate * 0.28, fail_rate * 0.12])[0]

                # Retention is a stochastic outcome of interacting factors, not a threshold.
                logit = (
                    2.6
                    + 1.5 * ability
                    + 0.05 * (attendance_rate - 85.0)
                    - 1.7 * (gwa - 2.5)
                    - 1.4 * failed_subjects
                    - 0.9 * fin_stress
                    - 0.6 * pressure
                    - 0.9 * year_drift
                    + student_effect
                )
                retained = 1 if rng.random() < _sig(logit) else 0
                if rng.random() < 0.04:  # imperfect historical records
                    retained = 1 - retained

                records.append({
                    "academic_year": ACADEMIC_YEARS[year],
                    "year_level": level_year + 1,
                    "scholarship_type": scholarship_type,
                    "socio_status": socio_status,
                    "annual_income": annual_income,
                    "gwa": gwa,
                    "failed_subjects": failed_subjects,
                    "units_enrolled": units_enrolled,
                    "attendance_rate": attendance_rate,
                    "semester_performance": semester_performance,
                    "retained": retained,
                })
                if retained == 0:
                    break  # lost the grant; no record in the following year
    return pd.DataFrame(records)


def load_or_generate():
    """Use the real imported history if present, otherwise generate a 3-year demo set."""
    if os.path.exists(Config.DATASET_PATH):
        try:
            df = pd.read_csv(Config.DATASET_PATH)
            if "retained" in df.columns and len(df) >= 20:
                print(f"[data] Using imported history: {len(df)} records from {Config.DATASET_PATH}")
                return df
        except Exception as e:
            print(f"[warn] Could not read {Config.DATASET_PATH} ({e}); regenerating demo data.")
    df = generate_history()
    df.to_csv(Config.DATASET_PATH, index=False)
    print(f"[data] No real history found; generated {len(df)} demo records -> {Config.DATASET_PATH}")
    return df


def train_and_report():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.DATASET_PATH), exist_ok=True)

    df = load_or_generate()

    columns = build_columns()
    X = df_to_matrix(df, columns)
    y = df["retained"].to_numpy()

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=10, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    }

    results = {}
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

    inner = best_model.named_steps["logisticregression"] if hasattr(best_model, "named_steps") else best_model
    if isinstance(inner, (DecisionTreeClassifier, RandomForestClassifier)):
        feats = inner.feature_importances_.tolist()
    else:
        feats = np.abs(inner.coef_[0]).tolist()
    importance = sorted((f, v) for f, v in zip(columns, feats))[::-1]

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

    # Interpretability view: if the best model is not itself a single tree, fit a
    # small Decision Tree as a transparent surrogate so the rules stay explainable.
    if isinstance(inner, DecisionTreeClassifier):
        rules_model = inner
    else:
        rules_model = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42)
        rules_model.fit(X_tr, y_tr)
        print(f"\n[info] best model is {best_name}; exporting interpretable "
              f"Decision Tree surrogate rules for the explanation view.")
    _export_rules(rules_model, columns)

    print(f"\nBest model: {best_name}  -> saved to {Config.MODELS_DIR}")
    print(f"Training dataset ({len(df)} records) -> {Config.DATASET_PATH}")
    return artifact


def _export_rules(tree, columns):
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
