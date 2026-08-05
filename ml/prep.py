"""Shared preprocessing / encoding for the retention model.

The model features mirror the feature-selection table in the manuscript:
GWA, Failed Subjects, Units Enrolled, Attendance Rate, Scholarship Type,
Socioeconomic Status, Year Level, and Semester Performance.
"""
import numpy as np
import pandas as pd

CATEGORICAL = ["scholarship_type", "socio_status", "year_level"]

CATEGORY_LEVELS = {
    "scholarship_type": ["Academic", "CHED_GIA", "Municipal", "DOST"],
    "socio_status": ["Low", "Lower-Middle", "Middle", "Upper-Middle"],
    "year_level": [1, 2, 3, 4],
}

NUMERIC = [
    "gwa",
    "failed_subjects",
    "units_enrolled",
    "attendance_rate",
    "semester_performance",
]


def validate_row(row: dict):
    """Coerce/normalise a raw input dict into a clean record."""
    for lvl in CATEGORY_LEVELS["year_level"]:
        if row.get("year_level") == str(lvl):
            row["year_level"] = lvl
    row["year_level"] = int(row.get("year_level") or 1)
    if row["year_level"] not in CATEGORY_LEVELS["year_level"]:
        row["year_level"] = 1
    for cat in ("scholarship_type", "socio_status"):
        if row.get(cat) not in CATEGORY_LEVELS[cat]:
            row[cat] = CATEGORY_LEVELS[cat][0]
    row["gwa"] = float(row.get("gwa") or 3.0)
    row["failed_subjects"] = int(row.get("failed_subjects") or 0)
    row["units_enrolled"] = int(row.get("units_enrolled") or 15)
    row["attendance_rate"] = float(row.get("attendance_rate") or 90.0)
    row["semester_performance"] = float(row.get("semester_performance") or row["gwa"])
    return row


def _column_to_category(col: str):
    """Map an encoded column name (e.g. 'socio_status_Middle') to (cat, level)."""
    for cat in CATEGORICAL:
        prefix = cat + "_"
        if col.startswith(prefix):
            return cat, col[len(prefix):]
    return None


def encode_row(row: dict, columns: list):
    """Turn one cleaned record into a numeric feature vector matching `columns`."""
    row = validate_row(dict(row))
    vec = []
    for col in columns:
        if col in NUMERIC:
            vec.append(float(row[col]))
        else:
            pair = _column_to_category(col)
            if pair is None:
                raise ValueError(f"Unknown encoded column: {col}")
            cat, lvl = pair
            vec.append(1.0 if row[cat] == lvl else 0.0)
    return np.asarray(vec, dtype=np.float64).reshape(1, -1)


def build_columns():
    cols = list(NUMERIC)
    for cat in CATEGORICAL:
        cols += [f"{cat}_{lv}" for lv in CATEGORY_LEVELS[cat]]
    return cols


def df_to_matrix(df: pd.DataFrame, columns: list):
    X = np.zeros((len(df), len(columns)), dtype=np.float64)
    for i, (_, row) in enumerate(df.iterrows()):
        X[i] = encode_row(dict(row), columns).ravel()
    return X
