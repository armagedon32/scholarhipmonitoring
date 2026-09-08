"""Generate data/dataset_attributes.xlsx - a reference for the manuscript/records.

Documents the 9 original attributes of records.xlsx (with their role in the
retention model) and the additional standardized fields introduced only during
preprocessing, so panelists can clearly distinguish source vs processed data.
Also includes a dataset profile sheet computed from the actual training CSV.

Run:  python make_attributes_reference.py
"""
import os
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config

ORIGINAL_ATTRS = [
    ("1", "Name", "Student's full name", "Identifier", "Not used as a predictor"),
    ("2", "Province", "Student's province (ZAMBALES for Kolehiyo ng Subic)", "Informational", "Not used as a predictor"),
    ("3", "GWA", "General weighted average (lower is better)", "Predictor", "Feature (numeric)"),
    ("4", "Units", "Units enrolled for the term", "Predictor", "Feature (numeric)"),
    ("5", "Failed Subject", "Number of failed subjects", "Predictor", "Feature (numeric)"),
    ("6", "Annual Family Income", "Family's annual income (PHP)", "Predictor", "Feature (numeric)"),
    ("7", "Attendance", "Class attendance rate (%)", "Predictor", "Feature (numeric)"),
    ("8", "Socio Economic Status", "Low / Lower-Middle / Middle / Upper-Middle", "Predictor", "Feature (categorical)"),
    ("9", "Prediction", "Retention outcome (Retained / At-Risk)", "Target / Output", "Label being predicted"),
]

PROCESSED_FIELDS = [
    ("student_name", "Derived from Name", "copied with standardized capitalization"),
    ("scholarship_type", "Added (default 'Academic')", "not directly collected; standardized program label"),
    ("semester_performance", "Set equal to GWA", "standardized term-performance score"),
    ("academic_year", "Set to 'Historical'", "informational only; ignored by the model"),
    ("year_level", "Set to 0", "not directly collected in the source file"),
]

HEADER_FILL = PatternFill("solid", fgColor="1B4F8A")
HEADER_FONT = Font(bold=True, color="FFFFFF")
SECTION_FILL = PatternFill("solid", fgColor="D9E2F2")
PROCESSED_FILL = PatternFill("solid", fgColor="FDF3D0")


def main():
    import pandas as pd

    src = Config.DATASET_PATH
    df = pd.read_csv(src) if os.path.exists(src) else None
    n = len(df) if df is not None else 0
    retained = int((df["retained"] == 1).sum()) if df is not None and "retained" in df.columns else 0
    at_risk = n - retained

    wb = Workbook()

    ws = wb.active
    ws.title = "Attributes"
    ws.append(["ORIGINAL SOURCE ATTRIBUTES (records.xlsx)"])
    ws["A1"].fill = SECTION_FILL
    ws["A1"].font = Font(bold=True)
    ws.append(["#", "Attribute", "Meaning", "Role in Model", "Type / Notes"])
    for i, cell in enumerate(ws[2], 1):
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    for row in ORIGINAL_ATTRS:
        ws.append(row)

    ws.append([])
    ws.append(["PROCESSED / STANDARDIZED FIELDS (introduced during preprocessing)"])
    ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=5)
    row = ws.cell(row=ws.max_row, column=1)
    row.fill = SECTION_FILL
    row.font = Font(bold=True)
    ws.append(["Field", "How it is produced", "Note (NOT collected directly from the Excel)"])
    # fill in merged header colors
    for c in range(1, 6):
        ws.cell(row=ws.max_row, column=c).fill = SECTION_FILL
    for field, how, note in PROCESSED_FIELDS:
        rr = ws.max_row + 1
        ws.append([field, how, note])
        for c in range(1, 4):
            cell = ws.cell(row=rr, column=c)
            cell.fill = PROCESSED_FILL
            cell.alignment = Alignment(vertical="top")

    for i, w in enumerate([5, 30, 52, 28, 46], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows():
        for cell in row:
            if cell.row > 2:
                cell.alignment = Alignment(vertical="top", wrap_text=(cell.column >= 3))

    prof = wb.create_sheet("Dataset Profile")
    prof.append(["Dataset Profile (training data: scholar_data.csv)"])
    prof["A1"].fill = SECTION_FILL
    prof["A1"].font = Font(bold=True)
    rows = [
        ("Source file", os.path.basename(src)),
        ("Number of records (instances)", n),
        ("Retained (1)", retained),
        ("At-Risk (0)", at_risk),
        ("Retained share", f"{retained / n * 100:.1f}%" if n else "-"),
        ("At-Risk share", f"{at_risk / n * 100:.1f}%" if n else "-"),
    ]
    for k, v in rows:
        prof.append([k, v])
    if df is not None:
        numeric = ["gwa", "failed_subjects", "units_enrolled", "attendance_rate", "annual_income",
                   "semester_performance"]
        prof.append([])
        prof.append(["Numeric feature summary"])
        prof["A7"].fill = SECTION_FILL
        prof.append(["Feature", "Min", "Max", "Mean"])
        for i, cell in enumerate(prof[8], 1):
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
        for feat in numeric:
            if feat in df.columns:
                prof.append([feat, round(float(df[feat].min()), 2),
                             round(float(df[feat].max()), 2),
                             round(float(df[feat].mean()), 2)])
        prof.append([])
        prof.append(["Categorical feature summary"])
        prof.cell(row=prof.max_row, column=1).fill = SECTION_FILL
        prof.append(["Feature", "Distinct values"])
        for i, cell in enumerate(prof[prof.max_row], 1):
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
        for feat in ["scholarship_type", "socio_status"]:
            if feat in df.columns:
                vals = ", ".join(str(v) for v in sorted(df[feat].dropna().unique()))
                prof.append([feat, vals])
    prof.column_dimensions["A"].width = 30
    prof.column_dimensions["B"].width = 70

    out = os.path.join(Config.DATA_DIR, "dataset_attributes.xlsx")
    wb.save(out)
    print(f"Reference written -> {out}")


if __name__ == "__main__":
    main()