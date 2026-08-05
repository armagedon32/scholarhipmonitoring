"""Database layer: schema, connection helpers, and demo seeding.

Uses SQLite. Passwords are hashed with werkzeug (PBKDF2).
"""
import os
import sqlite3
from contextlib import contextmanager

from werkzeug.security import generate_password_hash

from config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name     TEXT NOT NULL,
    email         TEXT,
    role          TEXT NOT NULL CHECK(role IN ('student','faculty','coordinator','it_expert','admin')),
    is_active     INTEGER DEFAULT 1,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scholarships (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    code                TEXT UNIQUE NOT NULL,
    name                TEXT NOT NULL,
    description         TEXT,
    requirements        TEXT,
    gwa_threshold       REAL DEFAULT 2.50,
    max_failed_subjects INTEGER DEFAULT 0,
    is_active           INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS applications (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id     INTEGER NOT NULL REFERENCES users(id),
    scholarship_id   INTEGER NOT NULL REFERENCES scholarships(id),
    gwa              REAL,
    failed_subjects  INTEGER DEFAULT 0,
    units_enrolled   INTEGER DEFAULT 0,
    attendance_rate  REAL,
    socio_status     TEXT,
    annual_income    REAL,
    year_level       INTEGER,
    documents        TEXT,
    status           TEXT DEFAULT 'pending',
    eligibility      TEXT,
    remarks          TEXT,
    reviewed_by      INTEGER REFERENCES users(id),
    reviewed_at      TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scholars (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id        INTEGER NOT NULL UNIQUE REFERENCES users(id),
    scholarship_id    INTEGER REFERENCES scholarships(id),
    application_id    INTEGER REFERENCES applications(id),
    year_level        INTEGER,
    status            TEXT DEFAULT 'active',
    retention_status  TEXT,
    risk_score        REAL,
    last_predicted_at TEXT,
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS performance_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scholar_id          INTEGER NOT NULL REFERENCES scholars(id),
    semester            TEXT,
    gwa                 REAL NOT NULL,
    failed_subjects     INTEGER DEFAULT 0,
    units_enrolled      INTEGER DEFAULT 0,
    attendance_rate     REAL,
    semester_performance REAL,
    submitted_by        INTEGER REFERENCES users(id),
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS notifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id),
    title      TEXT NOT NULL,
    message    TEXT,
    type       TEXT DEFAULT 'info',
    is_read    INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id),
    action     TEXT,
    details    TEXT,
    ip         TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_applications_applicant ON applications(applicant_id);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_perf_scholar ON performance_records(scholar_id);
CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id);
"""


@contextmanager
def get_db():
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    if os.path.exists(Config.DATABASE):
        os.remove(Config.DATABASE)
    with get_db() as db:
        db.executescript(SCHEMA)


def migrate():
    """Add columns introduced after first deploy to existing databases."""
    with get_db() as db:
        cols = {r["name"] for r in db.execute("PRAGMA table_info(applications)").fetchall()}
        if "annual_income" not in cols:
            db.execute("ALTER TABLE applications ADD COLUMN annual_income REAL")


def _seed(conn):
    users = [
        ("coordinator", "coordinator", "Maria Santos", "maria.santos@knsubic.edu.ph", "coordinator"),
        ("admin", "admin123", "Admin User", "admin@knsubic.edu.ph", "admin"),
        ("faculty", "faculty", "Prof. Jose Ramos", "jose.ramos@knsubic.edu.ph", "faculty"),
        ("faculty2", "faculty", "Prof. Ana Reyes", "ana.reyes@knsubic.edu.ph", "faculty"),
        ("itexpert", "itexpert", "Engr. Daniel Cruz", "daniel.cruz@knsubic.edu.ph", "it_expert"),
    ]
    for username, pw, name, email, role in users:
        conn.execute(
            "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?,?,?,?,?)",
            (username, generate_password_hash(pw), name, email, role),
        )

    scholarships = [
        ("KnS-ACAD", "Kolehiyo ng Subic Academic Scholarship",
         "Merit-based scholarship for scholars with outstanding academic standing.",
         "GWA <= 2.00; no failing grade", 2.00, 0),
        ("KnS-CHED", "CHED Grants-in-Aid (GIA)",
         "Commission on Higher Education grants for enrolled scholars.",
         "GWA <= 2.50; max 2 failed subjects", 2.50, 2),
        ("KnS-MUNI", "Subic Municipal Scholarship",
         "Local government scholarship for residents of Subic, Zambales.",
         "GWA <= 2.50; max 1 failed subject", 2.50, 1),
        ("KnS-DOST", "DOST-SEI Scholarship",
         "Department of Science and Technology scholarship for science & tech programs.",
         "GWA <= 2.00; no failing grade", 2.00, 0),
    ]
    for code, name, desc, req, gwa, fail in scholarships:
        conn.execute(
            "INSERT INTO scholarships (code, name, description, requirements, gwa_threshold, max_failed_subjects) VALUES (?,?,?,?,?,?)",
            (code, name, desc, req, gwa, fail),
        )


def seed_demo():
    with get_db() as db:
        _seed(db)
    # Student demo accounts are created with deterministic credentials.
    _seed_students()


def _seed_students():
    """Create a handful of student demo accounts (password = 'student')."""
    with get_db() as db:
        students = [
            ("stu01", "Maria Reyes", "maria.reyes@knsubic.edu.ph"),
            ("stu02", "John Michael Santos", "john.santos@knsubic.edu.ph"),
            ("stu03", "Angela Cruz", "angela.cruz@knsubic.edu.ph"),
            ("stu04", "Carlo Bautista", "carlo.bautista@knsubic.edu.ph"),
            ("stu05", "Nicole Aquino", "nicole.aquino@knsubic.edu.ph"),
            ("stu06", "Paolo Mendoza", "paolo.mendoza@knsubic.edu.ph"),
        ]
        for uname, name, email in students:
            cur = db.execute(
                "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?,?,?,?,?)",
                (uname, generate_password_hash("student"), name, email, "student"),
            )
            _create_demo_application(db, cur.lastrowid)


def _create_demo_application(db, student_id):
    import random
    random.seed(student_id)
    scholarship_id = random.randint(1, 4)
    gwa = round(random.uniform(1.1, 3.2), 2)
    failed = random.randint(0, 3)
    units = random.randint(15, 21)
    attendance = round(random.uniform(55, 100), 1)
    socio = random.choice(["Low", "Lower-Middle", "Middle", "Upper-Middle"])
    year = random.randint(1, 4)
    annual_income = random.choice([60000, 120000, 250000, 450000])
    db.execute(
        """INSERT INTO applications
           (applicant_id, scholarship_id, gwa, failed_subjects, units_enrolled,
            attendance_rate, socio_status, annual_income, year_level, documents,
            status, eligibility)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (student_id, scholarship_id, gwa, failed, units, attendance, socio,
         annual_income, year, "Grade_Report.pdf, Good_Moral.pdf", "pending",
         "Eligible" if gwa <= 2.5 and failed <= 2 else "Ineligible"),
    )
