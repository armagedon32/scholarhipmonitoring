"""Web-Based Scholarship Application and Performance Monitoring Platform
with Retention Modeling System Using Classification Algorithms.

Kolehiyo ng Subic -- research prototype for the doctoral study of
Catherine Mae A. Figuerrez (La Consolacion University Philippines).
"""
import csv
import datetime
import io
import os
import secrets

from flask import (Flask, abort, flash, g, redirect, render_template,
                   request, Response, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from config import Config
from database import get_db, init_db, migrate, seed_demo
from ml.predict import model

app = Flask(__name__)
app.config.from_object(Config)

TARGETS = {"accuracy": 0.85, "f1": 0.80}
ROLENAMES = {
    "student": "Student-Scholar",
    "faculty": "Faculty Member",
    "coordinator": "Scholarship Coordinator",
    "it_expert": "IT Expert",
    "admin": "Administrator",
}


def _bootstrap():
    """Create and seed the database on first run (local dev and Railway)."""
    if not os.path.exists(Config.DATABASE):
        os.makedirs(os.path.dirname(Config.DATABASE) or ".", exist_ok=True)
        init_db()
        seed_demo()
    migrate()


_bootstrap()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_user(user_id=None):
    if user_id is None:
        return None
    with get_db() as db:
        row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def login_required(view):
    def wrapped(*args, **kwargs):
        if "uid" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        g.user = get_user(session["uid"])
        if g.user is None:
            session.clear()
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped


def roles_required(*roles):
    def deco(view):
        def wrapped(*args, **kwargs):
            if "uid" not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login"))
            g.user = get_user(session["uid"])
            if g.user is None or g.user["role"] not in roles:
                abort(403)
            return view(*args, **kwargs)
        wrapped.__name__ = view.__name__
        return wrapped
    return deco


def audit(action, details=""):
    with get_db() as db:
        db.execute(
            "INSERT INTO audit_logs (user_id, action, details, ip) VALUES (?,?,?,?)",
            (session.get("uid"), action, details, request.remote_addr),
        )


def notify(user_id, title, message, ntype="info"):
    with get_db() as db:
        db.execute(
            "INSERT INTO notifications (user_id, title, message, type) VALUES (?,?,?,?)",
            (user_id, title, message, ntype),
        )


def unread_count(user_id):
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) c FROM notifications WHERE user_id=? AND is_read=0",
            (user_id,)).fetchone()
    return row["c"]


def map_scholarship_type(code):
    code = (code or "").upper()
    if "DOST" in code:
        return "DOST"
    if "CHED" in code:
        return "CHED_GIA"
    if "MUNI" in code or "MUNICIPAL" in code:
        return "Municipal"
    return "Academic"


def auto_eligibility(gwa, failed, threshold, max_failed):
    """Check eligibility. A None threshold/max_failed means 'no academic requirement'
    (e.g. LGU scholarship whose only qualification is residency in Zambales)."""
    try:
        ok = True
        if threshold is not None:
            ok = ok and gwa <= threshold
        if max_failed is not None:
            ok = ok and failed <= max_failed
        return ok
    except TypeError:
        return False


def scholarship_window(sch):
    """Return (is_open, message) based on the application start/deadline dates."""
    sch = dict(sch)
    today = datetime.date.today().isoformat()
    if sch.get("apply_start") and today < sch["apply_start"]:
        return False, f"Applications open on {sch['apply_start']}"
    if sch.get("apply_deadline") and today > sch["apply_deadline"]:
        return False, f"Application deadline passed on {sch['apply_deadline']}"
    return True, ""


def scholarship_row(scholar_id):
    with get_db() as db:
        s = db.execute("""
            SELECT s.*, u.full_name, u.email, u.username,
                   sch.code AS sch_code, sch.name AS sch_name,
                   sch.gwa_threshold, sch.max_failed_subjects,
                   a.socio_status, a.gwa AS app_gwa, a.failed_subjects AS app_failed,
                   a.attendance_rate AS app_att, a.units_enrolled AS app_units,
                   a.annual_income AS app_income,
                   a.year_level AS app_year
            FROM scholars s
            JOIN users u ON u.id = s.student_id
            LEFT JOIN scholarships sch ON sch.id = s.scholarship_id
            LEFT JOIN applications a ON a.id = s.application_id
            WHERE s.id = ?
        """, (scholar_id,)).fetchone()
        if s is None:
            return None
        s = dict(s)
        perf = db.execute(
            "SELECT * FROM performance_records WHERE scholar_id=? ORDER BY id DESC LIMIT 1",
            (scholar_id,)).fetchone()
        s["latest_perf"] = dict(perf) if perf else None
        return s


def feature_row(s):
    """Build the classifier input vector for a scholar row dict."""
    lp = s.get("latest_perf") or {}
    gwa = lp.get("gwa") if lp.get("gwa") is not None else (s.get("app_gwa") or 3.0)
    failed = lp.get("failed_subjects") if lp.get("failed_subjects") is not None else (s.get("app_failed") or 0)
    units = lp.get("units_enrolled") if lp.get("units_enrolled") is not None else (s.get("app_units") or 15)
    att = lp.get("attendance_rate") if lp.get("attendance_rate") is not None else (s.get("app_att") or 90.0)
    sp = lp.get("semester_performance") if lp.get("semester_performance") is not None else gwa
    return {
        "gwa": gwa,
        "failed_subjects": failed,
        "units_enrolled": units,
        "attendance_rate": att,
        "scholarship_type": map_scholarship_type(s.get("sch_code")),
        "socio_status": s.get("socio_status") or "Middle",
        "annual_income": s.get("app_income") or 250000,
        "semester_performance": sp,
    }


def run_prediction(scholar_id, force=True):
    s = scholarship_row(scholar_id)
    if s is None:
        return None
    status, prob = model.predict(feature_row(s))
    risk = round((1 - prob) * 100, 1)
    with get_db() as db:
        db.execute(
            """UPDATE scholars SET retention_status=?, risk_score=?, last_predicted_at=datetime('now')
               WHERE id=?""", (status, risk, scholar_id))
    return {"status": status, "risk": risk, "prob": round(prob, 4)}


def all_scholars_with_latest():
    with get_db() as db:
        rows = db.execute("""
            SELECT s.*, u.full_name, u.username, u.email,
                   sch.code AS sch_code, sch.name AS sch_name,
                   sch.gwa_threshold, sch.max_failed_subjects,
                   a.socio_status, a.gwa AS app_gwa, a.failed_subjects AS app_failed,
                   a.attendance_rate AS app_att, a.units_enrolled AS app_units,
                   a.annual_income AS app_income,
                   a.year_level AS app_year,
                   a.created_at AS app_submitted, a.reviewed_at AS app_approved
            FROM scholars s
            JOIN users u ON u.id = s.student_id
            LEFT JOIN scholarships sch ON sch.id = s.scholarship_id
            LEFT JOIN applications a ON a.id = s.application_id
            ORDER BY s.risk_score IS NULL, s.risk_score DESC, u.full_name
        """).fetchall()
        out = []
        for r in rows:
            r = dict(r)
            lp = db.execute(
                "SELECT * FROM performance_records WHERE scholar_id=? ORDER BY id DESC LIMIT 1",
                (r["id"],)).fetchone()
            r["latest_perf"] = dict(lp) if lp else None
            out.append(r)
        return out


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if "uid" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as db:
            row = db.execute("SELECT * FROM users WHERE username=? AND is_active=1",
                             (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            session["uid"] = row["id"]
            audit("LOGIN", f"{row['full_name']} logged in")
            if Config.MFA_ENABLED and row["role"] in ("coordinator", "admin"):
                session["mfa_code"] = secrets.randbelow(1000000)
                session["mfa_user"] = row["id"]
                session.pop("uid", None)
                return redirect(url_for("mfa"))
            return redirect(url_for("home"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/mfa", methods=["GET", "POST"])
def mfa():
    if "mfa_user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        code = request.form.get("code", "")
        if code.isdigit() and int(code) == session.get("mfa_code"):
            session["uid"] = session.pop("mfa_user")
            session.pop("mfa_code", None)
            audit("LOGIN", "Coordinator/Admin login completed with MFA")
            return redirect(url_for("home"))
        flash("Invalid code. Try again.", "error")
    return render_template("mfa.html", code="%06d" % session.get("mfa_code", 0))


@app.route("/logout")
def logout():
    if "uid" in session:
        audit("LOGOUT", "User logged out")
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    role = g.user["role"]
    if role == "student":
        return redirect(url_for("student_dashboard"))
    if role == "faculty":
        return redirect(url_for("faculty_dashboard"))
    return redirect(url_for("coord_dashboard"))


@app.context_processor
def inject_globals():
    unread = unread_count(session.get("uid")) if session.get("uid") else 0
    ep = request.endpoint or ""
    sections = {
        "student": ("student_dashboard", "student_apply"),
        "faculty": ("faculty_dashboard", "faculty_scholars", "faculty_scholar"),
        "apps": ("coord_applications",),
        "scholars": ("coord_scholars", "coord_scholar_detail", "coord_scholar_predict"),
        "scholarships": ("coord_scholarships",),
        "predictions": ("coord_predictions",),
        "reports": ("coord_reports",),
        "model": ("coord_model",),
        "users": ("coord_users",),
        "audit": ("coord_audit",),
        "notifications": ("notifications",),
    }
    active_section = next((k for k, eps in sections.items() if ep in eps), "")
    return {
        "rolenames": ROLENAMES,
        "unread_count": unread,
        "user": get_user(session.get("uid")),
        "active_section": active_section,
        "model_name": model.selected_algorithm(),
        "model_metrics": model.metrics().get(model.selected_algorithm(), {}),
        "targets": TARGETS,
    }


# --------------------------------------------------------------------------
# Notifications (all roles)
# --------------------------------------------------------------------------

@app.route("/notifications")
@login_required
def notifications():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM notifications WHERE user_id=? ORDER BY is_read, id DESC",
            (g.user["id"],)).fetchall()
        db.execute("UPDATE notifications SET is_read=1 WHERE user_id=?", (g.user["id"],))
    return render_template("notifications.html", notifications=[dict(r) for r in rows])


@app.route("/notifications/<int:nid>/read")
@login_required
def notification_read(nid):
    with get_db() as db:
        db.execute("UPDATE notifications SET is_read=1 WHERE id=? AND user_id=?",
                   (nid, g.user["id"]))
    return redirect(url_for("notifications"))


# --------------------------------------------------------------------------
# Student
# --------------------------------------------------------------------------

@app.route("/student")
@roles_required("student")
def student_dashboard():
    with get_db() as db:
        apps = db.execute("""
            SELECT a.*, s.name, s.code FROM applications a
            JOIN scholarships s ON s.id = a.scholarship_id
            WHERE a.applicant_id=? AND a.status='approved'
            ORDER BY a.id DESC""", (g.user["id"],)).fetchall()
        scholar = db.execute("""
            SELECT s.*, sch.name AS sch_name, sch.code AS sch_code,
                   sch.gwa_threshold, sch.max_failed_subjects
            FROM scholars s LEFT JOIN scholarships sch ON sch.id = s.scholarship_id
            WHERE s.student_id=?""", (g.user["id"],)).fetchone()
    scholar = dict(scholar) if scholar else None
    if scholar:
        perf_rows = None
        with get_db() as db:
            perf_rows = db.execute(
                "SELECT * FROM performance_records WHERE scholar_id=? ORDER BY id DESC",
                (scholar["id"],)).fetchall()
        scholar["perf"] = [dict(r) for r in perf_rows]
    return render_template("student/dashboard.html",
                           applications=[dict(r) for r in apps],
                           scholar=scholar)


@app.route("/student/apply", methods=["GET", "POST"])
@roles_required("student")
def student_apply():
    with get_db() as db:
        scholarships = db.execute(
            "SELECT * FROM scholarships WHERE is_active=1 ORDER BY name").fetchall()
    if request.method == "POST":
        try:
            gwa = float(request.form["gwa"])
            failed = int(request.form["failed_subjects"])
            units = int(request.form["units_enrolled"])
            att = float(request.form["attendance_rate"])
            socio = request.form["socio_status"]
            annual_income = float(request.form["annual_income"])
            sch_id = int(request.form["scholarship_id"])
        except (KeyError, ValueError):
            flash("Please fill in all required fields with valid values.", "error")
            return redirect(url_for("student_apply"))
        if not 1.0 <= gwa <= 5.0:
            flash("GWA must be between 1.00 and 5.00.", "error")
            return redirect(url_for("student_apply"))
        with get_db() as db:
            sch = db.execute("SELECT * FROM scholarships WHERE id=? AND is_active=1",
                             (sch_id,)).fetchone()
        if sch is None:
            flash("Selected scholarship is not available.", "error")
            return redirect(url_for("student_apply"))
        window_open, window_msg = scholarship_window(sch)
        if not window_open:
            flash(f"Cannot apply for {sch['name']}: {window_msg}.", "error")
            return redirect(url_for("student_apply"))
        # Save uploaded documents beside the database (persists on the Railway volume).
        saved_names = []
        for f in request.files.getlist("document_uploads"):
            if not f or not f.filename:
                continue
            fname = secure_filename(f.filename)
            if not fname:
                continue
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext not in Config.ALLOWED_EXTENSIONS:
                flash(f"File type not allowed for {f.filename}. "
                      f"Allowed: {', '.join(sorted(Config.ALLOWED_EXTENSIONS))}.", "error")
                return redirect(url_for("student_apply"))
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            dest = os.path.join(Config.UPLOAD_FOLDER, fname)
            f.save(dest)
            saved_names.append(fname)
        typed = [d.strip() for d in request.form.get("documents", "").split(",") if d.strip()]
        all_docs = ", ".join(typed + saved_names) if (typed or saved_names) else ""
        with get_db() as db:
            eligible = auto_eligibility(gwa, failed, sch["gwa_threshold"], sch["max_failed_subjects"])
            if not eligible:
                flash(
                    f"You are not eligible for {sch['name']} based on the scholarship requirements "
                    f"(GWA <= {sch['gwa_threshold']}, max {sch['max_failed_subjects']} failed subject(s)).",
                    "error")
                return redirect(url_for("student_apply"))
            db.execute("""
                INSERT INTO applications
                (applicant_id, scholarship_id, gwa, failed_subjects, units_enrolled,
                 attendance_rate, socio_status, annual_income, documents, status, eligibility)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (g.user["id"], sch_id, gwa, failed, units, att, socio, annual_income,
                 all_docs, "pending",
                 "Eligible" if eligible else "Ineligible"))
        audit("APPLY", f"{g.user['full_name']} submitted an application")
        notify(g.user["id"], "Application Submitted",
               "Your scholarship application has been received and is pending review.", "success")
        flash("Application submitted successfully!", "success")
        return redirect(url_for("student_dashboard"))
    sch_list = []
    for r in scholarships:
        d = dict(r)
        d["window_open"], d["window_msg"] = scholarship_window(r)
        sch_list.append(d)
    return render_template("student/apply.html", scholarships=sch_list)


# --------------------------------------------------------------------------
# Faculty
# --------------------------------------------------------------------------

@app.route("/faculty")
@roles_required("faculty")
def faculty_dashboard():
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) c FROM scholars WHERE status='active'").fetchone()["c"]
        at_risk = db.execute(
            "SELECT COUNT(*) c FROM scholars WHERE status='active' AND retention_status='At-Risk'"
        ).fetchone()["c"]
        recent = db.execute("""
            SELECT s.id, u.full_name, sch.name AS sch_name, p.semester, p.gwa,
                   p.failed_subjects, p.attendance_rate, p.created_at
            FROM performance_records p
            JOIN scholars s ON s.id = p.scholar_id
            JOIN users u ON u.id = s.student_id
            LEFT JOIN scholarships sch ON sch.id = s.scholarship_id
            ORDER BY p.id DESC LIMIT 8""").fetchall()
    return render_template("faculty/dashboard.html",
                           total=total, at_risk=at_risk,
                           recent=[dict(r) for r in recent])


@app.route("/faculty/scholars")
@roles_required("faculty")
def faculty_scholars():
    rows = all_scholars_with_latest()
    return render_template("faculty/scholars.html", scholars=rows)


@app.route("/faculty/scholar/<int:scholar_id>", methods=["GET", "POST"])
@roles_required("faculty")
def faculty_scholar(scholar_id):
    s = scholarship_row(scholar_id)
    if s is None:
        abort(404)
    if request.method == "POST":
        try:
            gwa = float(request.form["gwa"])
            failed = int(request.form["failed_subjects"])
            units = int(request.form["units_enrolled"])
            att = float(request.form["attendance_rate"])
            sp = float(request.form["semester_performance"])
        except (KeyError, ValueError):
            flash("Invalid values entered.", "error")
            return redirect(url_for("faculty_scholar", scholar_id=scholar_id))
        semester = request.form.get("semester", "1st Sem")
        with get_db() as db:
            db.execute("""
                INSERT INTO performance_records
                (scholar_id, semester, gwa, failed_subjects, units_enrolled,
                 attendance_rate, semester_performance, submitted_by)
                VALUES (?,?,?,?,?,?,?,?)""",
                (scholar_id, semester, gwa, failed, units, att, sp, g.user["id"]))
        audit("PERF_ADD", f"Performance added for scholar #{scholar_id} by {g.user['full_name']}")
        flash("Performance record saved. Retention status will be recomputed.", "success")
        return redirect(url_for("faculty_scholar", scholar_id=scholar_id))
    with get_db() as db:
        perf = db.execute(
            "SELECT * FROM performance_records WHERE scholar_id=? ORDER BY id DESC",
            (scholar_id,)).fetchall()
    return render_template("faculty/scholar.html", s=s, perf=[dict(r) for r in perf])


# --------------------------------------------------------------------------
# Coordinator / Admin
# --------------------------------------------------------------------------

@app.route("/coordinator")
@roles_required("coordinator", "admin", "it_expert")
def coord_dashboard():
    with get_db() as db:
        stats = db.execute("""
            SELECT
              (SELECT COUNT(*) FROM applications) AS total_apps,
              (SELECT COUNT(*) FROM applications WHERE status='pending') AS pending_apps,
              (SELECT COUNT(*) FROM applications WHERE status='approved') AS approved_apps,
              (SELECT COUNT(*) FROM scholars WHERE status='active') AS active_scholars,
              (SELECT COUNT(*) FROM scholars WHERE status='active' AND retention_status='At-Risk') AS at_risk,
              (SELECT COUNT(*) FROM users WHERE role='student') AS student_count
        """).fetchone()
        recent_apps = db.execute("""
            SELECT a.*, u.full_name, s.name AS sch_name FROM applications a
            JOIN users u ON u.id = a.applicant_id
            JOIN scholarships s ON s.id = a.scholarship_id
            ORDER BY a.id DESC LIMIT 6""").fetchall()
        recent_scholars = db.execute("""
            SELECT s.*, u.full_name, sch.name AS sch_name FROM scholars s
            JOIN users u ON u.id = s.student_id
            LEFT JOIN scholarships sch ON sch.id = s.scholarship_id
            ORDER BY s.risk_score IS NULL, s.risk_score DESC LIMIT 6""").fetchall()
        scholarships = db.execute(
            "SELECT * FROM scholarships ORDER BY is_active DESC, name").fetchall()
    return render_template("coordinator/dashboard.html",
                           stats=dict(stats),
                           recent_apps=[dict(r) for r in recent_apps],
                           recent_scholars=[dict(r) for r in recent_scholars],
                           scholarships=[dict(r) for r in scholarships])


@app.route("/coordinator/applications")
@roles_required("coordinator", "admin", "it_expert")
def coord_applications():
    with get_db() as db:
        rows = db.execute("""
            SELECT a.*, u.full_name, u.username, s.name AS sch_name, s.code AS sch_code,
                   s.gwa_threshold, s.max_failed_subjects
            FROM applications a
            JOIN users u ON u.id = a.applicant_id
            JOIN scholarships s ON s.id = a.scholarship_id
            ORDER BY CASE a.status WHEN 'pending' THEN 0 WHEN 'reviewing' THEN 1 ELSE 2 END, a.id DESC
        """).fetchall()
    return render_template("coordinator/applications.html", apps=[dict(r) for r in rows])


@app.route("/coordinator/applications/clear-pending", methods=["POST"])
@roles_required("coordinator", "admin")
def coord_clear_pending():
    with get_db() as db:
        rows = db.execute(
            "SELECT id FROM applications WHERE status='pending'").fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            flash("No pending applications to clear.", "info")
            return redirect(url_for("coord_applications"))
        placeholders = ",".join("?" * len(ids))
        db.execute(f"DELETE FROM applications WHERE id IN ({placeholders})", ids)
    audit("CLEAR", f"{g.user['full_name']} cleared {len(ids)} pending application(s)")
    flash(f"Cleared {len(ids)} pending application(s).", "success")
    return redirect(url_for("coord_applications"))


@app.route("/coordinator/applications/<int:aid>/review", methods=["POST"])
@roles_required("coordinator", "admin")
def coord_review(aid):
    decision = request.form.get("decision")
    remarks = request.form.get("remarks", "").strip()
    if decision not in ("approved", "rejected"):
        flash("Invalid decision.", "error")
        return redirect(url_for("coord_applications"))

    scholar_id = None
    applicant_id = None
    with get_db() as db:
        app_row = db.execute(
            "SELECT * FROM applications WHERE id=?", (aid,)).fetchone()
        if app_row is None:
            abort(404)
        applicant_id = app_row["applicant_id"]
        db.execute(
            """UPDATE applications SET status=?, remarks=?, reviewed_by=?, reviewed_at=datetime('now')
               WHERE id=?""", (decision, remarks, g.user["id"], aid))
        if decision == "approved":
            existing = db.execute(
                "SELECT id FROM scholars WHERE student_id=?",
                (applicant_id,)).fetchone()
            if existing is None:
                cur = db.execute("""
                    INSERT INTO scholars (student_id, scholarship_id, application_id, year_level, status)
                    VALUES (?,?,?,?, 'active')""",
                    (applicant_id, app_row["scholarship_id"], aid,
                     app_row["year_level"] or 1))
                scholar_id = cur.lastrowid
            else:
                scholar_id = existing["id"]

    if decision == "approved":
        notify(applicant_id, "Application Approved",
               "Congratulations! Your scholarship application has been approved.", "success")
        audit("APPROVE", f"Application #{aid} approved; scholar #{scholar_id} created")
        run_prediction(scholar_id)
    else:
        audit("REJECT", f"Application #{aid} rejected by {g.user['full_name']}")
        notify(applicant_id, "Application Rejected",
               "We regret to inform you that your application was not approved."
               + (f" Reason: {remarks}" if remarks else ""), "error")
    flash(f"Application #{aid} marked as {decision}.", "success")
    return redirect(url_for("coord_applications"))


@app.route("/coordinator/scholars")
@roles_required("coordinator", "admin", "it_expert")
def coord_scholars():
    rows = all_scholars_with_latest()
    return render_template("coordinator/scholars.html", scholars=rows)


@app.route("/coordinator/scholars/<int:scholar_id>")
@roles_required("coordinator", "admin", "it_expert")
def coord_scholar_detail(scholar_id):
    s = scholarship_row(scholar_id)
    if s is None:
        abort(404)
    with get_db() as db:
        perf = db.execute(
            "SELECT * FROM performance_records WHERE scholar_id=? ORDER BY id DESC",
            (scholar_id,)).fetchall()
    return render_template("coordinator/scholar.html", s=s, perf=[dict(r) for r in perf])


@app.route("/coordinator/scholars/<int:scholar_id>/predict")
@roles_required("coordinator", "admin", "it_expert")
def coord_scholar_predict(scholar_id):
    s = scholarship_row(scholar_id)
    if s is None:
        abort(404)
    result = run_prediction(scholar_id)
    audit("PREDICT", f"Retention prediction run for scholar #{scholar_id}")
    flash(f"Prediction complete: {result['status']} (risk {result['risk']}%).", "info")
    return redirect(url_for("coord_scholar_detail", scholar_id=scholar_id))


@app.route("/coordinator/predictions")
@roles_required("coordinator", "admin", "it_expert")
def coord_predictions():
    rows = all_scholars_with_latest()
    pending = [r for r in rows if not r["retention_status"]]
    if request.args.get("run_all"):
        for r in rows:
            run_prediction(r["id"])
        audit("PREDICT_BATCH", f"Batch prediction run for {len(rows)} scholars")
        flash(f"Retention predictions updated for {len(rows)} scholars.", "success")
        return redirect(url_for("coord_predictions"))
    summary = {
        "retained": sum(1 for r in rows if r["retention_status"] == "Retained"),
        "at_risk": sum(1 for r in rows if r["retention_status"] == "At-Risk"),
        "pending": len(pending),
        "total": len(rows),
    }
    return render_template("coordinator/predictions.html", scholars=rows, summary=summary)


@app.route("/coordinator/reports")
@roles_required("coordinator", "admin", "it_expert")
def coord_reports():
    rows = all_scholars_with_latest()
    with get_db() as db:
        apps = db.execute("""
            SELECT a.*, u.full_name, s.name AS sch_name FROM applications a
            JOIN users u ON u.id = a.applicant_id
            JOIN scholarships s ON s.id = a.scholarship_id
            ORDER BY a.id DESC""").fetchall()
    return render_template("coordinator/reports.html",
                           applicants=[dict(r) for r in apps], scholars=rows)


def _csv_response(filename, headers, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/coordinator/reports/export/<kind>")
@roles_required("coordinator", "admin", "it_expert")
def coord_export(kind):
    if kind == "applicants":
        with get_db() as db:
            rows = db.execute("""
                SELECT u.full_name, s.name, a.status, a.eligibility, a.gwa,
                       a.failed_subjects, a.annual_income, a.created_at FROM applications a
                JOIN users u ON u.id=a.applicant_id
                JOIN scholarships s ON s.id=a.scholarship_id ORDER BY a.id""").fetchall()
        return _csv_response("applicant_list.csv",
                             ["Applicant", "Scholarship", "Status", "Eligibility", "GWA", "Failed Subjects",
                              "Annual Income (PHP)", "Submitted"],
                             [list(r) for r in rows])
    if kind == "roster":
        rows = all_scholars_with_latest()
        return _csv_response("scholar_roster.csv",
                             ["Scholar", "Scholarship", "Status",
                              "Application Submitted", "Application Approved"],
                             [[r["full_name"], r["sch_name"], r["status"],
                               r["app_submitted"] or "", r["app_approved"] or ""] for r in rows])
    if kind == "performance":
        rows = all_scholars_with_latest()
        return _csv_response("performance_summary.csv",
                             ["Scholar", "Semester", "GWA", "Failed", "Units", "Attendance", "Semester Perf"],
                             [[r["full_name"], (r["latest_perf"] or {}).get("semester", "-"),
                               (r["latest_perf"] or {}).get("gwa", "-"),
                               (r["latest_perf"] or {}).get("failed_subjects", "-"),
                               (r["latest_perf"] or {}).get("units_enrolled", "-"),
                               (r["latest_perf"] or {}).get("attendance_rate", "-"),
                               (r["latest_perf"] or {}).get("semester_performance", "-")] for r in rows])
    if kind == "retention":
        rows = all_scholars_with_latest()
        return _csv_response("retention_report.csv",
                             ["Scholar", "Scholarship", "Status of Student", "Risk Score (%)",
                              "Date Submitted", "Date Approved"],
                             [[r["full_name"], r["sch_name"], r["retention_status"] or "Pending",
                               r["risk_score"] or "", r["app_submitted"] or "", r["app_approved"] or ""] for r in rows])
    abort(404)


@app.route("/coordinator/notify", methods=["POST"])
@roles_required("coordinator", "admin")
def coord_notify():
    title = request.form.get("title", "").strip()
    message = request.form.get("message", "").strip()
    target = request.form.get("target", "all")
    if not title or not message:
        flash("Title and message are required.", "error")
        return redirect(url_for("coord_dashboard"))
    with get_db() as db:
        if target == "at_risk":
            ids = [r["user_id"] for r in db.execute(
                "SELECT student_id AS user_id FROM scholars WHERE retention_status='At-Risk'")]
        elif target == "all":
            ids = [r["id"] for r in db.execute("SELECT id FROM users WHERE role='student'")]
        else:
            ids = [int(target)]
    for uid in ids:
        notify(uid, title, message, "alert")
    audit("NOTIFY", f"Notification sent to {len(ids)} student(s): {title}")
    flash(f"Notification sent to {len(ids)} recipient(s).", "success")
    return redirect(url_for("coord_dashboard"))


def _dataset_info():
    """Return (records, school_years) of the training dataset, with sensible fallbacks."""
    if os.path.exists(Config.DATASET_PATH):
        try:
            with open(Config.DATASET_PATH, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            if rows:
                years = {r.get("academic_year") for r in rows if r.get("academic_year")}
                years.discard("Historical")
                if not years:
                    years = {"3+ years"}
                return len(rows), len(years)
        except Exception:
            pass
    return 601, 3


@app.route("/coordinator/model")
@roles_required("coordinator", "admin", "it_expert")
def coord_model():
    selected = model.selected_algorithm()
    metrics = model.metrics().get(selected, {})
    importance = model.feature_importance()
    rules_path = os.path.join(Config.MODELS_DIR, "decision_rules.txt")
    rules = []
    if os.path.exists(rules_path):
        with open(rules_path, encoding="utf-8") as fh:
            rules = [l for l in fh.read().splitlines() if l.strip()][:40]
    return render_template("coordinator/model.html",
                           selected=selected, metrics=metrics,
                           importance=importance, rules=rules,
                           record_count=_dataset_info()[0],
                           years_trained=_dataset_info()[1])


@app.route("/coordinator/users", methods=["GET", "POST"])
@roles_required("coordinator", "admin")
def coord_users():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        role = request.form.get("role")
        if role not in ROLENAMES:
            role = "student"
        if not username or not password or not full_name:
            flash("Username, password and full name are required.", "error")
            return redirect(url_for("coord_users"))
        try:
            with get_db() as db:
                db.execute(
                    "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?,?,?,?,?)",
                    (username, generate_password_hash(password), full_name, email, role))
            audit("USER_ADD", f"User created: {username} ({role})")
            flash(f"User {username} created.", "success")
        except Exception:
            flash("Username already exists.", "error")
        return redirect(url_for("coord_users"))
    with get_db() as db:
        rows = db.execute("SELECT id, username, full_name, email, role, is_active FROM users ORDER BY role, id").fetchall()
    return render_template("coordinator/users.html", users=[dict(r) for r in rows])


@app.route("/coordinator/scholarships", methods=["GET", "POST"])
@roles_required("coordinator", "admin")
def coord_scholarships():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "toggle":
            sid = int(request.form.get("scholarship_id", 0))
            with get_db() as db:
                row = db.execute(
                    "SELECT is_active FROM scholarships WHERE id=?", (sid,)).fetchone()
                if row is not None:
                    db.execute("UPDATE scholarships SET is_active=? WHERE id=?",
                               (1 - row["is_active"], sid))
            audit("SCHOLARSHIP_TOGGLE", f"Scholarship #{sid} active state toggled")
            flash("Scholarship availability updated.", "success")
            return redirect(url_for("coord_scholarships"))
        code = request.form.get("code", "").strip()
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        requirements = request.form.get("requirements", "").strip()
        apply_start = request.form.get("apply_start", "").strip() or None
        apply_deadline = request.form.get("apply_deadline", "").strip() or None
        try:
            gwa_raw = request.form.get("gwa_threshold", "").strip()
            failed_raw = request.form.get("max_failed_subjects", "").strip()
            gwa_threshold = float(gwa_raw) if gwa_raw else None
            max_failed_subjects = int(failed_raw) if failed_raw else None
        except (TypeError, ValueError):
            flash("Invalid GWA threshold or max failed subjects.", "error")
            return redirect(url_for("coord_scholarships"))
        if not code or not name:
            flash("Scholarship code and name are required.", "error")
            return redirect(url_for("coord_scholarships"))
        sid = request.form.get("scholarship_id")
        try:
            with get_db() as db:
                if sid:
                    db.execute("""UPDATE scholarships SET code=?, name=?, description=?,
                                  requirements=?, gwa_threshold=?, max_failed_subjects=?,
                                  apply_start=?, apply_deadline=?
                                  WHERE id=?""",
                               (code, name, description, requirements,
                                gwa_threshold, max_failed_subjects,
                                apply_start, apply_deadline, sid))
                else:
                    db.execute("""INSERT INTO scholarships
                                  (code, name, description, requirements,
                                   gwa_threshold, max_failed_subjects,
                                   apply_start, apply_deadline, is_active)
                                  VALUES (?,?,?,?,?,?,?,?,1)""",
                               (code, name, description, requirements,
                                gwa_threshold, max_failed_subjects,
                                apply_start, apply_deadline))
            if sid:
                audit("SCHOLARSHIP_EDIT", f"Scholarship updated: {code}")
                flash(f"Scholarship {code} updated.", "success")
            else:
                audit("SCHOLARSHIP_ADD", f"Scholarship added: {code}")
                flash(f"Scholarship {code} added.", "success")
        except Exception:
            flash("Scholarship code already exists.", "error")
        return redirect(url_for("coord_scholarships"))
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM scholarships ORDER BY is_active DESC, code").fetchall()
    edit_id = request.args.get("edit", type=int)
    editing = None
    if edit_id:
        with get_db() as db:
            editing = db.execute(
                "SELECT * FROM scholarships WHERE id=?", (edit_id,)).fetchone()
    return render_template("coordinator/scholarships.html",
                           scholarships=[dict(r) for r in rows],
                           editing=dict(editing) if editing else None)


@app.route("/coordinator/audit")
@roles_required("coordinator", "admin", "it_expert")
def coord_audit():
    with get_db() as db:
        rows = db.execute("""
            SELECT l.*, u.full_name FROM audit_logs l
            LEFT JOIN users u ON u.id = l.user_id
            ORDER BY l.id DESC LIMIT 100""").fetchall()
    return render_template("coordinator/audit.html", logs=[dict(r) for r in rows])


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403,
                           message="You do not have permission to access this page."), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


if __name__ == "__main__":
    import threading
    import webbrowser

    # Single-process mode when launched from run.bat (FLASK_USE_RELOADER=0).
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "1") == "1"
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", "5000"))
    if not use_reloader and not os.environ.get("PORT"):
        # Local convenience: open the browser only after the server has bound.
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=use_reloader)
