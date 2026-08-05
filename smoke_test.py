"""Smoke test: exercises every route with the Flask test client."""
import os
import sys

os.environ.setdefault("MFA_ENABLED", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from database import init_db, seed_demo, get_db

if os.path.exists(Config.DATABASE):
    os.remove(Config.DATABASE)
init_db()
seed_demo()

from app import app

client = app.test_client()
results = []

def check(label, resp, expect=200):
    ok = resp.status_code == expect
    results.append((label, ok, resp.status_code))
    return resp

# Public
check("GET /login", client.get("/"))

# Approve one application first so scholars exist for later flows.
client.post("/", data={"username": "coordinator", "password": "coordinator"})
with client.session_transaction() as sess:
    code = sess["mfa_code"]
client.post("/mfa", data={"code": str(code).zfill(6)}, follow_redirects=True)
with get_db() as db:
    aid = db.execute("SELECT id FROM applications WHERE status='pending' LIMIT 1").fetchone()["id"]
check("approve application (seed)", client.post(
    f"/coordinator/applications/{aid}/review",
    data={"decision": "approved", "remarks": "Meets criteria"},
    follow_redirects=True))
client.get("/logout")

with get_db() as db:
    sid = db.execute("SELECT id FROM scholars LIMIT 1").fetchone()["id"]

# Student flow
r = client.post("/", data={"username": "stu01", "password": "student"}, follow_redirects=True)
check("student login", r)
check("student dashboard", client.get("/student"))
check("student apply form", client.get("/student/apply"))
r = client.post("/student/apply", data={
    "scholarship_id": "2", "gwa": "1.85", "failed_subjects": "0",
    "units_enrolled": "18", "attendance_rate": "92", "socio_status": "Low",
    "documents": "Report.pdf"}, follow_redirects=True)
check("student apply submit", r)
check("notifications", client.get("/notifications"))
check("logout", client.get("/logout", follow_redirects=True))

# Faculty flow
client.post("/", data={"username": "faculty", "password": "faculty"})
check("faculty dashboard", client.get("/faculty"))
check("faculty scholars", client.get("/faculty/scholars"))
check("faculty scholar detail", client.get(f"/faculty/scholar/{sid}"))
r = client.post(f"/faculty/scholar/{sid}", data={
    "semester": "1st Semester", "gwa": "2.20", "failed_subjects": "1",
    "units_enrolled": "18", "attendance_rate": "80", "semester_performance": "2.30"},
    follow_redirects=True)
check("faculty add performance", r)
client.get("/logout")

# Coordinator flow (has MFA)
client.post("/", data={"username": "coordinator", "password": "coordinator"})
r = client.get("/mfa")
check("MFA page", r)
# need the code from session
with client.session_transaction() as sess:
    code = sess["mfa_code"]
r = client.post("/mfa", data={"code": str(code).zfill(6)}, follow_redirects=True)
check("MFA verify", r)
check("coordinator dashboard", client.get("/coordinator"))
check("applications", client.get("/coordinator/applications"))
check("scholars", client.get("/coordinator/scholars"))
check("scholar detail", client.get(f"/coordinator/scholars/{sid}"))
check("run prediction", client.get(f"/coordinator/scholars/{sid}/predict", follow_redirects=True))
check("predictions dashboard", client.get("/coordinator/predictions"))
check("predictions run_all", client.get("/coordinator/predictions?run_all=1", follow_redirects=True))
check("reports", client.get("/coordinator/reports"))
for k in ("applicants", "roster", "performance", "retention"):
    check(f"export {k}", client.get(f"/coordinator/reports/export/{k}"))
check("model page", client.get("/coordinator/model"))
check("users", client.get("/coordinator/users"))
check("audit", client.get("/coordinator/audit"))
r = client.post("/coordinator/notify", data={"title": "Renewal", "message": "Submit renewal docs", "target": "all"},
                follow_redirects=True)
check("broadcast notify", r)
client.get("/logout")

# Admin
client.post("/", data={"username": "admin", "password": "admin123"})
with client.session_transaction() as sess:
    code = sess["mfa_code"]
client.post("/mfa", data={"code": str(code).zfill(6)}, follow_redirects=True)
check("admin dashboard", client.get("/coordinator"))
client.get("/logout")

# Review a pending application
client.post("/", data={"username": "coordinator", "password": "coordinator"})
with client.session_transaction() as sess:
    code = sess["mfa_code"]
client.post("/mfa", data={"code": str(code).zfill(6)}, follow_redirects=True)
with get_db() as db:
    aid = db.execute("SELECT id FROM applications WHERE status='pending' LIMIT 1").fetchone()["id"]
check("approve application", client.post(
    f"/coordinator/applications/{aid}/review",
    data={"decision": "approved", "remarks": "Meets criteria"},
    follow_redirects=True))
check("predictions after approval", client.get("/coordinator/predictions"))

# Access control: student cannot access coordinator page
client.get("/logout")
client.post("/", data={"username": "stu01", "password": "student"})
check("student blocked from coordinator (403)", client.get("/coordinator"), expect=403)

print("\n--- SMOKE TEST RESULTS ---")
fails = 0
for label, ok, status in results:
    print(f"{'PASS' if ok else 'FAIL'}  {status}  {label}")
    if not ok:
        fails += 1
print(f"\n{len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
