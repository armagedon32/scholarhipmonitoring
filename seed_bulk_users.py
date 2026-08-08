"""Bulk-create user accounts so you don't have to add them one by one.

By default this creates 100 student accounts (password: 'student'), each with a
demo application so their dashboards and the coordinator's review queue are
already populated. Existing usernames are skipped.

Usage:
  python seed_bulk_users.py                  # 100 students (stu001..stu100)
  python seed_bulk_users.py --count 200      # 200 students
  python seed_bulk_users.py --role faculty --count 8
  python seed_bulk_users.py --count 100 --no-apps   # accounts only, no applications
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from werkzeug.security import generate_password_hash

from database import _create_demo_application, get_db

FIRST_NAMES = [
    "Maria", "Juan", "Jose", "Angela", "Carlo", "Nicole", "Paolo", "Kyla",
    "Rafael", "Sofia", "Miguel", "Andrea", "Mark", "Dianne", "Joshua", "Camille",
    "Gabriel", "Erica", "Daniel", "Patricia", "Christian", "Jasmine", "Adrian", "Mariel",
]
LAST_NAMES = [
    "Reyes", "Santos", "Cruz", "Bautista", "Aquino", "Mendoza", "Garcia",
    "Torres", "Ramos", "Villanueva", "Del Rosario", "Domingo", "Salazar",
    "Navarro", "Vergara", "Lopez", "Flores", "Castillo", "Roxas", "De Leon",
]
PASSWORD = {
    "student": "student",
    "faculty": "faculty",
    "coordinator": "coordinator",
    "it_expert": "itexpert",
    "admin": "admin123",
}


def generate_name(seed):
    import random
    rng = random.Random(seed)
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def main():
    ap = argparse.ArgumentParser(description="Bulk-create school accounts")
    ap.add_argument("--count", type=int, default=100, help="Number of accounts (default 100)")
    ap.add_argument("--role", choices=list(PASSWORD), default="student", help="Account role")
    ap.add_argument("--start", type=int, default=1, help="Sequence start for usernames")
    ap.add_argument("--no-apps", action="store_true",
                    help="Skip demo applications (student role only)")
    args = ap.parse_args()

    created = skipped = 0
    with get_db() as db:
        for i in range(args.start, args.start + args.count):
            uname = f"{args.role}{i:03d}"
            exists = db.execute("SELECT 1 FROM users WHERE username=?", (uname,)).fetchone()
            if exists:
                skipped += 1
                continue
            name = generate_name(f"{args.role}-{i}")
            email = f"{uname}@knsubic.edu.ph"
            cur = db.execute(
                "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?,?,?,?,?)",
                (uname, generate_password_hash(PASSWORD[args.role]), name, email, args.role))
            uid = cur.lastrowid
            if args.role == "student" and not args.no_apps:
                _create_demo_application(db, uid)
            created += 1

    print(f"Created {created} {args.role} account(s) [{args.role}{args.start:03d}.."
          f"{args.role}{args.start + args.count - 1:03d}], password = {PASSWORD[args.role]}")
    if skipped:
        print(f"Skipped {skipped} existing username(s).")


if __name__ == "__main__":
    main()