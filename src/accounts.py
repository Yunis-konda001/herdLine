"""Sign-up and sign-in. Passwords are stored as a hash only."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone

from src.config import OUTPUT_DIR

DB_PATH = OUTPUT_DIR / "herdline.db"


def _connect() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                farm_name TEXT NOT NULL,
                farmer_name TEXT NOT NULL,
                herd_size INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, _, _ = stored.partition("$")
    if not salt or "$" not in stored:
        return False
    return secrets.compare_digest(hash_password(password, salt), stored)


def _public(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "email": row["email"],
        "farmName": row["farm_name"],
        "farmerName": row["farmer_name"],
        "herdSize": row["herd_size"],
        "contact": row["email"],
    }


def create_user(
    email: str,
    password: str,
    farm_name: str,
    farmer_name: str,
    herd_size: int,
) -> dict | None:
    # Returns None if the email is already used
    init_db()
    created = datetime.now(timezone.utc).isoformat()
    try:
        with _connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (email, password_hash, farm_name, farmer_name, herd_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    email.strip().lower(),
                    hash_password(password),
                    farm_name.strip(),
                    farmer_name.strip(),
                    herd_size,
                    created,
                ),
            )
            user_id = cur.lastrowid
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    except sqlite3.IntegrityError:
        return None
    return _public(row) if row else None


def authenticate(email: str, password: str) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _public(row)


def get_user(user_id: int) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _public(row) if row else None
