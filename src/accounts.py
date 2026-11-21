"""Sign-up and sign-in. Passwords are stored as a hash only."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone

from src.config import OUTPUT_DIR

# Define the path to the local SQLite database file.
DB_PATH = OUTPUT_DIR / "herdline.db"


def _connect() -> sqlite3.Connection:
    # Ensure the output directory exists before creating the DB.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Return rows as dictionaries so we can access columns by name.
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    # Create the users table if it does not already exist.
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
    # Generate a random 16-byte salt if one is not provided.
    salt = salt or secrets.token_hex(16)
    # Hash the password using PBKDF2-HMAC-SHA256 with 200,000 iterations.
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    )
    # Store the salt and the hash together, separated by a dollar sign.
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    # Extract the original salt from the stored string.
    salt, _, _ = stored.partition("$")
    if not salt or "$" not in stored:
        return False
    # Hash the input password with the same salt and compare securely.
    return secrets.compare_digest(hash_password(password, salt), stored)


def _public(row: sqlite3.Row) -> dict:
    # Convert a database row into a safe public dictionary (hiding the password hash).
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
    # Get the current time in UTC format.
    created = datetime.now(timezone.utc).isoformat()
    try:
        with _connect() as conn:
            # Insert the new user into the database.
            cur = conn.execute(
                """
                INSERT INTO users (email, password_hash, farm_name, farmer_name, herd_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    email.strip().lower(),
                    hash_password(password), # Hash the password before saving
                    farm_name.strip(),
                    farmer_name.strip(),
                    herd_size,
                    created,
                ),
            )
            user_id = cur.lastrowid
            # Fetch the newly created row to return it.
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    except sqlite3.IntegrityError:
        # Catch the error if the email already exists in the database.
        return None
    return _public(row) if row else None


def authenticate(email: str, password: str) -> dict | None:
    init_db()
    with _connect() as conn:
        # Find the user by their email address.
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    # If user doesn't exist, or the password doesn't match, return None.
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _public(row)


def get_user(user_id: int) -> dict | None:
    init_db()
    with _connect() as conn:
        # Fetch a user's details using their unique ID.
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _public(row) if row else None