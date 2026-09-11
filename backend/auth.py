import hashlib
import secrets
from jose import jwt
from datetime import datetime, timedelta
from backend.database import get_connection

SECRET_KEY = "secure-ai-chatbot-secret-key"
ALGORITHM = "HS256"

# Kept from the original single-admin setup so nothing that already
# depends on this login breaks. It's now just the seed row for the
# 'owner' role in the users table instead of a hardcoded check.
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD_HASH = "bd46a64696b878b262597cfa9daa290b:1029d960a60514de4427d9cf0a7ef6d8ca74443d7c0f5f63e802404d8ac9980c"


def hash_password(password):
    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    ).hex()

    return salt + ":" + password_hash


def verify_password(password, stored_password):

    salt, stored_hash = stored_password.split(":")

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    ).hex()

    return secrets.compare_digest(
        password_hash,
        stored_hash
    )


def create_token(username, role="employee"):

    data = {
        "sub": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=2)
    }

    return jwt.encode(
        data,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def verify_token(token: str) -> dict:
    """Returns {"username": ..., "role": ...}. Role defaults to
    'employee' for tokens issued before roles existed, so old tokens
    don't crash — they just get treated as the least-privileged role."""

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")

        if username is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

        return {"username": username, "role": payload.get("role", "employee")}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )


# ---------------------------------------------------------------------
# User accounts (owner + employees). This is new: the project previously
# had exactly one hardcoded login and no concept of separate employees,
# which the authorized-document-edit feature needs (it has to tell
# Employee A's key apart from Employee B's).
# ---------------------------------------------------------------------

def seed_default_admin():
    """Make sure the existing admin/admin123 login keeps working after
    upgrading to a real users table, seeded with the 'owner' role."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD_HASH, "owner", datetime.utcnow().isoformat()),
        )
        connection.commit()
    connection.close()


def get_user(username: str):
    """Returns (username, password_hash, role) or None."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT username, password_hash, role FROM users WHERE username = ?",
        (username,),
    )
    row = cursor.fetchone()
    connection.close()
    return row


def user_exists(username: str) -> bool:
    return get_user(username) is not None


def create_user(username: str, password: str, role: str) -> None:
    if role not in ("owner", "employee"):
        raise ValueError("role must be 'owner' or 'employee'")
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, hash_password(password), role, datetime.utcnow().isoformat()),
    )
    connection.commit()
    connection.close()


def list_users(role: str = None):
    connection = get_connection()
    cursor = connection.cursor()
    if role:
        cursor.execute("SELECT username, role, created_at FROM users WHERE role = ?", (role,))
    else:
        cursor.execute("SELECT username, role, created_at FROM users")
    rows = cursor.fetchall()
    connection.close()
    return [{"username": r[0], "role": r[1], "created_at": r[2]} for r in rows]