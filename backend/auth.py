import hashlib
import secrets
import smtplib
import os
from email.mime.text import MIMEText
from jose import jwt
from datetime import datetime, timedelta
from backend.database import get_connection

SECRET_KEY = "secure-ai-chatbot-secret-key"
ALGORITHM = "HS256"

# The one and only fixed owner account. It's the seed row for the
# 'owner' role in the users table — the only role that can see the
# Security Logs view. Hash below is pbkdf2_hmac(sha256, "admin_123",
# salt, 100000), i.e. this is just "admin" / "admin_123" stored salted
# + hashed instead of in plaintext.
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD_HASH = "707a382de7be0cd4b747d57587a7d033:f5a065b8addb4679d0ec5de27971c856de6a03bf995358adb943b22b7497ea09"


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


def create_user(username: str, password: str, role: str, email: str = None) -> None:
    if role not in ("owner", "employee"):
        raise ValueError("role must be 'owner' or 'employee'")
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO users (username, password_hash, role, created_at, email) VALUES (?, ?, ?, ?, ?)",
        (username, hash_password(password), role, datetime.utcnow().isoformat(), email),
    )
    connection.commit()
    connection.close()


def get_user_by_email(email: str):
    """Returns (username, password_hash, role, email) or None."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT username, password_hash, role, email FROM users WHERE email = ?",
        (email,),
    )
    row = cursor.fetchone()
    connection.close()
    return row


def email_in_use(email: str) -> bool:
    return get_user_by_email(email) is not None


# ---------------------------------------------------------------------
# Self-signup
#
# Anyone can create their own account with an email, username and
# password. Self-signed-up accounts are always role='employee' — the
# only 'owner' account is the fixed seeded admin/admin_123 login above.
# Employees can still create authorization keys for other employees
# (see backend/main.py), they just aren't the single account that sees
# the Security Logs view.
# ---------------------------------------------------------------------

# The owner username is reserved so nobody can self-signup and shadow
# the one fixed admin/admin_123 account.
RESERVED_USERNAMES = {DEFAULT_ADMIN_USERNAME}


def signup_user(username: str, email: str, password: str) -> None:
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        raise ValueError("Username, email and password are all required.")
    if username in RESERVED_USERNAMES:
        raise ValueError("That username is reserved.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError("Enter a valid email address.")
    if user_exists(username):
        raise ValueError("That username is already taken.")
    if email_in_use(email):
        raise ValueError("An account with that email already exists.")

    create_user(username, password, role="employee", email=email)


# ---------------------------------------------------------------------
# Forgot password: a 6-digit code is emailed to the account's address.
# Only the hash of the code is ever stored, and it expires in 15
# minutes / is single-use, same pattern as authorization keys.
# ---------------------------------------------------------------------

RESET_CODE_TTL_MINUTES = 15


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def send_email(to_email: str, subject: str, body: str) -> None:
    """Sends an email via SMTP if SMTP_HOST/SMTP_USER/SMTP_PASSWORD are
    configured in the environment. If they aren't configured (e.g. on a
    laptop during development), the message is written to security.log
    instead so the flow can still be tested end-to-end without a real
    mail server."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)

    if not (smtp_host and smtp_user and smtp_password):
        with open("security.log", "a", encoding="utf-8") as f:
            f.write(
                f'{{"event": "EMAIL_NOT_CONFIGURED", "to": "{to_email}", '
                f'"subject": "{subject}", "body": {body!r}, '
                f'"timestamp": "{datetime.utcnow().isoformat()}"}}\n'
            )
        return

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = to_email

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_from, [to_email], message.as_string())


def request_password_reset(email: str) -> None:
    """Always succeeds from the caller's point of view (no user
    enumeration) — if the email matches an account, a code is generated
    and emailed; if it doesn't, nothing happens."""
    email = email.strip().lower()
    row = get_user_by_email(email)
    if row is None:
        return

    username = row[0]
    code = f"{secrets.randbelow(1_000_000):06d}"
    now = datetime.utcnow()
    expires_at = now + timedelta(minutes=RESET_CODE_TTL_MINUTES)

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO password_reset_codes (username, code_hash, created_at, expires_at, used) "
        "VALUES (?, ?, ?, ?, 0)",
        (username, _hash_code(code), now.isoformat(), expires_at.isoformat()),
    )
    connection.commit()
    connection.close()

    send_email(
        email,
        "Your password reset code",
        f"Your password reset code is {code}. It expires in {RESET_CODE_TTL_MINUTES} minutes. "
        f"If you didn't request this, you can ignore this email.",
    )


def reset_password_with_code(email: str, code: str, new_password: str) -> None:
    email = email.strip().lower()
    if len(new_password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    row = get_user_by_email(email)
    if row is None:
        raise ValueError("Invalid or expired code.")
    username = row[0]

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT id, code_hash, expires_at, used FROM password_reset_codes "
        "WHERE username = ? ORDER BY id DESC LIMIT 1",
        (username,),
    )
    reset_row = cursor.fetchone()

    if reset_row is None:
        connection.close()
        raise ValueError("Invalid or expired code.")

    reset_id, code_hash, expires_at, used = reset_row

    if used or datetime.utcnow() > datetime.fromisoformat(expires_at):
        connection.close()
        raise ValueError("Invalid or expired code.")

    if not secrets.compare_digest(_hash_code(code), code_hash):
        connection.close()
        raise ValueError("Invalid or expired code.")

    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (hash_password(new_password), username),
    )
    cursor.execute(
        "UPDATE password_reset_codes SET used = 1 WHERE id = ?",
        (reset_id,),
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