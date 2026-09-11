import sqlite3
import os
from datetime import datetime

DB_PATH = "database/documents.db"


def get_connection():
    """Return a new SQLite connection to the database file."""
    return sqlite3.connect(DB_PATH)


def _ensure_column(cursor, table: str, column: str, coltype: str):
    """Add a column if it doesn't already exist. Safe to call every
    startup — lets us evolve the schema (versioning/authorization
    features) without breaking a database created by an older version
    of this project."""
    existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def create_database():
    """Create SQLite database and tables (idempotent: safe to call on
    every startup, and safe to run against a database that already has
    the old, pre-versioning schema)."""

    os.makedirs("database", exist_ok=True)

    connection = get_connection()
    cursor = connection.cursor()

    # --- original table, untouched shape -----------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        original_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        upload_time TEXT NOT NULL,
        backup_path TEXT NOT NULL
    )
    """)

    # --- versioning / ownership columns, added on top -----------------
    # document_key groups every version of "the same document" together
    # (defaults to the original filename). version starts at 1 and
    # increments on each authorized edit. parent_document_id links a
    # version back to the one it replaced, so history is never lost.
    _ensure_column(cursor, "documents", "document_key", "TEXT")
    _ensure_column(cursor, "documents", "version", "INTEGER DEFAULT 1")
    _ensure_column(cursor, "documents", "parent_document_id", "INTEGER")
    _ensure_column(cursor, "documents", "created_by", "TEXT")
    _ensure_column(cursor, "documents", "owner_username", "TEXT")
    _ensure_column(cursor, "documents", "authorization_id", "INTEGER")

    # Backfill document_key for any pre-existing rows so old data keeps
    # working with the new version-grouping logic.
    cursor.execute(
        "UPDATE documents SET document_key = filename WHERE document_key IS NULL"
    )

    # --- users -----------------------------------------------------
    # 'owner' = document owner/administrator, can generate authorization
    # keys. 'employee' = can only edit a document with a valid key.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # --- authorizations ----------------------------------------------
    # The raw key is NEVER stored — only key_hash (SHA-256 of the raw
    # key). document_key + employee_username scope the key so it can
    # only ever authorize that one employee on that one document.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS authorizations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key_hash TEXT NOT NULL,
        document_key TEXT NOT NULL,
        employee_username TEXT NOT NULL,
        permission TEXT NOT NULL DEFAULT 'EDIT',
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT,
        max_uses INTEGER,
        uses_count INTEGER NOT NULL DEFAULT 0,
        revoked INTEGER NOT NULL DEFAULT 0
    )
    """)

    # --- audit log -----------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        action TEXT NOT NULL,
        username TEXT,
        document_key TEXT,
        document_id INTEGER,
        authorization_id INTEGER,
        old_hash TEXT,
        new_hash TEXT,
        result TEXT NOT NULL,
        details TEXT
    )
    """)

    # --- chat history -----------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        sender TEXT NOT NULL CHECK(sender IN ('user','bot')),
        message TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """)

    # BUG FIX: this function never committed or closed its connection.
    # CREATE TABLE statements happened to survive because SQLite
    # auto-commits DDL in some cases, but the "UPDATE documents SET
    # document_key = filename ..." backfill above is a plain DML
    # statement — without an explicit commit it was silently thrown
    # away every single time, and the connection was leaked (never
    # closed). This was also the root cause of the app crashing on a
    # fresh database with "no such table: users": seed_default_admin()
    # opened a brand-new connection before this one's CREATE TABLE
    # users was ever committed.
    connection.commit()
    connection.close()


def add_chat_message(user_id: str, sender: str, message: str) -> None:
    """Insert a chat message into the chat_history table.
    `sender` must be either 'user' or 'bot'."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (user_id, sender, message, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, sender, message, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id: str, limit: int = 300) -> list[dict]:
    """Return this user's chat transcript, oldest first, so the frontend
    can render it top-to-bottom exactly like it was typed."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sender, message, timestamp FROM chat_history
        WHERE user_id = ? ORDER BY id ASC LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"sender": r[0], "message": r[1], "timestamp": r[2]} for r in rows]


def clear_chat_history(user_id: str) -> None:
    """Wipe this user's chat transcript (used by the 'New chat' action)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
