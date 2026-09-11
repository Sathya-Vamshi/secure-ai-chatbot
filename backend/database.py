import sqlite3
import os
from datetime import datetime
from typing import Optional

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
    # 'owner' = the single fixed admin/admin_123 account, sees the
    # Security Logs view. 'employee' = every self-signed-up user; can
    # edit a document with a valid key, and can generate authorization
    # keys for other employees too.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # email is added as a nullable column (rather than in the CREATE
    # TABLE above) so this stays safe to run against a database created
    # by an older version of this project that has no email column yet.
    _ensure_column(cursor, "users", "email", "TEXT")

    # Case-insensitive-ish uniqueness for email, but only enforced for
    # rows that actually have one set — the seeded 'admin' owner account
    # has no email and shouldn't block anything.
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email "
        "ON users(email) WHERE email IS NOT NULL"
    )

    # --- password reset codes -----------------------------------------
    # Used by the forgot-password flow. Only a hash of the one-time code
    # is stored (never the raw code) — same pattern as authorization
    # keys below. A row is single-use and short-lived.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS password_reset_codes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        used INTEGER NOT NULL DEFAULT 0
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

    # --- chat conversations (one account, many chats) --------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT 'New chat',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
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
    _ensure_column(cursor, "chat_history", "conversation_id", "INTEGER")

    # Namespace document families per owner so two accounts can each have
    # their own "report.txt" without seeing or overwriting each other.
    cursor.execute(
        "UPDATE documents SET owner_username = COALESCE(NULLIF(owner_username, ''), created_by) "
        "WHERE owner_username IS NULL OR owner_username = ''"
    )
    cursor.execute(
        """
        UPDATE documents
        SET document_key = owner_username || '::' || document_key
        WHERE owner_username IS NOT NULL
          AND document_key IS NOT NULL
          AND instr(document_key, '::') = 0
        """
    )
    cursor.execute(
        """
        UPDATE authorizations
        SET document_key = created_by || '::' || document_key
        WHERE created_by IS NOT NULL
          AND document_key IS NOT NULL
          AND instr(document_key, '::') = 0
        """
    )

    # Existing transcripts become one private conversation per user.
    cursor.execute(
        "SELECT DISTINCT user_id FROM chat_history WHERE conversation_id IS NULL"
    )
    for (uid,) in cursor.fetchall():
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (uid, "Previous chat", now, now),
        )
        conv_id = cursor.lastrowid
        cursor.execute(
            "UPDATE chat_history SET conversation_id = ? WHERE user_id = ? AND conversation_id IS NULL",
            (conv_id, uid),
        )

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


def create_conversation(user_id: str, title: str = "New chat") -> dict:
    now = datetime.now().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (user_id, title, now, now),
    )
    conv_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": conv_id, "user_id": user_id, "title": title, "created_at": now, "updated_at": now}


def list_conversations(user_id: str) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, created_at, updated_at FROM conversations
        WHERE user_id = ? ORDER BY updated_at DESC, id DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
        for r in rows
    ]


def get_conversation(conversation_id: int, user_id: str) -> Optional[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}


def delete_conversation(conversation_id: int, user_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM chat_history WHERE conversation_id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    cur.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def add_chat_message(user_id: str, sender: str, message: str, conversation_id: int) -> None:
    """Insert a chat message into this user's conversation only."""
    now = datetime.now().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    conv = cur.fetchone()
    if conv is None:
        conn.close()
        raise ValueError("Conversation not found.")
    cur.execute(
        "INSERT INTO chat_history (user_id, sender, message, timestamp, conversation_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, sender, message, now, conversation_id),
    )
    title = conv[1]
    if sender == "user" and (not title or title in ("New chat", "Previous chat")):
        title = message.strip().splitlines()[0][:48] or title
    cur.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
        (title, now, conversation_id, user_id),
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id: str, conversation_id: int, limit: int = 300) -> list[dict]:
    """Return one conversation's transcript for this user only."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sender, message, timestamp FROM chat_history
        WHERE user_id = ? AND conversation_id = ?
        ORDER BY id ASC LIMIT ?
        """,
        (user_id, conversation_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"sender": r[0], "message": r[1], "timestamp": r[2]} for r in rows]


def clear_chat_history(user_id: str, conversation_id: Optional[int] = None) -> None:
    """Wipe one conversation, or every conversation belonging to this user."""
    conn = get_connection()
    cur = conn.cursor()
    if conversation_id is None:
        cur.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
    else:
        cur.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        cur.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
    conn.commit()
    conn.close()


def count_user_documents(username: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM documents
        WHERE owner_username = ? AND status != 'SUPERSEDED'
        """,
        (username,),
    )
    count = cur.fetchone()[0]
    conn.close()
    return count
