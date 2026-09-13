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

    # Explicit, off-by-default permission before Aegis will access
    # anything about the local system (currently: the server clock, for
    # accurate date/time answers instead of the model guessing). Never
    # granted implicitly — see backend/system_info.py.
    _ensure_column(cursor, "users", "allow_system_info", "INTEGER NOT NULL DEFAULT 0")

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

    # Who/where a logged action came from. Added as nullable columns so
    # this stays safe against a database created by an older version of
    # this project. ip_address is the caller's network address as seen
    # by the server; mac_address is a best-effort lookup that only
    # succeeds when the caller is on the same local network segment as
    # the server (see backend/authorization.py) — it's simply not
    # possible to learn a client's MAC address over the open internet,
    # browsers never expose it, so this will legitimately be empty for
    # most remote users.
    _ensure_column(cursor, "audit_log", "ip_address", "TEXT")
    _ensure_column(cursor, "audit_log", "user_agent", "TEXT")
    _ensure_column(cursor, "audit_log", "mac_address", "TEXT")

    # --- chats -----------------------------------------------------------
    # A "chat" is one conversation thread that belongs to exactly one
    # account. chat_history rows are grouped under a chat_id so a user
    # can have several separate conversations and switch between them,
    # instead of one single ever-growing transcript.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
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

    # chat_id is added on top (nullable) so any pre-existing chat_history
    # rows from before multi-chat support keep working — they just live
    # in a "legacy" bucket (chat_id IS NULL) that the old endpoints still
    # serve, while every new message is filed under a real chat.
    _ensure_column(cursor, "chat_history", "chat_id", "INTEGER")

    # --- per-account "active document" -------------------------------
    # Which document a chat should treat as "the" document right now.
    # One row per account; defaults to nothing (falls back to "most
    # recently uploaded" — see versioning.resolve_active_key()).
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_settings(
        user_id TEXT PRIMARY KEY,
        active_document_key TEXT
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


def _chat_title_from(message: str) -> str:
    """Turn the first message of a new conversation into a short,
    human-readable title for the sidebar list, the way most chat apps
    do — trimmed and capped so it never wraps to more than one line."""
    title = " ".join(message.strip().split())
    if not title:
        return "New chat"
    return title[:57] + "…" if len(title) > 57 else title


def create_chat(user_id: str, title: str = None) -> int:
    """Start a brand-new, empty conversation for this account and
    return its id. Chats are always private to the account that owns
    them — user_id is what enforces that everywhere below."""
    now = datetime.now().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chats (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (user_id, title or "New chat", now, now),
    )
    conn.commit()
    chat_id = cur.lastrowid
    conn.close()
    return chat_id


def list_chats(user_id: str) -> list[dict]:
    """This account's conversations, most recently active first — what
    populates the 'switch between chats' list in the sidebar."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, created_at, updated_at FROM chats "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]


def get_chat(user_id: str, chat_id: int) -> dict | None:
    """A single chat, but ONLY if it belongs to user_id — this is the
    ownership check every chat-scoped endpoint relies on, same pattern
    as document ownership."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, created_at, updated_at FROM chats WHERE id = ? AND user_id = ?",
        (chat_id, user_id),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}


def touch_chat(chat_id: int, retitle_from: str = None) -> None:
    """Bump a chat's updated_at (so it floats to the top of the list)
    and, if it's still using the default title, set a real title from
    the first message sent in it."""
    conn = get_connection()
    cur = conn.cursor()
    if retitle_from:
        cur.execute(
            "UPDATE chats SET updated_at = ?, "
            "title = CASE WHEN title = 'New chat' THEN ? ELSE title END "
            "WHERE id = ?",
            (datetime.now().isoformat(), _chat_title_from(retitle_from), chat_id),
        )
    else:
        cur.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (datetime.now().isoformat(), chat_id))
    conn.commit()
    conn.close()


def delete_chat(user_id: str, chat_id: int) -> bool:
    """Delete a chat and every message in it. Returns False (and does
    nothing) if this account doesn't own that chat."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
    deleted = cur.rowcount > 0
    if deleted:
        cur.execute("DELETE FROM chat_history WHERE chat_id = ? AND user_id = ?", (chat_id, user_id))
    conn.commit()
    conn.close()
    return deleted


def add_chat_message(user_id: str, sender: str, message: str, chat_id: int = None) -> None:
    """Insert a chat message into the chat_history table.
    `sender` must be either 'user' or 'bot'. `chat_id` files it under a
    specific conversation; leaving it None keeps old, pre-multi-chat
    behavior (a single legacy bucket) for backward compatibility."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (user_id, sender, message, timestamp, chat_id) VALUES (?, ?, ?, ?, ?)",
        (user_id, sender, message, datetime.now().isoformat(), chat_id),
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id: str, chat_id: int = None, limit: int = 300) -> list[dict]:
    """This user's transcript, oldest first, so the frontend can render
    it top-to-bottom exactly like it was typed. Pass chat_id to get one
    specific conversation; omit it to get the legacy (pre-multi-chat)
    bucket only, for backward compatibility."""
    conn = get_connection()
    cur = conn.cursor()
    if chat_id is not None:
        cur.execute(
            """
            SELECT sender, message, timestamp FROM chat_history
            WHERE user_id = ? AND chat_id = ? ORDER BY id ASC LIMIT ?
            """,
            (user_id, chat_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT sender, message, timestamp FROM chat_history
            WHERE user_id = ? AND chat_id IS NULL ORDER BY id ASC LIMIT ?
            """,
            (user_id, limit),
        )
    rows = cur.fetchall()
    conn.close()
    return [{"sender": r[0], "message": r[1], "timestamp": r[2]} for r in rows]


def clear_chat_history(user_id: str, chat_id: int = None) -> None:
    """Wipe a transcript. With no chat_id, wipes only the legacy bucket
    (kept for backward compatibility with the old 'New chat' behavior);
    to delete a real chat entirely, use delete_chat() instead."""
    conn = get_connection()
    cur = conn.cursor()
    if chat_id is not None:
        cur.execute("DELETE FROM chat_history WHERE user_id = ? AND chat_id = ?", (user_id, chat_id))
    else:
        cur.execute("DELETE FROM chat_history WHERE user_id = ? AND chat_id IS NULL", (user_id,))
    conn.commit()
    conn.close()


# --- per-account active document ------------------------------------------

def set_active_document(user_id: str, document_key: str) -> None:
    """Remember which document this account's chat should treat as
    active from now on — set explicitly (picked from the Chat view's
    document switcher) or automatically (the assistant noticed the
    question was about a different uploaded document)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_settings (user_id, active_document_key) VALUES (?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET active_document_key = excluded.active_document_key",
        (user_id, document_key),
    )
    conn.commit()
    conn.close()


def get_active_document_key(user_id: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT active_document_key FROM user_settings WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None
