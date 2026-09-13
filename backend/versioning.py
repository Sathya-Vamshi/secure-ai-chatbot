"""
Document versioning + the integrity gate in front of the RAG pipeline.

Key design decision: an authorized edit NEVER overwrites the existing
file on disk or mutates an existing `documents` row's hash. It always
writes a brand-new file (Security_Policy__v2.txt) and inserts a brand
new `documents` row. The old row's file and hash are left completely
untouched, marked SUPERSEDED but not deleted.

That one decision is what makes tamper-detection unambiguous: since a
legitimate authorized edit is structurally incapable of changing an
EXISTING row's on-disk hash, any time we find an existing row whose
current file hash no longer matches its stored hash, that can only
mean the file was changed directly on disk/DB, bypassing the
authorized-edit endpoint entirely — which is exactly what "tampering"
means here (spec TEST 9).
"""

import os
import re
from datetime import datetime

from backend.database import get_connection, get_active_document_key
from backend.security import calculate_hash, create_backup
from backend.authorization import write_audit


def _row_to_dict(row):
    (doc_id, filename, filepath, original_hash, status, upload_time,
     backup_path, document_key, version, parent_document_id, created_by,
     owner_username, authorization_id) = row
    return {
        "id": doc_id, "filename": filename, "filepath": filepath,
        "original_hash": original_hash, "status": status,
        "upload_time": upload_time, "backup_path": backup_path,
        "document_key": document_key, "version": version,
        "parent_document_id": parent_document_id, "created_by": created_by,
        "owner_username": owner_username, "authorization_id": authorization_id,
    }


_SELECT_COLS = """
    id, filename, filepath, original_hash, status, upload_time, backup_path,
    document_key, version, parent_document_id, created_by, owner_username,
    authorization_id
"""


def get_document(document_id: int):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(f"SELECT {_SELECT_COLS} FROM documents WHERE id = ?", (document_id,))
    row = cursor.fetchone()
    connection.close()
    return _row_to_dict(row) if row else None


def get_active_version(document_key: str):
    """The current, non-superseded version of a document family — the
    one that should be served to RAG and to /chat."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT {_SELECT_COLS} FROM documents
        WHERE document_key = ? AND status != 'SUPERSEDED'
        ORDER BY id DESC LIMIT 1
    """, (document_key,))
    row = cursor.fetchone()
    connection.close()
    return _row_to_dict(row) if row else None


def list_document_versions(document_key: str):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT {_SELECT_COLS} FROM documents
        WHERE document_key = ? ORDER BY version ASC
    """, (document_key,))
    rows = cursor.fetchall()
    connection.close()
    return [_row_to_dict(r) for r in rows]


def list_documents_grouped():
    """One entry per document family, showing its current active
    version, across ALL accounts. Kept for internal/back-compat use —
    prefer list_documents_visible_to() for anything user-facing."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT document_key FROM documents")
    keys = [r[0] for r in cursor.fetchall()]
    connection.close()
    return [get_active_version(k) for k in keys if get_active_version(k)]


def list_documents_visible_to(username: str):
    """Each account only sees its own uploaded documents — plus any
    document someone else has explicitly shared with it via a valid
    authorization key (so the recipient of a key can still find that
    document to act on it). This is what powers the per-account
    'My Documents' list."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT DISTINCT document_key FROM documents WHERE owner_username = ?
        UNION
        SELECT DISTINCT document_key FROM authorizations
        WHERE employee_username = ? AND revoked = 0
    """, (username, username))
    keys = [r[0] for r in cursor.fetchall()]
    connection.close()
    return [get_active_version(k) for k in keys if get_active_version(k)]


def verify_document_integrity(document_id: int, ip_address: str = None, user_agent: str = None) -> dict:
    """Recomputes the hash of a specific version's file and compares it
    to the hash trusted at the time that version was created/edited.
    On mismatch, persists status=TAMPERED and writes an audit record —
    this is the check that must run before ANY content from this row
    reaches the RAG pipeline or the LLM."""
    doc = get_document(document_id)
    if doc is None:
        return {"ok": False, "status": "NOT_FOUND", "message": "Document not found."}

    if not os.path.exists(doc["filepath"]):
        return {"ok": False, "status": "MISSING", "message": "Document file is missing from storage."}

    current_hash = calculate_hash(doc["filepath"])

    if current_hash == doc["original_hash"]:
        # Self-heal a stale TAMPERED flag: if the hash matches again (e.g.
        # after a restore from backup), the content is verified-safe right
        # now, regardless of what status was persisted last time.
        status = "SUPERSEDED" if doc["status"] == "SUPERSEDED" else "SAFE"
        if status != doc["status"]:
            _set_status(document_id, status)
        return {
            "ok": True, "status": status, "document": doc,
            "current_hash": current_hash,
        }

    # Hash mismatch. Per spec: before declaring tampering, check whether
    # there's a matching AUTHORIZED_EDIT audit record for this exact
    # transition. In normal operation this branch is never hit (authorized
    # edits create a NEW row instead of mutating this one) — this is a
    # defensive second check, not the primary authorization mechanism.
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT id FROM audit_log
        WHERE action = 'AUTHORIZED_EDIT' AND document_id = ?
              AND old_hash = ? AND new_hash = ? AND result = 'SUCCESS'
        LIMIT 1
    """, (document_id, doc["original_hash"], current_hash))
    authorized_match = cursor.fetchone()
    connection.close()

    if authorized_match:
        return {
            "ok": True, "status": "SAFE", "document": doc,
            "current_hash": current_hash,
        }

    # Genuine, unauthorized tampering.
    _set_status(document_id, "TAMPERED")
    write_audit(
        "UNAUTHORIZED_MODIFICATION", document_key=doc["document_key"],
        document_id=document_id, old_hash=doc["original_hash"],
        new_hash=current_hash, result="TAMPERED",
        details="Hash mismatch with no matching authorized-edit record. Blocked.",
        ip_address=ip_address, user_agent=user_agent,
    )
    return {
        "ok": False, "status": "TAMPERED", "document": doc,
        "current_hash": current_hash,
        "message": "Document integrity verification failed. The document was modified without valid authorization and has been blocked.",
    }


def _set_status(document_id: int, status: str) -> None:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("UPDATE documents SET status = ? WHERE id = ?", (status, document_id))
    connection.commit()
    connection.close()


def _is_shared_with(document_key: str, username: str) -> bool:
    """True if someone has ever issued username an authorization key for
    this document family — the same rule main.py uses to decide whether
    a non-owner can see/act on a document."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM authorizations WHERE document_key = ? AND employee_username = ? LIMIT 1",
        (document_key, username),
    )
    row = cursor.fetchone()
    connection.close()
    return row is not None


def resolve_active_key(current_user: str) -> str:
    """Which document_key counts as this account's active document right
    now. Prefers an explicitly-set active document (from the Chat view's
    switcher, or auto-switched because a question was clearly about a
    different document) as long as it still resolves and the account
    can still access it; otherwise falls back to whichever document
    family this account most recently uploaded — the original,
    upload-order-only behavior."""
    stored_key = get_active_document_key(current_user)
    if stored_key:
        doc = get_active_version(stored_key)
        if doc is not None and (doc["owner_username"] == current_user or _is_shared_with(stored_key, current_user)):
            return stored_key
        # Stored key no longer resolves (document deleted, access revoked,
        # etc.) — silently fall through to the upload-recency default
        # instead of surfacing a broken reference.

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT document_key FROM documents WHERE owner_username = ? ORDER BY id DESC LIMIT 1",
        (current_user,),
    )
    row = cursor.fetchone()
    connection.close()
    return row[0] if row else None


_SWITCH_INTENT_RE = re.compile(
    r"\b(switch|change|use|open|load|activate|work on|focus on|talk about|"
    r"regarding|refer(?:ring)? to|instead of|rather than)\b",
    re.IGNORECASE,
)
_FILENAME_STOPWORDS = {
    "the", "a", "an", "of", "and", "to", "for", "in", "on", "with",
    "doc", "document", "file", "txt", "pdf", "docx", "csv", "final", "new", "old", "copy",
}


def _tokenize_filename(filename: str) -> set:
    """Distinctive words in a filename, ignoring the extension and common
    filler words — 'Quarterly_Report_2024.txt' -> {'quarterly','report','2024'}."""
    base = os.path.splitext(filename)[0]
    parts = re.split(r"[^a-zA-Z0-9]+", base.lower())
    return {p for p in parts if p and p not in _FILENAME_STOPWORDS and len(p) >= 3}


def has_switch_intent(message: str) -> bool:
    """True if the message itself carries a 'switch to / use / open …'
    verb — used to react even when the matched document turns out to
    already be the active one (see chat_document_reference())."""
    return bool(message and _SWITCH_INTENT_RE.search(message.lower()))


def chat_document_reference(visible_docs: list, message: str):
    """Best-effort read of which of my documents (if any) this message
    is about — covers both an explicit request ("switch to the
    invoice") and an implicit one (just asking a question that's
    clearly about another uploaded document). Returns the matching
    document dict, or None if nothing in the message points clearly
    enough at a specific document.

    Deliberately conservative: a bare word that happens to overlap isn't
    enough on its own unless the message also carries switch-like intent
    or (nearly) the whole filename is mentioned — the goal is to feel
    smart, not to flip documents on a coincidental word match. The
    caller (not this function) decides what "matches the currently
    active document" should mean, so an explicit "switch to X" where X
    is already active can still get a sensible reply instead of being
    silently dropped."""
    if not visible_docs or not message:
        return None

    msg_lower = message.lower()
    # Filenames typically separate words with '_' or '-', and people
    # often type them that way too ("switch to secure_ai_chatbot doc").
    # Regex \b doesn't break on '_' (it's a word character), so without
    # this normalization a token like "chatbot" would never be found
    # inside "secure_ai_chatbot" — normalizing both sides the same way
    # turns that into a plain, boundary-safe word match.
    msg_normalized = re.sub(r"[_\-]+", " ", msg_lower)
    explicit_intent = bool(_SWITCH_INTENT_RE.search(msg_lower))

    best_doc, best_score = None, 0
    for doc in visible_docs:
        filename = doc.get("filename") or ""
        if not filename:
            continue
        base = os.path.splitext(filename)[0]

        score = 0
        if filename.lower() in msg_lower or (len(base) >= 3 and base.lower() in msg_lower):
            # The filename (or its base name) is basically spelled out —
            # about as clear a signal as free text gets.
            score = 100
        else:
            tokens = _tokenize_filename(filename)
            if tokens:
                hits = sum(1 for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", msg_normalized))
                if hits:
                    if explicit_intent:
                        # Intent is already clear from the verb ("switch to…") —
                        # any real token overlap is enough to identify which doc.
                        score = 10 * hits
                    else:
                        # No switch-y verb: require matching most of what makes
                        # this filename distinctive, so a single shared common
                        # word across two unrelated documents doesn't trigger
                        # a surprise switch. Short (1-2 word) filenames need
                        # every one of their words matched; longer ones need
                        # all but one.
                        needed = len(tokens) if len(tokens) <= 2 else max(2, len(tokens) - 1)
                        score = 10 * hits if hits >= needed else 0

        if score > best_score:
            best_score, best_doc = score, doc

    if best_doc is None or best_score == 0:
        return None
    return best_doc


def resolve_active_document(current_user: str, ip_address: str = None, user_agent: str = None) -> dict:
    """What /chat actually calls. Finds THIS account's active document
    (explicitly set, or the most recently uploaded one — see
    resolve_active_key()), verifies its active version's integrity, and
    only returns text if that check passes. Falls back to
    storage/test.txt (unverified, matches old demo behavior) if this
    account has no document at all."""
    key = resolve_active_key(current_user)

    if key is None:
        fallback = "storage/test.txt"
        if os.path.exists(fallback):
            with open(fallback, "r", encoding="utf-8", errors="ignore") as f:
                return {"ok": True, "text": f.read(), "document_id": None, "status": "UNVERIFIED_DEMO"}
        return {"ok": False, "status": "NO_DOCUMENT", "message": "No document has been uploaded yet."}

    active = get_active_version(key)
    if active is None:
        return {"ok": False, "status": "NO_DOCUMENT", "message": "No document has been uploaded yet."}

    check = verify_document_integrity(active["id"], ip_address=ip_address, user_agent=user_agent)
    if not check["ok"]:
        return {"ok": False, "status": check["status"], "message": check.get("message", "Document blocked."), "document_id": active["id"]}

    with open(active["filepath"], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    return {
        "ok": True, "text": text, "document_id": active["id"], "status": active["status"],
        "document_key": key, "filename": active["filename"],
    }


def create_document_version(document_key: str, new_text: str, created_by: str,
                             authorization_id: int, ip_address: str = None,
                             user_agent: str = None) -> dict:
    """Writes new_text as the next version of document_key, supersedes
    the current active version (file + hash untouched, just marked),
    and returns the audit-relevant details for the caller."""
    current = get_active_version(document_key)
    if current is None:
        raise ValueError(f"No existing document found for '{document_key}'")

    next_version = current["version"] + 1
    base, ext = os.path.splitext(document_key)
    new_filename = f"{base}__v{next_version}{ext}"
    new_filepath = os.path.join("storage", new_filename)

    with open(new_filepath, "w", encoding="utf-8") as f:
        f.write(new_text)

    new_hash = calculate_hash(new_filepath)
    backup_path = create_backup(new_filepath)

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO documents
        (filename, filepath, original_hash, status, upload_time, backup_path,
         document_key, version, parent_document_id, created_by, owner_username,
         authorization_id)
        VALUES (?, ?, ?, 'SAFE', ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        new_filename, new_filepath, new_hash, datetime.now().isoformat(),
        backup_path, document_key, next_version, current["id"], created_by,
        current["owner_username"], authorization_id,
    ))
    connection.commit()
    new_id = cursor.lastrowid
    connection.close()

    # Old version stays on disk with its original hash intact — only its
    # status changes, so it remains available for audit/history exactly
    # as the spec requires ("do not delete the previous version").
    _set_status(current["id"], "SUPERSEDED")

    write_audit(
        "AUTHORIZED_EDIT", username=created_by, document_key=document_key,
        document_id=new_id, authorization_id=authorization_id,
        old_hash=current["original_hash"], new_hash=new_hash, result="SUCCESS",
        details=f"v{current['version']} -> v{next_version}",
        ip_address=ip_address, user_agent=user_agent,
    )

    return {
        "document_id": new_id, "version": next_version,
        "old_version": current["version"], "old_hash": current["original_hash"],
        "new_hash": new_hash, "filepath": new_filepath,
    }
