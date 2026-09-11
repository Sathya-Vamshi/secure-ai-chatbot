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
from datetime import datetime

from backend.database import get_connection
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
    version. Used to populate the owner's 'select document' dropdown."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT document_key FROM documents")
    keys = [r[0] for r in cursor.fetchall()]
    connection.close()
    return [get_active_version(k) for k in keys if get_active_version(k)]


def verify_document_integrity(document_id: int) -> dict:
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


def resolve_active_document() -> dict:
    """What /chat actually calls. Finds the most recently touched
    document family, verifies its active version's integrity, and only
    returns text if that check passes. Falls back to storage/test.txt
    (unverified, matches old demo behavior) if nothing has ever been
    uploaded."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT document_key FROM documents ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    connection.close()

    if row is None:
        fallback = "storage/test.txt"
        if os.path.exists(fallback):
            with open(fallback, "r", encoding="utf-8", errors="ignore") as f:
                return {"ok": True, "text": f.read(), "document_id": None, "status": "UNVERIFIED_DEMO"}
        return {"ok": False, "status": "NO_DOCUMENT", "message": "No document has been uploaded yet."}

    active = get_active_version(row[0])
    if active is None:
        return {"ok": False, "status": "NO_DOCUMENT", "message": "No document has been uploaded yet."}

    check = verify_document_integrity(active["id"])
    if not check["ok"]:
        return {"ok": False, "status": check["status"], "message": check.get("message", "Document blocked."), "document_id": active["id"]}

    with open(active["filepath"], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    return {"ok": True, "text": text, "document_id": active["id"], "status": active["status"]}


def create_document_version(document_key: str, new_text: str, created_by: str,
                             authorization_id: int) -> dict:
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
    )

    return {
        "document_id": new_id, "version": next_version,
        "old_version": current["version"], "old_hash": current["original_hash"],
        "new_hash": new_hash, "filepath": new_filepath,
    }
