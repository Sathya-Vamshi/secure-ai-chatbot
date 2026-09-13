"""
Authorized document modification.

This is the "authentication key" feature: a document owner/admin can
generate a key that lets ONE specific employee EDIT ONE specific
document, optionally with an expiry date and/or a max-use count. This
module is the single place that:

  - generates and hashes authorization keys (never stores the raw key)
  - verifies a submitted key against every required rule at once
  - rate-limits and logs failed attempts
  - writes structured audit log rows for every authorized/denied edit

Security principle this file exists to enforce (from the spec):
    IF hash mismatch:
        verify the EXACT authorization: user + document + permission +
        validity + authorization status
        THEN decide whether it is authorized.
NOT "does this employee have any key at all". Every check below is
scoped to the specific (document_key, employee_username) pair, so a
key can never be reused for a different employee or a different
document, even by accident.
"""

import time
import subprocess
import platform
import re as _re
from datetime import datetime, timezone

from backend.database import get_connection
from backend.security import hash_secret, generate_authorization_key

# ---------------------------------------------------------------------
# Rate limiting for failed key attempts (in-memory, mirrors the
# in-memory _chat_cache pattern already used in main.py — no new
# infra needed for this).
# ---------------------------------------------------------------------
_FAILED_ATTEMPTS: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW_SECONDS = 15 * 60
RATE_LIMIT_MAX_ATTEMPTS = 5


def is_rate_limited(username: str) -> bool:
    now = time.time()
    attempts = _FAILED_ATTEMPTS.get(username, [])
    attempts = [t for t in attempts if now - t < RATE_LIMIT_WINDOW_SECONDS]
    _FAILED_ATTEMPTS[username] = attempts
    return len(attempts) >= RATE_LIMIT_MAX_ATTEMPTS


def record_failed_attempt(username: str) -> None:
    _FAILED_ATTEMPTS.setdefault(username, []).append(time.time())


def reset_attempts(username: str) -> None:
    _FAILED_ATTEMPTS.pop(username, None)


# ---------------------------------------------------------------------
# Best-effort MAC address lookup
#
# IMPORTANT, please read before relying on this: an HTTP server can
# NEVER learn a client's MAC address over the open internet — browsers
# don't send it, and it isn't part of the IP/TCP/HTTP protocols once
# traffic has crossed a router. A MAC address only exists on the local
# network segment between two directly-connected devices.
#
# What IS possible, and what this function does: if the server and the
# person accessing it happen to be on the SAME local network (e.g. an
# office LAN, a home network, or "server on Mac, client on the same
# WiFi"), the operating system's own ARP table already knows the MAC
# address that corresponds to that IP, because it had to find it out
# to deliver packets on the local segment. We just ask the OS for it.
# For anyone accessing over the internet / a different network, this
# will correctly come back empty — there's no way around that, and no
# tool actually gets around it despite what some products imply.
# ---------------------------------------------------------------------
_MAC_CACHE: dict[str, tuple[float, str | None]] = {}
_MAC_CACHE_TTL_SECONDS = 120


def lookup_mac_address(ip_address: str | None) -> str | None:
    if not ip_address or ip_address in ("127.0.0.1", "::1", "localhost", "testclient"):
        return None

    cached = _MAC_CACHE.get(ip_address)
    if cached and (time.time() - cached[0]) < _MAC_CACHE_TTL_SECONDS:
        return cached[1]

    mac = None
    mac_pattern = _re.compile(r"([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}")
    try:
        system = platform.system()
        if system == "Windows":
            output = subprocess.run(["arp", "-a", ip_address], capture_output=True, text=True, timeout=1.5).stdout
        else:
            # Works on both Linux and macOS.
            output = subprocess.run(["arp", "-n", ip_address], capture_output=True, text=True, timeout=1.5).stdout
        match = mac_pattern.search(output or "")
        if match:
            mac = match.group(0).upper().replace("-", ":")
    except Exception:
        # ARP tooling missing/unavailable/sandboxed — this is expected
        # in a lot of environments (containers, cloud hosts). Just
        # means we don't get a MAC for this request, nothing breaks.
        mac = None

    _MAC_CACHE[ip_address] = (time.time(), mac)
    return mac


# ---------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------

def write_audit(action: str, username: str = None, document_key: str = None,
                 document_id: int = None, authorization_id: int = None,
                 old_hash: str = None, new_hash: str = None,
                 result: str = "SUCCESS", details: str = "",
                 ip_address: str = None, user_agent: str = None,
                 mac_address: str = None) -> None:
    if mac_address is None and ip_address:
        mac_address = lookup_mac_address(ip_address)

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO audit_log
        (timestamp, action, username, document_key, document_id,
         authorization_id, old_hash, new_hash, result, details,
         ip_address, user_agent, mac_address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        action, username, document_key, document_id,
        authorization_id, old_hash, new_hash, result, details,
        ip_address, user_agent, mac_address,
    ))
    connection.commit()
    connection.close()


_AUDIT_COLUMNS = ["id", "timestamp", "action", "username", "document_key",
                   "document_id", "authorization_id", "old_hash", "new_hash",
                   "result", "details", "ip_address", "user_agent", "mac_address"]


def get_audit_log(limit: int = 200) -> list[dict]:
    """Every account's activity — the Owner's view."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT {", ".join(_AUDIT_COLUMNS)}
        FROM audit_log ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    connection.close()
    return [dict(zip(_AUDIT_COLUMNS, row)) for row in rows]


def get_audit_log_for_user(username: str, limit: int = 200) -> list[dict]:
    """One account's OWN activity only — what every non-owner account
    sees. Deliberately does not expose other accounts' rows, even ones
    that mention a shared document, so an employee can't use their own
    audit view to snoop on someone else's activity."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT {", ".join(_AUDIT_COLUMNS)}
        FROM audit_log WHERE username = ? ORDER BY id DESC LIMIT ?
    """, (username, limit))
    rows = cursor.fetchall()
    connection.close()
    return [dict(zip(_AUDIT_COLUMNS, row)) for row in rows]


# ---------------------------------------------------------------------
# Authorization key lifecycle
# ---------------------------------------------------------------------

def create_authorization(document_key: str, employee_username: str, created_by: str,
                          permission: str = "EDIT", expires_at: str = None,
                          max_uses: int = None) -> tuple[int, str]:
    """Creates the authorization record and returns (authorization_id,
    raw_key). The raw key is returned ONCE, here, and never again —
    only its hash is persisted."""
    raw_key = generate_authorization_key()
    key_hash = hash_secret(raw_key)

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO authorizations
        (key_hash, document_key, employee_username, permission, created_by,
         created_at, expires_at, max_uses, uses_count, revoked)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
    """, (
        key_hash, document_key, employee_username, permission, created_by,
        datetime.now(timezone.utc).isoformat(), expires_at, max_uses,
    ))
    connection.commit()
    authorization_id = cursor.lastrowid
    connection.close()

    write_audit(
        "KEY_GENERATED", username=created_by, document_key=document_key,
        authorization_id=authorization_id, result="SUCCESS",
        details=f"Issued to {employee_username}, permission={permission}, expires_at={expires_at}, max_uses={max_uses}",
    )

    return authorization_id, raw_key


def has_authorization_for(document_key: str, employee_username: str) -> bool:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT id FROM authorizations
        WHERE document_key = ? AND employee_username = ? AND revoked = 0
        LIMIT 1
        """,
        (document_key, employee_username),
    )
    row = cursor.fetchone()
    connection.close()
    return row is not None


def list_authorizations(username: str, document_key: str = None) -> list[dict]:
    """Never returns key_hash. Only keys this user created or received."""
    connection = get_connection()
    cursor = connection.cursor()
    if document_key:
        cursor.execute("""
            SELECT id, document_key, employee_username, permission, created_by,
                   created_at, expires_at, max_uses, uses_count, revoked
            FROM authorizations
            WHERE document_key = ?
              AND (created_by = ? OR employee_username = ?)
            ORDER BY id DESC
        """, (document_key, username, username))
    else:
        cursor.execute("""
            SELECT id, document_key, employee_username, permission, created_by,
                   created_at, expires_at, max_uses, uses_count, revoked
            FROM authorizations
            WHERE created_by = ? OR employee_username = ?
            ORDER BY id DESC
        """, (username, username))
    rows = cursor.fetchall()
    connection.close()
    cols = ["id", "document_key", "employee_username", "permission", "created_by",
            "created_at", "expires_at", "max_uses", "uses_count", "revoked"]
    return [dict(zip(cols, row)) for row in rows]


def revoke_authorization(authorization_id: int, revoked_by: str) -> bool:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "SELECT document_key, created_by, employee_username FROM authorizations WHERE id = ?",
        (authorization_id,),
    )
    row = cursor.fetchone()
    if row is None:
        connection.close()
        return False
    if revoked_by not in (row[1], row[2]) and not row[0].startswith(f"{revoked_by}::"):
        connection.close()
        return False
    cursor.execute("UPDATE authorizations SET revoked = 1 WHERE id = ?", (authorization_id,))
    connection.commit()
    connection.close()
    write_audit(
        "KEY_REVOKED", username=revoked_by, document_key=row[0],
        authorization_id=authorization_id, result="SUCCESS",
    )
    return True


def find_document_for_key(employee_username: str, raw_key: str, permission: str = "EDIT"):
    """The one-step redemption lookup: given ONLY the raw key and the
    account redeeming it, work out which document it unlocks — the
    person pasting the key should never have to already know or pick
    the right document first.

    Deliberately does NOT consume a use; it's a preview/discovery step.
    The use is only consumed later, in verify_and_consume_authorization,
    when the edit is actually submitted — so just unlocking a document
    to look at it can never burn through a single-use key.

    Returns (True, document_key, authorization_id, "OK") on success, or
    (False, None, None, reason) on failure.
    """
    key_hash = hash_secret(raw_key)
    now = datetime.now(timezone.utc)

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT id, document_key, key_hash, permission, expires_at, max_uses, uses_count, revoked
        FROM authorizations
        WHERE employee_username = ?
    """, (employee_username,))
    candidates = cursor.fetchall()
    connection.close()

    for (auth_id, document_key, stored_hash, stored_permission, expires_at,
         max_uses, uses_count, revoked) in candidates:

        if stored_hash != key_hash:
            continue  # not this record — try the next candidate, if any

        if revoked:
            return False, None, None, "This authorization has been revoked."

        if stored_permission != permission:
            return False, None, None, f"This key does not grant '{permission}' permission."

        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at)
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                if now > expiry:
                    return False, None, None, "This authorization key has expired."
            except ValueError:
                pass  # malformed date shouldn't hard-crash the request

        if max_uses is not None and uses_count >= max_uses:
            return False, None, None, "This authorization key has already been used the maximum number of times."

        return True, document_key, auth_id, "OK"

    # No candidate matched at all (wrong key, or a key meant for a
    # different account). One generic message either way, so we don't
    # leak which part was wrong.
    return False, None, None, "That authorization key isn't valid for your account."


def verify_and_consume_authorization(document_key: str, employee_username: str,
                                      raw_key: str, permission: str = "EDIT"):
    """Checks every rule required by the spec, scoped to this exact
    (document, employee) pair:
      - a matching, non-revoked record exists
      - the key hash matches
      - it hasn't expired
      - it hasn't exceeded max_uses
      - the permission matches

    On success: increments uses_count (consumes one use) and returns
    (True, authorization_id, "OK").
    On failure: returns (False, None, reason) and does NOT consume
    anything — a failed guess must never eat into a legitimate use.
    """
    key_hash = hash_secret(raw_key)
    now = datetime.now(timezone.utc)

    connection = get_connection()
    cursor = connection.cursor()
    # Scoped to this document_key + this employee_username: a key for
    # Document A can never match a lookup for Document B, and a key
    # issued to Employee A can never match a lookup for Employee B —
    # even if the raw key strings were somehow identical.
    cursor.execute("""
        SELECT id, key_hash, permission, expires_at, max_uses, uses_count, revoked
        FROM authorizations
        WHERE document_key = ? AND employee_username = ?
    """, (document_key, employee_username))
    candidates = cursor.fetchall()

    for (auth_id, stored_hash, stored_permission, expires_at, max_uses,
         uses_count, revoked) in candidates:

        if stored_hash != key_hash:
            continue  # not this record — try the next candidate, if any

        if revoked:
            connection.close()
            return False, None, "This authorization has been revoked."

        if stored_permission != permission:
            connection.close()
            return False, None, f"This key does not grant '{permission}' permission."

        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at)
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                if now > expiry:
                    connection.close()
                    return False, None, "This authorization key has expired."
            except ValueError:
                pass  # malformed date shouldn't hard-crash the request

        if max_uses is not None and uses_count >= max_uses:
            connection.close()
            return False, None, "This authorization key has already been used the maximum number of times."

        # All checks passed — consume one use now, inside this same
        # connection, so a single-use key can't be raced/reused.
        cursor.execute(
            "UPDATE authorizations SET uses_count = uses_count + 1 WHERE id = ?",
            (auth_id,),
        )
        connection.commit()
        connection.close()
        return True, auth_id, "OK"

    connection.close()
    # No candidate matched at all (wrong key, wrong employee, or wrong
    # document — all collapse to the same generic message so we don't
    # leak which part was wrong).
    return False, None, "Invalid authorization key for this document and user."
