import hashlib
import shutil
import os
import secrets
import string


def hash_secret(raw: str) -> str:
    """SHA-256 of a string (as opposed to calculate_hash, which is for
    files). Used for authorization keys: we store this hash, never the
    raw key, exactly like calculate_hash never stores the file itself."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_KEY_ALPHABET = string.ascii_uppercase + string.digits


def generate_authorization_key() -> str:
    """Cryptographically secure random key, formatted like
    AUTH-7K92-M4PX. secrets.choice (not random.choice) is what makes
    this unpredictable enough to use as a credential."""
    group = lambda: "".join(secrets.choice(_KEY_ALPHABET) for _ in range(4))
    return f"AUTH-{group()}-{group()}"


def calculate_hash(filepath):
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as file:
        while chunk := file.read(4096):
            sha256.update(chunk)

    return sha256.hexdigest()


def create_backup(filepath):
    """Create trusted backup of uploaded file, preserving per-user folders
    so two accounts never overwrite each other's backups."""

    if os.path.normpath(filepath).startswith(os.path.normpath("storage") + os.sep):
        rel = os.path.relpath(filepath, "storage")
    else:
        rel = os.path.basename(filepath)

    backup_path = os.path.join("backup", rel)
    os.makedirs(os.path.dirname(backup_path) or "backup", exist_ok=True)
    shutil.copy2(filepath, backup_path)

    return backup_path