"""Decrypt a HAL leaderboard upload (agent-evals/hal_traces on Hugging Face) without installing the harness.

HAL encrypts uploaded traces so that benchmark answers are not scraped by accident; the password is public in the
harness source (hal/utils/decrypt.py, "hal1234") and the scheme is PBKDF2-HMAC-SHA256 (480,000 iterations, 16-byte
salt shipped beside the ciphertext) feeding a Fernet key. Each zip member is a JSON object {"encrypted_data", "salt"}.

usage: python hal_decrypt.py <zip or .json.encrypted> [out.json]
"""
import base64
import json
import sys
import zipfile
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

PASSWORD = b"hal1234"


def _key(salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    return base64.urlsafe_b64encode(kdf.derive(PASSWORD))


def decrypt_blob(blob: dict) -> bytes:
    salt = base64.b64decode(blob["salt"])
    token = base64.b64decode(blob["encrypted_data"])
    return Fernet(_key(salt)).decrypt(token)


def decrypt_member(data: bytes):
    """Decrypt one encrypted member (the bytes of a *.encrypted file); returns parsed JSON or raw bytes."""
    blob = json.loads(data.decode("utf-8"))
    plain = decrypt_blob(blob)
    try:
        return json.loads(plain.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return plain


def decrypt_upload(path) -> dict:
    """Return the decrypted *_UPLOAD.json object from a trace zip (the member whose name ends with UPLOAD.json.encrypted)."""
    path = Path(path)
    if path.suffix == ".encrypted":
        return decrypt_member(path.read_bytes())
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("UPLOAD.json.encrypted")] or [n for n in z.namelist() if n.endswith(".encrypted")]
        if not names:
            raise ValueError(f"no encrypted member in {path.name}: {z.namelist()[:5]}")
        return decrypt_member(z.read(names[0]))


def describe(obj, depth=0, max_items=6):
    pad = "  " * depth
    if isinstance(obj, dict):
        for k in list(obj)[:max_items]:
            v = obj[k]
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__} ({len(v)})")
                if depth < 2:
                    describe(v, depth + 1)
            else:
                print(f"{pad}{k}: {json.dumps(v)[:100]}")
        if len(obj) > max_items:
            print(f"{pad}... {len(obj) - max_items} more keys")
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0]:")
        describe(obj[0], depth + 1)


if __name__ == "__main__":
    src = sys.argv[1]
    out = decrypt_upload(src)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", sys.argv[2])
    if isinstance(out, dict):
        print("top-level keys:", list(out.keys()))
        describe(out)
    else:
        print("decrypted", len(out), "bytes (not JSON)")
