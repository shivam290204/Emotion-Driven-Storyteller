"""security_utils.py
Local encryption helpers to protect emotion analytics at rest.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_DIR = Path(".secure")
_KEY_PATH = _KEY_DIR / "analytics.key"
_KEY_BITS = 256
_NONCE_SIZE = 12


def _ensure_store() -> None:
    _KEY_DIR.mkdir(parents=True, exist_ok=True)


def get_encryption_key() -> bytes:
    _ensure_store()
    if not _KEY_PATH.exists():
        key = AESGCM.generate_key(bit_length=_KEY_BITS)
        _KEY_PATH.write_bytes(base64.b64encode(key))
        return key
    try:
        data = _KEY_PATH.read_bytes()
        return base64.b64decode(data, validate=True)
    except Exception as error:  # pragma: no cover - corrupted key fallback
        key = AESGCM.generate_key(bit_length=_KEY_BITS)
        _KEY_PATH.write_bytes(base64.b64encode(key))
        return key


def encrypt_text(plaintext: str) -> str:
    key = get_encryption_key()
    aes = AESGCM(key)
    nonce = os.urandom(_NONCE_SIZE)
    cipher = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
    payload = nonce + cipher
    return base64.b64encode(payload).decode("ascii")


def decrypt_text(token: str) -> str:
    key = get_encryption_key()
    data = base64.b64decode(token.encode("ascii"), validate=True)
    nonce = data[:_NONCE_SIZE]
    cipher = data[_NONCE_SIZE:]
    aes = AESGCM(key)
    plaintext = aes.decrypt(nonce, cipher, None)
    return plaintext.decode("utf-8")


def destroy_key_material() -> None:
    if _KEY_PATH.exists():
        _KEY_PATH.unlink()
    try:
        _KEY_DIR.rmdir()
    except OSError:
        pass
