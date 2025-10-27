"""profiles.py
Utility helpers for storing and retrieving personalized user context."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from security_utils import decrypt_text, encrypt_text

_STORE_PATH = Path("user_profiles.json")


@dataclass
class UserProfile:
    name: str
    favorite_places: List[str]
    friends: List[str]
    interests: List[str]
    notes: str = ""
    culture: str = "global"

    def to_context(self) -> Dict[str, Iterable[str] | str]:
        return {
            "name": self.name,
            "favorite_places": self.favorite_places,
            "friends": self.friends,
            "interests": self.interests,
            "notes": self.notes,
            "culture": self.culture,
        }


def _read_store(path: Path = _STORE_PATH) -> Dict[str, UserProfile]:
    if not path.exists():
        return {}
    try:
        encoded = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    encoded = encoded.strip()
    if not encoded:
        return {}

    try:
        decoded = decrypt_text(encoded)
    except Exception:
        decoded = encoded

    try:
        raw = json.loads(decoded)
    except json.JSONDecodeError:
        return {}

    profiles: Dict[str, UserProfile] = {}
    for key, data in raw.items():
        profiles[key] = UserProfile(
            name=data.get("name", key),
            favorite_places=list(data.get("favorite_places", [])),
            friends=list(data.get("friends", [])),
            interests=list(data.get("interests", [])),
            notes=data.get("notes", ""),
            culture=data.get("culture", "global"),
        )
    return profiles


def load_profiles(path: Path = _STORE_PATH) -> Dict[str, UserProfile]:
    return _read_store(path)


def save_profiles(profiles: Dict[str, UserProfile], path: Path = _STORE_PATH) -> None:
    serialisable = {name: asdict(profile) for name, profile in profiles.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(serialisable, indent=2)
    encoded = encrypt_text(payload)
    path.write_text(encoded, encoding="utf-8")


def upsert_profile(profile: UserProfile, path: Path = _STORE_PATH) -> None:
    profiles = load_profiles(path)
    profiles[profile.name] = profile
    save_profiles(profiles, path)


def delete_profile(name: str, path: Path = _STORE_PATH) -> None:
    profiles = load_profiles(path)
    if name in profiles:
        profiles.pop(name)
        save_profiles(profiles, path)


def purge_profiles(path: Path = _STORE_PATH) -> None:
    if path.exists():
        path.unlink()
