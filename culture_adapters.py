"""culture_adapters.py
Utilities for adapting emotion handling and storytelling to cultural contexts.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, Mapping, Optional

DEFAULT_CULTURE = "global"

SUPPORTED_CULTURES: Dict[str, str] = {
    "global": "Global",
    "indian": "Indian",
    "japanese": "Japanese",
    "american": "American",
}

_CULTURE_LIBRARY: Dict[str, Dict[str, object]] = {
    "global": {
        "probability_weights": {},
        "story": {
            "style": "Use an inclusive, universally relatable tone.",
            "settings": ["a peaceful park", "a friendly café", "a shared digital space"],
            "idioms": ["shared journey", "finding common ground"],
            "names": ["Alex", "Sam", "Jordan"],
            "greeting": "Hey there",
            "language": "English",
        },
    },
    "indian": {
        "probability_weights": {
            "face": {"neutral": 0.9, "happy": 1.1, "sad": 1.05, "angry": 0.95},
            "voice": {"neutral": 0.95, "happy": 1.05, "sad": 1.1, "angry": 0.9},
            "text": {"neutral": 0.9, "happy": 1.05, "sad": 1.1},
        },
        "story": {
            "style": "Lean into warmth, family ties, and community resilience.",
            "settings": ["a monsoon evening in Mumbai", "a festive Delhi market", "the ghats of Varanasi"],
            "idioms": ["dil se", "jugaad", "rang birangi"],
            "names": ["Aarav", "Diya", "Ishan", "Meera"],
            "greeting": "Namaste",
            "language": "English with Hindi phrases",
        },
    },
    "japanese": {
        "probability_weights": {
            "face": {"neutral": 1.15, "happy": 0.95, "sad": 1.05},
            "voice": {"neutral": 1.1, "happy": 0.95, "sad": 1.05},
            "text": {"neutral": 1.1, "happy": 0.95, "fear": 1.05},
        },
        "story": {
            "style": "Emphasize subtle emotional shifts, harmony, and reflective pacing.",
            "settings": ["a serene Kyoto garden", "a lantern-lit festival", "a quiet Tokyo café"],
            "idioms": ["mono no aware", "gambatte", "kokoro"],
            "names": ["Haruki", "Sakura", "Ren", "Yuna"],
            "greeting": "Konnichiwa",
            "language": "English with Japanese expressions",
        },
    },
    "american": {
        "probability_weights": {
            "face": {"happy": 1.1, "surprise": 1.05, "neutral": 0.95},
            "voice": {"happy": 1.1, "angry": 1.05, "neutral": 0.9},
            "text": {"happy": 1.1, "surprise": 1.05, "fear": 0.95},
        },
        "story": {
            "style": "Highlight individual agency, optimistic arcs, and direct dialogue.",
            "settings": ["a road trip on Route 66", "a bustling New York street", "a cozy Seattle bookstore"],
            "idioms": ["go the extra mile", "silver lining", "weather the storm"],
            "names": ["Taylor", "Jordan", "Morgan", "Avery"],
            "greeting": "Hi",
            "language": "English",
        },
    },
}


def normalize_culture(code: Optional[str]) -> str:
    if not code:
        return DEFAULT_CULTURE
    lowered = code.strip().lower()
    return lowered if lowered in SUPPORTED_CULTURES else DEFAULT_CULTURE


def adjust_probabilities(
    probabilities: Mapping[str, float],
    culture: Optional[str],
    *,
    modality: str,
) -> Dict[str, float]:
    normalized = normalize_culture(culture)
    if not probabilities:
        return {}

    weights = (
        _CULTURE_LIBRARY.get(normalized, {})
        .get("probability_weights", {})  # type: ignore[union-attr]
        .get(modality, {})  # type: ignore[union-attr]
    )

    weighted: Dict[str, float] = {}
    for label, value in probabilities.items():
        base = float(value)
        if base <= 0:
            continue
        multiplier = float(weights.get(label.lower(), 1.0))
        weighted[label.lower()] = base * multiplier

    if not weighted:
        return {label.lower(): float(value) for label, value in probabilities.items() if float(value) > 0}

    total = sum(weighted.values())
    if total <= 0:
        return {label: 0.0 for label in weighted}

    return {label: round((value / total) * 100, 1) for label, value in weighted.items()}


def culture_story_directives(culture: Optional[str]) -> Dict[str, object]:
    normalized = normalize_culture(culture)
    data = _CULTURE_LIBRARY.get(normalized, {})
    story = data.get("story", {})  # type: ignore[assignment]
    if not isinstance(story, dict):
        return {"culture": SUPPORTED_CULTURES.get(normalized, normalized.title())}

    # Randomize a couple of elements for variety without breaking determinism too much.
    settings = story.get("settings", [])
    idioms = story.get("idioms", [])
    names = story.get("names", [])

    def pick_items(items: Iterable[str], count: int) -> Iterable[str]:
        pool = [item for item in items if item]
        if not pool:
            return []
        return random.sample(pool, k=min(count, len(pool)))

    return {
        "culture": SUPPORTED_CULTURES.get(normalized, normalized.title()),
        "style": story.get("style", ""),
        "settings": list(pick_items(settings, 2)),
        "idioms": list(pick_items(idioms, 2)),
        "names": list(pick_items(names, 2)),
        "greeting": story.get("greeting", ""),
        "language": story.get("language", "English"),
    }
