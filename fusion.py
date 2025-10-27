"""fusion.py
Combine modality-specific emotion predictions into a unified outcome."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple


def normalize_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        return {label: 0.0 for label in probabilities}
    return {label: value / total for label, value in probabilities.items()}


def fuse_emotions(
    modalities: Iterable[Tuple[str, float, Dict[str, float]]],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Fuse emotion predictions.

    Each modality tuple is (source_name, confidence_percent, probability_map).
    """

    aggregated = defaultdict(float)
    weights = weights or {}

    for source, confidence, probs in modalities:
        if not probs:
            continue
        weight = weights.get(source, 1.0)
        normalized = normalize_probabilities(probs)
        for emotion, prob in normalized.items():
            aggregated[emotion] += weight * prob * (confidence / 100.0)

    if not aggregated:
        return None, 0.0, {}

    fused_probs = normalize_probabilities(dict(aggregated))
    dominant_emotion = max(fused_probs.items(), key=lambda kv: kv[1])
    confidence_percent = round(dominant_emotion[1] * 100, 1)

    return dominant_emotion[0], confidence_percent, {
        label: round(value * 100, 1) for label, value in fused_probs.items()
    }