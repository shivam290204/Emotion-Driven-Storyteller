"""text_analyzer.py
Analyze written input and infer emotion labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

try:  # transformers is optional at runtime
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


class TextEmotionAnalyzer:
    """Leverage a Hugging Face text-classification model to score emotions."""

    LABEL_MAP = {
        "joy": "happy",
        "positive": "happy",
        "love": "happy",
        "optimism": "happy",
        "admiration": "happy",
        "gratitude": "happy",
        "anger": "angry",
        "annoyance": "angry",
        "disgust": "angry",
        "fear": "fear",
        "sadness": "sad",
        "negative": "sad",
        "pessimism": "sad",
        "surprise": "surprise",
        "neutral": "neutral",
        "other": "neutral",
    }

    def __init__(
        self,
        *,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._classifier: Optional[Callable[..., Any]] = None

    def _ensure_classifier(self):
        if self._classifier is None:
            if pipeline is None:
                raise ImportError(
                    "transformers is required for text emotion analysis. Install it to enable this feature."
                )
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
            )

    def analyze_emotion(self, text: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Return dominant emotion, confidence percentage, and probability breakdown."""
        if not text or not text.strip():
            return None, 0.0, {}

        self._ensure_classifier()
        classifier = self._classifier
        if classifier is None:  # pragma: no cover - defensive
            return None, 0.0, {}
        outputs = classifier(
            text.strip(),
            return_all_scores=True,
            truncation=True,
            max_length=self.max_length,
        )

        # transformers returns list[list[dict]] when return_all_scores=True
        if not outputs:
            return None, 0.0, {}
        scores = outputs[0]

        grouped = defaultdict(float)
        for item in scores:
            label = item.get("label")
            score = item.get("score", 0.0)
            if not isinstance(label, str):
                continue
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = 0.0
            mapped_label = self.LABEL_MAP.get(label.lower(), label.lower())
            grouped[mapped_label] += score_value

        if not grouped:
            return None, 0.0, {}

        probabilities = {k: v * 100 for k, v in grouped.items()}
        dominant_label = max(probabilities.items(), key=lambda kv: kv[1])
        return dominant_label[0], dominant_label[1], probabilities

    @staticmethod
    def format_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
        total = sum(probabilities.values())
        if total <= 0:
            return probabilities
        return {label: round((value / total) * 100, 1) for label, value in probabilities.items()}
