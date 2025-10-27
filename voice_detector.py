"""voice_detector.py
Capture audio from the system microphone and estimate vocal emotion."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:  # Audio capture is optional; guard import for environments without sounddevice
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore

try:  # Lazy import so the module degrades gracefully when transformers lacks audio deps
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


class VoiceEmotionDetector:
    """Record microphone input and classify emotion using a Hugging Face model."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        duration_seconds: int = 5,
        model_name: str = "superb/wav2vec2-base-superb-er",
    ) -> None:
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.model_name = model_name
        self._classifier: Optional[Callable[[Any], Iterable[Any]]] = None

    def _ensure_classifier(self) -> None:
        if self._classifier is None:
            if pipeline is None:
                raise ImportError(
                    "transformers is required for audio classification. Install it to enable voice analysis."
                )
            self._classifier = pipeline(
                "audio-classification",
                model=self.model_name,
            )

    def record_audio(self) -> np.ndarray:
        """Record audio from the default microphone."""
        if sd is None:
            raise ImportError(
                "sounddevice is required for audio capture. Install it or disable voice analysis."
            )

        num_frames = int(self.sample_rate * self.duration_seconds)
        recording = sd.rec(
            num_frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return recording.squeeze()

    def analyze_emotion(self, audio: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Classify the supplied audio array and return the dominant emotion."""
        if audio.size == 0 or not np.any(np.isfinite(audio)):
            return None, 0.0, {}

        self._ensure_classifier()
        assert self._classifier is not None
        outputs = self._classifier({"array": audio.tolist(), "sampling_rate": self.sample_rate})

        if not isinstance(outputs, Iterable):
            return None, 0.0, {}

        probabilities: Dict[str, float] = {}
        for item in outputs:
            if isinstance(item, dict):
                label = item.get("label")
                score = item.get("score", 0.0)
            else:
                label = getattr(item, "label", None)
                score = getattr(item, "score", 0.0)

            if isinstance(label, str):
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    numeric_score = 0.0
                probabilities[label.lower()] = numeric_score * 100

        if not probabilities:
            return None, 0.0, {}

        dominant = max(probabilities.items(), key=lambda kv: kv[1])
        return dominant[0], dominant[1], probabilities

    def capture_and_analyze(self) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Convenience wrapper to record audio before classifying."""
        try:
            audio = self.record_audio()
        except Exception as error:
            print(f"Audio capture failed: {error}")
            return None, 0.0, {}

        try:
            return self.analyze_emotion(audio)
        except Exception as error:
            print(f"Audio emotion analysis failed: {error}")
            return None, 0.0, {}

    @staticmethod
    def format_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
        """Normalize probabilities to sum to 100 for display purposes."""
        total = sum(probabilities.values())
        if total <= 0:
            return dict(probabilities)
        return {label: round((value / total) * 100, 1) for label, value in probabilities.items()}