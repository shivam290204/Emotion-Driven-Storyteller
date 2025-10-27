from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class EmotionDetector:
    DEFAULT_BACKEND: str

    def __init__(self) -> None: ...

    @property
    def available_emotions(self) -> Tuple[str, ...]: ...

    def detect_emotion(self, frame: Any) -> Tuple[Optional[str], float, Any, Dict[str, float]]: ...

    def start_webcam_scan(
        self,
        confidence_threshold: float = ...,
        timeout_seconds: int = ...,
    ) -> Optional[Tuple[str, float, Dict[str, float]]]: ...

    def start_group_scan(
        self,
        confidence_threshold: float = ...,
        timeout_seconds: int = ...,
        min_participants: int = ...,
    ) -> Optional[List[Dict[str, Any]]]: ...

    def last_detection(self) -> Tuple[Optional[str], float]: ...

    def last_probabilities(self) -> Dict[str, float]: ...

    def last_group_results(self) -> List[Dict[str, Any]]: ...

    @staticmethod
    def format_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]: ...

    def _analyze_frame(self, frame: Any) -> Tuple[Any, List[Dict[str, Any]]]: ...
