"""emotion_detector.py
Provides real-time facial emotion detection using the webcam and DeepFace.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
from deepface import DeepFace


class EmotionDetector:
    """Detect emotions from webcam frames leveraging DeepFace analysis."""

    DEFAULT_BACKEND = "opencv"

    def __init__(self) -> None:
        """Prepare the detector; the DeepFace model loads lazily on first use."""
        self._last_emotion: Optional[str] = None
        self._last_confidence: float = 0.0
        self._last_probabilities: Dict[str, float] = {}
        self._last_group_results: List[Dict[str, Any]] = []
        self._available_emotions = [
            "happy",
            "sad",
            "angry",
            "surprise",
            "fear",
            "neutral",
        ]
        print("EmotionDetector ready. DeepFace model loads on first analysis.")

    @property
    def available_emotions(self) -> Tuple[str, ...]:
        """Return emotions exposed by the detector for manual fallback UI."""
        return tuple(self._available_emotions)

    def detect_emotion(self, frame: Any) -> Tuple[Optional[str], float, Any, Dict[str, float]]:
        """Analyze a single frame and annotate it with bounding box and labels."""
        try:
            annotated, results = self._analyze_frame(frame)
        except Exception as error:  # DeepFace can throw when no face is detected
            print(f"DeepFace analysis failed: {error}")
            return None, 0.0, frame, {}

        if not results:
            return None, 0.0, annotated, {}

        primary = results[0]
        dominant_emotion = primary.get("emotion")
        confidence = float(primary.get("confidence", 0.0))
        probabilities = dict(primary.get("probabilities", {}))

        self._last_emotion = dominant_emotion
        self._last_confidence = confidence
        self._last_probabilities = probabilities

        return dominant_emotion, confidence, annotated, probabilities

    def start_webcam_scan(
        self,
        confidence_threshold: float = 40.0,
        timeout_seconds: int = 15,
    ) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """
        Open the webcam, stream frames, and return the first emotion above the threshold.
        """

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None

        print("Webcam scan started. Press 'q' to cancel early.")
        start_time = time.time()

        detected: Optional[Tuple[str, float, Dict[str, float]]] = None

        while True:
            if (time.time() - start_time) > timeout_seconds:
                print("Scan timed out without confident prediction.")
                break

            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            frame = cv2.flip(frame, 1)
            emotion, confidence, annotated_frame, probabilities = self.detect_emotion(frame)

            if emotion and confidence >= confidence_threshold:
                detected = (emotion, confidence, probabilities)
                cv2.imshow("Emotion Scan", annotated_frame)
                cv2.waitKey(1000)
                break

            cv2.imshow("Emotion Scan - Press 'q' to quit", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Scan cancelled by user.")
                break

        cap.release()
        cv2.destroyAllWindows()

        return detected

    def start_group_scan(
        self,
        confidence_threshold: float = 40.0,
        timeout_seconds: int = 15,
        min_participants: int = 1,
    ) -> Optional[List[Dict[str, Any]]]:
        """Capture multiple faces and return per-participant emotion details."""

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam for group scan.")
            return None

        print("Group scan started. Press 'q' to cancel early.")
        start_time = time.time()
        group_results: Optional[List[Dict[str, Any]]] = None

        while True:
            if (time.time() - start_time) > timeout_seconds:
                print("Group scan timed out without confident prediction.")
                break

            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            frame = cv2.flip(frame, 1)

            try:
                annotated_frame, results = self._analyze_frame(frame)
            except Exception as error:
                print(f"DeepFace group analysis failed: {error}")
                annotated_frame = frame
                results = []

            if results:
                confidences = [float(entry.get("confidence", 0.0)) for entry in results]
                best_confidence = max(confidences) if confidences else 0.0
                if len(results) >= min_participants and best_confidence >= confidence_threshold:
                    group_results = results
                    cv2.imshow("Group Emotion Scan", annotated_frame)
                    cv2.waitKey(1000)
                    break

            cv2.imshow("Group Emotion Scan - Press 'q' to quit", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Group scan cancelled by user.")
                break

        cap.release()
        cv2.destroyAllWindows()

        self._last_group_results = group_results or []
        return group_results

    def last_detection(self) -> Tuple[Optional[str], float]:
        """Return the most recent emotion and confidence detected."""
        return self._last_emotion, self._last_confidence

    def last_probabilities(self) -> Dict[str, float]:
        """Return the last probability distribution captured."""
        return self._last_probabilities

    def last_group_results(self) -> List[Dict[str, Any]]:
        """Return the previous group analysis results."""
        return self._last_group_results

    @staticmethod
    def format_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
        if not probabilities:
            return {}
        total = sum(probabilities.values())
        if total <= 0:
            return {label: 0.0 for label in probabilities}
        return {
            label.lower(): round((value / total) * 100, 1)
            for label, value in probabilities.items()
        }

    def _analyze_frame(self, frame: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """Run DeepFace analysis and return annotated frame plus participant details."""

        analysis: Any = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=self.DEFAULT_BACKEND,
        )

        annotated = frame.copy()
        analyses: List[Dict[str, Any]] = []

        if isinstance(analysis, list):
            analyses = [item for item in analysis if isinstance(item, dict)]
        elif isinstance(analysis, dict):
            if "emotion" in analysis:
                analyses = [analysis]
            else:
                analyses = [
                    value
                    for value in analysis.values()
                    if isinstance(value, dict) and value.get("emotion")
                ]

        participants: List[Dict[str, Any]] = []

        for index, item in enumerate(analyses):
            dominant = str(item.get("dominant_emotion") or "").lower()
            emotion_scores: Dict[str, float] = dict(item.get("emotion") or {})
            region: Dict[str, int] = dict(item.get("region") or {})
            formatted_probs = self.format_probabilities(emotion_scores)
            confidence = (
                float(formatted_probs.get(dominant, 0.0))
                if dominant
                else 0.0
            )

            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("w", 0))
            h = int(region.get("h", 0))

            if w and h:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label_text = dominant.title() if dominant else "Unknown"
            cv2.putText(
                annotated,
                f"{label_text} ({confidence:.1f}%)",
                (max(x, 10), max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            participants.append(
                {
                    "id": index + 1,
                    "label": f"Friend {index + 1}",
                    "emotion": dominant,
                    "confidence": confidence,
                    "probabilities": formatted_probs,
                    "source": "face",
                    "region": region,
                }
            )

        participants.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return annotated, participants
