"""emotion_forecaster.py
Generate near-term emotion forecasts from historical logs for proactive alerts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from analytics_logger import fetch_events


@dataclass
class ForecastInsight:
    label: str
    timestamp: datetime
    emotion: str
    confidence: float
    alert_level: str
    message: str
    insights: List[str]
    secondary: Optional[Tuple[str, float]] = None


class EmotionForecaster:
    """Heuristic forecaster that projects likely moods over the next few time windows."""

    SLOT_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
        {"label": "Later today", "offset_hours": 6},
        {"label": "Tomorrow", "offset_hours": 24},
        {"label": "Upcoming few days", "offset_hours": 72},
    )

    RECENCY_WINDOW_HOURS = 36
    RECENCY_HALFLIFE_HOURS = 12

    def __init__(self, *, history_limit: int = 720) -> None:
        self.history_limit = history_limit

    def generate_forecast(self, now: Optional[datetime] = None) -> List[ForecastInsight]:
        history = self._load_history()
        if history.empty:
            return []

        now_ts = pd.Timestamp(now or datetime.now())
        prepared = self._prepare(history, now_ts)

        recency_weights = self._recent_distribution(prepared)
        if not recency_weights:
            recency_weights = self._overall_distribution(prepared)

        forecasts: List[ForecastInsight] = []
        previous_emotion: Optional[str] = None

        for slot in self.SLOT_DEFINITIONS:
            target_time = now_ts + pd.Timedelta(hours=slot["offset_hours"])
            slot_label = self._resolve_slot_label(slot, target_time)
            pattern_weights = self._pattern_distribution(prepared, target_time)
            combined = self._combine_distributions(recency_weights, pattern_weights)
            if not combined:
                combined = self._overall_distribution(prepared)
            if not combined:
                continue

            ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
            top_emotion, top_score = ranked[0]
            secondary = ranked[1] if len(ranked) > 1 else None

            confidence = round(min(top_score * 100, 100.0), 1)
            alert = self._alert_profile(top_emotion)

            insights = self._build_insights(
                slot_time=target_time,
                emotion=top_emotion,
                recency_weights=recency_weights,
                pattern_weights=pattern_weights,
                combined_weights=combined,
                secondary=secondary,
            )

            if previous_emotion and previous_emotion != top_emotion:
                insights.append(
                    f"Shift from {previous_emotion.title()} trend → {top_emotion.title()} outlook; prepare for the change."
                )

            previous_emotion = top_emotion

            forecasts.append(
                ForecastInsight(
                    label=slot_label,
                    timestamp=target_time.to_pydatetime(),
                    emotion=top_emotion,
                    confidence=confidence,
                    alert_level=alert["level"],
                    message=alert["message"],
                    insights=insights,
                    secondary=secondary,
                )
            )

        return forecasts

    def _load_history(self) -> pd.DataFrame:
        df = fetch_events(limit=self.history_limit)
        if df.empty:
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        return df[df["emotion"].notna()]

    def _prepare(self, df: pd.DataFrame, now_ts: pd.Timestamp) -> pd.DataFrame:
        prepared = df.copy()
        prepared["emotion"] = prepared["emotion"].astype(str).str.lower()
        prepared["confidence"] = prepared.get("confidence", pd.Series([0.0] * len(prepared))).astype(float)
        prepared["hours_ago"] = prepared["timestamp"].apply(
            lambda ts: (now_ts - ts).total_seconds() / 3600.0 if pd.notnull(ts) else float("inf")
        )
        prepared["weekday"] = prepared["timestamp"].dt.day_name()
        prepared["hour_block"] = (prepared["timestamp"].dt.hour // 6) * 6
        prepared.sort_values("timestamp", ascending=False, inplace=True)
        return prepared

    def _recent_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        recent = df[df["hours_ago"] <= self.RECENCY_WINDOW_HOURS].copy()
        if recent.empty:
            return {}
        recent["weight"] = recent["hours_ago"].apply(
            lambda hours: math.exp(-hours / self.RECENCY_HALFLIFE_HOURS)
        )
        scores = recent.groupby("emotion")["weight"].sum().to_dict()
        return self._normalize(scores)

    def _pattern_distribution(self, df: pd.DataFrame, target: pd.Timestamp) -> Dict[str, float]:
        weekday = target.day_name()
        block = (target.hour // 6) * 6

        subset = df[(df["weekday"] == weekday) & (df["hour_block"] == block)]
        if subset.empty:
            subset = df[df["hour_block"] == block]
        if subset.empty:
            subset = df
        scores = subset.groupby("emotion")["confidence"].mean().to_dict()
        return self._normalize(scores)

    def _overall_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        counts = {
            emotion: float(size)
            for emotion, size in df.groupby("emotion").size().to_dict().items()
        }
        return self._normalize(counts)

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        filtered = {emotion: max(float(value), 0.0) for emotion, value in scores.items() if value and value > 0}
        total = sum(filtered.values())
        if total <= 0:
            return {}
        return {emotion: value / total for emotion, value in filtered.items()}

    @staticmethod
    def _combine_distributions(
        recent: Dict[str, float],
        pattern: Dict[str, float],
        *,
        recent_weight: float = 0.6,
        pattern_weight: float = 0.4,
    ) -> Dict[str, float]:
        emotions = set(recent) | set(pattern)
        if not emotions:
            return {}
        combined = {
            emotion: recent_weight * recent.get(emotion, 0.0) + pattern_weight * pattern.get(emotion, 0.0)
            for emotion in emotions
        }
        return EmotionForecaster._normalize(combined)

    def _build_insights(
        self,
        *,
        slot_time: pd.Timestamp,
        emotion: str,
        recency_weights: Dict[str, float],
        pattern_weights: Dict[str, float],
        combined_weights: Dict[str, float],
        secondary: Optional[Tuple[str, float]],
    ) -> List[str]:
        bucket_label = self._time_bucket(slot_time.hour)
        recency_share = recency_weights.get(emotion, 0.0) * 100
        pattern_share = pattern_weights.get(emotion, 0.0) * 100
        combined_share = combined_weights.get(emotion, 0.0) * 100

        insights = [
            f"Recent {emotion.title()} signals account for {recency_share:.0f}% of the forecast strength.",
            f"Typical {slot_time.day_name()} {bucket_label} patterns add {pattern_share:.0f}% support.",
            f"Overall likelihood sits near {combined_share:.0f}% based on available history.",
        ]

        if secondary:
            secondary_emotion, secondary_score = secondary
            gap = combined_share - (secondary_score * 100)
            if gap < 12:
                insights.append(
                    f"Watch for {secondary_emotion.title()} as a secondary possibility (within {abs(gap):.0f}% of the leader)."
                )

        return insights

    @staticmethod
    def _alert_profile(emotion: str) -> Dict[str, str]:
        emotion = emotion.lower()
        suggestions = {
            "happy": {
                "level": "positive",
                "message": "Ride the upbeat energy—plan something celebratory or share the joy with someone.",
            },
            "surprise": {
                "level": "info",
                "message": "Stay flexible—surprises may pop up, so leave a little room in your schedule.",
            },
            "neutral": {
                "level": "info",
                "message": "A balanced window ahead—use it to maintain healthy routines and rest.",
            },
            "sad": {
                "level": "warning",
                "message": "Line up supportive rituals—message a friend, prepare comforting music, or plan a walk.",
            },
            "angry": {
                "level": "critical",
                "message": "Consider proactive stress relief: breathing breaks, journaling, or a quick workout.",
            },
            "fear": {
                "level": "warning",
                "message": "Note possible anxiety triggers—schedule grounding moments and reduce unnecessary commitments.",
            },
        }
        return suggestions.get(
            emotion,
            {
                "level": "info",
                "message": "Keep mindful of upcoming moments and check in with yourself ahead of time.",
            },
        )

    @staticmethod
    def _time_bucket(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    @staticmethod
    def _resolve_slot_label(slot: Dict[str, Any], target: pd.Timestamp) -> str:
        if slot["label"] == "Later today" and target.date() != pd.Timestamp.now().date():
            return f"Soon ({target.strftime('%A %H:%M')})"
        if slot["label"] == "Tomorrow":
            return f"Tomorrow ({target.strftime('%A')})"
        if slot["label"] == "Upcoming few days":
            return f"{target.strftime('%A')} outlook"
        return slot["label"]
