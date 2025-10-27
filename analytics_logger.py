"""analytics_logger.py
Persistent logging of emotion events and retrieval for analytics."""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from security_utils import decrypt_text, destroy_key_material, encrypt_text

_DB_PATH = Path("emotion_insights.db")


def initialize_database(db_path: Path = _DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emotion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                details TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                action TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS story_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT,
                action TEXT NOT NULL,
                story TEXT NOT NULL,
                profile_name TEXT,
                generator TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS game_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                choice_label TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reward INTEGER NOT NULL,
                details TEXT
            )
            """
        )
        conn.commit()


@contextmanager
def _connect(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _encrypt_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    return encrypt_text(text)


def _decrypt_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if not text:
        return text
    try:
        return decrypt_text(text)
    except Exception:
        return text


def log_emotion_event(
    timestamp: str,
    source: str,
    emotion: str,
    confidence: float,
    details: Optional[str] = None,
    db_path: Path = _DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO emotion_events (timestamp, source, emotion, confidence, details) VALUES (?, ?, ?, ?, ?)",
            (
                timestamp,
                _encrypt_field(source),
                _encrypt_field(emotion),
                confidence,
                _encrypt_field(details),
            ),
        )
        conn.commit()


def log_recommendation_event(
    timestamp: str,
    emotion: str,
    category: str,
    title: str,
    action: str,
    metadata: Optional[str] = None,
    db_path: Path = _DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO recommendation_events (timestamp, emotion, category, title, action, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                _encrypt_field(emotion),
                _encrypt_field(category),
                _encrypt_field(title),
                _encrypt_field(action),
                _encrypt_field(metadata),
            ),
        )
        conn.commit()


def fetch_events(
    limit: Optional[int] = None,
    sources: Optional[Iterable[str]] = None,
    db_path: Path = _DB_PATH,
) -> pd.DataFrame:
    apply_limit = limit if not sources else None
    query = "SELECT timestamp, source, emotion, confidence, details FROM emotion_events ORDER BY timestamp DESC"
    params: List[Any] = []
    if apply_limit:
        query += " LIMIT ?"
        params.append(apply_limit)

    with _connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params or None)

    if df.empty:
        return df

    for column in ("source", "emotion", "details"):
        df[column] = df[column].apply(_decrypt_field)

    if sources:
        normalized = {str(src).lower() for src in sources}
        df = df[df["source"].fillna("").str.lower().isin(normalized)]

    if limit and df.shape[0] > limit:
        df = df.head(limit)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.reset_index(drop=True)


def emotion_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["emotion", "count", "percentage"])
    counts = df["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


def daily_trends(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.assign(date=df["timestamp"].dt.date)
        .groupby(["date", "emotion"]) ["confidence"].mean()
        .reset_index()
    )


def fetch_recommendation_feedback(
    emotion: Optional[str] = None,
    db_path: Path = _DB_PATH,
) -> pd.DataFrame:
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT emotion, category, title, action FROM recommendation_events",
            conn,
        )

    if df.empty:
        return df

    for column in ("emotion", "category", "title", "action"):
        df[column] = df[column].apply(_decrypt_field)

    if emotion:
        target = emotion.lower()
        df = df[df["emotion"].fillna("").str.lower() == target]

    if df.empty:
        return pd.DataFrame(columns=["emotion", "category", "title", "action", "count"])

    aggregated = (
        df.groupby(["emotion", "category", "title", "action"]).size().reset_index(name="count")
    )
    return aggregated


def log_story_feedback(
    timestamp: str,
    emotion: Optional[str],
    action: str,
    story: str,
    *,
    profile_name: Optional[str] = None,
    generator: str,
    db_path: Path = _DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO story_feedback (timestamp, emotion, action, story, profile_name, generator)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                _encrypt_field(emotion),
                _encrypt_field(action),
                _encrypt_field(story),
                _encrypt_field(profile_name),
                _encrypt_field(generator),
            ),
        )
        conn.commit()


def log_game_event(
    timestamp: str,
    emotion: str,
    difficulty: str,
    choice_label: str,
    outcome: str,
    reward: int,
    *,
    details: Optional[str] = None,
    db_path: Path = _DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO game_events (timestamp, emotion, difficulty, choice_label, outcome, reward, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                _encrypt_field(emotion),
                _encrypt_field(difficulty),
                _encrypt_field(choice_label),
                _encrypt_field(outcome),
                reward,
                _encrypt_field(details),
            ),
        )
        conn.commit()


def purge_all_data(*, db_path: Path = _DB_PATH, clear_key: bool = True) -> None:
    if db_path.exists():
        db_path.unlink()
    if clear_key:
        destroy_key_material()