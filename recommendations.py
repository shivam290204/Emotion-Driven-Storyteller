"""recommendations.py
Rule-based wellness suggestions with simple feedback-aware ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional

from analytics_logger import fetch_recommendation_feedback


@dataclass(frozen=True)
class Recommendation:
    title: str
    description: str
    url: str
    provider: str


_DEFAULT_LIBRARY: Mapping[str, Mapping[str, List[Recommendation]]] = {
    "happy": {
        "Music": [
            Recommendation(
                "Feel-Good Vibes",
                "Upbeat pop and indie tracks to keep the momentum going.",
                "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
                "Spotify",
            ),
            Recommendation(
                "Sunshine Acoustic",
                "Bright acoustic covers that match an energetic mood.",
                "https://music.youtube.com/playlist?list=PLvhY1wQUSr8",  # truncated list ID
                "YouTube Music",
            ),
        ],
        "Exercise": [
            Recommendation(
                "HIIT Express",
                "15-minute high-intensity workout to channel that excitement.",
                "https://www.youtube.com/watch?v=ml6cT4AZdqI",
                "YouTube",
            ),
            Recommendation(
                "Dance Cardio",
                "Follow-along dance class to celebrate the good vibes.",
                "https://www.youtube.com/watch?v=4sPq0-G1n9Y",
                "YouTube",
            ),
        ],
        "Podcasts": [
            Recommendation(
                "How I Built This",
                "Inspirational founder stories to fuel your optimism.",
                "https://podcasts.apple.com/podcast/how-i-built-this-with-guy-raz/id1150510297",
                "NPR",
            ),
            Recommendation(
                "On Purpose with Jay Shetty",
                "Quick hits of gratitude and positivity.",
                "https://open.spotify.com/show/5EqqB52m2bsr4k1Ii7sStc",
                "Spotify",
            ),
        ],
    },
    "sad": {
        "Music": [
            Recommendation(
                "Gentle Piano",
                "Soothing instrumentals to accompany reflection.",
                "https://open.spotify.com/playlist/37i9dQZF1DWSlw12ofHcMM",
                "Spotify",
            ),
            Recommendation(
                "Comfort Classics",
                "Soft pop ballads when you need a comforting soundtrack.",
                "https://music.youtube.com/playlist?list=PL63F0C78739B09958",
                "YouTube Music",
            ),
        ],
        "Meditation": [
            Recommendation(
                "10-Minute Self-Compassion",
                "Guided practice focused on kindness toward yourself.",
                "https://www.youtube.com/watch?v=IeblJdB2-Vo",
                "Great Meditation",
            ),
            Recommendation(
                "Calm Body Scan",
                "Ease tension with a calming body scan session.",
                "https://www.calm.com/programs/3",
                "Calm",
            ),
        ],
        "Podcasts": [
            Recommendation(
                "Terrible, Thanks for Asking",
                "Real stories validating complicated feelings.",
                "https://www.ttfa.org/listen",
                "TTFA",
            ),
            Recommendation(
                "Unlocking Us",
                "Brené Brown explores emotions and human connection.",
                "https://open.spotify.com/show/4P86ZzHf7EOlRG7do9LkKZ",
                "Spotify",
            ),
        ],
    },
    "angry": {
        "Exercise": [
            Recommendation(
                "Kickboxing Release",
                "Cardio kickboxing session to burn off frustration.",
                "https://www.youtube.com/watch?v=1pcqnxkmwLM",
                "YouTube",
            ),
            Recommendation(
                "Power Yoga",
                "Channel intensity into a strong vinyasa flow.",
                "https://www.doyogawithme.com/yoga-classes/power-yoga-strength-and-flexibility",
                "DoYogaWithMe",
            ),
        ],
        "Meditation": [
            Recommendation(
                "Cooling Breath Practice",
                "Pranayama technique to settle the nervous system.",
                "https://www.youtube.com/watch?v=J5YhTHL-QgQ",
                "Yoga With Adriene",
            ),
            Recommendation(
                "Compassion Meditation",
                "Shift perspective with a 12-minute compassion session.",
                "https://www.youtube.com/watch?v=cHH0A8SKCf8",
                "Mindful",
            ),
        ],
        "Podcasts": [
            Recommendation(
                "The Happiness Lab",
                "Science-backed tools to manage tough emotions.",
                "https://pushkin.fm/podcasts/the-happiness-lab",
                "Pushkin",
            ),
            Recommendation(
                "On Purpose - Healthy Anger",
                "Episode focused on reframing anger constructively.",
                "https://open.spotify.com/episode/6e5ieRP70dNDOx6jttuPAo",
                "Spotify",
            ),
        ],
    },
    "fear": {
        "Meditation": [
            Recommendation(
                "Grounding Breath",
                "Box-breath meditation to regain a sense of safety.",
                "https://www.youtube.com/watch?v=7X49wco6e5w",
                "Headspace",
            ),
            Recommendation(
                "Anxiety SOS",
                "Short guided reset for anxious moments.",
                "https://insighttimer.com/candacevandell/guided-meditations/anxiety-sos",
                "Insight Timer",
            ),
        ],
        "Podcasts": [
            Recommendation(
                "Therapy for Black Girls",
                "Tools and expert advice for navigating anxious thoughts.",
                "https://open.spotify.com/show/1k8Y30uq2LX8F0pS5Y0qgC",
                "Spotify",
            ),
            Recommendation(
                "The Calm Collective",
                "Stories about grief, change, and courage.",
                "https://podcasts.apple.com/podcast/the-calm-collective/id1363600262",
                "Apple Podcasts",
            ),
        ],
        "Games": [
            Recommendation(
                "Monument Valley",
                "Relaxing puzzle game with mindful pacing.",
                "https://www.ustwo.com/games/monument-valley",
                "ustwo games",
            ),
            Recommendation(
                "Alto's Odyssey",
                "Endless sandboarding adventure with calming visuals.",
                "https://altoadventure.com/alto-odyssey",
                "Snowman",
            ),
        ],
    },
    "surprise": {
        "Music": [
            Recommendation(
                "Discover Weekly",
                "Let Spotify surface unexpected tracks tailored to you.",
                "https://open.spotify.com/playlist/37i9dQZEVXcVhNLoQEqplR",
                "Spotify",
            ),
            Recommendation(
                "Global Beats",
                "Explore lively world music for a spontaneous mood.",
                "https://music.youtube.com/playlist?list=PLMC9KNkIncKtsacKpgMb0CVq40QJ4atNA",
                "YouTube Music",
            ),
        ],
        "Podcasts": [
            Recommendation(
                "TED Radio Hour",
                "Curated talks to fuel curiosity.",
                "https://www.npr.org/podcasts/510298/ted-radio-hour",
                "NPR",
            ),
            Recommendation(
                "Stuff You Should Know",
                "Deep dives into topics you never saw coming.",
                "https://www.iheart.com/podcast/stuff-you-should-know-20922291/",
                "iHeart",
            ),
        ],
        "Experiences": [
            Recommendation(
                "GeoGuessr",
                "Guess locations from street views—embrace the surprise!",
                "https://www.geoguessr.com/",
                "GeoGuessr",
            ),
            Recommendation(
                "Digital Escape Room",
                "Collaborative puzzle adventure for a novel thrill.",
                "https://www.mysteryescaperoom.com/digital-escape-rooms",
                "Mystery Escape Room",
            ),
        ],
    },
    "neutral": {
        "Music": [
            Recommendation(
                "Lo-Fi Focus",
                "Ambient beats for relaxed productivity.",
                "https://open.spotify.com/playlist/37i9dQZF1DX3PFzdbtx1Us",
                "Spotify",
            ),
            Recommendation(
                "Calm Background",
                "Instrumental backdrop for staying centered.",
                "https://music.youtube.com/playlist?list=PLH8vjQx76zGxeqmoeZaZX6mE6oPD58LlK",
                "YouTube Music",
            ),
        ],
        "Meditation": [
            Recommendation(
                "Daily Mindfulness",
                "Five-minute mindful check-in to stay balanced.",
                "https://www.youtube.com/watch?v=ZToicYcHIOU",
                "Headspace",
            ),
            Recommendation(
                "Calm Playlists",
                "Ambient soundscapes for gentle focus.",
                "https://app.relaxmelodies.com/",
                "Relax Melodies",
            ),
        ],
        "Exercise": [
            Recommendation(
                "Desk Stretch Reset",
                "Mobility routine to refresh between tasks.",
                "https://www.youtube.com/watch?v=KdQ7VZbOR6g",
                "MadFit",
            ),
            Recommendation(
                "Nature Walk Challenge",
                "Use AllTrails to discover a new local path.",
                "https://www.alltrails.com/",
                "AllTrails",
            ),
        ],
    },
}


class RecommendationEngine:
    """Simple rule engine that reorders suggestions based on feedback history."""

    def __init__(
        self,
        library: Optional[Mapping[str, Mapping[str, List[Recommendation]]]] = None,
        fallback_emotion: str = "neutral",
    ) -> None:
        self._library = library or _DEFAULT_LIBRARY
        self._fallback = fallback_emotion

    def supported_emotions(self) -> List[str]:
        return sorted(self._library.keys())

    def get_recommendations(
        self,
        emotion: Optional[str],
        limit_per_category: int = 2,
    ) -> Mapping[str, List[Recommendation]]:
        mood = (emotion or self._fallback).lower()
        options = self._library.get(mood, self._library.get(self._fallback, {}))
        if not options:
            return {}

        feedback = _preference_scores(mood)

        ranked: Dict[str, List[Recommendation]] = {}
        for category, items in options.items():
            sorted_items = sorted(
                items,
                key=lambda rec: feedback.get((category, rec.title), 0.0),
                reverse=True,
            )
            ranked[category] = sorted_items[:limit_per_category]
        return ranked


def _preference_scores(emotion: str) -> Dict[tuple[str, str], float]:
    df = fetch_recommendation_feedback(emotion)
    if df.empty:
        return {}

    grouped: MutableMapping[tuple[str, str], Dict[str, float]] = {}
    for _, row in df.iterrows():
        key = (row["category"], row["title"])
        actions = grouped.setdefault(key, {"liked": 0.0, "dismissed": 0.0})
        action = row["action"].lower()
        if action == "liked":
            actions["liked"] = float(row["count"])
        elif action == "dismissed":
            actions["dismissed"] = float(row["count"])
        elif action == "opened":
            actions.setdefault("opened", 0.0)
            actions["opened"] += float(row["count"])

    scores: Dict[tuple[str, str], float] = {}
    for key, counts in grouped.items():
        likes = counts.get("liked", 0.0)
        dismissals = counts.get("dismissed", 0.0)
        opened_bonus = counts.get("opened", 0.0) * 0.2
        scores[key] = likes - dismissals + opened_bonus
    return scores
