# app.py
# Streamlit dashboard for the Emotion-Driven Interactive Storyteller.

from __future__ import annotations

from datetime import datetime
import html
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from emotion_detector import EmotionDetector
from story_generator import StoryGenerator
from voice_detector import VoiceEmotionDetector
from text_analyzer import TextEmotionAnalyzer
from fusion import fuse_emotions
from culture_adapters import (
    SUPPORTED_CULTURES,
    adjust_probabilities,
    culture_story_directives,
    normalize_culture,
)
from analytics_logger import (
    initialize_database,
    log_emotion_event,
    log_recommendation_event,
    log_story_feedback,
    log_game_event,
    fetch_events,
    emotion_summary,
    daily_trends,
    purge_all_data,
)
from recommendations import RecommendationEngine
from profiles import UserProfile, delete_profile, load_profiles, purge_profiles, upsert_profile
from game_engine import EmotionAdaptiveGame
from emotion_forecaster import EmotionForecaster, ForecastInsight


st.set_page_config(
    page_title="Emotion-Driven Storyteller",
    page_icon="🎭",
    layout="wide",
)

EMOTION_COLORS: Dict[str, str] = {
    "happy": "#F9A826",
    "sad": "#4B6CB7",
    "angry": "#E63946",
    "surprise": "#9D4EDD",
    "fear": "#2A9D8F",
    "neutral": "#6C757D",
}

ALERT_STYLES: Dict[str, Dict[str, str]] = {
    "critical": {"icon": "🚨", "accent": "#E63946", "bg": "#FCE8EA"},
    "warning": {"icon": "⚠️", "accent": "#F4A261", "bg": "#FFF2DF"},
    "positive": {"icon": "🌈", "accent": "#2A9D8F", "bg": "#E1F6EF"},
    "info": {"icon": "ℹ️", "accent": "#457B9D", "bg": "#E6F0FB"},
}


@st.cache_resource(show_spinner=False)
def get_emotion_detector() -> EmotionDetector:
    initialize_database()
    return EmotionDetector()


@st.cache_resource(show_spinner=False)
def get_story_generator() -> StoryGenerator:
    return StoryGenerator()


@st.cache_resource(show_spinner=False)
def get_voice_detector() -> VoiceEmotionDetector:
    return VoiceEmotionDetector()


@st.cache_resource(show_spinner=False)
def get_text_analyzer() -> TextEmotionAnalyzer:
    return TextEmotionAnalyzer()


@st.cache_resource(show_spinner=False)
def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine()


@st.cache_resource(show_spinner=False)
def get_game_engine() -> EmotionAdaptiveGame:
    return EmotionAdaptiveGame()


@st.cache_resource(show_spinner=False)
def get_emotion_forecaster() -> EmotionForecaster:
    return EmotionForecaster()


def initialize_state() -> None:
    defaults = {
        "current_story": "",
        "current_emotion": "",
        "current_confidence": 0.0,
        "history": [],
        "ai_used": False,
        "voice_emotion": "",
        "voice_confidence": 0.0,
        "voice_probs": {},
        "face_probs": {},
        "text_input": "",
        "text_emotion": "",
        "text_confidence": 0.0,
        "text_probs": {},
        "fused_emotion": "",
        "fused_confidence": 0.0,
        "fused_probs": {},
        "recommendation_shown_keys": [],
        "recommendation_feedback_message": "",
        "active_profile_name": "",
        "profile_edit_mode": False,
        "profile_form_target": "",
        "profile_form_name": "",
        "profile_form_places": "",
        "profile_form_friends": "",
        "profile_form_interests": "",
        "profile_form_notes": "",
        "profile_form_culture": "global",
        "story_feedback_message": "",
        "story_strategy": "dominant",
        "group_participants": [],
        "group_emotion": "",
        "group_confidence": 0.0,
        "group_probs": {},
        "group_story": "",
        "group_ai_used": False,
        "group_story_feedback": "",
        "group_aggregation": "majority",
        "group_min_faces": 2,
        "game_active": False,
        "game_session": None,
        "game_result": None,
        "game_selected_choice": "",
        "game_score": 0,
        "game_history": [],
        "game_difficulty_pref": "auto",
        "game_auto_refresh": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "profiles" not in st.session_state:
        st.session_state["profiles"] = load_profiles()


def emotion_badge(emotion: str, confidence: float) -> str:
    color = EMOTION_COLORS.get(emotion.lower(), "#6C757D")
    return (
        f"<span style='background-color:{color}; color:#fff; padding:4px 10px;"
        f" border-radius:16px; font-weight:600;'>"
        f"{emotion.title()} • {confidence:.1f}%</span>"
    )


def build_story(
    story_generator: StoryGenerator,
    emotion: str,
    *,
    use_ai: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    profile_context: Optional[Mapping[str, Any]] = None,
    emotion_blend: Optional[Dict[str, float]] = None,
    story_strategy: str = "dominant",
) -> Tuple[str, bool]:
    seed_story = story_generator.select_story(emotion)
    prompt = story_generator.craft_personalized_prompt(  # type: ignore[attr-defined]
        emotion,
        seed_story,
        profile_context,
        emotion_blend=emotion_blend,
        story_strategy=story_strategy,
    )
    if use_ai:
        ai_story = story_generator.generate_ai_story(
            emotion,
            seed_story=seed_story,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if ai_story:
            return ai_story, True
    personalized = story_generator.personalize_template_story(  # type: ignore[attr-defined]
        seed_story,
        emotion,
        profile_context,
        emotion_blend=emotion_blend,
        story_strategy=story_strategy,
    )
    return personalized, False


def build_group_story(
    story_generator: StoryGenerator,
    participants: Sequence[Mapping[str, Any]],
    *,
    dominant_emotion: Optional[str],
    use_ai: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    story_strategy: str,
    emotion_blend: Optional[Dict[str, float]] = None,
    culture_hint: Optional[str] = None,
) -> Tuple[str, bool]:
    roster = list(participants)
    guiding_emotion = (dominant_emotion or _fallback_group_emotion(roster) or "neutral").lower()
    culture_code = normalize_culture(culture_hint)
    seed_story = story_generator.select_story(guiding_emotion)
    prompt = story_generator.craft_group_prompt(
        guiding_emotion,
        roster,
        emotion_blend=emotion_blend,
        story_strategy=story_strategy,
        culture_hint=culture_code,
    )
    if use_ai:
        ai_story = story_generator.generate_ai_story(
            guiding_emotion,
            seed_story=seed_story,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if ai_story:
            return ai_story, True

    personalized = story_generator.personalize_group_template(
        seed_story,
        guiding_emotion,
        roster,
        emotion_blend=emotion_blend,
        story_strategy=story_strategy,
        culture_hint=culture_code,
    )
    return personalized, False


def _render_forecast_section(forecaster: EmotionForecaster) -> None:
    st.markdown("### Emotion Outlook")
    try:
        forecasts = forecaster.generate_forecast()
    except Exception as error:
        st.warning(f"Unable to generate emotion forecasts right now: {error}")
        return

    if not forecasts:
        st.caption("Forecasts appear after logging more emotion events across the day.")
        return

    st.caption("Projected moods for upcoming windows so you can plan supportive actions in advance.")
    for forecast in forecasts:
        _render_forecast_card(forecast)


def _render_forecast_card(forecast: ForecastInsight) -> None:
    style = ALERT_STYLES.get(forecast.alert_level, ALERT_STYLES["info"])
    icon = style.get("icon", "ℹ️")
    accent = style.get("accent", "#457B9D")
    background = style.get("bg", "#E6F0FB")
    timestamp_str = forecast.timestamp.strftime("%a %H:%M")
    badge_html = emotion_badge(forecast.emotion, forecast.confidence)
    message_text = html.escape(forecast.message)
    emotion_label = html.escape(forecast.emotion.title())

    insights = list(forecast.insights or [])
    if forecast.secondary:
        secondary_emotion, secondary_share = forecast.secondary
        secondary_text = (
            f"Secondary watch: {secondary_emotion.title()} ({secondary_share * 100:.0f}% likelihood)"
        )
        insights.append(secondary_text)

    insights_html = "".join(f"<li>{html.escape(item)}</li>" for item in insights)
    if not insights_html:
        insights_html = "<li>Keep logging check-ins to sharpen future forecasts.</li>"

    card_html = f"""
    <div style=\"background-color:{background}; border-left:4px solid {accent}; padding:12px 14px; margin-bottom:12px; border-radius:6px;\">
        <div style=\"display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;\">
            <span style=\"font-weight:600; color:{accent};\">{icon} {html.escape(forecast.label)} • {emotion_label} outlook</span>
            <span style=\"font-size:0.85rem; color:#555;\">{html.escape(timestamp_str)}</span>
        </div>
        <div style=\"margin-bottom:6px;\">{badge_html}</div>
        <p style=\"margin:0 0 8px 0; color:#333;\">{message_text}</p>
        <ul style=\"margin:0; padding-left:18px; color:#333;\">{insights_html}</ul>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def main() -> None:
    initialize_state()
    purge_notice = st.session_state.pop("purge_notice", False)

    detector = get_emotion_detector()
    generator = get_story_generator()
    text_analyzer = get_text_analyzer()
    recommender = get_recommendation_engine()
    forecaster = get_emotion_forecaster()

    st.title("🎭 Emotion-Driven Interactive Storyteller")
    st.caption("Detect how you feel, hear your story, and save the moments that move you.")
    st.divider()
    st.info(
        "All emotion analysis runs locally. Logs are encrypted on your device and never uploaded."
    )
    if purge_notice:
        st.success("History erased. Fresh encryption keys are now active.")

    with st.sidebar:
        st.header("Configuration")
        confidence_threshold = st.slider(
            "Emotion confidence threshold",
            min_value=10.0,
            max_value=90.0,
            value=45.0,
            help="Minimum confidence percentage required to accept an emotion.",
        )
        timeout_seconds = st.slider(
            "Webcam scan timeout (seconds)",
            min_value=5,
            max_value=30,
            value=15,
        )

        st.subheader("Narration")
        enable_tts = st.checkbox("Enable narration", value=True)
        tts_rate = st.slider("Speech rate", 120, 220, 160)
        tts_volume = st.slider("Volume", 0.4, 1.0, 0.9)
        generator.set_tts_preferences(tts_rate, tts_volume, voice=None)

        st.subheader("Story Engine")
        use_ai = st.checkbox("Generate with AI (requires transformers)", value=False)
        max_tokens = st.slider("Max new tokens", 80, 400, 220)
        temperature = st.slider("Creativity (temperature)", 0.3, 1.2, 0.9)
        top_p = st.slider("Nucleus sampling (top_p)", 0.1, 1.0, 0.92)
        strategy_options = {
            "Follow dominant emotion": "dominant",
            "Blend top intensities": "blend",
        }
        current_strategy = st.session_state.get("story_strategy", "dominant")
        strategy_label = next(
            (label for label, value in strategy_options.items() if value == current_strategy),
            "Follow dominant emotion",
        )
        selected_strategy = st.radio(
            "Story mood strategy",
            list(strategy_options.keys()),
            index=list(strategy_options.keys()).index(strategy_label),
            help="Blend multiple emotions for nuanced narratives or stay with the strongest mood.",
        )
        st.session_state["story_strategy"] = strategy_options[selected_strategy]

        st.subheader("Voice Analysis")
        enable_voice_capture = st.checkbox("Enable voice emotion capture", value=True)
        voice_duration = st.slider("Recording duration (seconds)", 3, 10, 5)
        voice_detector = get_voice_detector() if enable_voice_capture else None
        if voice_detector:
            voice_detector.duration_seconds = voice_duration

        st.subheader("Text Analysis")
        enable_text_analysis = st.checkbox("Enable text emotion analysis", value=True)

        st.subheader("Wellness Companion")
        enable_recommendations = st.checkbox(
            "Show mood-based suggestions",
            value=True,
            help="Surface music, podcasts, movement, or mindfulness content for the detected mood.",
        )
        recommendations_per_category = st.slider(
            "Suggestions per category",
            min_value=1,
            max_value=4,
            value=2,
        )

        st.subheader("Manual emotion")
        available_emotions = detector.available_emotions
        manual_emotion = st.selectbox(
            "Choose an emotion",
            options=["(Select emotion)"] + list(available_emotions),
        )
        manual_generate = st.button("✨ Generate story without webcam")

        if manual_emotion != "(Select emotion)" and st.button(
            "➕ Add manual emotion to group",
            key="manual_group_add",
            use_container_width=True,
        ):
            manual_probs = _apply_cultural_adjustment({manual_emotion: 100.0}, "text")
            dominant_manual, dominant_manual_conf = _dominant_from_probabilities(
                manual_probs,
                fallback=(manual_emotion, 100.0),
            )
            manual_emotion = dominant_manual or manual_emotion
            _append_group_participant(
                label=f"Manual #{len(st.session_state.get('group_participants', [])) + 1}",
                emotion=manual_emotion,
                confidence=dominant_manual_conf if dominant_manual_conf else 100.0,
                probabilities=manual_probs,
                source="manual",
            )
            _log_group_event("manual")
            st.success(f"Added {manual_emotion.title()} to the group session.")

        st.subheader("Group Mode")
        group_options = {
            "Majority vibe (vote)": "majority",
            "Average blend (mean)": "average",
            "Strongest signal (spotlight)": "strongest",
        }
        current_group_method = st.session_state.get("group_aggregation", "majority")
        current_group_label = next(
            (label for label, value in group_options.items() if value == current_group_method),
            "Majority vibe (vote)",
        )
        selected_group_label = st.selectbox(
            "Aggregation mode",
            list(group_options.keys()),
            index=list(group_options.keys()).index(current_group_label),
            help="Choose how to combine multiple emotions when building a group summary.",
        )
        selected_group_method = group_options[selected_group_label]
        if selected_group_method != st.session_state.get("group_aggregation"):
            st.session_state["group_aggregation"] = selected_group_method
            _update_group_summary()
        else:
            st.session_state["group_aggregation"] = selected_group_method

        min_faces = st.slider(
            "Minimum faces to capture",
            min_value=1,
            max_value=5,
            value=st.session_state.get("group_min_faces", 2),
            help="Require at least this many faces before completing a group scan.",
        )
        if min_faces != st.session_state.get("group_min_faces"):
            st.session_state["group_min_faces"] = min_faces

        if st.button("Reset group session", use_container_width=True, key="reset_group_sidebar"):
            _clear_group_session()

        st.subheader("Personal profile")
        profiles = _get_profiles()
        profile_options = ["(No profile)"] + sorted(profiles.keys())
        active_profile = st.session_state.get("active_profile_name", "")
        default_index = profile_options.index(active_profile) if active_profile in profile_options else 0
        selected_profile = st.selectbox(
            "Active profile",
            profile_options,
            index=default_index,
            help="Select whose context should tailor the story prompts.",
        )
        _activate_profile(selected_profile if selected_profile != "(No profile)" else "")

        profile_action_cols = st.columns(3)
        if profile_action_cols[0].button("New", use_container_width=True):
            _start_profile_edit(None)
        if selected_profile != "(No profile)" and profile_action_cols[1].button(
            "Edit",
            use_container_width=True,
        ):
            _start_profile_edit(selected_profile)
        if selected_profile != "(No profile)" and profile_action_cols[2].button(
            "Delete",
            use_container_width=True,
        ):
            _delete_profile(selected_profile)

        if st.session_state.get("profile_edit_mode"):
            _render_profile_form()
        else:
            preview = _get_profile_context()
            if preview:
                st.caption(_profile_preview_text(preview))

        st.subheader("Emotion-Adaptive Game")
        difficulty_options = {
            "Auto-match intensity": "auto",
            "Gentle (supportive)": "gentle",
            "Balanced (steady)": "balanced",
            "Dynamic (high energy)": "dynamic",
        }
        pref_value = st.session_state.get("game_difficulty_pref", "auto")
        pref_label = next((label for label, value in difficulty_options.items() if value == pref_value), "Auto-match intensity")
        selected_label = st.selectbox(
            "Game intensity mode",
            list(difficulty_options.keys()),
            index=list(difficulty_options.keys()).index(pref_label),
            help="Pick how challenging the adaptive mini game should feel.",
        )
        st.session_state["game_difficulty_pref"] = difficulty_options[selected_label]

        auto_refresh_current = st.session_state.get("game_auto_refresh", True)
        auto_refresh_toggle = st.checkbox(
            "Auto-refresh with new mood detections",
            value=auto_refresh_current,
            help="When enabled, each new detected emotion spins up a fresh scenario automatically.",
        )
        st.session_state["game_auto_refresh"] = auto_refresh_toggle

        if st.button("🎮 Launch adaptive game", use_container_width=True):
            if not _start_game_session():
                st.warning("Detect or choose an emotion first to adapt the game.")

        if st.session_state.get("game_history"):
            if st.button("Reset game score", type="secondary", use_container_width=True):
                _end_game_session(reset_score=True)

        st.subheader("Privacy & Security")
        st.caption(
            "Inference happens offline and encrypted logs stay under your control."
        )
        if st.button(
            "🧹 Erase all saved history",
            use_container_width=True,
            key="purge_history_button",
        ):
            purge_all_data()
            purge_profiles()
            initialize_database()
            st.session_state.clear()
            st.session_state["purge_notice"] = True
            st.experimental_rerun()  # type: ignore[attr-defined]

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("### Live Emotion Scan")
        st.write(
            "Launch the webcam scanner to capture a frame, analyse your mood, and craft a story."
        )
        start_scan = st.button("🎬 Start Emotion Scan", use_container_width=True)

        if start_scan:
            with st.spinner("Accessing webcam. A preview window will open; press 'q' to cancel."):
                detection = detector.start_webcam_scan(
                    confidence_threshold=confidence_threshold,
                    timeout_seconds=timeout_seconds,
                )

            if detection:
                detected_emotion, confidence, face_probabilities = detection
                base_probs = face_probabilities or {detected_emotion or "neutral": confidence}
                adjusted_face_probs = _apply_cultural_adjustment(base_probs, "face")
                fallback_conf = confidence if confidence else max(base_probs.values() or [0.0])
                dominant, dominant_conf = _dominant_from_probabilities(
                    adjusted_face_probs,
                    fallback=(detected_emotion or "neutral", fallback_conf),
                )
                detected_emotion = dominant or (detected_emotion or "neutral")
                confidence = dominant_conf if dominant_conf else fallback_conf
                st.success(
                    f"Detected **{detected_emotion.title()}** with confidence {confidence:.1f}%"
                )
                log_emotion_event(
                    datetime.now().isoformat(),
                    "face",
                    detected_emotion,
                    confidence,
                    details=_emotion_details_json(adjusted_face_probs, source="face"),
                )
                st.session_state["face_probs"] = adjusted_face_probs
                story_strategy = st.session_state.get("story_strategy", "dominant")
                story, ai_used = build_story(
                    generator,
                    detected_emotion,
                    use_ai=use_ai,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    profile_context=_get_profile_context(),
                    emotion_blend=adjusted_face_probs,
                    story_strategy=story_strategy,
                )
                st.session_state.update(
                    {
                        "current_story": story,
                        "current_emotion": detected_emotion,
                        "current_confidence": confidence,
                        "ai_used": ai_used,
                    }
                )
                _append_history(story, detected_emotion, confidence, ai_used)

                if enable_tts:
                    generator.narrate_story(story)  # type: ignore[attr-defined]
                log_emotion_event(
                    datetime.now().isoformat(),
                    "story",
                    detected_emotion,
                    confidence,
                    details=_emotion_details_json(adjusted_face_probs, source="story_face"),
                )
                if st.session_state.get("game_auto_refresh"):
                    _start_game_session(
                        emotion_override=detected_emotion,
                        confidence=confidence,
                        auto_trigger=True,
                    )

            else:
                last_emotion, last_conf = detector.last_detection()
                st.warning(
                    "No confident emotion detected. Adjust the threshold or use manual selection."
                )
                if last_emotion:
                    st.info(
                        f"Last partial detection: {last_emotion.title()} at {last_conf:.1f}% confidence."
                    )

        if manual_generate:
            if manual_emotion == "(Select emotion)":
                st.warning("Pick an emotion from the dropdown first.")
            else:
                manual_blend = _story_emotion_blend(manual_emotion) or {manual_emotion: 100.0}
                manual_blend = _apply_cultural_adjustment(manual_blend, "text")
                dominant_manual, dominant_manual_conf = _dominant_from_probabilities(
                    manual_blend,
                    fallback=(manual_emotion, 100.0),
                )
                manual_emotion = dominant_manual or manual_emotion
                manual_confidence = dominant_manual_conf if dominant_manual_conf else 0.0
                story, ai_used = build_story(
                    generator,
                    manual_emotion,
                    use_ai=use_ai,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    profile_context=_get_profile_context(),
                    emotion_blend=manual_blend,
                    story_strategy=st.session_state.get("story_strategy", "dominant"),
                )
                st.session_state.update(
                    {
                        "current_story": story,
                        "current_emotion": manual_emotion,
                        "current_confidence": manual_confidence,
                        "ai_used": ai_used,
                    }
                )
                _append_history(story, manual_emotion, manual_confidence, ai_used)
                st.success(f"Generated story for **{manual_emotion.title()}**.")
                if enable_tts:
                    generator.narrate_story(story)  # type: ignore[attr-defined]
                log_emotion_event(
                    datetime.now().isoformat(),
                    "story_manual",
                    manual_emotion,
                    manual_confidence,
                    details=_emotion_details_json(manual_blend, source="story_manual"),
                )
                if st.session_state.get("game_auto_refresh"):
                    _start_game_session(
                        emotion_override=manual_emotion,
                        confidence=manual_confidence,
                        auto_trigger=True,
                    )

        st.markdown("### Group Emotion Scan")
        st.write(
            "Scan multiple faces at once to capture a shared vibe for collaborative stories."
        )
        if st.button("👥 Scan group faces", use_container_width=True):
            min_faces_required = st.session_state.get("group_min_faces", 2)
            with st.spinner(
                "Accessing webcam for group scan. A preview window will open; press 'q' to cancel."
            ):
                group_detection = detector.start_group_scan(
                    confidence_threshold=confidence_threshold,
                    timeout_seconds=timeout_seconds,
                    min_participants=min_faces_required,
                )

            if group_detection:
                _set_group_participants(group_detection)
                _log_group_event("face")
                shared_emotion = st.session_state.get("group_emotion") or "mixed"
                shared_confidence = st.session_state.get("group_confidence", 0.0)
                st.success(
                    f"Captured {len(group_detection)} participants. Shared mood: {shared_emotion.title()} ({shared_confidence:.1f}%)."
                )
            else:
                st.warning("No confident group emotion detected. Try repositioning or lowering the threshold.")

        st.markdown("### Voice Emotion")
        if enable_voice_capture and voice_detector:
            if st.button("🎤 Record voice sample", use_container_width=True):
                with st.spinner(f"Recording for {voice_duration} seconds..."):
                    voice_emotion, voice_confidence, voice_probs = voice_detector.capture_and_analyze()

                if voice_emotion:
                    formatted_voice_probs = VoiceEmotionDetector.format_probabilities(voice_probs)
                    adjusted_voice_probs = _apply_cultural_adjustment(
                        formatted_voice_probs or {voice_emotion: voice_confidence},
                        "voice",
                    )
                    fallback_voice_conf = voice_confidence if voice_confidence else max(
                        (formatted_voice_probs or {}).values() or [0.0]
                    )
                    dominant_voice, dominant_conf = _dominant_from_probabilities(
                        adjusted_voice_probs,
                        fallback=(voice_emotion, fallback_voice_conf),
                    )
                    voice_emotion = dominant_voice or voice_emotion
                    voice_confidence = dominant_conf if dominant_conf else fallback_voice_conf
                    st.session_state.update(
                        {
                            "voice_emotion": voice_emotion,
                            "voice_confidence": voice_confidence,
                            "voice_probs": adjusted_voice_probs,
                        }
                    )
                    st.success(
                        f"Dominant voice emotion: {voice_emotion.title()} at {voice_confidence:.1f}% confidence."
                    )
                    log_emotion_event(
                        datetime.now().isoformat(),
                        "voice",
                        voice_emotion,
                        voice_confidence,
                        details=_emotion_details_json(adjusted_voice_probs, source="voice"),
                    )
                    if st.session_state.get("game_auto_refresh"):
                        _start_game_session(
                            emotion_override=voice_emotion,
                            confidence=voice_confidence,
                            auto_trigger=True,
                        )
                    if st.button(
                        "➕ Add voice mood to group",
                        key="voice_group_add",
                        use_container_width=True,
                    ):
                        _append_group_participant(
                            label=f"Voice #{len(st.session_state.get('group_participants', [])) + 1}",
                            emotion=voice_emotion,
                            confidence=voice_confidence,
                            probabilities=adjusted_voice_probs,
                            source="voice",
                        )
                        _log_group_event("voice")
                        st.success("Voice emotion added to the group session.")
                else:
                    st.warning(
                        "Could not detect a clear vocal emotion. Ensure microphone access is granted and background noise is minimal."
                    )
        else:
            st.caption("Voice capture disabled in the sidebar.")

        st.markdown("### Text Emotion")
        st.markdown("### Fused Emotion")
        face_weight = st.slider("Weight: Face", 0.1, 2.0, 1.0, step=0.1)
        voice_weight = st.slider("Weight: Voice", 0.1, 2.0, 1.0, step=0.1)
        text_weight = st.slider("Weight: Text", 0.1, 2.0, 1.0, step=0.1)
        fusion_weights = {"face": face_weight, "voice": voice_weight, "text": text_weight}
        if st.button("🔗 Fuse available emotions", use_container_width=True):
            modalities = []
            face_emotion = st.session_state.get("current_emotion")
            face_conf = st.session_state.get("current_confidence", 0.0)
            if face_emotion and face_conf:
                modalities.append(
                    (
                        "face",
                        face_conf,
                        st.session_state.get("face_probs", {}) or {face_emotion: face_conf},
                    )
                )
            voice_emotion = st.session_state.get("voice_emotion")
            if voice_emotion:
                modalities.append(
                    (
                        "voice",
                        st.session_state["voice_confidence"],
                        st.session_state.get("voice_probs", {}),
                    )
                )
            text_emotion = st.session_state.get("text_emotion")
            if text_emotion:
                modalities.append(
                    (
                        "text",
                        st.session_state["text_confidence"],
                        st.session_state.get("text_probs", {}),
                    )
                )

            if not modalities:
                st.warning("Generate at least one modality before fusing.")
            else:
                fused = fuse_emotions(modalities, fusion_weights)
                fused_emotion, fused_confidence, fused_probs = fused
                if fused_emotion:
                    adjusted_fused_probs = _apply_cultural_adjustment(fused_probs, "fused")
                    dominant_fused, dominant_fused_conf = _dominant_from_probabilities(
                        adjusted_fused_probs,
                        fallback=(fused_emotion, fused_confidence),
                    )
                    fused_emotion = dominant_fused or fused_emotion
                    fused_confidence = dominant_fused_conf if dominant_fused_conf else fused_confidence
                    fused_probs = adjusted_fused_probs or fused_probs
                    st.session_state.update(
                        {
                            "fused_emotion": fused_emotion,
                            "fused_confidence": fused_confidence,
                            "fused_probs": fused_probs,
                        }
                    )
                    st.success(
                        f"Fused result: {fused_emotion.title()} • {fused_confidence:.1f}% confidence"
                    )
                    log_emotion_event(
                        datetime.now().isoformat(),
                        "fusion",
                        fused_emotion,
                        fused_confidence,
                        details=_emotion_details_json(fused_probs, source="fusion"),
                    )
                    if st.session_state.get("game_auto_refresh"):
                        _start_game_session(
                            emotion_override=fused_emotion,
                            confidence=fused_confidence,
                            auto_trigger=True,
                        )
                else:
                    st.warning("Fusion did not yield a dominant emotion. Try adjusting weights.")
        if enable_text_analysis:
            text_value = st.text_area(
                "Describe how you're feeling or what happened today",
                value=st.session_state.get("text_input", ""),
                height=160,
            )
            text_value = text_value or ""
            st.session_state["text_input"] = text_value

            if st.button("📝 Analyze text emotion", use_container_width=True):
                clean_text = text_value.strip()
                if not clean_text:
                    st.warning("Enter some text before running the analysis.")
                else:
                    with st.spinner("Analyzing text sentiment..."):
                        try:
                            (
                                text_emotion,
                                text_confidence,
                                text_probs,
                            ) = text_analyzer.analyze_emotion(clean_text)
                        except ImportError as error:
                            st.error(str(error))
                            text_emotion = None
                            text_confidence = 0.0
                            text_probs = {}

                    if text_emotion:
                        formatted_text_probs = TextEmotionAnalyzer.format_probabilities(text_probs)
                        adjusted_text_probs = _apply_cultural_adjustment(
                            formatted_text_probs or {text_emotion: text_confidence},
                            "text",
                        )
                        fallback_text_conf = text_confidence if text_confidence else max(
                            (formatted_text_probs or {}).values() or [0.0]
                        )
                        dominant_text, dominant_text_conf = _dominant_from_probabilities(
                            adjusted_text_probs,
                            fallback=(text_emotion, fallback_text_conf),
                        )
                        text_emotion = dominant_text or text_emotion
                        text_confidence = dominant_text_conf if dominant_text_conf else fallback_text_conf
                        st.session_state.update(
                            {
                                "text_emotion": text_emotion,
                                "text_confidence": text_confidence,
                                "text_probs": adjusted_text_probs,
                            }
                        )
                        st.success(
                            f"Text suggests **{text_emotion.title()}** with {text_confidence:.1f}% confidence."
                        )
                        log_emotion_event(
                            datetime.now().isoformat(),
                            "text",
                            text_emotion,
                            text_confidence,
                            details=_emotion_details_json(adjusted_text_probs, source="text"),
                        )
                        if st.session_state.get("game_auto_refresh"):
                            _start_game_session(
                                emotion_override=text_emotion,
                                confidence=text_confidence,
                                auto_trigger=True,
                            )
                        if st.button(
                            "➕ Add text mood to group",
                            key="text_group_add",
                            use_container_width=True,
                        ):
                            _append_group_participant(
                                label=f"Text #{len(st.session_state.get('group_participants', [])) + 1}",
                                emotion=text_emotion,
                                confidence=text_confidence,
                                probabilities=adjusted_text_probs,
                                source="text",
                            )
                            _log_group_event("text")
                            st.success("Text emotion added to the group session.")
                    else:
                        st.warning("Could not infer an emotion from the provided text.")
        else:
            st.caption("Text analysis disabled in the sidebar.")

    with col_right:
        _render_forecast_section(forecaster)

        st.markdown("### Story Output")
        if st.session_state.get("current_story"):
            badge_html = emotion_badge(
                st.session_state["current_emotion"],
                st.session_state["current_confidence"],
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            context_details = _get_profile_context()
            if context_details:
                st.caption(_profile_preview_text(context_details))
            if st.session_state.get("ai_used"):
                st.caption("Generated with Hugging Face text-generation pipeline")
            st.write(st.session_state["current_story"])
            if st.session_state.get("story_feedback_message"):
                st.info(st.session_state["story_feedback_message"])
                st.session_state["story_feedback_message"] = ""
            feedback_cols = st.columns(2)
            if feedback_cols[0].button(
                "👍 Loved it",
                use_container_width=True,
                key="story_feedback_like",
            ):
                _handle_story_feedback("liked")
            if feedback_cols[1].button(
                "👎 Needs work",
                use_container_width=True,
                key="story_feedback_dislike",
            ):
                _handle_story_feedback("disliked")
            if st.button("🔊 Replay narration", use_container_width=True, key="replay"):
                if enable_tts:
                    generator.narrate_story(st.session_state["current_story"])  # type: ignore[attr-defined]
                else:
                    st.info("Enable narration in the sidebar to replay audio.")
        else:
            st.info("Stories you generate will appear here with emotion insights.")

        st.markdown("### Story History")
        if st.session_state["history"]:
            with st.expander("Show previous stories", expanded=False):
                for entry in reversed(st.session_state["history"]):
                    badge_html = emotion_badge(entry["emotion"], entry["confidence"])
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.caption(
                        f"{entry['timestamp']} • {'AI' if entry['ai_used'] else 'Template'} story"
                    )
                    st.write(entry["story"])
                    st.divider()
        else:
            st.caption("No stories yet. Start the scanner or generate one manually!")

        st.markdown("### Face Analysis")
        face_probs = st.session_state.get("face_probs", {})
        if face_probs:
            _display_probability_breakdown(face_probs)
        else:
            st.caption("Run the webcam scanner to see facial emotion intensities.")

        st.markdown("### Voice Analysis")
        if st.session_state.get("voice_emotion"):
            badge_html = emotion_badge(
                st.session_state["voice_emotion"],
                st.session_state["voice_confidence"],
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            probs = st.session_state.get("voice_probs", {})
            if probs:
                _display_probability_breakdown(probs)
        else:
            st.caption("Record a voice sample to view vocal emotion insights.")

        st.markdown("### Text Analysis")
        if st.session_state.get("text_emotion"):
            badge_html = emotion_badge(
                st.session_state["text_emotion"],
                st.session_state["text_confidence"],
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            text_probs = st.session_state.get("text_probs")
            if text_probs:
                _display_probability_breakdown(text_probs)
        else:
            st.caption("Analyze some text to view sentiment insights.")

        st.markdown("### Fused Emotion")
        if st.session_state.get("fused_emotion"):
            badge_html = emotion_badge(
                st.session_state["fused_emotion"],
                st.session_state["fused_confidence"],
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            fused_probs = st.session_state.get("fused_probs", {})
            if fused_probs:
                _display_probability_breakdown(fused_probs)
        else:
            st.caption(
                "Combine available modalities using the fusion controls to see the aggregated emotion."
            )

        st.markdown("### Group Experience")
        group_participants = st.session_state.get("group_participants", [])
        if group_participants:
            st.caption(f"{len(group_participants)} participants in the current group session.")
            st.table(_group_participants_dataframe(group_participants))

            group_emotion = st.session_state.get("group_emotion")
            if group_emotion:
                badge_html = emotion_badge(
                    group_emotion,
                    st.session_state.get("group_confidence", 0.0),
                )
                st.markdown(badge_html, unsafe_allow_html=True)

            group_probs = st.session_state.get("group_probs", {})
            if group_probs:
                _display_probability_breakdown(group_probs)

            if st.button(
                "📖 Generate collaborative story",
                use_container_width=True,
                key="group_story_button",
            ):
                story, ai_used = build_group_story(
                    generator,
                    group_participants,
                    dominant_emotion=st.session_state.get("group_emotion"),
                    use_ai=use_ai,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    story_strategy=st.session_state.get("story_strategy", "dominant"),
                    emotion_blend=group_probs,
                    culture_hint=_current_culture(),
                )
                st.session_state.update(
                    {
                        "group_story": story,
                        "group_ai_used": ai_used,
                    }
                )
                log_emotion_event(
                    datetime.now().isoformat(),
                    "group_story",
                    st.session_state.get("group_emotion") or "mixed",
                    st.session_state.get("group_confidence", 0.0),
                    details=_emotion_details_json(
                        group_probs,
                        source="group_story",
                        extra={
                            "participants": group_participants,
                            "strategy": st.session_state.get("story_strategy", "dominant"),
                        },
                    ),
                )
                if enable_tts:
                    generator.narrate_story(story)  # type: ignore[attr-defined]

            if st.session_state.get("group_story"):
                if st.session_state.get("group_ai_used"):
                    st.caption("Generated with Hugging Face text-generation pipeline")
                st.write(st.session_state["group_story"])

            if st.button(
                "Clear group session",
                use_container_width=True,
                key="clear_group_main",
            ):
                _clear_group_session()
                st.success("Group session cleared.")
        else:
            st.caption("Capture multiple moods to unlock collaborative stories.")

        st.markdown("### Emotion-Adaptive Game")
        game_session = st.session_state.get("game_session")
        game_result = st.session_state.get("game_result")
        if st.session_state.get("game_active") and game_session:
            st.subheader(game_session.get("title", ""))
            st.caption(
                f"Mood: {game_session.get('emotion', 'unknown').title()} • Mode: {game_session.get('difficulty', '').title()}"
            )
            st.write(game_session.get("intro", ""))
            st.info(game_session.get("challenge", ""))
            mechanic_hint = game_session.get("mechanic_hint")
            if mechanic_hint:
                st.caption(mechanic_hint)
            soundtrack = game_session.get("soundtrack")
            if soundtrack:
                st.audio(soundtrack)

            if not game_result:
                choices = game_session.get("choices", [])
                if choices:
                    choice_map = {choice["id"]: choice for choice in choices}
                    selected_key = st.radio(
                        "Choose your move",
                        list(choice_map.keys()),
                        format_func=lambda key: choice_map[key]["text"],
                        key="game_choice_radio",
                    )
                    st.session_state["game_selected_choice"] = selected_key
                    if st.button("Lock in move", use_container_width=True) and selected_key:
                        _resolve_game_choice(str(selected_key))
                else:
                    st.warning("No choices available for this scenario.")
            else:
                reward = int(game_result.get("reward", 0))
                outcome_text = game_result.get("outcome", "")
                if reward >= 0:
                    st.success(outcome_text)
                else:
                    st.warning(outcome_text)
                st.metric("Total game score", st.session_state.get("game_score", 0))
                control_cols = st.columns(2)
                if control_cols[0].button("Next scenario", use_container_width=True):
                    _start_game_session(
                        emotion_override=game_session.get("emotion"),
                        confidence=float(game_session.get("confidence", 0.0) or 0.0),
                    )
                if control_cols[1].button("End session", use_container_width=True):
                    _end_game_session()
        else:
            if st.session_state.get("game_history"):
                st.caption(
                    f"Total game score: {st.session_state.get('game_score', 0)} • Launch a new scenario to keep the streak going."
                )
            else:
                st.caption("Launch the adaptive game to unlock mood-reactive mini quests.")

        if st.session_state.get("game_history"):
            with st.expander("Game highlights", expanded=False):
                for event in reversed(st.session_state.get("game_history", [])):
                    st.write(
                        f"{event['timestamp']} — {event['emotion'].title()} ({event['difficulty']})"
                    )
                    st.caption(f"Choice: {event['choice']} | Outcome: {event['outcome']}")
                    st.write(f"Score change: {event['reward']}")
                    st.divider()

        st.markdown("### Wellness Companion")
        if not enable_recommendations:
            st.caption("Enable mood-based suggestions in the sidebar to see curated ideas.")
        else:
            active_emotion = _resolve_primary_emotion()
            if not active_emotion:
                st.caption("Detect, narrate, or fuse an emotion to unlock personalized inspiration.")
            else:
                rec_map = recommender.get_recommendations(
                    active_emotion,
                    limit_per_category=recommendations_per_category,
                )
                if not rec_map:
                    st.caption("No recommendations available yet. Try again later.")
                else:
                    if st.session_state.get("recommendation_feedback_message"):
                        st.info(st.session_state["recommendation_feedback_message"])
                        st.session_state["recommendation_feedback_message"] = ""
                    st.caption(
                        f"Inspired by your **{active_emotion.title()}** mood—save what resonates!"
                    )
                    for category, items in rec_map.items():
                        st.subheader(category)
                        for rec in items:
                            key = f"{active_emotion}:{category}:{rec.title}"
                            if key not in st.session_state["recommendation_shown_keys"]:
                                st.session_state["recommendation_shown_keys"].append(key)
                                log_recommendation_event(
                                    datetime.now().isoformat(),
                                    active_emotion,
                                    category,
                                    rec.title,
                                    "shown",
                                    metadata=rec.url,
                                )
                            st.markdown(
                                f"**{rec.title}** — {rec.description} _(via {rec.provider})_"
                            )
                            if rec.url:
                                st.markdown(f"[Open link]({rec.url})")
                            feedback_cols = st.columns(2)
                            if feedback_cols[0].button(
                                "👍 Helpful",
                                key=f"like_{key}",
                            ):
                                log_recommendation_event(
                                    datetime.now().isoformat(),
                                    active_emotion,
                                    category,
                                    rec.title,
                                    "liked",
                                    metadata=rec.url,
                                )
                                st.session_state["recommendation_feedback_message"] = (
                                    f"Saved that you liked {rec.title}."
                                )
                            if feedback_cols[1].button(
                                "👎 Skip",
                                key=f"skip_{key}",
                            ):
                                log_recommendation_event(
                                    datetime.now().isoformat(),
                                    active_emotion,
                                    category,
                                    rec.title,
                                    "dismissed",
                                    metadata=rec.url,
                                )
                                st.session_state["recommendation_feedback_message"] = (
                                    f"We'll show fewer picks like {rec.title}."
                                )
                            st.divider()


def _get_profiles() -> Dict[str, UserProfile]:
    profiles = st.session_state.get("profiles", {})
    return profiles if isinstance(profiles, dict) else {}


def _activate_profile(name: str) -> None:
    if st.session_state.get("active_profile_name") != name:
        st.session_state["active_profile_name"] = name
        st.session_state["recommendation_shown_keys"] = []


def _start_profile_edit(name: Optional[str]) -> None:
    profiles = _get_profiles()
    st.session_state["profile_edit_mode"] = True
    if name and name in profiles:
        profile = profiles[name]
        st.session_state["profile_form_name"] = profile.name
        st.session_state["profile_form_places"] = ", ".join(profile.favorite_places)
        st.session_state["profile_form_friends"] = ", ".join(profile.friends)
        st.session_state["profile_form_interests"] = ", ".join(profile.interests)
        st.session_state["profile_form_notes"] = profile.notes
        st.session_state["profile_form_target"] = profile.name
        st.session_state["profile_form_culture"] = normalize_culture(profile.culture)
    else:
        st.session_state["profile_form_name"] = ""
        st.session_state["profile_form_places"] = ""
        st.session_state["profile_form_friends"] = ""
        st.session_state["profile_form_interests"] = ""
        st.session_state["profile_form_notes"] = ""
        st.session_state["profile_form_target"] = ""
        st.session_state["profile_form_culture"] = "global"


def _delete_profile(name: str) -> None:
    delete_profile(name)
    profiles = dict(_get_profiles())
    profiles.pop(name, None)
    st.session_state["profiles"] = profiles
    if st.session_state.get("active_profile_name") == name:
        st.session_state["active_profile_name"] = ""
    st.session_state["profile_edit_mode"] = False
    st.experimental_rerun()  # type: ignore[attr-defined]


def _render_profile_form() -> None:
    with st.form("profile_form", clear_on_submit=False):
        name = st.text_input("Name", value=st.session_state.get("profile_form_name", "")) or ""
        places = st.text_area(
            "Favorite places",
            value=st.session_state.get("profile_form_places", ""),
            help="Comma-separated list (e.g., Delhi, Shimla)",
        ) or ""
        friends = st.text_area(
            "Close friends",
            value=st.session_state.get("profile_form_friends", ""),
            help="Comma-separated names",
        ) or ""
        interests = st.text_area(
            "Interests",
            value=st.session_state.get("profile_form_interests", ""),
            help="Comma-separated hobbies or passions",
        ) or ""
        notes = st.text_area(
            "Additional notes",
            value=st.session_state.get("profile_form_notes", ""),
            help="Any custom context you'd like the stories to consider.",
        ) or ""

        culture_default = st.session_state.get("profile_form_culture", "global")
        culture_codes = list(SUPPORTED_CULTURES.keys())
        culture_index = culture_codes.index(culture_default) if culture_default in culture_codes else 0
        culture_code = st.selectbox(
            "Cultural background",
            culture_codes,
            index=culture_index,
            format_func=lambda code: SUPPORTED_CULTURES[code],
            help="Tailor emotion interpretation and storytelling tone to this cultural lens.",
        )
        st.session_state["profile_form_culture"] = culture_code

        form_cols = st.columns(2)
        save_clicked = form_cols[0].form_submit_button("Save profile")
        cancel_clicked = form_cols[1].form_submit_button("Cancel", type="secondary")

    if save_clicked:
        cleaned_name = name.strip()
        if not cleaned_name:
            st.warning("Name is required for a profile.")
            return
        normalized_culture = normalize_culture(culture_code)
        profile = UserProfile(
            name=cleaned_name,
            favorite_places=_split_to_list(places),
            friends=_split_to_list(friends),
            interests=_split_to_list(interests),
            notes=notes.strip(),
            culture=normalized_culture,
        )
        upsert_profile(profile)
        profiles = dict(_get_profiles())
        profiles[profile.name] = profile
        st.session_state["profiles"] = profiles
        st.session_state["active_profile_name"] = profile.name
        st.session_state["profile_form_culture"] = normalized_culture
        st.session_state["profile_edit_mode"] = False
        st.success(f"Saved profile for {profile.name}.")
        st.experimental_rerun()  # type: ignore[attr-defined]
    elif cancel_clicked:
        st.session_state["profile_edit_mode"] = False
        st.experimental_rerun()  # type: ignore[attr-defined]


def _profile_preview_text(context: Mapping[str, Any]) -> str:
    name = str(context.get("name", "")).strip() or "the listener"
    segments = [f"Stories tailored for {name}."]
    places = context.get("favorite_places") or []
    interests = context.get("interests") or []
    friends = context.get("friends") or []
    notes = str(context.get("notes", "")).strip()

    def _format(items: Iterable[str], label: str) -> None:
        coll = ", ".join(str(item) for item in items if str(item))
        if coll:
            segments.append(f"{label}: {coll}")

    _format(places, "Places")
    _format(interests, "Interests")
    _format(friends, "People")
    if notes:
        segments.append(notes)
    culture = context.get("culture")
    if isinstance(culture, str) and culture:
        normalized = normalize_culture(culture)
        culture_label = SUPPORTED_CULTURES.get(normalized, culture.title())
        segments.append(f"Cultural lens: {culture_label}")
        directives = culture_story_directives(normalized)
        if isinstance(directives, Mapping):
            settings = directives.get("settings") or []
            if isinstance(settings, str):
                settings_list = [settings]
            elif isinstance(settings, Iterable):
                settings_list = [str(item) for item in settings if item]
            else:
                settings_list = [str(settings)]
            if settings_list:
                segments.append("Inspiration: " + ", ".join(settings_list))
    return " • ".join(segments)


def _get_profile_context() -> Optional[Dict[str, Any]]:
    profiles = _get_profiles()
    active = st.session_state.get("active_profile_name")
    if active and active in profiles:
        return profiles[active].to_context()
    return None


def _split_to_list(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _story_emotion_blend(primary: Optional[str] = None) -> Optional[Dict[str, float]]:
    for key in ("fused_probs", "face_probs", "voice_probs", "text_probs"):
        value = st.session_state.get(key)
        if isinstance(value, dict) and value:
            return value
    if primary:
        return {primary: 100.0}
    return None


def _current_culture() -> str:
    context = _get_profile_context() or {}
    culture_code = context.get("culture") if isinstance(context, Mapping) else None
    return normalize_culture(culture_code if isinstance(culture_code, str) else None)


def _apply_cultural_adjustment(
    probabilities: Optional[Mapping[str, float]],
    modality: str,
) -> Dict[str, float]:
    if not probabilities:
        return {}
    return adjust_probabilities(probabilities, _current_culture(), modality=modality)


def _dominant_from_probabilities(
    probabilities: Mapping[str, float],
    fallback: Optional[Tuple[str, float]] = None,
) -> Tuple[Optional[str], float]:
    filtered = {label: float(value) for label, value in probabilities.items() if float(value) > 0}
    if filtered:
        dominant_label, dominant_value = max(filtered.items(), key=lambda kv: kv[1])
        return dominant_label, float(dominant_value)
    if fallback:
        return fallback[0], float(fallback[1])
    return None, 0.0


def _emotion_details_json(
    probabilities: Optional[Mapping[str, float]],
    *,
    source: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    culture_code = _current_culture()
    payload: Dict[str, Any] = {
        "culture": culture_code,
        "culture_label": SUPPORTED_CULTURES.get(culture_code, culture_code.title()),
        "probabilities": {
            str(label): float(value)
            for label, value in (probabilities or {}).items()
        },
    }
    if source:
        payload["modality"] = source
    if extra:
        payload.update({key: value for key, value in extra.items()})
    return json.dumps(payload)


def _append_group_participant(
    *,
    label: str,
    emotion: str,
    confidence: float,
    probabilities: Optional[Mapping[str, float]] = None,
    source: str,
) -> None:
    participant = {
        "label": label,
        "emotion": emotion.lower() if emotion else "",
        "confidence": float(confidence),
        "probabilities": dict(probabilities or {}),
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "culture": _current_culture(),
    }
    st.session_state.setdefault("group_participants", []).append(participant)
    _update_group_summary()


def _set_group_participants(participants: Sequence[Mapping[str, Any]]) -> None:
    formatted: List[Dict[str, Any]] = []
    for item in participants:
        label = str(item.get("label") or f"Friend {item.get('id', len(formatted) + 1)}")
        emotion = str(item.get("emotion") or "").lower()
        confidence = float(item.get("confidence", 0.0))
        probabilities = dict(item.get("probabilities") or {})
        source = str(item.get("source") or "face")
        adjusted_probs = _apply_cultural_adjustment(probabilities, source)
        dominant_entry, dominant_conf = _dominant_from_probabilities(
            adjusted_probs,
            fallback=(emotion, confidence),
        )
        formatted.append(
            {
                "label": label,
                "emotion": (dominant_entry or emotion),
                "confidence": dominant_conf if dominant_conf else confidence,
                "probabilities": adjusted_probs,
                "source": source,
                "culture": item.get("culture") or _current_culture(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    st.session_state["group_participants"] = formatted
    _update_group_summary()


def _update_group_summary() -> None:
    participants: List[Dict[str, Any]] = list(st.session_state.get("group_participants", []))
    if not participants:
        st.session_state.update(
            {
                "group_emotion": "",
                "group_confidence": 0.0,
                "group_probs": {},
            }
        )
        return

    aggregation = st.session_state.get("group_aggregation", "majority")

    if aggregation == "strongest":
        strongest = max(participants, key=lambda item: float(item.get("confidence", 0.0)))
        emotion = strongest.get("emotion", "")
        confidence = float(strongest.get("confidence", 0.0))
        probs = _normalize_probabilities(strongest.get("probabilities") or {})
    else:
        if aggregation == "majority":
            votes = Counter()
            for item in participants:
                vote = str(item.get("emotion") or "").lower()
                if vote:
                    votes[vote] += 1
            if votes:
                total_votes = sum(votes.values())
                emotion, count = votes.most_common(1)[0]
                confidence = round((count / total_votes) * 100, 1)
                probs = {
                    label: round((value / total_votes) * 100, 1)
                    for label, value in votes.items()
                }
            else:
                emotion = _fallback_group_emotion(participants) or ""
                confidence = 0.0
                probs = {}
        else:  # average
            aggregated: Dict[str, float] = {}
            contributors = 0
            for item in participants:
                item_probs = item.get("probabilities")
                if isinstance(item_probs, Mapping) and item_probs:
                    contributors += 1
                    for label, value in item_probs.items():
                        aggregated[label.lower()] = aggregated.get(label.lower(), 0.0) + float(value)
            if aggregated and contributors:
                average_probs = {
                    label: value / contributors
                    for label, value in aggregated.items()
                }
                probs = _normalize_probabilities(average_probs)
                if probs:
                    emotion = max(probs.items(), key=lambda kv: kv[1])[0]
                    confidence = float(probs.get(emotion, 0.0))
                else:
                    emotion = _fallback_group_emotion(participants) or ""
                    confidence = 0.0
            else:
                emotion = _fallback_group_emotion(participants) or ""
                confidence = 0.0
                probs = {}

    confidence = max(0.0, min(float(confidence), 100.0))
    st.session_state.update(
        {
            "group_emotion": emotion or "mixed",
            "group_confidence": confidence,
            "group_probs": probs,
            "group_story": "",
            "group_ai_used": False,
        }
    )


def _clear_group_session() -> None:
    st.session_state["group_participants"] = []
    st.session_state["group_emotion"] = ""
    st.session_state["group_confidence"] = 0.0
    st.session_state["group_probs"] = {}
    st.session_state["group_story"] = ""
    st.session_state["group_ai_used"] = False


def _log_group_event(source: str) -> None:
    participants = st.session_state.get("group_participants", [])
    if not participants:
        return
    payload = {
        "participants": participants,
        "aggregation": st.session_state.get("group_aggregation", "majority"),
        "blend": st.session_state.get("group_probs", {}),
        "culture": _current_culture(),
        "culture_label": SUPPORTED_CULTURES.get(_current_culture(), _current_culture().title()),
    }
    log_emotion_event(
        datetime.now().isoformat(),
        f"group_{source}",
        st.session_state.get("group_emotion") or "mixed",
        st.session_state.get("group_confidence", 0.0),
        details=json.dumps(payload),
    )


def _fallback_group_emotion(participants: Sequence[Mapping[str, Any]]) -> Optional[str]:
    counts = Counter()
    for item in participants:
        emotion = str(item.get("emotion") or "").lower()
        if not emotion:
            continue
        counts[emotion] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _group_participants_dataframe(participants: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records = []
    for index, item in enumerate(participants, start=1):
        probs = item.get("probabilities") or {}
        summary = _summarize_probabilities(probs)
        records.append(
            {
                "#": index,
                "Label": item.get("label", f"Friend {index}"),
                "Emotion": str(item.get("emotion") or "").title(),
                "Confidence": round(float(item.get("confidence", 0.0)), 1),
                "Snapshot": summary,
                "Source": item.get("source", "face"),
                "Culture": SUPPORTED_CULTURES.get(
                    normalize_culture(item.get("culture") if isinstance(item.get("culture"), str) else None),
                    str(item.get("culture") or "").title(),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _summarize_probabilities(probabilities: Mapping[str, float]) -> str:
    if not probabilities:
        return "-"
    top_items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return ", ".join(f"{label.title()} {value:.0f}%" for label, value in top_items if value > 0)


def _normalize_probabilities(probabilities: Mapping[str, float]) -> Dict[str, float]:
    if not probabilities:
        return {}
    total = sum(float(value) for value in probabilities.values())
    if total <= 0:
        return {}
    return {
        str(label).lower(): round((float(value) / total) * 100, 1)
        for label, value in probabilities.items()
        if float(value) > 0
    }


def _handle_story_feedback(action: str) -> None:
    story = st.session_state.get("current_story", "").strip()
    if not story:
        return
    emotion = (
        st.session_state.get("current_emotion")
        or st.session_state.get("fused_emotion")
        or st.session_state.get("voice_emotion")
        or st.session_state.get("text_emotion")
    )
    profile_name = st.session_state.get("active_profile_name") or None
    generator_source = "ai" if st.session_state.get("ai_used") else "template"
    log_story_feedback(
        datetime.now().isoformat(),
        emotion,
        action,
        story,
        profile_name=profile_name,
        generator=generator_source,
    )
    st.session_state["story_feedback_message"] = "Thanks! Future stories will reflect your taste."


def _start_game_session(
    emotion_override: Optional[str] = None,
    *,
    confidence: float = 0.0,
    auto_trigger: bool = False,
) -> bool:
    emotion = (emotion_override or _resolve_primary_emotion())
    if not emotion:
        return False

    if not confidence:
        confidence_candidates = [
            st.session_state.get("fused_confidence"),
            st.session_state.get("current_confidence"),
            st.session_state.get("voice_confidence"),
            st.session_state.get("text_confidence"),
        ]
        for candidate in confidence_candidates:
            if candidate:
                confidence = float(candidate)
                break

    engine = get_game_engine()
    preference = st.session_state.get("game_difficulty_pref", "auto")
    scenario = engine.prepare_session(
        emotion,
        difficulty=None if preference == "auto" else preference,
        confidence=confidence,
    )
    payload = scenario.to_payload()
    payload["confidence"] = confidence
    payload["trigger"] = "auto" if auto_trigger else "manual"

    st.session_state["game_active"] = True
    st.session_state["game_session"] = payload
    st.session_state["game_result"] = None
    st.session_state["game_selected_choice"] = ""
    st.session_state.setdefault("game_history", [])
    st.session_state.setdefault("game_score", 0)
    if "game_choice_radio" in st.session_state:
        del st.session_state["game_choice_radio"]
    return True


def _resolve_game_choice(choice_id: str) -> None:
    scenario = st.session_state.get("game_session") or {}
    choices = scenario.get("choices", [])
    choice = next((item for item in choices if item.get("id") == choice_id), None)
    if not choice:
        return

    timestamp = datetime.now()
    reward = int(choice.get("reward", 0))
    outcome = choice.get("outcome", "")
    choice_text = choice.get("text", "")
    st.session_state["game_result"] = {
        "choice": choice_text,
        "reward": reward,
        "outcome": outcome,
    }
    st.session_state["game_selected_choice"] = choice_id
    st.session_state["game_score"] = st.session_state.get("game_score", 0) + reward

    event = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": scenario.get("emotion", ""),
        "difficulty": scenario.get("difficulty", ""),
        "choice": choice_text,
        "outcome": outcome,
        "reward": reward,
    }
    st.session_state.setdefault("game_history", []).append(event)
    log_game_event(
        timestamp.isoformat(),
        event["emotion"] or "unknown",
        event["difficulty"] or "balanced",
        choice_text,
        outcome,
        reward,
        details=scenario.get("title"),
    )


def _end_game_session(*, reset_score: bool = False) -> None:
    st.session_state["game_active"] = False
    st.session_state["game_session"] = None
    st.session_state["game_result"] = None
    st.session_state["game_selected_choice"] = ""
    if "game_choice_radio" in st.session_state:
        del st.session_state["game_choice_radio"]
    if reset_score:
        st.session_state["game_score"] = 0
        st.session_state["game_history"] = []


def _probability_dataframe(probabilities: Dict[str, float]) -> pd.DataFrame:
    if not probabilities:
        return pd.DataFrame(columns=["Emotion", "Confidence (%)"])
    items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(items, columns=["Emotion", "Confidence (%)"])


def _display_probability_breakdown(probabilities: Dict[str, float]) -> None:
    if not probabilities:
        return
    df = _probability_dataframe(probabilities)
    st.bar_chart(df.set_index("Emotion"))
    st.table(df)


def _append_history(story: str, emotion: str, confidence: float, ai_used: bool) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append(
        {
            "story": story,
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": timestamp,
            "ai_used": ai_used,
        }
    )


def _resolve_primary_emotion() -> Optional[str]:
    candidates = [
        st.session_state.get("fused_emotion"),
        st.session_state.get("current_emotion"),
        st.session_state.get("voice_emotion"),
        st.session_state.get("text_emotion"),
    ]
    for value in candidates:
        if value:
            return value
    return None


if __name__ == "__main__":
    main()

