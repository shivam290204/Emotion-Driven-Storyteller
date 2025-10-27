# story_generator.py
# Select stories from templates or generate them via Hugging Face, plus narration.

from __future__ import annotations

import random
from collections.abc import Iterable as IterableABC
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pyttsx3

from story_data import STORY_TEMPLATES
from culture_adapters import culture_story_directives, normalize_culture

try:  # Optional dependency used for AI generation
    from transformers import pipeline  # type: ignore
except ImportError:  # pragma: no cover
    pipeline = None


class StoryGenerator:
    """Generate narrative content based on emotion with optional AI support."""

    def __init__(
        self,
        *,
        default_voice: Optional[str] = None,
        default_rate: int = 150,
        default_volume: float = 0.9,
        hf_model_name: str = "distilgpt2",
    ) -> None:
        self.story_templates = STORY_TEMPLATES
        self._hf_model_name = hf_model_name
        self._tts_rate = default_rate
        self._tts_volume = default_volume
        self._tts_voice = default_voice

        try:
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
        except Exception as error:
            print(f"Error initializing text-to-speech engine: {error}")
            self.tts_engine = None

    def _configure_tts(self) -> None:
        if not self.tts_engine:
            return
        if self._tts_voice:
            self.tts_engine.setProperty("voice", self._tts_voice)
        self.tts_engine.setProperty("rate", self._tts_rate)
        self.tts_engine.setProperty("volume", self._tts_volume)

    @property
    def huggingface_model(self) -> str:
        return self._hf_model_name

    def set_tts_preferences(self, rate: int, volume: float, voice: Optional[str]) -> None:
        self._tts_rate = rate
        self._tts_volume = max(0.0, min(volume, 1.0))
        self._tts_voice = voice
        self._configure_tts()

    def select_story(self, emotion: str) -> str:
        if emotion not in self.story_templates:
            emotion = "neutral"
        stories = self.story_templates.get(emotion, ["No story found for this mood."])
        return random.choice(stories)

    def generate_ai_story(
        self,
        emotion: str,
        seed_story: Optional[str] = None,
        *,
        prompt: Optional[str] = None,
        max_new_tokens: int = 220,
        temperature: float = 0.9,
        top_p: float = 0.92,
    ) -> Optional[str]:
        try:
            generator = _load_text_generator(self._hf_model_name)
        except ImportError as error:
            print(f"Transformers pipeline unavailable: {error}")
            return None
        except Exception as error:
            print(f"Failed to load Hugging Face model: {error}")
            return None

        prompt_text = (prompt or seed_story or self.select_story(emotion)).strip()

        try:
            outputs = generator(
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **_generation_kwargs(generator),
            )
            if outputs:
                text = outputs[0]["generated_text"].strip()
                return text
        except Exception as error:
            print(f"AI story generation failed: {error}")

        return None

    def personalize_template_story(
        self,
        base_story: str,
        emotion: str,
        profile_context: Optional[Mapping[str, Any]] = None,
        *,
        emotion_blend: Optional[Dict[str, float]] = None,
        story_strategy: str = "dominant",
    ) -> str:
        additions = []

        if profile_context:
            name = str(profile_context.get("name", "")).strip()
            places = _normalise_iterable(profile_context.get("favorite_places"))
            friends = _normalise_iterable(profile_context.get("friends"))
            interests = _normalise_iterable(profile_context.get("interests"))
            notes = str(profile_context.get("notes", "")).strip()

            mood = emotion.lower()
            if name:
                additions.append(
                    f"{name} leans into the {mood} feeling, letting it color the next moments."
                )
            if places:
                joined_places = ", ".join(places)
                additions.append(
                    f"They picture cherished places like {joined_places}, drawing comfort."
                )
            if friends:
                joined_friends = ", ".join(friends)
                additions.append(
                    f"Friends such as {joined_friends} come to mind as trusted companions."
                )
            if interests:
                joined_interests = ", ".join(interests)
                additions.append(
                    f"Personal passions—{joined_interests}—offer an outlet to express this mood."
                )
            if notes:
                additions.append(notes)

        if emotion_blend:
            blend_text, interplay = _describe_emotion_blend(emotion_blend, story_strategy)
            if blend_text:
                additions.append(blend_text)
            if interplay:
                additions.append(interplay)

        culture_code = normalize_culture(
            profile_context.get("culture") if isinstance(profile_context, Mapping) else None
        )
        culture_additions = _culture_enrichment(culture_code)
        if culture_additions:
            additions.extend(culture_additions)

        if not additions:
            return base_story

        personalization = " ".join(additions)
        return f"{base_story}\n\n{personalization}"

    def craft_personalized_prompt(
        self,
        emotion: str,
        base_story: str,
        profile_context: Optional[Mapping[str, Any]] = None,
        *,
        emotion_blend: Optional[Dict[str, float]] = None,
        story_strategy: str = "dominant",
    ) -> str:
        lines = [
            "You are an empathetic storyteller.",
            (
                "Write a vivid, emotionally intelligent story that helps the listener explore a "
                f"sense of {emotion}."
            ),
        ]

        culture_code = normalize_culture(
            profile_context.get("culture") if isinstance(profile_context, Mapping) else None
        )
        directives = culture_story_directives(culture_code)

        if profile_context:
            name = str(profile_context.get("name", "")).strip() or "the listener"
            lines.append(f"Center the story on {name}.")

            def append_line(label: str, items: Any) -> None:
                if items is None:
                    return
                if isinstance(items, str):
                    cleaned = items.strip()
                    if cleaned:
                        lines.append(f"{label}: {cleaned}")
                    return
                collection = ", ".join(_normalise_iterable(items))
                if collection:
                    lines.append(f"{label}: {collection}")

            append_line("Favorite places", profile_context.get("favorite_places"))
            append_line("Close friends", profile_context.get("friends"))
            append_line("Interests", profile_context.get("interests"))
            append_line("Notes", profile_context.get("notes"))
        else:
            lines.append("Invent supportive characters and comforting locations.")

        if directives:
            culture_label = directives.get("culture")
            style = directives.get("style")
            if culture_label and culture_code != "global":
                lines.append(f"Cultural backdrop: {culture_label}.")
            if isinstance(style, str) and style:
                lines.append(style)
            settings = _normalise_iterable(directives.get("settings"))
            if settings:
                lines.append(f"Consider settings like {', '.join(settings)}.")
            idioms = _normalise_iterable(directives.get("idioms"))
            if idioms:
                lines.append(f"Weave in idioms or expressions such as {', '.join(idioms)}.")
            names = _normalise_iterable(directives.get("names"))
            if names:
                lines.append(f"Introduce characters with names like {', '.join(names)}.")
            greeting = directives.get("greeting")
            if isinstance(greeting, str) and greeting:
                lines.append(f"Optionally open with a greeting like '{greeting}'.")
            language_hint = directives.get("language")
            if isinstance(language_hint, str) and language_hint and culture_code != "global":
                lines.append(f"Blend in brief {language_hint} phrases respectfully.")

        base_excerpt = base_story.strip()
        if base_excerpt:
            lines.append("Inspiration excerpt: " + base_excerpt)

        if emotion_blend:
            blend_text, interplay = _describe_emotion_blend(emotion_blend, story_strategy)
            if blend_text:
                lines.append("Emotion intensity breakdown: " + blend_text)
            if interplay:
                lines.append(interplay)
        else:
            lines.append("Acknowledge any subtle secondary emotions that might appear.")

        lines.append(
            "Deliver 2-3 paragraphs with sensory detail and a hopeful closing reflection."
        )
        return "\n".join(lines)

    def craft_group_prompt(
        self,
        dominant_emotion: str,
        participants: Sequence[Mapping[str, Any]],
        *,
        emotion_blend: Optional[Dict[str, float]] = None,
        story_strategy: str = "dominant",
        culture_hint: Optional[str] = None,
    ) -> str:
        dominant = dominant_emotion or "a mix of feelings"
        lines: List[str] = [
            "You are an empathetic narrator guiding a collaborative adventure for friends.",
            f"The shared emotional center leans toward {dominant}.",
            "Create a vivid tale where each participant's mood shapes their character's choices, and the group resolves a challenge together.",
        ]

        culture_code = normalize_culture(culture_hint)
        directives = culture_story_directives(culture_code)

        roster = list(participants)
        if roster:
            for index, participant in enumerate(roster, start=1):
                name = str(participant.get("label") or f"Participant {index}")
                mood = str(participant.get("emotion") or "neutral")
                confidence = float(participant.get("confidence", 0.0))
                lines.append(f"{name}: primary emotion {mood} ({confidence:.0f}%).")
                secondary = _top_secondary_emotion(
                    participant.get("probabilities") or {},
                    exclude=mood,
                )
                if secondary:
                    lines.append(
                        f"{name} also shows hints of {secondary[0]} ({secondary[1]:.0f}%)."
                    )
        else:
            lines.append(
                "Invent a diverse trio who each bring a distinct mood into the shared moment."
            )

        if directives:
            culture_label = directives.get("culture")
            style = directives.get("style")
            if culture_label and culture_code != "global":
                lines.append(f"Anchor the world in {culture_label} context.")
            if isinstance(style, str) and style:
                lines.append(style)
            settings = _normalise_iterable(directives.get("settings"))
            if settings:
                lines.append("Suggested shared settings: " + ", ".join(settings))
            idioms = _normalise_iterable(directives.get("idioms"))
            if idioms:
                lines.append(
                    "Encourage dialogue using expressions such as " + ", ".join(idioms)
                )
            language_hint = directives.get("language")
            if isinstance(language_hint, str) and language_hint and culture_code != "global":
                lines.append(f"Include brief {language_hint} phrases authentically.")

        if emotion_blend:
            blend_text, interplay = _describe_emotion_blend(emotion_blend, story_strategy)
            if blend_text:
                lines.append("Group emotion breakdown: " + blend_text)
            if interplay:
                lines.append(interplay)
        else:
            lines.append(
                "Allow layered emotions to surface as subtle cues between the characters."
            )

        lines.extend(
            [
                "Write in 3-4 paragraphs with alternating focus between characters.",
                "Balance dialogue and sensory detail, ending with a shared insight the group carries forward.",
            ]
        )
        return "\n".join(lines)

    def personalize_group_template(
        self,
        base_story: str,
        dominant_emotion: str,
        participants: Sequence[Mapping[str, Any]],
        *,
        emotion_blend: Optional[Dict[str, float]] = None,
        story_strategy: str = "dominant",
        culture_hint: Optional[str] = None,
    ) -> str:
        roster = list(participants)
        if not roster and not emotion_blend:
            return base_story

        additions: List[str] = []

        for index, participant in enumerate(roster, start=1):
            name = str(participant.get("label") or f"Friend {index}")
            mood = str(participant.get("emotion") or "neutral")
            confidence = float(participant.get("confidence", 0.0))
            description = (
                f"{name} channels {mood} energy (around {confidence:.0f}%) and steers part of the adventure."
            )
            secondary = _top_secondary_emotion(
                participant.get("probabilities") or {},
                exclude=mood,
            )
            if secondary:
                description += (
                    f" Beneath the surface, a note of {secondary[0].lower()} ({secondary[1]:.0f}%) colors their choices."
                )
            additions.append(description)

        if emotion_blend:
            blend_text, interplay = _describe_emotion_blend(emotion_blend, story_strategy)
            if blend_text:
                additions.append(f"Together they sense a shared blend of {blend_text}.")
            if interplay:
                additions.append(interplay)

        if dominant_emotion:
            additions.append(
                f"They agree to honor the {dominant_emotion} tone while giving space for everyone to be heard."
            )
        else:
            additions.append(
                "They promise to weave every feeling into a supportive group arc."
            )

        culture_code = normalize_culture(culture_hint)
        culture_additions = _culture_enrichment(culture_code)
        if culture_additions:
            additions.extend(culture_additions)

        personalization = " ".join(additions)
        return f"{base_story}\n\n{personalization}" if personalization else base_story

    def narrate_story(self, text: str) -> bool:
        if not text:
            return False
        if not self.tts_engine:
            print("Text-to-speech engine not available.")
            return False
        try:
            self._configure_tts()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as error:
            print(f"Error during narration: {error}")
            return False


@lru_cache(maxsize=1)
def _load_text_generator(model_name: str):
    if pipeline is None:
        raise ImportError(
            "transformers is not installed. Install it to enable AI story generation."
        )
    return pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
    )


def _generation_kwargs(generator):
    tokenizer = getattr(generator, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        model = getattr(generator, "model", None)
        config = getattr(model, "config", None)
        pad_token_id = getattr(config, "eos_token_id", None)

    return {"pad_token_id": pad_token_id} if pad_token_id is not None else {}


def _describe_emotion_blend(
    emotion_blend: Dict[str, float],
    story_strategy: str,
) -> Tuple[str, Optional[str]]:
    if not emotion_blend:
        return "", None

    sorted_items = sorted(emotion_blend.items(), key=lambda kv: kv[1], reverse=True)
    blend_text = ", ".join(
        f"{label.title()} ({value:.0f}%)" for label, value in sorted_items if value > 0
    )

    interplay: Optional[str]
    if story_strategy == "blend" and len(sorted_items) > 1:
        primary, secondary = sorted_items[0], sorted_items[1]
        interplay = (
            f"Show how {primary[0].title()} mingles with undertones of {secondary[0].title()}, "
            "illustrating layered emotions."
        )
    elif sorted_items:
        primary = sorted_items[0]
        interplay = (
            f"Keep {primary[0].title()} as the guiding tone while gently acknowledging the other "
            "feelings in subtle cues."
        )
    else:
        interplay = None

    return blend_text, interplay


def _normalise_iterable(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if isinstance(value, IterableABC):
        items: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _top_secondary_emotion(
    probabilities: Mapping[str, float],
    *,
    exclude: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    if not probabilities:
        return None
    cleaned_exclude = (exclude or "").lower()
    sorted_items = sorted(
        probabilities.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for label, value in sorted_items:
        if label.lower() == cleaned_exclude:
            continue
        if value <= 0:
            continue
        return label.title(), float(value)
    return None


def _culture_enrichment(culture_code: str) -> List[str]:
    directives = culture_story_directives(culture_code)
    if not isinstance(directives, Mapping):
        return []
    additions: List[str] = []
    culture_label = directives.get("culture")
    if culture_label and culture_code != "global":
        additions.append(f"The narrative leans into {culture_label} heritage and rhythms.")
    style = directives.get("style")
    if isinstance(style, str) and style:
        additions.append(style)
    settings = _normalise_iterable(directives.get("settings"))
    if settings:
        additions.append(f"Scenes unfold across {', '.join(settings)}.")
    idioms = _normalise_iterable(directives.get("idioms"))
    if idioms:
        additions.append(f"Expressions like {', '.join(idioms)} surface naturally.")
    names = _normalise_iterable(directives.get("names"))
    if names:
        additions.append(f"Companions such as {', '.join(names)} join the journey.")
    greeting = directives.get("greeting")
    if isinstance(greeting, str) and greeting:
        additions.append(f"They often open with '{greeting}'.")
    language_hint = directives.get("language")
    if (
        isinstance(language_hint, str)
        and language_hint
        and culture_code != "global"
    ):
        additions.append(f"Phrases echoing {language_hint} add authenticity.")
    return additions

