"""game_engine.py
Emotion-aware mini game scenarios that adapt difficulty and flavor per mood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class GameChoice:
    identifier: str
    text: str
    outcome: str
    reward: int
    tone: str

    def to_payload(self) -> Dict[str, str | int]:
        return {
            "id": self.identifier,
            "text": self.text,
            "outcome": self.outcome,
            "reward": self.reward,
            "tone": self.tone,
        }


@dataclass
class GameScenario:
    emotion: str
    difficulty: str
    title: str
    intro: str
    challenge: str
    mechanic_hint: str
    choices: List[GameChoice]
    soundtrack: Optional[str] = None
    ambience: Optional[str] = None

    def to_payload(self) -> Dict[str, object]:
        return {
            "emotion": self.emotion,
            "difficulty": self.difficulty,
            "title": self.title,
            "intro": self.intro,
            "challenge": self.challenge,
            "mechanic_hint": self.mechanic_hint,
            "choices": [choice.to_payload() for choice in self.choices],
            "soundtrack": self.soundtrack,
            "ambience": self.ambience,
        }


class EmotionAdaptiveGame:
    """Return pre-authored interactive beats tuned to the current mood."""

    def __init__(self) -> None:
        self._scenarios = _SCENARIO_LIBRARY
        self._fallback_emotion = "neutral"

    def prepare_session(
        self,
        emotion: str,
        *,
        difficulty: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> GameScenario:
        mood = emotion.lower()
        bank = self._scenarios.get(mood) or self._scenarios[self._fallback_emotion]
        diff = self._normalise_difficulty(difficulty)
        if diff is None:
            diff = self.suggest_difficulty(mood, confidence or 0.0)
        if diff not in bank:
            diff = next(iter(bank.keys()))

        data = bank[diff]
        choices = [
            GameChoice(
                identifier=item["id"],
                text=item["text"],
                outcome=item["outcome"],
                reward=int(item.get("reward", 0)),
                tone=item.get("tone", "supportive"),
            )
            for item in data["choices"]
        ]
        return GameScenario(
            emotion=mood,
            difficulty=diff,
            title=data["title"],
            intro=data["intro"],
            challenge=data["challenge"],
            mechanic_hint=data["mechanic_hint"],
            choices=choices,
            soundtrack=data.get("soundtrack"),
            ambience=data.get("ambience"),
        )

    def suggest_difficulty(self, emotion: str, confidence: float) -> str:
        if emotion in {"sad", "fear"} or confidence <= 35:
            return "gentle"
        if emotion in {"happy", "angry"} and confidence >= 60:
            return "dynamic"
        if emotion == "surprise" and confidence >= 55:
            return "dynamic"
        return "balanced"

    def available_difficulties(self, emotion: str) -> Iterable[str]:
        mood = emotion.lower()
        return self._scenarios.get(mood, self._scenarios[self._fallback_emotion]).keys()

    def _normalise_difficulty(self, difficulty: Optional[str]) -> Optional[str]:
        if not difficulty:
            return None
        mapping = {
            "auto": None,
            "gentle": "gentle",
            "soothe": "gentle",
            "relaxed": "gentle",
            "balanced": "balanced",
            "steady": "balanced",
            "dynamic": "dynamic",
            "challenge": "dynamic",
        }
        return mapping.get(difficulty.lower(), None)


def _scenario(
    *,
    title: str,
    intro: str,
    challenge: str,
    mechanic_hint: str,
    choices: List[Dict[str, Any]],
    soundtrack: Optional[str],
    ambience: Optional[str],
) -> Dict[str, Any]:
    return {
        "title": title,
        "intro": intro,
        "challenge": challenge,
        "mechanic_hint": mechanic_hint,
        "choices": choices,
        "soundtrack": soundtrack,
        "ambience": ambience,
    }


_SCENARIO_LIBRARY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "happy": {
        "gentle": _scenario(
            title="Lantern Grove Warm-Up",
            intro=(
                "You drift through a grove of floating lanterns where everyone moves at a relaxed pace. "
                "Your upbeat mood attracts curious festival sprites."
            ),
            challenge="Choose how you share your spark with the grove tonight.",
            mechanic_hint="Gentle mode reduces time pressure and favors collaborative outcomes.",
            choices=[
                {
                    "id": "guide_walk",
                    "text": "Lead a mellow lantern walk with soft claps to the beat.",
                    "outcome": "The crowd mirrors you, creating a ripple of steady joy that keeps nerves calm.",
                    "reward": 10,
                    "tone": "supportive",
                },
                {
                    "id": "story_circle",
                    "text": "Gather people for a gratitude circle under the brightest tree.",
                    "outcome": "Shared memories brighten the grove, strengthening your sense of connection.",
                    "reward": 12,
                    "tone": "reflective",
                },
                {
                    "id": "music_shift",
                    "text": "Tune the background music to a chill hop playlist and sway gently.",
                    "outcome": "The festival DJ nods approval, easing everyone into a comfortable groove.",
                    "reward": 8,
                    "tone": "calming",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/10/25/audio_23dccf1f92.mp3?filename=sunny-travel-122114.mp3",
            ambience=None,
        ),
        "balanced": _scenario(
            title="Sky Lantern Rally",
            intro=(
                "The plaza is ready for the evening release of lanterns. Teams rely on your upbeat energy to coordinate the lift-off."
            ),
            challenge="Pick the role that channels your current happiness best.",
            mechanic_hint="Balanced mode blends strategy and rhythm mini challenges.",
            choices=[
                {
                    "id": "conductor",
                    "text": "Conduct the orchestra of lantern bearers with dynamic gestures.",
                    "outcome": "Your cues launch a wave of lanterns that paints constellations across the sky.",
                    "reward": 16,
                    "tone": "energetic",
                },
                {
                    "id": "puzzle_team",
                    "text": "Solve a pattern puzzle to unlock limited-edition lantern designs.",
                    "outcome": "A quick deduction unlocks shimmering variants that astonish the crowd.",
                    "reward": 18,
                    "tone": "strategic",
                },
                {
                    "id": "dance_circle",
                    "text": "Start an improvisational dance circle with festival goers.",
                    "outcome": "Your moves turn the plaza into a shared choreography of joy.",
                    "reward": 15,
                    "tone": "expressive",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/11/01/audio_52cea90b7d.mp3?filename=happy-travel-122365.mp3",
            ambience=None,
        ),
        "dynamic": _scenario(
            title="Aurora Sprint",
            intro=(
                "Game masters open the legendary Aurora Gate. Only the most radiant spirit can sprint across the sky bridges to ignite the finale."
            ),
            challenge="Decide how boldly you embrace the high-energy challenge.",
            mechanic_hint="Dynamic mode increases risk-reward and quick decision windows.",
            choices=[
                {
                    "id": "speed_dash",
                    "text": "Dash across the bridges, hitting rhythm pads for bonus flares.",
                    "outcome": "You chain perfect hits, triggering an aurora wave that leaves onlookers speechless.",
                    "reward": 24,
                    "tone": "high_energy",
                },
                {
                    "id": "combo_team",
                    "text": "Coordinate a combo run with fellow sprinters to multiply the light show.",
                    "outcome": "Team synergy doubles the spectacle and cements your status as festival MVP.",
                    "reward": 26,
                    "tone": "cooperative",
                },
                {
                    "id": "improv_stage",
                    "text": "Freestyle on the central stage, weaving story beats into high-octane dance.",
                    "outcome": "Your expressive storytelling keeps the crowd roaring until sunrise.",
                    "reward": 22,
                    "tone": "improvisational",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2023/01/11/audio_3eb15c93ab.mp3?filename=energetic-epic-adventure-132551.mp3",
            ambience=None,
        ),
    },
    "sad": {
        "gentle": _scenario(
            title="Rainy Library Retreat",
            intro="A quiet library café welcomes you with soft lamplight and friendly nods from the staff.",
            challenge="Choose the restorative activity that feels right in this moment.",
            mechanic_hint="Gentle mode emphasizes soothing choices and lowers challenge stakes.",
            choices=[
                {
                    "id": "tea_break",
                    "text": "Blend a calming tea and journal a few comforting thoughts.",
                    "outcome": "Warm steam and kind words help you process the heaviness without rushing it away.",
                    "reward": 10,
                    "tone": "restorative",
                },
                {
                    "id": "memory_album",
                    "text": "Curate a photo album of gentle memories with a librarian guide.",
                    "outcome": "Each page reminds you of supportive moments that steady your breathing.",
                    "reward": 11,
                    "tone": "nostalgic",
                },
                {
                    "id": "acoustic_corner",
                    "text": "Request a soft acoustic song from the corner musician.",
                    "outcome": "The melody sways through the room, giving everyone permission to exhale.",
                    "reward": 9,
                    "tone": "soothing",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/10/03/audio_a45b437969.mp3?filename=lofi-study-ambient-119419.mp3",
            ambience=None,
        ),
        "balanced": _scenario(
            title="Harbor of Echoes",
            intro="You board a gentle ferry where passengers share stories of resilient comebacks.",
            challenge="Select how you transform the melancholy into motion.",
            mechanic_hint="Balanced mode offers narrative reflection with light strategy elements.",
            choices=[
                {
                    "id": "message_bottle",
                    "text": "Send a message in a bottle containing hopes for the week ahead.",
                    "outcome": "The current carries your words, reminding you that feelings flow and shift.",
                    "reward": 14,
                    "tone": "hopeful",
                },
                {
                    "id": "mentor_chat",
                    "text": "Join a mentor NPC for a coaching dialogue over warm broth.",
                    "outcome": "They offer grounded prompts that gently reframe the chapter you are in.",
                    "reward": 16,
                    "tone": "guided",
                },
                {
                    "id": "art_installation",
                    "text": "Collaborate on a light sculpture that visualizes shared emotions.",
                    "outcome": "The sculpture glows softly, showing that sadness can still illuminate.",
                    "reward": 15,
                    "tone": "creative",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2023/04/03/audio_9d0e6f2469.mp3?filename=moody-lounge-144178.mp3",
            ambience=None,
        ),
        "dynamic": _scenario(
            title="Symphony of Release",
            intro=(
                "An impromptu ensemble invites you to lead the crescendo that transforms sorrow into strength."
            ),
            challenge="Decide how you will channel the depth of feeling into a powerful release.",
            mechanic_hint="Dynamic mode introduces bigger emotional swings with higher rewards.",
            choices=[
                {
                    "id": "spotlight_ballad",
                    "text": "Perform a raw ballad that crescendos into a powerful anthem.",
                    "outcome": "Listeners rise with you, turning quiet tears into shared resolve.",
                    "reward": 20,
                    "tone": "cathartic",
                },
                {
                    "id": "volunteer_dash",
                    "text": "Lead a spontaneous volunteer drive to brighten the harbor docks.",
                    "outcome": "Action-focused steps help everyone see tangible progress.",
                    "reward": 22,
                    "tone": "purposeful",
                },
                {
                    "id": "mystery_puzzle",
                    "text": "Crack the harbor's lighthouse riddle to reveal a hope beacon.",
                    "outcome": "The beacon ignites, signposting safe passage for ships and hearts alike.",
                    "reward": 21,
                    "tone": "adventurous",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2023/03/24/audio_623ba95858.mp3?filename=emotional-inspiring-piano-142793.mp3",
            ambience=None,
        ),
    },
    "angry": {
        "gentle": _scenario(
            title="Forge Cooldown",
            intro="The Ember Forge invites you to temper raw sparks into focused intent.",
            challenge="Pick the ritual that helps you reshape the heat.",
            mechanic_hint="Gentle mode slows pacing and promotes grounding techniques.",
            choices=[
                {
                    "id": "breathsmith",
                    "text": "Hammer molten metal in sync with measured breathing patterns.",
                    "outcome": "Each exhale shapes a calming charm that steadies your pulse.",
                    "reward": 11,
                    "tone": "grounding",
                },
                {
                    "id": "cool_stream",
                    "text": "Trace glowing runes into a cool stream to dissipate excess heat.",
                    "outcome": "Steam rises, carrying frustration away in shimmering wisps.",
                    "reward": 10,
                    "tone": "restorative",
                },
                {
                    "id": "ally_circle",
                    "text": "Invite trusted allies to share quick grounding mantras.",
                    "outcome": "Their steady voices anchor you before the next bout.",
                    "reward": 12,
                    "tone": "supportive",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/03/15/audio_baa0fa0341.mp3?filename=calm-and-peaceful-ambient-10199.mp3",
            ambience=None,
        ),
        "balanced": _scenario(
            title="Ember Arena Match",
            intro="Rivals await your command decisions. The arena senses your focused fire.",
            challenge="Select the tactic that channels intensity into mastery.",
            mechanic_hint="Balanced mode mixes tactical choices with timing-based payoffs.",
            choices=[
                {
                    "id": "feint_combo",
                    "text": "Execute a feint into a precise counter combo.",
                    "outcome": "The crowd gasps as your control eclipses raw aggression.",
                    "reward": 18,
                    "tone": "strategic",
                },
                {
                    "id": "team_command",
                    "text": "Coordinate squad positions to flank the arena boss.",
                    "outcome": "Clear orders turn chaos into a disciplined victory.",
                    "reward": 17,
                    "tone": "cooperative",
                },
                {
                    "id": "focus_sigil",
                    "text": "Engrave a focus sigil mid-battle to sustain your stamina.",
                    "outcome": "The sigil pulses, channeling heat into precise strikes.",
                    "reward": 16,
                    "tone": "calculated",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/10/12/audio_44bc1d00ba.mp3?filename=epic-cinematic-sport-trap-121068.mp3",
            ambience=None,
        ),
        "dynamic": _scenario(
            title="Inferno Boss Rush",
            intro="A towering ember titan challenges you to a final duel of reflexes and resolve.",
            challenge="Pick the high-stakes move that fits your sharpened focus.",
            mechanic_hint="Dynamic mode raises stakes with larger score swings.",
            choices=[
                {
                    "id": "limit_break",
                    "text": "Trigger a limit-break combo that risks stamina for massive payoff.",
                    "outcome": "The titan staggers as sparks rain like comets behind you.",
                    "reward": 27,
                    "tone": "intense",
                },
                {
                    "id": "ally_summon",
                    "text": "Summon the forge spirits for a synchronized strike pattern.",
                    "outcome": "Elemental allies weave around you, amplifying each blow.",
                    "reward": 25,
                    "tone": "spectacular",
                },
                {
                    "id": "parry_maze",
                    "text": "Enter a parry maze, deflecting rapid-fire attacks flawlessly.",
                    "outcome": "Perfect timing turns the boss frenzy into your highlight reel.",
                    "reward": 26,
                    "tone": "technical",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2023/04/24/audio_b5bd45bf35.mp3?filename=epic-action-trailer-146356.mp3",
            ambience=None,
        ),
    },
    "fear": {
        "gentle": _scenario(
            title="Glowhaven Sanctuary",
            intro="You arrive at Glowhaven, where bioluminescent flora guide every step.",
            challenge="Choose the reassurance ritual that helps you move forward.",
            mechanic_hint="Gentle mode prioritizes reassurance and clear guidance.",
            choices=[
                {
                    "id": "path_light",
                    "text": "Light small wayfinding orbs one by one with a calm mantra.",
                    "outcome": "Each orb brightens the path, shrinking the shadows around you.",
                    "reward": 9,
                    "tone": "soothing",
                },
                {
                    "id": "guardian_whisper",
                    "text": "Listen to the guardian owl recite protective stories.",
                    "outcome": "Its measured tone reassures your racing thoughts.",
                    "reward": 10,
                    "tone": "comforting",
                },
                {
                    "id": "breath_puzzle",
                    "text": "Solve a simple breath-synced sigil puzzle.",
                    "outcome": "Each solved symbol releases calming aromatics into the air.",
                    "reward": 11,
                    "tone": "focused",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2021/09/03/audio_5bf29b7d73.mp3?filename=deep-relax-113310.mp3",
            ambience=None,
        ),
        "balanced": _scenario(
            title="Midnight Labyrinth",
            intro="A living maze shifts with your heartbeat. Guardians promise safe passage if you listen closely.",
            challenge="Select how you will navigate the labyrinth tonight.",
            mechanic_hint="Balanced mode mixes light puzzles with courage boosts.",
            choices=[
                {
                    "id": "echo_map",
                    "text": "Use echo-location taps to chart the next corridors.",
                    "outcome": "Patterns emerge, revealing a gentle path to the heart of the maze.",
                    "reward": 15,
                    "tone": "analytical",
                },
                {
                    "id": "companion_call",
                    "text": "Invite a spectral companion to walk beside you.",
                    "outcome": "Their presence steadies your steps and keeps the maze cooperative.",
                    "reward": 14,
                    "tone": "supportive",
                },
                {
                    "id": "courage_chant",
                    "text": "Lead a steady chant that turns the maze walls translucent.",
                    "outcome": "Translucent walls reveal shortcuts bathed in safe light.",
                    "reward": 16,
                    "tone": "empowering",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2023/02/24/audio_70ba651675.mp3?filename=calm-meditation-136260.mp3",
            ambience=None,
        ),
        "dynamic": _scenario(
            title="Phantom Chase",
            intro="Shadow phantoms prowl. The bravest players turn the chase into a triumphant reveal.",
            challenge="Pick the daring move that converts fear into fleet-footed success.",
            mechanic_hint="Dynamic mode rewards bold reveals and confident timing.",
            choices=[
                {
                    "id": "lightwall",
                    "text": "Sprint ahead to raise a brilliant lightwall behind you.",
                    "outcome": "The phantoms dissolve as the lightwall seals the path.",
                    "reward": 23,
                    "tone": "decisive",
                },
                {
                    "id": "trickster_turn",
                    "text": "Feign a stumble, then loop behind to tag the phantom leader.",
                    "outcome": "Your clever reversal turns the hunters into allies.",
                    "reward": 22,
                    "tone": "clever",
                },
                {
                    "id": "guardian_beacon",
                    "text": "Activate a guardian beacon sequence to escort the group out.",
                    "outcome": "Beacons ignite in sequence, ending the haunt with a cheering escort.",
                    "reward": 24,
                    "tone": "heroic",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/08/09/audio_c4ce0d3248.mp3?filename=uplifting-ambient-114184.mp3",
            ambience=None,
        ),
    },
    "surprise": {
        "gentle": _scenario(
            title="Curio Alley Stroll",
            intro="You wander through an alley of whimsical curio carts that respond to your curiosity.",
            challenge="Pick the playful discovery that feels right now.",
            mechanic_hint="Gentle mode spotlights exploration without pressure to optimize.",
            choices=[
                {
                    "id": "fortune_seed",
                    "text": "Plant a fortune seed and watch an unexpected truth bloom.",
                    "outcome": "The seed sprouts a hologram of a delightful future surprise.",
                    "reward": 12,
                    "tone": "whimsical",
                },
                {
                    "id": "pocket_museum",
                    "text": "Open a pocket museum drawer labeled 'Serendipity'.",
                    "outcome": "A mini diorama reenacts a joyful twist from your past week.",
                    "reward": 11,
                    "tone": "nostalgic",
                },
                {
                    "id": "hidden_melody",
                    "text": "Follow a hidden melody and hum along.",
                    "outcome": "The alley lights respond in sync, creating a surprise duet.",
                    "reward": 13,
                    "tone": "playful",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/10/24/audio_663a08dc6f.mp3?filename=quirky-quest-122078.mp3",
            ambience=None,
        ),
        "balanced": _scenario(
            title="Portal Plaza Challenge",
            intro="Five shimmering portals flicker in and out. Each hides a different twist.",
            challenge="Pick how you explore the unknown energy.",
            mechanic_hint="Balanced mode blends surprise reveals with light risk management.",
            choices=[
                {
                    "id": "puzzle_portal",
                    "text": "Decode glyphs to open the mystery puzzle portal.",
                    "outcome": "Inside, you solve a clever riddle that rewards keen intuition.",
                    "reward": 17,
                    "tone": "curious",
                },
                {
                    "id": "co_op_portal",
                    "text": "Invite nearby players to co-op through an adventure portal.",
                    "outcome": "Team synergy earns bonus loot and shared laughter.",
                    "reward": 18,
                    "tone": "collaborative",
                },
                {
                    "id": "random_portal",
                    "text": "Jump into the glitching portal with zero intel.",
                    "outcome": "You land on a cloud trampoline mini-game and collect surprise prizes.",
                    "reward": 16,
                    "tone": "spontaneous",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/03/24/audio_7eadda48a1.mp3?filename=arcade-fantasy-10540.mp3",
            ambience=None,
        ),
        "dynamic": _scenario(
            title="Quantum Heist",
            intro="A reality-hopping crew offers you the wildcard role in a caper across timelines.",
            challenge="Choose the high-stakes gambit that capitalizes on your surprise momentum.",
            mechanic_hint="Dynamic mode rewards quick adaptation and daring choices.",
            choices=[
                {
                    "id": "time_swap",
                    "text": "Swap places with your future self for a perfect steal.",
                    "outcome": "Future-you winks as the heist succeeds with impossible timing.",
                    "reward": 24,
                    "tone": "bold",
                },
                {
                    "id": "glitch_hack",
                    "text": "Hack the quantum vault using improvisational rhythm taps.",
                    "outcome": "Syncopated beats open hidden compartments brimming with artifacts.",
                    "reward": 23,
                    "tone": "inventive",
                },
                {
                    "id": "mirror_maze",
                    "text": "Lead foes through a mirror maze and reappear at the prize.",
                    "outcome": "Your misdirection leaves everyone cheering at the unexpected finale.",
                    "reward": 25,
                    "tone": "cunning",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2023/04/15/audio_fc006b61ff.mp3?filename=future-dystopia-145584.mp3",
            ambience=None,
        ),
    },
    "neutral": {
        "gentle": _scenario(
            title="Zen Atrium",
            intro="A minimalist atrium offers balanced lighting, soft fountains, and gentle focus.",
            challenge="Select the centering routine that appeals right now.",
            mechanic_hint="Gentle mode keeps interactions low stakes and mindful.",
            choices=[
                {
                    "id": "breath_walk",
                    "text": "Take a square-breath walking meditation around the fountain.",
                    "outcome": "Your mind clears, ready for whichever direction the day turns.",
                    "reward": 10,
                    "tone": "mindful",
                },
                {
                    "id": "mini_garden",
                    "text": "Arrange sand and stones in a mini zen garden.",
                    "outcome": "Balanced patterns mirror the evenness you feel inside.",
                    "reward": 11,
                    "tone": "calming",
                },
                {
                    "id": "ambient_mix",
                    "text": "Mix ambient loops to craft the room's soundtrack.",
                    "outcome": "The soundscape steadies everyone's pace without forcing a mood.",
                    "reward": 9,
                    "tone": "creative",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/06/20/audio_410e68fa3a.mp3?filename=ambient-piano-111527.mp3",
            ambience=None,
        ),
        "balanced": _scenario(
            title="Crossroads Observatory",
            intro="From the observatory, paths to every mood kingdom unfold.",
            challenge="Decide which balanced action prepares you for any shift.",
            mechanic_hint="Balanced mode offers moderate payoff with adaptable outcomes.",
            choices=[
                {
                    "id": "map_planning",
                    "text": "Chart routes to each kingdom, noting helpful allies.",
                    "outcome": "Detailed notes keep you prepared regardless of the next emotion swing.",
                    "reward": 14,
                    "tone": "strategic",
                },
                {
                    "id": "supply_swap",
                    "text": "Trade supplies between travelers to balance everyone's packs.",
                    "outcome": "Shared resources ensure the group stays ready for adventure.",
                    "reward": 15,
                    "tone": "cooperative",
                },
                {
                    "id": "signal_setup",
                    "text": "Install signal flares tailored to each mood's color.",
                    "outcome": "Flares prime the observatory to respond instantly to future shifts.",
                    "reward": 13,
                    "tone": "prepared",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2022/01/05/audio_b8dcf3fbb7.mp3?filename=ambient-106005.mp3",
            ambience=None,
        ),
        "dynamic": _scenario(
            title="Momentum Mixer",
            intro="Engineers invite you to remix the room to match the next big feeling swing.",
            challenge="Choose the dynamic experiment you want to run.",
            mechanic_hint="Dynamic mode adds tempo shifts and combo bonuses.",
            choices=[
                {
                    "id": "tempo_lab",
                    "text": "Test rapid tempo changes to prep the room for excitement spikes.",
                    "outcome": "Quick switches keep everyone engaged without losing composure.",
                    "reward": 20,
                    "tone": "innovative",
                },
                {
                    "id": "energy_bridge",
                    "text": "Build an energy bridge linking calm and hype zones.",
                    "outcome": "The bridge lets players glide between moods without whiplash.",
                    "reward": 19,
                    "tone": "adaptive",
                },
                {
                    "id": "scenario_sim",
                    "text": "Run rapid-fire scenario sims to train for any twist.",
                    "outcome": "Simulated surprises tune your reflexes for the next emotion shift.",
                    "reward": 21,
                    "tone": "versatile",
                },
            ],
            soundtrack="https://cdn.pixabay.com/download/audio/2023/04/20/audio_629b029d79.mp3?filename=ambient-technology-146055.mp3",
            ambience=None,
        ),
    },
}
