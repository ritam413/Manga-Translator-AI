import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import numpy as np

from core.models import TextRegion
from quality.bubble_shape import BubbleShape

logger = logging.getLogger("quality.tone_engine")


class Tone(str, Enum):
    SHOUT = "SHOUT"
    THOUGHT = "THOUGHT"
    WHISPER = "WHISPER"
    NARRATOR = "NARRATOR"
    STANDARD = "STANDARD"


@dataclass
class ToneScore:
    tone: Tone
    scores: Dict[Tone, float]
    reasons: List[str] = field(default_factory=list)


# Font mappings for each tone
FONT_MAP = {
    Tone.SHOUT: ["impact.ttf", "arialbd.ttf"],
    Tone.THOUGHT: ["comicbd.ttf", "calibrib.ttf"],
    Tone.WHISPER: ["calibri.ttf", "segoeui.ttf", "tahoma.ttf"],
    Tone.NARRATOR: ["arial.ttf", "calibri.ttf"],
    Tone.STANDARD: ["Bangers-Regular.ttf", "impact.ttf", "arialbd.ttf"],
}


# A standard color palette for speaker differentiation
PALETTE = [
    "#E6194B",  # Red
    "#3CB44B",  # Green
    "#FFE119",  # Yellow
    "#4363D8",  # Blue
    "#F58231",  # Orange
    "#911EB4",  # Purple
    "#42D4F4",  # Cyan
    "#F032E6",  # Magenta
]


def color_for_speaker(speaker_id: str) -> Optional[str]:
    """
    Deterministically assigns a hex color code to a speaker using their ID.
    If speaker_id is empty, returns None (caller falls back to default text color).
    """
    if not speaker_id or not speaker_id.strip():
        return None
    
    idx = int(hashlib.md5(speaker_id.encode("utf-8")).hexdigest(), 16) % len(PALETTE)
    return PALETTE[idx]


def score_tone(region: TextRegion, bubble_shape: BubbleShape) -> ToneScore:
    """
    Evaluates multiple weighted signals (bubble shape, punctuation, case ratio, geometry)
    and uses a weighted voting system to determine the optimal tone for typeset text.
    """
    scores = {
        Tone.SHOUT: 0.0,
        Tone.THOUGHT: 0.0,
        Tone.WHISPER: 0.0,
        Tone.NARRATOR: 0.0,
        Tone.STANDARD: 0.0,
    }
    reasons = []

    # 1. Bubble Shape Signals
    if bubble_shape == BubbleShape.SPIKY:
        scores[Tone.SHOUT] += 3.0
        reasons.append("Bubble shape is SPIKY (+3.0 SHOUT)")
    elif bubble_shape == BubbleShape.CLOUD:
        scores[Tone.THOUGHT] += 3.0
        reasons.append("Bubble shape is CLOUD (+3.0 THOUGHT)")
    elif bubble_shape == BubbleShape.RECTANGLE:
        scores[Tone.NARRATOR] += 2.0
        reasons.append("Bubble shape is RECTANGLE (+2.0 NARRATOR)")
    elif bubble_shape == BubbleShape.OVAL:
        scores[Tone.STANDARD] += 1.0
        reasons.append("Bubble shape is OVAL (+1.0 STANDARD)")

    # 2. Punctuation signals from translated text
    text = region.translated_text or ""
    
    # Exclamations indicate SHOUT
    if "!!!" in text or "!!" in text or "!?" in text or "?!" in text:
        scores[Tone.SHOUT] += 2.5
        reasons.append("Strong exclamation patterns (e.g. !!, !!!, !?) (+2.5 SHOUT)")
    elif "!" in text:
        scores[Tone.SHOUT] += 1.0
        reasons.append("Contains single exclamation mark (!) (+1.0 SHOUT)")

    # Ellipses indicate WHISPER or THOUGHT
    if "..." in text or ".." in text or "…" in text:
        scores[Tone.WHISPER] += 1.5
        scores[Tone.THOUGHT] += 1.0
        reasons.append("Contains ellipses (...) (+1.5 WHISPER, +1.0 THOUGHT)")

    # Parentheses or brackets often indicate internal thoughts
    if "(?)" in text or "(...)" in text or ("(" in text and ")" in text):
        scores[Tone.THOUGHT] += 1.5
        reasons.append("Contains parentheses/thought indicators (+1.5 THOUGHT)")

    # 3. Case Ratio signals (for English)
    letters = [c for c in text if c.isalpha()]
    if letters:
        # Check if the text is entirely lowercase (indicates whispering/thoughtfulness)
        if text.islower() and len(letters) >= 3:
            scores[Tone.WHISPER] += 1.5
            reasons.append("Text is entirely lowercase (+1.5 WHISPER)")

    # 4. Bounding Box geometry
    box = region.box
    if box:
        w = box.width()
        h = box.height()
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Wide, thin rectangular boxes (often narration boxes at top/bottom)
        if aspect_ratio > 2.5:
            scores[Tone.NARRATOR] += 1.5
            reasons.append(f"Wide bounding box aspect ratio {aspect_ratio:.2f} (+1.5 NARRATOR)")

    # Standard baseline
    scores[Tone.STANDARD] += 0.5

    # Determine highest scoring tone with tie-breaking order: SHOUT > THOUGHT > WHISPER > NARRATOR > STANDARD
    tie_break_order = [Tone.SHOUT, Tone.THOUGHT, Tone.WHISPER, Tone.NARRATOR, Tone.STANDARD]
    best_tone = Tone.STANDARD
    best_score = -1.0
    for t in tie_break_order:
        if scores[t] > best_score:
            best_score = scores[t]
            best_tone = t

    return ToneScore(tone=best_tone, scores=scores, reasons=reasons)
