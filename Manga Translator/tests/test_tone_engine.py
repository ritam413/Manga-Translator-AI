import unittest
from core.models import TextRegion, BoundingBox
from quality.bubble_shape import BubbleShape
from quality.tone_engine import Tone, score_tone, color_for_speaker, PALETTE


class TestToneEngine(unittest.TestCase):
    def test_tone_enum(self):
        self.assertEqual(Tone.SHOUT, "SHOUT")
        self.assertEqual(Tone.THOUGHT, "THOUGHT")
        self.assertEqual(Tone.WHISPER, "WHISPER")
        self.assertEqual(Tone.NARRATOR, "NARRATOR")
        self.assertEqual(Tone.STANDARD, "STANDARD")

    def test_color_for_speaker(self):
        # Empty speaker ID should return None
        self.assertIsNone(color_for_speaker(""))
        self.assertIsNone(color_for_speaker("   "))
        self.assertIsNone(color_for_speaker(None))

        # Valid speaker ID should return a color from PALETTE
        color1 = color_for_speaker("speaker_01")
        self.assertIn(color1, PALETTE)

        # Deterministic check
        color2 = color_for_speaker("speaker_01")
        self.assertEqual(color1, color2)

        # Different speaker should ideally return a different color (or at least valid)
        color3 = color_for_speaker("speaker_02")
        self.assertIn(color3, PALETTE)

    def test_score_tone_bubble_shapes(self):
        box = BoundingBox(10, 10, 100, 100)
        
        # SPIKY bubble should score SHOUT
        region = TextRegion(id="r0", box=box, translated_text="Hello")
        tone_score = score_tone(region, BubbleShape.SPIKY)
        self.assertEqual(tone_score.tone, Tone.SHOUT)
        self.assertTrue(any("SPIKY" in r for r in tone_score.reasons))

        # CLOUD bubble should score THOUGHT
        tone_score = score_tone(region, BubbleShape.CLOUD)
        self.assertEqual(tone_score.tone, Tone.THOUGHT)
        self.assertTrue(any("CLOUD" in r for r in tone_score.reasons))

        # RECTANGLE bubble should score NARRATOR
        tone_score = score_tone(region, BubbleShape.RECTANGLE)
        self.assertEqual(tone_score.tone, Tone.NARRATOR)
        self.assertTrue(any("RECTANGLE" in r for r in tone_score.reasons))

    def test_score_tone_punctuation(self):
        box = BoundingBox(10, 10, 100, 100)
        
        # Multiple exclamation marks -> SHOUT
        region = TextRegion(id="r0", box=box, translated_text="STOP!!")
        tone_score = score_tone(region, BubbleShape.OVAL)
        self.assertEqual(tone_score.tone, Tone.SHOUT)
        self.assertTrue(any("exclamation" in r or "punctuation" in r for r in tone_score.reasons))

        # Parentheses -> THOUGHT
        region = TextRegion(id="r0", box=box, translated_text="(what is that?)")
        tone_score = score_tone(region, BubbleShape.OVAL)
        self.assertEqual(tone_score.tone, Tone.THOUGHT)

        # Ellipses -> WHISPER (since ellipses score WHISPER +1.5, THOUGHT +1.0)
        region = TextRegion(id="r0", box=box, translated_text="uh... wait...")
        tone_score = score_tone(region, BubbleShape.OVAL)
        self.assertEqual(tone_score.tone, Tone.WHISPER)

    def test_score_tone_case(self):
        box = BoundingBox(10, 10, 100, 100)
        # Mostly lowercase text -> WHISPER
        region = TextRegion(id="r0", box=box, translated_text="please keep it quiet")
        tone_score = score_tone(region, BubbleShape.OVAL)
        self.assertEqual(tone_score.tone, Tone.WHISPER)
        self.assertTrue(any("lowercase" in r for r in tone_score.reasons))

    def test_score_tone_geometry(self):
        # Wide aspect ratio -> NARRATOR
        box = BoundingBox(10, 10, 300, 50)  # aspect ratio = 290/40 = 7.25
        region = TextRegion(id="r0", box=box, translated_text="Narrator text here")
        tone_score = score_tone(region, BubbleShape.OVAL)
        self.assertEqual(tone_score.tone, Tone.NARRATOR)
        self.assertTrue(any("aspect ratio" in r for r in tone_score.reasons))

    def test_score_tone_tie_breaks(self):
        box = BoundingBox(10, 10, 100, 100)
        # SPIKY bubble (+3.0 SHOUT) vs ellipses (+1.5 WHISPER, +1.0 THOUGHT)
        # SHOUT should win (score 3.0 vs 1.5)
        region = TextRegion(id="r0", box=box, translated_text="wait...")
        tone_score = score_tone(region, BubbleShape.SPIKY)
        self.assertEqual(tone_score.tone, Tone.SHOUT)
        
        # Tie-break check: equal score of SHOUT and THOUGHT (e.g. if we artificially balance them)
        # SPIKY bubble (+3.0 SHOUT) vs CLOUD bubble (which is not possible since it's one shape).
        # Let's verify standard deterministic resolution order: SHOUT > THOUGHT > WHISPER > NARRATOR > STANDARD
        # If we have a region with aspect_ratio > 2.5 (+1.5 NARRATOR) and ellipses (+1.5 WHISPER, +1.0 THOUGHT)
        # WHISPER (1.5) vs NARRATOR (1.5). Since WHISPER is higher priority in tie break, WHISPER should win.
        box_wide = BoundingBox(10, 10, 300, 50)
        region_tie = TextRegion(id="r0", box=box_wide, translated_text="...")
        tone_score = score_tone(region_tie, BubbleShape.OVAL) # OVAL is +1.0 STANDARD
        # Scores:
        # WHISPER: 1.5 (ellipses)
        # NARRATOR: 1.5 (aspect ratio)
        # THOUGHT: 1.0 (ellipses)
        # STANDARD: 1.0 (oval) + 0.5 (baseline) = 1.5
        # Ties between WHISPER, NARRATOR, STANDARD. WHISPER > NARRATOR > STANDARD.
        self.assertEqual(tone_score.tone, Tone.WHISPER)
