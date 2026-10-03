"""
tests/test_stage8.py — Unit tests for Stage 8: Translation Subsystem

All tests use unittest.mock — NO real translation libraries or network calls.
Covers:
  - Construction & properties
  - load() behaviour (backends, idempotency, edge cases)
  - unload() behaviour
  - process_page() logic (success, fallback, empty text, both-fail, etc.)
  - StageController integration (marks phase_a_completed)
"""

import unittest
from unittest.mock import patch, MagicMock, call

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from workers.translator import TranslationWorker


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_page(*raw_texts):
    """Create a PageManifest with TextRegions whose raw_text values are given."""
    page = PageManifest(image_filename="page01.png", width=800, height=1200)
    page.text_regions = [
        TextRegion(
            id=f"r{i:04d}",
            box=BoundingBox(10 * i, 10, 80 * i + 80, 80),
            raw_text=txt,
        )
        for i, txt in enumerate(raw_texts)
    ]
    return page


def _make_manifest():
    return ProjectManifest(project_name="Test", pages={})


# ── 1. Construction & Properties ─────────────────────────────────────────────

class TestTranslationWorkerConstruction(unittest.TestCase):

    def test_worker_name(self):
        worker = TranslationWorker()
        self.assertEqual(worker.worker_name, "Translation Worker")

    def test_initial_not_loaded(self):
        worker = TranslationWorker()
        self.assertFalse(worker.is_loaded())

    def test_initial_backends_false(self):
        worker = TranslationWorker()
        self.assertFalse(worker._argos_available)
        self.assertFalse(worker._google_available)

    def test_default_language_pair(self):
        worker = TranslationWorker()
        self.assertEqual(worker._source_language, "ja")
        self.assertEqual(worker._target_language, "en")

    def test_custom_language_pair(self):
        worker = TranslationWorker(source_language="ko", target_language="en")
        self.assertEqual(worker._source_language, "ko")
        self.assertEqual(worker._target_language, "en")

    def test_prefer_offline_default(self):
        worker = TranslationWorker()
        self.assertTrue(worker._prefer_offline)

    def test_prefer_offline_false(self):
        worker = TranslationWorker(prefer_offline=False)
        self.assertFalse(worker._prefer_offline)


# ── 2. load() — happy paths ───────────────────────────────────────────────────

class TestTranslationWorkerLoad(unittest.TestCase):

    @patch("workers.translator.argostranslate")
    @patch("workers.translator.GoogleTranslator")
    def test_load_both_backends_available(self, mock_gt, mock_argos):
        """Both argos and google available → _loaded=True."""
        mock_argos.package.get_installed_packages.return_value = [
            MagicMock(from_code="ja", to_code="en")
        ]
        worker = TranslationWorker()
        worker.load()
        self.assertTrue(worker.is_loaded())
        self.assertTrue(worker._argos_available)
        self.assertTrue(worker._google_available)

    @patch("workers.translator.argostranslate")
    @patch("workers.translator.GoogleTranslator", new=None)
    def test_load_argos_only(self, mock_argos):
        """Only argos available → still loads successfully."""
        mock_argos.package.get_installed_packages.return_value = [
            MagicMock(from_code="ja", to_code="en")
        ]
        worker = TranslationWorker()
        worker.load()
        self.assertTrue(worker.is_loaded())
        self.assertTrue(worker._argos_available)
        self.assertFalse(worker._google_available)

    @patch("workers.translator.argostranslate", new=None)
    @patch("workers.translator.GoogleTranslator")
    def test_load_google_only(self, mock_gt):
        """Only google available → still loads successfully."""
        worker = TranslationWorker()
        worker.load()
        self.assertTrue(worker.is_loaded())
        self.assertFalse(worker._argos_available)
        self.assertTrue(worker._google_available)

    @patch("workers.translator.argostranslate", new=None)
    @patch("workers.translator.GoogleTranslator", new=None)
    @patch.object(TranslationWorker, "_probe_ollama", return_value=False)
    def test_load_no_backends_raises(self, mock_probe):
        """No backends at all → RuntimeError."""
        worker = TranslationWorker()
        with self.assertRaises(RuntimeError):
            worker.load()

    @patch("workers.translator.argostranslate")
    @patch("workers.translator.GoogleTranslator")
    def test_load_idempotent(self, mock_gt, mock_argos):
        """load() called twice must not double-initialise."""
        mock_argos.package.get_installed_packages.return_value = [
            MagicMock(from_code="ja", to_code="en")
        ]
        worker = TranslationWorker()
        worker.load()
        worker.load()  # second call
        # get_installed_packages called only once (during first load)
        mock_argos.package.get_installed_packages.assert_called_once()

    @patch("workers.translator.argostranslate")
    @patch("workers.translator.GoogleTranslator")
    def test_load_argos_pack_missing_no_install(self, mock_gt, mock_argos):
        """Argos pack not installed, install_argos_pkg=False → argos unavailable."""
        mock_argos.package.get_installed_packages.return_value = []  # no ja→en pack
        worker = TranslationWorker(install_argos_pkg=False)
        worker.load()
        self.assertFalse(worker._argos_available)
        self.assertTrue(worker._google_available)

    @patch("workers.translator.argostranslate")
    @patch("workers.translator.GoogleTranslator")
    def test_load_argos_pack_missing_with_install(self, mock_gt, mock_argos):
        """Argos pack not installed, install_argos_pkg=True → downloads & installs."""
        mock_argos.package.get_installed_packages.return_value = []  # not installed
        mock_pkg = MagicMock(from_code="ja", to_code="en")
        mock_pkg.download.return_value = "/tmp/fake_pack.argosmodel"
        mock_argos.package.get_available_packages.return_value = [mock_pkg]

        worker = TranslationWorker(install_argos_pkg=True)
        worker.load()

        mock_argos.package.update_package_index.assert_called_once()
        mock_argos.package.install_from_path.assert_called_once_with(
            "/tmp/fake_pack.argosmodel"
        )
        self.assertTrue(worker._argos_available)

    @patch("workers.translator.argostranslate")
    @patch("workers.translator.GoogleTranslator")
    def test_load_argos_exception_falls_back(self, mock_gt, mock_argos):
        """If argostranslate.package.get_installed_packages raises, argos=False."""
        mock_argos.package.get_installed_packages.side_effect = RuntimeError("boom")
        worker = TranslationWorker()
        worker.load()
        self.assertFalse(worker._argos_available)
        self.assertTrue(worker._google_available)


# ── 3. unload() ───────────────────────────────────────────────────────────────

class TestTranslationWorkerUnload(unittest.TestCase):

    @patch("workers.translator.argostranslate")
    @patch("workers.translator.GoogleTranslator")
    def test_unload_resets_state(self, mock_gt, mock_argos):
        mock_argos.package.get_installed_packages.return_value = [
            MagicMock(from_code="ja", to_code="en")
        ]
        worker = TranslationWorker()
        worker.load()
        self.assertTrue(worker.is_loaded())

        worker.unload()
        self.assertFalse(worker.is_loaded())
        self.assertFalse(worker._argos_available)
        self.assertFalse(worker._google_available)

    def test_unload_before_load_is_safe(self):
        """unload() before load() must not raise."""
        worker = TranslationWorker()
        worker.unload()  # should not raise
        self.assertFalse(worker.is_loaded())

    def test_unload_idempotent(self):
        """unload() twice must not raise."""
        worker = TranslationWorker()
        worker._loaded = True
        worker._argos_available = True
        worker._google_available = True
        worker.unload()
        worker.unload()  # second call — must not raise
        self.assertFalse(worker.is_loaded())


# ── 4. process_page() — not loaded ───────────────────────────────────────────

class TestProcessPageNotLoaded(unittest.TestCase):

    def test_raises_if_not_loaded(self):
        worker = TranslationWorker()
        page = _make_page("テスト")
        manifest = _make_manifest()
        with self.assertRaises(RuntimeError):
            worker.process_page(page, manifest)


# ── 5. process_page() — translation logic ────────────────────────────────────

class TestProcessPageTranslation(unittest.TestCase):

    def _loaded_worker(self, prefer_offline=True):
        """Return a worker with load() bypassed (backends set manually)."""
        worker = TranslationWorker(prefer_offline=prefer_offline)
        worker._loaded = True
        worker._argos_available = True
        worker._google_available = True
        return worker

    # ── 5a. Argos succeeds ───────────────────────────────────────────────────

    @patch("workers.translator.argostranslate")
    def test_translates_via_argos(self, mock_argos):
        """Primary backend (argos) succeeds → translated_text populated."""
        mock_argos.translate.translate.return_value = "Test sentence"

        worker = self._loaded_worker()
        page = _make_page("テスト文章")
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        mock_argos.translate.translate.assert_called_once_with(
            "テスト文章", "ja", "en"
        )
        self.assertEqual(result.text_regions[0].translated_text, "Test sentence")
        self.assertFalse(result.text_regions[0].needs_review)

    # ── 5b. Argos fails → Google fallback ────────────────────────────────────

    @patch("workers.translator.GoogleTranslator")
    @patch("workers.translator.argostranslate")
    def test_falls_back_to_google(self, mock_argos, mock_gt_cls):
        """Argos raises → GoogleTranslator used."""
        mock_argos.translate.translate.side_effect = RuntimeError("argos error")

        mock_gt_instance = MagicMock()
        mock_gt_instance.translate.return_value = "Fallback English"
        mock_gt_cls.return_value = mock_gt_instance

        worker = self._loaded_worker()
        page = _make_page("テスト")
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        mock_gt_cls.assert_called_once_with(source="ja", target="en")
        mock_gt_instance.translate.assert_called_once_with("テスト")
        self.assertEqual(result.text_regions[0].translated_text, "Fallback English")
        self.assertFalse(result.text_regions[0].needs_review)

    # ── 5c. Both fail → pass-through + needs_review ──────────────────────────

    @patch("workers.translator.GoogleTranslator")
    @patch("workers.translator.argostranslate")
    def test_both_fail_passthrough(self, mock_argos, mock_gt_cls):
        """Both backends fail → translated_text = raw_text, needs_review=True."""
        mock_argos.translate.translate.side_effect = RuntimeError("argos fail")
        mock_gt_instance = MagicMock()
        mock_gt_instance.translate.side_effect = RuntimeError("google fail")
        mock_gt_cls.return_value = mock_gt_instance

        worker = self._loaded_worker()
        page = _make_page("テスト")
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        region = result.text_regions[0]
        self.assertEqual(region.translated_text, "テスト")
        self.assertTrue(region.needs_review)

    # ── 5d. Empty raw_text skipped ────────────────────────────────────────────

    @patch("workers.translator.argostranslate")
    def test_empty_raw_text_skipped(self, mock_argos):
        """Regions with empty raw_text are skipped entirely."""
        worker = self._loaded_worker()
        page = _make_page("", "  ")  # blank / whitespace-only
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        mock_argos.translate.translate.assert_not_called()
        for region in result.text_regions:
            self.assertEqual(region.translated_text, "")
            self.assertFalse(region.needs_review)

    # ── 5e. Multiple regions translated independently ─────────────────────────

    @patch("workers.translator.argostranslate")
    def test_multiple_regions_all_translated(self, mock_argos):
        """Every region gets its own translation call."""
        mock_argos.translate.translate.side_effect = [
            "Translation 1",
            "Translation 2",
            "Translation 3",
        ]
        worker = self._loaded_worker()
        page = _make_page("テキスト1", "テキスト2", "テキスト3")
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        self.assertEqual(mock_argos.translate.translate.call_count, 3)
        self.assertEqual(result.text_regions[0].translated_text, "Translation 1")
        self.assertEqual(result.text_regions[1].translated_text, "Translation 2")
        self.assertEqual(result.text_regions[2].translated_text, "Translation 3")

    # ── 5f. Returns same PageManifest object ──────────────────────────────────

    @patch("workers.translator.argostranslate")
    def test_returns_same_page_object(self, mock_argos):
        mock_argos.translate.translate.return_value = "Hello"
        worker = self._loaded_worker()
        page = _make_page("こんにちは")
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        self.assertIs(result, page)

    # ── 5g. Per-region exception does not abort the page ─────────────────────

    @patch("workers.translator.GoogleTranslator")
    @patch("workers.translator.argostranslate")
    def test_per_region_exception_does_not_abort(self, mock_argos, mock_gt_cls):
        """If one region's translation totally fails, remaining regions continue."""
        mock_argos.translate.translate.side_effect = [
            RuntimeError("argos fail"),
            "Second region OK",
        ]
        mock_gt_instance = MagicMock()
        mock_gt_instance.translate.side_effect = RuntimeError("google fail")
        mock_gt_cls.return_value = mock_gt_instance

        worker = self._loaded_worker()
        page = _make_page("失敗", "成功")
        manifest = _make_manifest()

        # Must NOT raise
        result = worker.process_page(page, manifest)

        # First region both backends failed
        self.assertEqual(result.text_regions[0].translated_text, "失敗")
        self.assertTrue(result.text_regions[0].needs_review)

        # Second region succeeded via argos
        self.assertEqual(result.text_regions[1].translated_text, "Second region OK")
        self.assertFalse(result.text_regions[1].needs_review)

    # ── 5h. Google-first (prefer_offline=False) ───────────────────────────────

    @patch("workers.translator.GoogleTranslator")
    @patch("workers.translator.argostranslate")
    def test_prefer_online_calls_google_first(self, mock_argos, mock_gt_cls):
        """With prefer_offline=False, GoogleTranslator is called before argos."""
        mock_gt_instance = MagicMock()
        mock_gt_instance.translate.return_value = "Online result"
        mock_gt_cls.return_value = mock_gt_instance

        worker = TranslationWorker(prefer_offline=False)
        worker._loaded = True
        worker._argos_available = True
        worker._google_available = True

        page = _make_page("テスト")
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        # Google called first
        mock_gt_cls.assert_called_once_with(source="ja", target="en")
        # Argos never called
        mock_argos.translate.translate.assert_not_called()
        self.assertEqual(result.text_regions[0].translated_text, "Online result")

    # ── 5i. Argos returns empty string → falls back to google ─────────────────

    @patch("workers.translator.GoogleTranslator")
    @patch("workers.translator.argostranslate")
    def test_argos_empty_result_triggers_fallback(self, mock_argos, mock_gt_cls):
        """Argos returning empty string is treated as failure → try google."""
        mock_argos.translate.translate.return_value = ""
        mock_gt_instance = MagicMock()
        mock_gt_instance.translate.return_value = "Google result"
        mock_gt_cls.return_value = mock_gt_instance

        worker = self._loaded_worker()
        page = _make_page("テスト")
        manifest = _make_manifest()

        result = worker.process_page(page, manifest)

        self.assertEqual(result.text_regions[0].translated_text, "Google result")
        self.assertFalse(result.text_regions[0].needs_review)


# ── 6. StageController integration ───────────────────────────────────────────

class TestStageControllerIntegration(unittest.TestCase):
    """
    Verify TranslationWorker integrates correctly with StageController:
      - marks_phase="a" sets phase_a_completed=True
      - load() called once, unload() called once
      - Already-complete pages are skipped
      - save_manifest() called per successful page
    """

    def _make_engine(self, pages):
        """Build a mock ManifestEngine with the given PageManifest dict."""
        from core.models import ProjectManifest
        manifest = ProjectManifest(
            project_name="Test",
            pages={p.image_filename: p for p in pages},
        )
        engine = MagicMock()
        engine.manifest = manifest
        engine.save_manifest = MagicMock()
        engine.update_page = MagicMock()
        return engine

    @patch("workers.translator.argostranslate")
    def test_marks_phase_a_completed(self, mock_argos):
        """StageController with marks_phase='a' sets phase_a_completed."""
        mock_argos.translate.translate.return_value = "Hello"
        from stages.controller import StageController

        worker = TranslationWorker()
        worker._loaded = False
        worker._argos_available = False
        worker._google_available = False

        # Patch load/unload to avoid real backend checks
        worker.load = MagicMock(side_effect=lambda: setattr(worker, "_loaded", True) or setattr(worker, "_argos_available", True))
        worker.unload = MagicMock(side_effect=lambda: setattr(worker, "_loaded", False))

        page = _make_page("テスト")
        engine = self._make_engine([page])

        with patch.object(worker, "process_page", wraps=lambda p, m: p) as mock_pp:
            controller = StageController(
                engine=engine,
                worker=worker,
                stage_name="Phase-A Translation",
                marks_phase="a",
                run_quality_check=False,
                flush_vram=False,
            )
            report = controller.run()

        self.assertIn("page01.png", report.succeeded)
        # phase_a_completed should be set by StageController
        self.assertTrue(page.phase_a_completed)

    @patch("workers.translator.argostranslate")
    def test_load_and_unload_called_once(self, mock_argos):
        """Worker load() and unload() each called exactly once per run()."""
        from stages.controller import StageController

        worker = MagicMock(spec=TranslationWorker)
        worker.worker_name = "Translation Worker"
        worker.is_loaded.return_value = False

        page = _make_page("テスト")
        engine = self._make_engine([page])

        worker.process_page.return_value = page

        controller = StageController(
            engine=engine,
            worker=worker,
            stage_name="Phase-A Translation",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False,
        )
        controller.run()

        worker.load.assert_called_once()
        worker.unload.assert_called_once()

    @patch("workers.translator.argostranslate")
    def test_completed_pages_skipped(self, mock_argos):
        """Pages with phase_a_completed=True are skipped when force=False."""
        from stages.controller import StageController

        worker = MagicMock(spec=TranslationWorker)
        worker.worker_name = "Translation Worker"

        page_done = _make_page("テスト")
        page_done.phase_a_completed = True

        engine = self._make_engine([page_done])

        controller = StageController(
            engine=engine,
            worker=worker,
            stage_name="Phase-A Translation",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False,
        )
        report = controller.run()

        worker.process_page.assert_not_called()
        self.assertIn("page01.png", report.skipped)

    @patch("workers.translator.argostranslate")
    def test_save_manifest_called_per_page(self, mock_argos):
        """engine.save_manifest() is called once per successfully processed page."""
        from stages.controller import StageController

        worker = MagicMock(spec=TranslationWorker)
        worker.worker_name = "Translation Worker"

        page1 = _make_page("テスト1")
        page1.image_filename = "page01.png"
        page2 = _make_page("テスト2")
        page2.image_filename = "page02.png"

        engine = self._make_engine([page1, page2])
        worker.process_page.side_effect = lambda p, m: p

        controller = StageController(
            engine=engine,
            worker=worker,
            stage_name="Phase-A Translation",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False,
        )
        controller.run()

        self.assertEqual(engine.save_manifest.call_count, 2)



# ── 6. Ollama Client & Health Probe (Ticket 1) ────────────────────────────────

class TestOllamaClient(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_probe_ollama_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"models": [{"name": "richardyoung/qwen2.5-3b-instruct-abliterated:latest"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        worker = TranslationWorker()
        self.assertTrue(worker._probe_ollama())

    @patch("urllib.request.urlopen")
    def test_probe_ollama_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        worker = TranslationWorker()
        self.assertFalse(worker._probe_ollama())

    @patch("urllib.request.urlopen")
    def test_generate_ollama_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"response": "Hello world"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        worker = TranslationWorker()
        res = worker._generate_ollama("Translate: こんにちは")
        self.assertEqual(res, "Hello world")

    @patch("urllib.request.urlopen")
    def test_generate_ollama_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Timeout")
        worker = TranslationWorker()
        res = worker._generate_ollama("Translate: こんにちは")
        self.assertIsNone(res)



# ── 7. Page-Level JSON Batch Prompt & Multi-Tier Parser (Ticket 2) ─────────────

class TestOllamaBatchTranslation(unittest.TestCase):

    def test_parse_json_tier1_direct(self):
        raw = '{"r0000": "Hello", "r0001": "World"}'
        parsed = TranslationWorker._parse_json_batch_response(raw)
        self.assertEqual(parsed, {"r0000": "Hello", "r0001": "World"})

    def test_parse_json_tier2_markdown(self):
        raw = 'Here is your output:\n```json\n{"r0000": "Good morning", "r0001": "Everyone"}\n```\nEnjoy!'
        parsed = TranslationWorker._parse_json_batch_response(raw)
        self.assertEqual(parsed, {"r0000": "Good morning", "r0001": "Everyone"})

    def test_parse_json_tier3_substring(self):
        raw = 'Leading text {"r0000": "See you later"} trailing text'
        parsed = TranslationWorker._parse_json_batch_response(raw)
        self.assertEqual(parsed, {"r0000": "See you later"})

    def test_parse_json_tier4_regex(self):
        raw = 'Malformed json with no outer braces: "r0000": "Wait", "r0001": "Stop"'
        parsed = TranslationWorker._parse_json_batch_response(raw)
        self.assertEqual(parsed.get("r0000"), "Wait")
        self.assertEqual(parsed.get("r0001"), "Stop")

    def test_parse_json_empty_or_none(self):
        self.assertEqual(TranslationWorker._parse_json_batch_response(""), {})
        self.assertEqual(TranslationWorker._parse_json_batch_response(None), {})

    @patch.object(TranslationWorker, "_generate_ollama")
    def test_translate_page_batch_ollama_success(self, mock_gen):
        mock_gen.return_value = '{"r0000": "Hello", "r0001": "World"}'
        worker = TranslationWorker()
        res = worker._translate_page_batch_ollama({"r0000": "こんにちは", "r0001": "世界"})
        self.assertEqual(res, {"r0000": "Hello", "r0001": "World"})

    @patch.object(TranslationWorker, "_translate_page_batch_ollama")
    def test_process_page_ollama_batch_flow(self, mock_batch):
        mock_batch.return_value = {"r0000": "Hello", "r0001": "World"}
        worker = TranslationWorker(backend="ollama")
        worker._loaded = True
        worker._ollama_available = True

        page = _make_page("こんにちは", "世界")
        manifest = _make_manifest()

        updated_page = worker.process_page(page, manifest)
        self.assertEqual(updated_page.text_regions[0].translated_text, "Hello")
        self.assertEqual(getattr(updated_page.text_regions[0], "translation_source", None), "ollama_batch")
        self.assertEqual(updated_page.text_regions[1].translated_text, "World")
        self.assertEqual(getattr(updated_page.text_regions[1], "translation_source", None), "ollama_batch")

    @patch.object(TranslationWorker, "_translate_page_batch_ollama")
    @patch.object(TranslationWorker, "_translate_with_fallback")
    def test_process_page_ollama_batch_partial_dropped_key_fallback(self, mock_fallback, mock_batch):
        # Ollama only returned r0000, r0001 was dropped
        mock_batch.return_value = {"r0000": "Hello"}
        mock_fallback.return_value = "World Fallback"

        worker = TranslationWorker(backend="ollama")
        worker._loaded = True
        worker._ollama_available = True

        page = _make_page("こんにちは", "世界")
        manifest = _make_manifest()

        updated_page = worker.process_page(page, manifest)
        self.assertEqual(updated_page.text_regions[0].translated_text, "Hello")
        self.assertEqual(getattr(updated_page.text_regions[0], "translation_source", None), "ollama_batch")
        self.assertEqual(updated_page.text_regions[1].translated_text, "World Fallback")



# ── 8. Rolling Dialogue Buffer & Glossary Memory (Ticket 3) ───────────────────

class TestRollingDialogueAndGlossary(unittest.TestCase):

    def test_build_ollama_prompt_with_glossary_and_history(self):
        mock_tm = MagicMock()
        mock_tm.get_glossary.return_value = {"先輩": "Senpai", "勇者": "Hero"}

        worker = TranslationWorker(tm_db=mock_tm)
        worker._rolling_dialogue = [("おはよう", "Good morning"), ("行くぞ", "Let's go")]

        batch = {"r1": "待って、先輩！"}
        prompt = worker._build_ollama_prompt(batch, manga_id="test_series")

        self.assertIn("Series Glossary", prompt)
        self.assertIn("- 先輩 -> Senpai", prompt)
        self.assertIn("Recent Dialogue Context", prompt)
        self.assertIn("- JP: おはよう | EN: Good morning", prompt)
        self.assertIn('"r1": "待って、先輩！"', prompt)

    @patch.object(TranslationWorker, "_translate_page_batch_ollama")
    def test_rolling_dialogue_populated_across_pages(self, mock_batch):
        worker = TranslationWorker(backend="ollama")
        worker._loaded = True
        worker._ollama_available = True

        mock_batch.return_value = {"r0000": "Good morning", "r0001": "Senpai"}
        page1 = _make_page("おはよう", "先輩")
        manifest = _make_manifest()
        worker.process_page(page1, manifest)

        self.assertEqual(len(worker._rolling_dialogue), 2)
        self.assertEqual(worker._rolling_dialogue[0], ("おはよう", "Good morning"))
        self.assertEqual(worker._rolling_dialogue[1], ("先輩", "Senpai"))

        mock_batch.return_value = {"r0000": "Let's go"}
        page2 = _make_page("行こう")
        worker.process_page(page2, manifest)

        self.assertEqual(len(worker._rolling_dialogue), 3)
        self.assertEqual(worker._rolling_dialogue[2], ("行こう", "Let's go"))

    def test_unload_clears_rolling_dialogue(self):
        worker = TranslationWorker()
        worker._loaded = True
        worker._rolling_dialogue = [("おはよう", "Good morning")]
        worker.unload()
        self.assertEqual(len(worker._rolling_dialogue), 0)


# ── 9. Clean Fallback & CLI Config Integration (Tickets 4 & 5) ───────────────

class TestCleanFallbackAndCliIntegration(unittest.TestCase):

    def test_clean_fallback_direct_call(self):
        worker = TranslationWorker()
        worker._loaded = True
        worker._google_available = True

        with patch.object(worker, "_translate_via_google", return_value="Good morning"):
            res = worker._translate_with_clean_fallback("おはよう", "r0", "p1.png")
            self.assertEqual(res, "Good morning")

    def test_clean_fallback_dictionary_match(self):
        worker = TranslationWorker()
        worker._loaded = True
        res = worker._translate_with_clean_fallback("はい", "r0", "p1.png")
        self.assertEqual(res, "Yes")

    def test_context_fallback_forwards_to_clean_fallback(self):
        worker = TranslationWorker()
        worker._loaded = True
        with patch.object(worker, "_translate_with_clean_fallback", return_value="Forwarded Clean") as mock_clean:
            res = worker._translate_with_context_fallback("Some Prefix", "テスト", "r0", "p1.png")
            self.assertEqual(res, "Forwarded Clean")
            mock_clean.assert_called_once_with("テスト", "r0", "p1.png")

    def test_cli_config_models_translation_settings(self):
        from cli.config.models import CliConfig, TranslationSettings
        cfg = CliConfig()
        self.assertEqual(cfg.translation.backend, "auto")
        self.assertEqual(cfg.translation.ollama_url, "http://localhost:11434")
        self.assertIn("qwen2.5", cfg.translation.ollama_model)

    def test_worker_factory_builds_translation_worker_with_ollama(self):
        from cli.config.models import CliConfig
        from cli.worker_factory import build_translation_worker

        cfg = CliConfig()
        cfg.translation.backend = "ollama"
        cfg.translation.ollama_url = "http://127.0.0.1:11434"
        cfg.translation.ollama_model = "test-model:latest"

        worker = build_translation_worker(cfg)
        self.assertEqual(worker._backend, "ollama")
        self.assertEqual(worker._ollama_url, "http://127.0.0.1:11434")
        self.assertEqual(worker._ollama_model, "test-model:latest")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()

