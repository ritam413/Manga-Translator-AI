"""
tests/test_translator_ollama.py — Test suite for Ollama batch translation,
rolling dialogue consistency, clean NMT decoupling, and CLI settings.
"""

import unittest
from unittest.mock import patch, MagicMock
from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from workers.translator import TranslationWorker
from cli.config.models import CliConfig, TranslationSettings
from cli.worker_factory import build_translation_worker


class TestOllamaTranslationPipeline(unittest.TestCase):

    def setUp(self):
        self.worker = TranslationWorker(
            ollama_url="http://localhost:11434",
            ollama_model="richardyoung/qwen2.5-3b-instruct-abliterated:latest",
            backend="ollama",
        )
        self.worker._loaded = True
        self.worker._ollama_available = True

    def test_json_batch_parsing_all_four_tiers(self):
        # Tier 1: Direct JSON
        t1 = '{"r1": "Hello", "r2": "World"}'
        self.assertEqual(TranslationWorker._parse_json_batch_response(t1), {"r1": "Hello", "r2": "World"})

        # Tier 2: Markdown fence
        t2 = '```json\n{"r1": "Look out!", "r2": "Over there"}\n```'
        self.assertEqual(TranslationWorker._parse_json_batch_response(t2), {"r1": "Look out!", "r2": "Over there"})

        # Tier 3: Substring with surrounding conversational chatter
        t3 = 'Sure, here is your translation:\n{"r1": "Wait up", "r2": "Coming!"}\nHope this helps!'
        self.assertEqual(TranslationWorker._parse_json_batch_response(t3), {"r1": "Wait up", "r2": "Coming!"})

        # Tier 4: Quoted key-value line fallback
        t4 = '"r1": "Hey!"\n"r2": "Let\'s go now."'
        self.assertEqual(TranslationWorker._parse_json_batch_response(t4), {"r1": "Hey!", "r2": "Let's go now."})

    @patch.object(TranslationWorker, "_generate_ollama")
    def test_page_batch_prompt_and_translation(self, mock_gen):
        mock_gen.return_value = '{"r1": "Hey, Senpai!", "r2": "Good morning."}'

        # Mock TM with glossary
        mock_tm = MagicMock()
        mock_tm.get_glossary.return_value = {"先輩": "Senpai"}
        self.worker._tm_db = mock_tm
        self.worker._rolling_dialogue = [("おい", "Hey")]

        batch_input = {"r1": "ねえ、先輩！", "r2": "おはよう。"}
        res = self.worker._translate_page_batch_ollama(batch_input, manga_id="series_1")

        self.assertEqual(res, {"r1": "Hey, Senpai!", "r2": "Good morning."})
        prompt_arg = mock_gen.call_args[1]["prompt"]
        self.assertIn("Series Glossary", prompt_arg)
        self.assertIn("- 先輩 -> Senpai", prompt_arg)
        self.assertIn("Recent Dialogue Context", prompt_arg)
        self.assertIn("- JP: おい | EN: Hey", prompt_arg)

    @patch.object(TranslationWorker, "_translate_page_batch_ollama")
    def test_full_page_process_and_rolling_buffer_evolution(self, mock_batch):
        mock_batch.return_value = {"r1": "Let's head out!", "r2": "Alright!"}

        p1_regions = [
            TextRegion(id="r1", box=BoundingBox(0, 0, 10, 10), raw_text="出発しよう！"),
            TextRegion(id="r2", box=BoundingBox(0, 15, 10, 10), raw_text="了解！"),
        ]
        page1 = PageManifest(image_filename="001.jpg", width=1000, height=1500, text_regions=p1_regions)
        manifest = ProjectManifest(project_name="TestManga", pages={"001.jpg": page1})

        self.worker.process_page(page1, manifest)

        self.assertEqual(page1.text_regions[0].translated_text, "Let's head out!")
        self.assertEqual(page1.text_regions[1].translated_text, "Alright!")
        self.assertEqual(len(self.worker._rolling_dialogue), 2)
        self.assertEqual(self.worker._rolling_dialogue[0], ("出発しよう！", "Let's head out!"))

        # Process Page 2
        mock_batch.return_value = {"r3": "Be careful."}
        p2_regions = [TextRegion(id="r3", box=BoundingBox(0, 0, 10, 10), raw_text="気をつけて。")]
        page2 = PageManifest(image_filename="002.jpg", width=1000, height=1500, text_regions=p2_regions)
        manifest.pages["002.jpg"] = page2

        self.worker.process_page(page2, manifest)
        self.assertEqual(page2.text_regions[0].translated_text, "Be careful.")
        self.assertEqual(len(self.worker._rolling_dialogue), 3)

    def test_clean_fallback_no_delimiter_pollution(self):
        worker = TranslationWorker(prefer_offline=False, backend="google")
        worker._loaded = True
        worker._google_available = True

        with patch.object(worker, "_translate_via_google", return_value="Let's go together") as mock_google:
            res = worker._translate_with_clean_fallback("一緒に行こう", "r1", "p1.jpg")
            self.assertEqual(res, "Let's go together")
            mock_google.assert_called_once_with("一緒に行こう")

    def test_cli_config_factory_integration(self):
        config = CliConfig()
        config.translation.backend = "ollama"
        config.translation.ollama_model = "custom/model:v1"
        config.translation.ollama_url = "http://localhost:11434"

        worker = build_translation_worker(config)
        self.assertEqual(worker._backend, "ollama")
        self.assertEqual(worker._ollama_model, "custom/model:v1")
        self.assertEqual(worker._ollama_url, "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
