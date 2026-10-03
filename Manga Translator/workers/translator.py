"""
workers/translator.py — Stage 8: Phase A Translation Subsystem

Translates the Japanese text (raw_text) captured by Stage 7 (MangaOCR) into
English, writing the result to region.translated_text.

Translation strategy (in order of preference):
  1. **Argos Translate** (offline, no API key): fast, no network required.
     Falls back automatically if the ja→en language pack is not installed.
  2. **Deep Translator / GoogleTranslator** (online): used if Argos is
     unavailable or raises an exception for a specific region.
  3. **Raw pass-through**: if both backends fail for a region, translated_text
     is set to raw_text and needs_review is set True.

Subclasses BaseWorker so it plugs into StageController unchanged.
StageController must be constructed with marks_phase="a" — this is the
stage that sets phase_a_completed = True after all regions are translated.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Optional

# ── Offline backend: argostranslate ──────────────────────────────────────────
# Import at module level so patch("workers.translator.argostranslate") works.
try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:  # pragma: no cover
    argostranslate = None  # type: ignore[assignment]

# ── Online backend: deep_translator ──────────────────────────────────────────
# Import at module level so patch("workers.translator.GoogleTranslator") works.
try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover
    GoogleTranslator = None  # type: ignore[assignment]

from core.models import PageManifest, ProjectManifest
from workers.base import BaseWorker

logger = logging.getLogger("workers.translator")

# Language codes for the translation pair
_SRC_LANG = "ja"
_TGT_LANG = "en"

_COMMON_JA_FALLBACKS = {
    "その": "That",
    "この": "This",
    "あの": "That",
    "これ": "This",
    "それ": "That",
    "あれ": "That",
    "はい": "Yes",
    "いいえ": "No",
    "ああ": "Ah",
    "あっ": "Ah!",
    "えっ": "Eh!?",
    "うん": "Yeah",
    "ううん": "No",
    "おっ": "Oh",
    "はっ": "Ha!",
    "わっ": "Wah!",
    "きゃっ": "Kyah!",
    "ふぅ": "Phew",
    "へぇ": "Heh",
    "ほら": "Look",
    "ねえ": "Hey",
    "おい": "Hey",
    "ちょっと": "Wait",
    "なに": "What",
    "何": "What",
    "誰": "Who",
    "どこ": "Where",
    "いつ": "When",
    "どうして": "Why",
    "なぜ": "Why",
    "本当": "Really",
    "マジ": "Seriously",
    "嘘": "No way",
    "助けて": "Help",
    "お願い": "Please",
    "ありがとう": "Thank you",
    "ごめん": "Sorry",
}


def is_valid_en_translation(text: Optional[str]) -> bool:
    """Return True if text contains Latin/ASCII/numeric characters and is not primarily Japanese."""
    if not text or not text.strip():
        return False
    has_ascii_alnum = bool(re.search(r'[a-zA-Z0-9]', text))
    if not has_ascii_alnum:
        if re.search(r'^[!?.\s~-]+$', text.strip()):
            return True
        return False
    cjk_count = len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]', text))
    ascii_count = len(re.findall(r'[a-zA-Z0-9]', text))
    if cjk_count > ascii_count:
        return False
    return True


class TranslationWorker(BaseWorker):
    """
    Stage 8 worker — translates every TextRegion.raw_text (Japanese) to
    TextRegion.translated_text (English).

    Translation backends tried in order:
        1. argostranslate (offline neural MT)
        2. GoogleTranslator via deep_translator (online)
        3. Pass-through fallback (sets needs_review=True)

    Lifecycle (managed by VramManager / StageController):
        worker.load()                        # validates/warms up backends
        updated_page = worker.process_page(page, manifest)
        worker.unload()                      # releases references

    Args:
        source_language:  BCP-47 code for source language. Default "ja".
        target_language:  BCP-47 code for target language. Default "en".
        prefer_offline:   If True (default), try argostranslate first.
                          If False, go straight to online (GoogleTranslator).
        install_argos_pkg: If True, attempt to download+install the Argos
                           language pack during load() when it is missing.
                           Default False (assumes user has pre-installed it).
    """

    def __init__(
        self,
        source_language: str = _SRC_LANG,
        target_language: str = _TGT_LANG,
        prefer_offline: bool = True,
        install_argos_pkg: bool = False,
        tm_db: Optional[object] = None,
        fuzzy_threshold: float = 0.8,
        context_limit: int = 5,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "richardyoung/qwen2.5-3b-instruct-abliterated:latest",
        backend: str = "auto",  # "auto", "ollama", "tm", "google", "argos"
    ) -> None:
        self._source_language = source_language
        self._target_language = target_language
        self._prefer_offline = prefer_offline
        self._install_argos_pkg = install_argos_pkg
        self._tm_db = tm_db
        self._fuzzy_threshold = fuzzy_threshold
        self._context_limit = context_limit
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._backend = backend

        # Rolling dialogue memory buffer (for cross-page 100-page consistency)
        self._rolling_dialogue: list[tuple[str, str]] = []
        self._max_rolling_context: int = 8

        # Runtime state
        self._ollama_available: bool = False  # True when Ollama server & model match
        self._argos_available: bool = False   # True when ja->en pack is ready
        self._google_available: bool = False  # True when GoogleTranslator importable
        self._loaded: bool = False

    # ── BaseWorker interface ──────────────────────────────────────────────────

    @property
    def worker_name(self) -> str:
        return "Translation Worker"

    def load(self) -> None:
        """
        Validate translation backends. Idempotent — safe to call twice.

        Checks:
          - Ollama server connectivity & model discovery (/api/tags).
          - argostranslate is importable AND the required language pack is
            installed. If install_argos_pkg=True and the pack is absent,
            downloads and installs it automatically.
          - GoogleTranslator is importable as an online fallback.

        At least one backend must be available; otherwise raises RuntimeError.
        """
        if self._loaded:
            logger.info("[TranslationWorker] Already loaded — skipping.")
            return

        logger.info(
            "[TranslationWorker] Initialising translation backends "
            "(src=%s tgt=%s backend=%s prefer_offline=%s)…",
            self._source_language,
            self._target_language,
            self._backend,
            self._prefer_offline,
        )

        # ── 0. Check Ollama ───────────────────────────────────────────────────
        if self._backend in ("auto", "ollama"):
            self._ollama_available = self._probe_ollama()
            if self._ollama_available:
                logger.info(
                    "[TranslationWorker] Ollama backend active (model=%s, url=%s).",
                    self._ollama_model,
                    self._ollama_url,
                )
            elif self._backend == "ollama":
                logger.warning(
                    "[TranslationWorker] Ollama requested as backend, but probe failed at %s.",
                    self._ollama_url,
                )

        # ── 1. Check argostranslate ───────────────────────────────────────────
        if argostranslate is not None:
            self._argos_available = self._check_or_install_argos_pack()
        else:
            logger.warning(
                "[TranslationWorker] argostranslate not importable — "
                "offline backend unavailable."
            )
            self._argos_available = False

        # ── 2. Check GoogleTranslator ─────────────────────────────────────────
        if GoogleTranslator is not None:
            self._google_available = True
            logger.info("[TranslationWorker] GoogleTranslator (online) available.")
        else:
            logger.warning(
                "[TranslationWorker] deep_translator not importable — "
                "online backend unavailable."
            )
            self._google_available = False

        # ── 3. Require at least one backend ──────────────────────────────────
        if not self._ollama_available and not self._argos_available and not self._google_available:
            raise RuntimeError(
                "TranslationWorker: no translation backend is available. "
                "Ensure Ollama is running, or install argostranslate/deep-translator."
            )

        self._loaded = True
        logger.info(
            "[TranslationWorker] Ready — ollama=%s argos=%s google=%s.",
            self._ollama_available,
            self._argos_available,
            self._google_available,
        )

    def process_page(
        self,
        page: PageManifest,
        manifest: ProjectManifest,
    ) -> PageManifest:
        """
        Translate every TextRegion on the page.

        1. Filter non-empty regions.
        2. Attempt fast Page-Level JSON Batch Translation via Ollama if available.
        3. For any missing/invalid regions or if Ollama is unavailable,
           fall back to TM matches followed by per-region MT backends.
        4. Mutates and returns the PageManifest.
        """
        if not self._loaded:
            raise RuntimeError(
                "TranslationWorker.load() must be called before process_page()."
            )

        logger.info(
            "[TranslationWorker] Translating %d region(s) in: %s",
            len(page.text_regions),
            page.image_filename,
        )

        valid_regions = [r for r in page.text_regions if r.raw_text and r.raw_text.strip()]
        if not valid_regions:
            return page

        manga_id = getattr(manifest, "manga_id", "") or getattr(manifest, "project_name", "") or "default"

        # 1. Check exact TM hits first
        tm_exact_map: dict[str, str] = {}
        if self._tm_db is not None:
            for region in valid_regions:
                hit = self._tm_db.exact_match(manga_id, region.raw_text)
                if hit and is_valid_en_translation(hit):
                    tm_exact_map[region.id] = hit

        # 2. Try Page-Level JSON Batch Translation via Ollama for remaining regions
        ollama_batch_map: dict[str, str] = {}
        regions_for_ollama = [r for r in valid_regions if r.id not in tm_exact_map]

        if self._ollama_available and self._backend in ("auto", "ollama") and regions_for_ollama:
            batch_input = {r.id: r.raw_text for r in regions_for_ollama}
            ollama_batch_map = self._translate_page_batch_ollama(batch_input, manga_id=manga_id)

        # 3. Assign translations and run fallbacks for unresolved regions
        for region in page.text_regions:
            if not region.raw_text or not region.raw_text.strip():
                continue

            translated: Optional[str] = None
            source: str = "mt"

            if region.id in tm_exact_map:
                translated = tm_exact_map[region.id]
                source = "tm_exact"
                logger.info("[TranslationWorker] TM exact hit for %s: %r -> %r", region.id, region.raw_text, translated)
            elif region.id in ollama_batch_map and is_valid_en_translation(ollama_batch_map[region.id]):
                translated = ollama_batch_map[region.id]
                source = "ollama_batch"
                logger.debug("[TranslationWorker] Ollama batch hit for %s: %r -> %r", region.id, region.raw_text, translated)
            else:
                # Check fuzzy TM match
                if self._tm_db is not None:
                    fuzzy_res = self._tm_db.fuzzy_match(manga_id, region.raw_text, threshold=self._fuzzy_threshold)
                    if fuzzy_res and is_valid_en_translation(fuzzy_res[0]):
                        translated = fuzzy_res[0]
                        source = "tm_fuzzy"
                        logger.info("[TranslationWorker] TM fuzzy hit for %s: %r -> %r", region.id, region.raw_text, translated)

                # Fallback to single region translation (clean sentence, no delimiter injection)
                if translated is None:
                    translated = self._translate_with_clean_fallback(
                        region.raw_text, region.id, page.image_filename
                    )

            region.translated_text = translated
            setattr(region, "translation_source", source)

            if translated == region.raw_text or not is_valid_en_translation(translated):
                region.needs_review = True
                logger.warning(
                    "[TranslationWorker] Translation unresolved or invalid for region %s "
                    "in %s (translated=%r) — setting needs_review=True.",
                    region.id,
                    page.image_filename,
                    translated,
                )
            else:
                logger.debug(
                    "[TranslationWorker] %s -> %r (source: %s)", region.id, region.translated_text, source
                )

        # 4. Update rolling dialogue FIFO buffer for cross-page dialogue consistency
        for region in page.text_regions:
            if (
                region.raw_text
                and region.raw_text.strip()
                and region.translated_text
                and is_valid_en_translation(region.translated_text)
                and region.translated_text.strip() != region.raw_text.strip()
            ):
                self._rolling_dialogue.append((region.raw_text.strip(), region.translated_text.strip()))

        if len(self._rolling_dialogue) > self._max_rolling_context:
            self._rolling_dialogue = self._rolling_dialogue[-self._max_rolling_context:]

        logger.info(
            "[TranslationWorker] Finished page: %s", page.image_filename
        )
        return page

    def unload(self) -> None:
        """
        Release worker state. Idempotent.
        """
        if not self._loaded:
            logger.info("[TranslationWorker] Already unloaded — skipping.")
            return
        logger.info("[TranslationWorker] Unloading…")
        self._rolling_dialogue.clear()
        self._ollama_available = False
        self._argos_available = False
        self._google_available = False
        self._loaded = False
        logger.info("[TranslationWorker] Unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _probe_ollama(self) -> bool:
        """Non-blocking probe checking if Ollama server and target model are available."""
        try:
            req = urllib.request.Request(
                f"{self._ollama_url.rstrip('/')}/api/tags",
                headers={"User-Agent": "MangaTranslator"},
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    matched = any(
                        self._ollama_model in m
                        or m in self._ollama_model
                        or "qwen2.5" in m
                        for m in models
                    )
                    if matched or models:
                        logger.info(
                            "[TranslationWorker] Ollama probe succeeded. Available models: %s",
                            models,
                        )
                        return True
        except Exception as exc:
            logger.debug("[TranslationWorker] Ollama probe failed: %s", exc)
            return False
        return False

    def _generate_ollama(
        self,
        prompt: str,
        system: Optional[str] = None,
        format_json: bool = False,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """
        Send inference request to Ollama `/api/generate`.
        Returns raw text response or None on failure.
        """
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "repeat_penalty": 1.15,
            },
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        try:
            req = urllib.request.Request(
                f"{self._ollama_url.rstrip('/')}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "MangaTranslator",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    return res_data.get("response", "")
        except Exception as exc:
            logger.warning("[TranslationWorker] Ollama generation failed: %s", exc)
            return None
        return None

    def _build_ollama_prompt(
        self, batch_input: dict[str, str], manga_id: str = "default"
    ) -> str:
        """
        Assemble contextual localization prompt with series glossary and rolling dialogue history.
        """
        # 1. Extract active glossary terms
        glossary_items: list[str] = []
        if self._tm_db is not None:
            if hasattr(self._tm_db, "get_glossary"):
                glossary = self._tm_db.get_glossary(manga_id, limit=12)
                for jp, en in glossary.items():
                    glossary_items.append(f"- {jp} -> {en}")
            elif hasattr(self._tm_db, "manga_vocab") and manga_id in self._tm_db.manga_vocab:
                for jp, en in list(self._tm_db.manga_vocab[manga_id].items())[:12]:
                    glossary_items.append(f"- {jp} -> {en}")

        prompt_sections = [
            "You are an expert Japanese-to-English manga localizer and translator.",
            "Translate each Japanese dialogue entry into natural, expressive ENGLISH.",
            "Rules:",
            "- Output MUST be in English only (NO Japanese, NO Chinese).",
            "- Maintain consistent character names, pronouns, honorifics, and speaking styles.",
            "- Preserve manga character nuances, colloquialisms, and punctuation.",
            "- Return ONLY a valid JSON object mapping each region ID key to its translated English string value.",
        ]

        if glossary_items:
            prompt_sections.append("\nSeries Glossary (Character Names & Key Terminology):")
            prompt_sections.extend(glossary_items)

        if self._rolling_dialogue:
            prompt_sections.append("\nRecent Dialogue Context (Chronological Flow):")
            for jp, en in self._rolling_dialogue[-self._max_rolling_context:]:
                prompt_sections.append(f"- JP: {jp} | EN: {en}")

        prompt_sections.append("\nInput Dialogue Blocks to Translate (JSON):")
        prompt_sections.append(json.dumps(batch_input, ensure_ascii=False, indent=2))
        prompt_sections.append("\nOutput JSON (English):")

        return "\n".join(prompt_sections)

    def _translate_page_batch_ollama(
        self,
        batch_input: dict[str, str],
        manga_id: str = "default",
    ) -> dict[str, str]:
        """
        Translate a dictionary of {region_id: raw_ja_text} via Ollama in a single JSON pass.
        Returns a dict of {region_id: translated_en_text}.
        """
        if not batch_input:
            return {}

        prompt = self._build_ollama_prompt(batch_input, manga_id=manga_id)
        system = "You are a professional Japanese to English manga translator. Translate all dialogue into natural English. Always reply with valid JSON only."

        raw_response = self._generate_ollama(prompt=prompt, system=system, format_json=True, timeout=60.0)
        if not raw_response:
            # Fallback retry without format_json in case strict format mode fails
            raw_response = self._generate_ollama(prompt=prompt, system=system, format_json=False, timeout=60.0)

        if not raw_response:
            logger.warning("[TranslationWorker] Ollama batch translation returned empty response.")
            return {}

        parsed = self._parse_json_batch_response(raw_response)
        logger.info(
            "[TranslationWorker] Ollama batch translated %d/%d regions successfully.",
            len(parsed),
            len(batch_input),
        )
        return parsed

    @staticmethod
    def _parse_json_batch_response(text: str) -> dict[str, str]:
        """
        4-tier robust JSON response parser:
          Tier 1: Direct json.loads
          Tier 2: Markdown code fence extraction (```json ... ```)
          Tier 3: First '{' to last '}' boundary extraction
          Tier 4: Line-by-line regex key-value extraction
        """
        if not text or not text.strip():
            return {}

        clean_text = text.strip()

        # Tier 1: Direct JSON parse
        try:
            data = json.loads(clean_text)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v is not None}
        except Exception:
            pass

        # Tier 2: Markdown code fence block
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
        if code_block_match:
            try:
                data = json.loads(code_block_match.group(1).strip())
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items() if v is not None}
            except Exception:
                pass

        # Tier 3: Substring between first '{' and last '}'
        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            substring = clean_text[start_idx : end_idx + 1]
            try:
                data = json.loads(substring)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items() if v is not None}
            except Exception:
                pass

        # Tier 4: Regex key-value extraction (matching delimiter quote pair with backreference)
        result = {}
        kv_pattern = re.compile(r'["\']([a-zA-Z0-9_.-]+)["\']\s*:\s*(["\'])(.*?)\2')
        for match in kv_pattern.finditer(clean_text):
            k, v = match.group(1), match.group(3)
            if k and v is not None:
                result[k] = v.strip()

        return result

    def _check_or_install_argos_pack(self) -> bool:
        """
        Return True if the required argostranslate language pack is installed.
        If install_argos_pkg=True and the pack is absent, download & install it.
        """
        try:
            installed = argostranslate.package.get_installed_packages()
            pack_ready = any(
                p.from_code == self._source_language
                and p.to_code == self._target_language
                for p in installed
            )
            if pack_ready:
                logger.info(
                    "[TranslationWorker] Argos %s->%s language pack installed.",
                    self._source_language,
                    self._target_language,
                )
                return True

            if self._install_argos_pkg:
                logger.info(
                    "[TranslationWorker] Argos pack not found — downloading…"
                )
                try:
                    argostranslate.package.update_package_index()
                    available = argostranslate.package.get_available_packages()
                    target_pkg = next(
                        (
                            p
                            for p in available
                            if p.from_code == self._source_language
                            and p.to_code == self._target_language
                        ),
                        None,
                    )
                    if target_pkg:
                        argostranslate.package.install_from_path(
                            target_pkg.download()
                        )
                        logger.info(
                            "[TranslationWorker] Argos %s->%s pack installed.",
                            self._source_language,
                            self._target_language,
                        )
                        return True
                except Exception as exc:
                    logger.warning("[TranslationWorker] Argos package index download failed: %s", exc)

                # Fallback: Check local candidate file in cwd or parent
                for candidate_filename in ("translate-ja_en-1_1.argosmodel", "translate-ja_en.argosmodel"):
                    for search_dir in (".", "..", "assets/models", "../assets/models"):
                        candidate_path = os.path.join(search_dir, candidate_filename)
                        if os.path.isfile(candidate_path):
                            logger.info("[TranslationWorker] Found local Argos model at %s, installing…", candidate_path)
                            argostranslate.package.install_from_path(os.path.abspath(candidate_path))
                            return True

                logger.warning(
                    "[TranslationWorker] Argos %s->%s pack not in index.",
                    self._source_language,
                    self._target_language,
                )
                return False
            else:
                logger.warning(
                    "[TranslationWorker] Argos %s->%s pack not installed. "
                    "Run with install_argos_pkg=True or install manually.",
                    self._source_language,
                    self._target_language,
                )
                return False

        except Exception as exc:
            logger.warning(
                "[TranslationWorker] Argos backend init failed: %s", exc
            )
            return False

    def _translate_via_argos(self, text: str) -> str:
        """Translate text using argostranslate. Raises on failure."""
        return argostranslate.translate.translate(
            text, self._source_language, self._target_language
        )

    def _translate_via_google(self, text: str) -> str:
        """Translate text using deep_translator GoogleTranslator. Raises on failure."""
        translator = GoogleTranslator(
            source=self._source_language, target=self._target_language
        )
        return translator.translate(text)

    def _translate_with_clean_fallback(
        self, text: str, region_id: str, image_filename: str
    ) -> str:
        """
        Translate isolated clean sentence with neural MT backends (NO delimiter injections).
        Prevents repetition loops caused by multi-line context delimiters in Seq2Seq models.
        """
        return self._translate_with_fallback(text, region_id, image_filename)

    def _translate_with_fallback(
        self, text: str, region_id: str, image_filename: str
    ) -> str:
        """
        Try preferred backend first, then fallback, then return raw_text.
        Returns the translated string (or raw_text if both fail).
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return text

        # Check common fallback dictionary
        if clean_text in _COMMON_JA_FALLBACKS:
            fallback_val = _COMMON_JA_FALLBACKS[clean_text]
            logger.info("[TranslationWorker] %s translated via common dictionary: %r -> %r", region_id, text, fallback_val)
            return fallback_val

        # If Ollama is available, try single prompt generation before MT backends
        if self._ollama_available and self._backend in ("auto", "ollama"):
            try:
                single_prompt = (
                    "You are an expert manga translator. Translate this Japanese dialogue to natural English:\n"
                    f"{clean_text}\n"
                    "Output ONLY the English translation without explanation or quotes:"
                )
                ollama_res = self._generate_ollama(prompt=single_prompt, timeout=20.0)
                if ollama_res and ollama_res.strip() and is_valid_en_translation(ollama_res.strip()):
                    cleaned_res = ollama_res.strip().strip('"').strip("'")
                    logger.debug(
                        "[TranslationWorker] %s single fallback translated via Ollama: %r",
                        region_id,
                        cleaned_res,
                    )
                    return cleaned_res
            except Exception as exc:
                logger.debug("[TranslationWorker] Single Ollama fallback failed: %s", exc)

        backends = []
        if self._prefer_offline:
            if self._argos_available:
                backends.append(("argostranslate", self._translate_via_argos))
            if self._google_available:
                backends.append(("GoogleTranslator", self._translate_via_google))
        else:
            if self._google_available:
                backends.append(("GoogleTranslator", self._translate_via_google))
            if self._argos_available:
                backends.append(("argostranslate", self._translate_via_argos))

        for backend_name, backend_fn in backends:
            try:
                result = backend_fn(clean_text)
                if result and result.strip() and is_valid_en_translation(result):
                    logger.debug(
                        "[TranslationWorker] %s translated via %s",
                        region_id,
                        backend_name,
                    )
                    return result
                logger.warning(
                    "[TranslationWorker] %s: %s returned invalid or non-English result: %r",
                    region_id,
                    backend_name,
                    result,
                )
            except Exception as exc:
                logger.warning(
                    "[TranslationWorker] %s: %s failed for region %s in %s: %s",
                    backend_name,
                    backend_name,
                    region_id,
                    image_filename,
                    exc,
                )

        # Both backends failed — return raw_text (caller sets needs_review)
        return text

    def _translate_with_context_fallback(
        self, context_prefix: str, text: str, region_id: str, image_filename: str
    ) -> str:
        """Deprecated: forwards directly to clean fallback to prevent Seq2Seq repetition loops."""
        return self._translate_with_clean_fallback(text, region_id, image_filename)
