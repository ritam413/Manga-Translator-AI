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

import logging
import os
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

import re

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
    ) -> None:
        self._source_language = source_language
        self._target_language = target_language
        self._prefer_offline = prefer_offline
        self._install_argos_pkg = install_argos_pkg
        self._tm_db = tm_db
        self._fuzzy_threshold = fuzzy_threshold
        self._context_limit = context_limit

        # Runtime state
        self._argos_available: bool = False   # True when ja→en pack is ready
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
          - argostranslate is importable AND the required language pack is
            installed.  If install_argos_pkg=True and the pack is absent,
            downloads and installs it automatically.
          - GoogleTranslator is importable as an online fallback.

        At least one backend must be available; otherwise raises RuntimeError.
        """
        if self._loaded:
            logger.info("[TranslationWorker] Already loaded — skipping.")
            return

        logger.info(
            "[TranslationWorker] Initialising translation backends "
            "(src=%s tgt=%s prefer_offline=%s)…",
            self._source_language,
            self._target_language,
            self._prefer_offline,
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
        if not self._argos_available and not self._google_available:
            raise RuntimeError(
                "TranslationWorker: no translation backend is available. "
                "Install argostranslate==1.11.0 or deep-translator==1.11.4."
            )

        self._loaded = True
        logger.info(
            "[TranslationWorker] Ready — argos=%s google=%s.",
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

        For each region:
          1. Skip if raw_text is empty (nothing to translate).
          2. Attempt translation via the preferred backend, then the other.
          3. Write result to region.translated_text.
          4. If both backends fail: set translated_text = raw_text and
             needs_review = True (never raise from within the region loop).

        Returns the mutated PageManifest (same object).
        Must NOT write to disk.
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

        manga_id = getattr(manifest, "manga_id", "") or getattr(manifest, "project_name", "") or "default"

        for region in page.text_regions:
            if not region.raw_text or not region.raw_text.strip():
                logger.debug(
                    "[TranslationWorker] Skipping empty region %s", region.id
                )
                continue

            # 1. Check Translation Memory
            tm_hit = None
            source = "mt"
            if self._tm_db is not None:
                # Exact match
                hit = self._tm_db.exact_match(manga_id, region.raw_text)
                if hit and is_valid_en_translation(hit):
                    tm_hit = hit
                    source = "tm_exact"
                else:
                    # Fuzzy match
                    fuzzy_res = self._tm_db.fuzzy_match(manga_id, region.raw_text, threshold=self._fuzzy_threshold)
                    if fuzzy_res and is_valid_en_translation(fuzzy_res[0]):
                        tm_hit = fuzzy_res[0]
                        source = "tm_fuzzy"

            if tm_hit is not None:
                translated = tm_hit
                logger.info("[TranslationWorker] TM hit (%s) for %s: %r -> %r", source, region.id, region.raw_text, translated)
            else:
                # 2. Build context prefix and call context translation helper
                page_num = getattr(page, "page_number", 0)
                region_idx = getattr(region, "region_index", 0)
                
                from finetuning.pipeline_integration import build_story_position, build_context_prefix
                story_pos = build_story_position(manga_id, page_num, region_idx)
                context_prefix = build_context_prefix(self._tm_db, manga_id, story_pos, limit=self._context_limit)
                
                translated = self._translate_with_context_fallback(
                    context_prefix, region.raw_text, region.id, page.image_filename
                )

            region.translated_text = translated
            setattr(region, "translation_source", source)

            if translated == region.raw_text or not is_valid_en_translation(translated):
                # Both backends failed or output is not English — flag for review
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

        logger.info(
            "[TranslationWorker] Finished page: %s", page.image_filename
        )
        return page

    def unload(self) -> None:
        """
        Release worker state. Idempotent.
        (No heavy models to release — argostranslate & deep_translator are
        stateless library calls, not persistent in-memory objects.)
        """
        if not self._loaded:
            logger.info("[TranslationWorker] Already unloaded — skipping.")
            return
        logger.info("[TranslationWorker] Unloading…")
        self._argos_available = False
        self._google_available = False
        self._loaded = False
        logger.info("[TranslationWorker] Unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

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

            # Check local file in cwd or parent
            for candidate_filename in ("translate-ja_en-1_1.argosmodel", "translate-ja_en.argosmodel"):
                for search_dir in (".", "..", "assets/models", "../assets/models"):
                    candidate_path = os.path.join(search_dir, candidate_filename)
                    if os.path.isfile(candidate_path):
                        logger.info("[TranslationWorker] Found local Argos model at %s, installing…", candidate_path)
                        argostranslate.package.install_from_path(os.path.abspath(candidate_path))
                        return True

            if self._install_argos_pkg:
                logger.info(
                    "[TranslationWorker] Argos pack not found — downloading…"
                )
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
                else:
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

    def _translate_with_fallback(
        self, text: str, region_id: str, image_filename: str
    ) -> str:
        """
        Try preferred backend first, then fallback, then return raw_text.
        Returns the translated string (or raw_text if both fail).
        """
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
                result = backend_fn(text)
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

        # Check common fallback dictionary
        clean_key = text.strip()
        if clean_key in _COMMON_JA_FALLBACKS:
            fallback_val = _COMMON_JA_FALLBACKS[clean_key]
            logger.info("[TranslationWorker] %s translated via common dictionary: %r -> %r", region_id, text, fallback_val)
            return fallback_val

        # Both backends failed — return raw_text (caller sets needs_review)
        return text

    def _translate_with_context_fallback(
        self, context_prefix: str, text: str, region_id: str, image_filename: str
    ) -> str:
        """Translate text with prepended context, then extract target sentence."""
        if not context_prefix:
            return self._translate_with_fallback(text, region_id, image_filename)

        from finetuning.pipeline_integration import prepend_context
        query = prepend_context(context_prefix, text)

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
                translated_block = backend_fn(query)
                if translated_block and translated_block.strip():
                    import re
                    # Match variations of separator dashes or vertical bars
                    pattern = re.compile(r'(?:\n-+\n|\n\|{3,}\n|---)')
                    parts = pattern.split(translated_block)
                    if len(parts) > 1:
                        target_text = parts[-1].strip()
                        if target_text and is_valid_en_translation(target_text):
                            logger.debug(
                                "[TranslationWorker] %s translated via %s with context separator",
                                region_id,
                                backend_name,
                            )
                            return target_text
                            
                    # Fallback 1: Sentence tokenization
                    try:
                        import nltk
                        sentences = nltk.tokenize.sent_tokenize(translated_block)
                    except Exception:
                        sentences = [s.strip() for s in re.split(r'(?<=[.!?。！？])\s+', translated_block) if s.strip()]
                    
                    if sentences:
                        target_text = sentences[-1].strip()
                        if target_text and is_valid_en_translation(target_text):
                            logger.debug(
                                "[TranslationWorker] %s translated via %s with sentence tokenize fallback",
                                region_id,
                                backend_name,
                            )
                            return target_text
            except Exception as exc:
                logger.warning(
                    "[TranslationWorker] %s with context failed for %s in %s: %s",
                    backend_name,
                    region_id,
                    image_filename,
                    exc,
                )

        # Fallback 2: Direct translation on text alone
        logger.info(
            "[TranslationWorker] Context translation failed or yielded non-English target for %s. Retrying without context.",
            region_id
        )
        return self._translate_with_fallback(text, region_id, image_filename)
