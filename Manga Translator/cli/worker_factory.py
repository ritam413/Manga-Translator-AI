"""
cli/worker_factory.py — Centralised worker instantiation from CliConfig.

This is the ONLY place that maps CLI config values to worker constructor
kwargs. When a worker adds a new parameter, update this file — not every
command file.

No worker is instantiated at module level. All factories are pure functions.
"""

from __future__ import annotations

import os
from typing import Optional

from cli.config.models import CliConfig


def _resolve_path(base_dir: str, path: str) -> str:
    """Resolve a path relative to base_dir if it is not already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def build_text_detector(project_dir: str, config: CliConfig, model_version: str = "v2"):
    """Build a TextDetectorWorker from config."""
    from workers.text_detector import TextDetectorWorker

    input_dir = os.path.join(project_dir, config.pipeline.input_subdir)
    return TextDetectorWorker(
        input_dir=input_dir,
        gpu=config.pipeline.gpu,
        enable_box_merge=(model_version == "v2"),
    )


def build_ocr_worker(project_dir: str, config: CliConfig, model_version: str = "v2"):
    """Build an OcrWorker from config."""
    from workers.ocr_worker import OcrWorker

    input_dir = os.path.join(project_dir, config.pipeline.input_subdir)
    
    tm_db = None
    if config.finetuning.enabled:
        from finetuning.pipeline_integration import load_translation_memory
        tm_rel = config.finetuning.translation_memory_path
        if model_version == "v1":
            tm_rel = tm_rel.replace("translation_memory.json", "translation_memory_v1.json")
        elif model_version == "v2":
            tm_rel = tm_rel.replace("translation_memory.json", "translation_memory_v2.json")
        
        tm_path = _resolve_path(project_dir, tm_rel)
        tm_db = load_translation_memory(tm_path)

    return OcrWorker(
        input_dir=input_dir,
        tm_db=tm_db,
        enable_tm_correction=config.finetuning.enabled,
    )


def build_translation_worker(
    config: CliConfig,
    prefer_online: bool = False,
    project_dir: str = ".",
    model_version: str = "v2",
):
    """
    Build a TranslationWorker from config.

    Args:
        config:        CLI config (carries prefer_offline default).
        prefer_online: If True, overrides config.pipeline.prefer_offline → False.
        project_dir:   Project root directory (used to resolve TM path).
        model_version: Selected model version (v1 or v2).
    """
    from workers.translator import TranslationWorker

    prefer_offline = config.pipeline.prefer_offline and not prefer_online
    
    tm_db = None
    fuzzy_threshold = 0.8
    context_limit = 5
    if config.finetuning.enabled:
        from finetuning.pipeline_integration import load_translation_memory
        tm_rel = config.finetuning.translation_memory_path
        if model_version == "v1":
            tm_rel = tm_rel.replace("translation_memory.json", "translation_memory_v1.json")
        elif model_version == "v2":
            tm_rel = tm_rel.replace("translation_memory.json", "translation_memory_v2.json")
            
        tm_path = _resolve_path(project_dir, tm_rel)
        tm_db = load_translation_memory(tm_path)
        fuzzy_threshold = config.finetuning.fuzzy_match_threshold
        context_limit = config.finetuning.context_sentence_limit

    return TranslationWorker(
        source_language=config.translation.source_language,
        target_language=config.translation.target_language,
        prefer_offline=prefer_offline,
        tm_db=tm_db,
        fuzzy_threshold=fuzzy_threshold,
        context_limit=context_limit,
    )


def build_inpainter(project_dir: str, config: CliConfig):
    """Build an InpainterWorker from config."""
    from workers.inpainter import InpainterWorker

    input_dir = os.path.join(project_dir, config.pipeline.input_subdir)
    output_dir = os.path.join(project_dir, config.pipeline.output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    return InpainterWorker(
        input_dir=input_dir,
        output_dir=output_dir,
        inpaint_radius=config.pipeline.inpaint_radius,
        use_contour_inpaint=config.finetuning.enabled and config.finetuning.use_contour_inpaint,
        layout_model_min_confidence=config.finetuning.layout_model_min_confidence,
    )


def build_typesetter(project_dir: str, config: CliConfig, model_version: str = "v2"):
    """Build a TypesetterWorker from config."""
    from workers.typesetter import TypesetterWorker

    input_dir = os.path.join(project_dir, config.pipeline.input_subdir)
    output_dir = os.path.join(project_dir, config.pipeline.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    font_path = _resolve_path(project_dir, config.pipeline.font_path)
    chosen_font: Optional[str] = font_path if os.path.isfile(font_path) else None

    layout_model = None
    if config.finetuning.enabled:
        from finetuning.pipeline_integration import load_layout_model
        lm_rel = config.finetuning.layout_model_path
        if model_version == "v1":
            lm_rel = lm_rel.replace("layout_model.pkl", "layout_model_v1.pkl")
        elif model_version == "v2":
            lm_rel = lm_rel.replace("layout_model.pkl", "layout_model_v2.pkl")
            
        model_path = _resolve_path(project_dir, lm_rel)
        layout_model = load_layout_model(model_path)

    return TypesetterWorker(
        input_dir=input_dir,
        output_dir=output_dir,
        font_path=chosen_font,
        stroke_width=config.pipeline.stroke_width,
        layout_model=layout_model,
        layout_model_min_confidence=config.finetuning.layout_model_min_confidence,
        enable_box_expansion=(model_version == "v2"),
        enable_char_wrap=(model_version == "v1"),
        tone_engine_enabled=config.pipeline.tone_engine_enabled,
        max_words_per_line=config.pipeline.max_words_per_line,
    )
