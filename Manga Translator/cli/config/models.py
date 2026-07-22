"""
cli/config/models.py — Pydantic models for the CLI configuration layer.

These models are STRICTLY for CLI config/settings. They do NOT replace or
modify the core/models.py dataclasses (BoundingBox, TextRegion, etc.).

Storage format: TOML at {project_dir}/.manga-translator/config.toml
"""

from __future__ import annotations

import os
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# Use stdlib tomllib on Python 3.11+, fall back to tomli on 3.10
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef, import-not-found]

import tomli_w  # for writing TOML (stdlib has no writer)

from cli.config.defaults import config_path, DEFAULT_CONFIG_TOML


# ── Pydantic section models ────────────────────────────────────────────────

class GeneralSettings(BaseModel):
    """General project settings."""
    project_name: str = "Manga Project"
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper


class PipelineSettings(BaseModel):
    """AI pipeline execution settings."""
    gpu: bool = True
    inpaint_radius: int = Field(default=3, ge=1, le=20)
    prefer_offline: bool = True
    font_path: str = "assets/fonts/Bangers-Regular.ttf"
    stroke_width: int = Field(default=3, ge=0, le=10)
    input_subdir: str = "input"
    output_subdir: str = "output"
    tone_engine_enabled: bool = True
    max_words_per_line: Optional[int] = None


class TranslationSettings(BaseModel):
    """Translation language settings."""
    source_language: str = "ja"
    target_language: str = "en"


class FinetuningSettings(BaseModel):
    """Finetuning & pipeline integration settings (Phase 9 & 10)."""
    enabled: bool = True
    translation_memory_path: str = "assets/models/translation_memory.json"
    layout_model_path: str = "assets/models/layout_model.pkl"
    fuzzy_match_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    context_sentence_limit: int = Field(default=5, ge=1, le=20)
    use_contour_inpaint: bool = True
    layout_model_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CliConfig(BaseModel):
    """
    Root CLI configuration object.

    Loaded from .manga-translator/config.toml inside the project directory.
    Falls back to safe defaults if the file is absent or malformed.
    """
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    finetuning: FinetuningSettings = Field(default_factory=FinetuningSettings)

    # ── I/O ──────────────────────────────────────────────────────────────

    @classmethod
    def from_toml(cls, path: str) -> "CliConfig":
        """
        Load config from a TOML file.

        Returns default config if the file does not exist.
        On parse error, returns defaults and prints a warning (does not crash).
        """
        if not os.path.isfile(path):
            return cls()
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            return cls.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"Failed to parse config at {path!r}: {exc}. Using defaults.",
                stacklevel=2,
            )
            return cls()

    def save(self, path: str) -> None:
        """
        Write this config to a TOML file.

        Uses atomic write (tempfile + os.replace) for crash safety.
        Creates parent directories if they do not exist.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            tomli_w.dump(self.model_dump(exclude_none=True), f)
        os.replace(tmp_path, path)

    # ── Key-value access with dot notation ────────────────────────────────

    _SECTION_MAP: dict[str, str] = {
        "general": "general",
        "pipeline": "pipeline",
        "translation": "translation",
        "finetuning": "finetuning",
    }

    def get_value(self, key: str) -> Any:
        """
        Get a config value by dot-notation key (e.g., 'pipeline.gpu').

        Raises KeyError for unknown section or field.
        """
        parts = key.split(".", maxsplit=1)
        if len(parts) != 2:
            raise KeyError(
                f"Key must be in 'section.field' format, got: {key!r}. "
                f"Available sections: {list(self._SECTION_MAP)}"
            )
        section_name, field_name = parts
        section = getattr(self, section_name, None)
        if section is None:
            raise KeyError(
                f"Unknown section: {section_name!r}. "
                f"Available: {list(self._SECTION_MAP)}"
            )
        if not hasattr(section, field_name):
            raise KeyError(
                f"Unknown field: {field_name!r} in section {section_name!r}. "
                f"Available: {list(section.model_fields)}"
            )
        return getattr(section, field_name)

    def set_value(self, key: str, raw_value: str) -> "CliConfig":
        """
        Return a NEW CliConfig with a single field updated.

        Type-coerces raw_value to match the field's current type.
        Raises KeyError for unknown keys, ValueError for bad values.

        Args:
            key:       Dot-notation key, e.g., 'pipeline.inpaint_radius'
            raw_value: String representation of the new value.

        Returns:
            A new CliConfig instance with the updated field.
        """
        parts = key.split(".", maxsplit=1)
        if len(parts) != 2:
            raise KeyError(f"Key must be 'section.field', got: {key!r}")
        section_name, field_name = parts

        section = getattr(self, section_name, None)
        if section is None:
            raise KeyError(f"Unknown section: {section_name!r}")
        if not hasattr(section, field_name):
            raise KeyError(f"Unknown field: {field_name!r} in {section_name!r}")

        current_val = getattr(section, field_name)
        coerced: Any
        if isinstance(current_val, bool):
            coerced = raw_value.lower() in ("true", "1", "yes", "on")
        elif isinstance(current_val, int):
            coerced = int(raw_value)
        elif isinstance(current_val, float):
            coerced = float(raw_value)
        else:
            coerced = raw_value

        # Build a new config by modifying the data dict
        data = self.model_dump()
        data[section_name][field_name] = coerced
        return CliConfig.model_validate(data)

    def all_keys(self) -> list[tuple[str, str, Any]]:
        """
        Return all (section, field, value) triples for display.
        """
        rows = []
        for section_name in ("general", "pipeline", "translation", "finetuning"):
            section = getattr(self, section_name)
            for field_name in section.model_fields:
                rows.append((section_name, field_name, getattr(section, field_name)))
        return rows


# ── Convenience loaders ────────────────────────────────────────────────────

def load_config(project_dir: str) -> CliConfig:
    """Load CliConfig from the project's .manga-translator/config.toml."""
    return CliConfig.from_toml(config_path(project_dir))


def save_config(config: CliConfig, project_dir: str) -> None:
    """Persist CliConfig to the project's .manga-translator/config.toml."""
    config.save(config_path(project_dir))
