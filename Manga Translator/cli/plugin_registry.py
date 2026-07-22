"""
cli/plugin_registry.py — PluginRegistry extends StageRegistry.

This is Option B: subclassing without modifying stages/registry.py.

Plugin contract:
    A plugin is a Python file placed in .manga-translator/plugins/.
    It must expose a top-level function:

        def register(registry: "PluginRegistry") -> None: ...

    Inside register(), call registry.register_stage(...) to add stages.

    Example plugin (.manga-translator/plugins/my_ocr.py):

        from workers.base import BaseWorker

        class MyOcrWorker(BaseWorker):
            ...

        def register(registry):
            registry.register_stage(
                stage_id=11,
                name="my-ocr",
                description="Custom OCR backend",
                worker_class=MyOcrWorker,
                phase="a",
            )
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import List, Optional, Tuple, Type

from stages.registry import StageRegistry
from workers.base import BaseWorker

logger = logging.getLogger("cli.plugin_registry")


class PluginRegistry(StageRegistry):
    """
    Drop-in replacement for StageRegistry that:
      1. Starts with the same 5 built-in stages (via super().__init__()).
      2. Scans the plugin directory and loads any .py files found.
      3. Calls plugin.register(self) for each loaded plugin.
      4. Exposes register_stage() for plugins to call.
    """

    def __init__(self, plugin_dir: Optional[str] = None) -> None:
        super().__init__()  # loads built-in stages, no existing code touched
        if plugin_dir and os.path.isdir(plugin_dir):
            self._load_plugins(plugin_dir)

    # ── Plugin API ────────────────────────────────────────────────────────

    def register_stage(
        self,
        stage_id: int,
        name: str,
        description: str,
        worker_class: Type[BaseWorker],
        phase: str,
    ) -> None:
        """
        Add a new stage to the registry. Called by plugins inside register().

        Args:
            stage_id:    Unique integer ID (must not collide with 6-10).
            name:        CLI name, e.g., 'my-ocr' (must be unique).
            description: Short human-readable description.
            worker_class: BaseWorker subclass (not yet instantiated).
            phase:       'a' or 'b'.

        Raises:
            ValueError: If stage_id or name already exists.
        """
        existing_ids = {s[0] for s in self._stages}
        existing_names = {s[1] for s in self._stages}
        if stage_id in existing_ids:
            raise ValueError(
                f"Plugin stage_id {stage_id} conflicts with existing stage."
            )
        if name in existing_names:
            raise ValueError(
                f"Plugin stage name {name!r} conflicts with existing stage."
            )
        self._stages.append((stage_id, name, description, worker_class, phase))
        logger.info("[PluginRegistry] Registered plugin stage: %s (id=%d)", name, stage_id)

    # ── Plugin loader ─────────────────────────────────────────────────────

    def _load_plugins(self, plugin_dir: str) -> None:
        """Scan plugin_dir for .py files and load each one."""
        py_files = sorted(
            f for f in os.listdir(plugin_dir) if f.endswith(".py") and not f.startswith("_")
        )
        if not py_files:
            logger.debug("[PluginRegistry] No plugins found in %s", plugin_dir)
            return

        for filename in py_files:
            filepath = os.path.join(plugin_dir, filename)
            self._load_single_plugin(filepath)

    def _load_single_plugin(self, filepath: str) -> None:
        """Load a single plugin .py file and call its register() function."""
        module_name = f"manga_plugin_{os.path.splitext(os.path.basename(filepath))[0]}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                logger.warning("[PluginRegistry] Cannot load plugin: %s", filepath)
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            if not hasattr(module, "register"):
                logger.warning(
                    "[PluginRegistry] Plugin %s has no register() function — skipped.",
                    filepath,
                )
                return

            module.register(self)
            logger.info("[PluginRegistry] Loaded plugin: %s", filepath)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[PluginRegistry] Failed to load plugin %s: %s",
                filepath,
                exc,
                exc_info=True,
            )
