import os
import json
import shutil
import tempfile
import logging
from typing import List, Dict, Optional, Tuple, Set, Callable
from PIL import Image

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest

logger = logging.getLogger("ManifestEngine")

class ManifestEngine:
    """
    Engine to manage and synchronize project manifests with the underlying files on disk.
    Ensures safe reading/writing and VRAM/RAM optimization.
    """
    def __init__(self, project_dir: str, input_subdir: str = "input", manifest_filename: str = "project_manifest.json"):
        """
        Initializes the ManifestEngine with paths.
        
        Args:
            project_dir: Root directory of the project.
            input_subdir: Name of the subdirectory containing raw images.
            manifest_filename: Name of the JSON file containing project metadata.
        """
        self.project_dir = os.path.abspath(project_dir)
        self.input_subdir = input_subdir
        self.manifest_filename = manifest_filename
        self.manifest_path = os.path.join(self.project_dir, self.manifest_filename)
        self.input_dir = os.path.join(self.project_dir, self.input_subdir)
        self.manifest: Optional[ProjectManifest] = None

    def load_or_create(self, project_name: str = "Manga Project", source_lang: str = "ja", target_lang: str = "en") -> ProjectManifest:
        """
        Loads the project manifest from disk if it exists, or initializes a new one.
        Also ensures the project directory and input subdirectory exist.
        
        Args:
            project_name: Fallback project name for a new project.
            source_lang: Fallback source language.
            target_lang: Fallback target language.
            
        Returns:
            The loaded or newly created ProjectManifest object.
        """
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)

        if os.path.exists(self.manifest_path):
            try:
                self.manifest = ProjectManifest.load_from_file(self.manifest_path)
                if not getattr(self.manifest, "manga_id", ""):
                    self.manifest.manga_id = project_name or "default"
                logger.info(f"Loaded existing project manifest from {self.manifest_path}")
            except Exception as e:
                logger.error(f"Error loading manifest from {self.manifest_path}: {e}. Initializing a new one.")
                self.manifest = ProjectManifest(
                    project_name=project_name,
                    source_language=source_lang,
                    target_language=target_lang,
                    manga_id=project_name or "default"
                )
        else:
            logger.info(f"Manifest not found. Initializing a new project manifest at {self.manifest_path}")
            self.manifest = ProjectManifest(
                project_name=project_name,
                source_language=source_lang,
                target_language=target_lang,
                manga_id=project_name or "default"
            )
        return self.manifest

    def save_manifest(self, backup: bool = True) -> None:
        """
        Saves the current manifest to self.manifest_path.
        Uses a temporary file and atomic replace to prevent data corruption.
        If backup is True, also creates/updates project_manifest.json.bak.
        
        Args:
            backup: Whether to create a backup file of the previous manifest.
        """
        if not self.manifest:
            raise ValueError("No manifest loaded. Call load_or_create() first.")

        os.makedirs(self.project_dir, exist_ok=True)

        # Create a temp file in the project directory to ensure atomic renaming on the same volume
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.project_dir, suffix=".tmp")
        try:
            os.close(tmp_fd)
            self.manifest.save_to_file(tmp_path)

            if backup and os.path.exists(self.manifest_path):
                backup_path = self.manifest_path + ".bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                shutil.copy2(self.manifest_path, backup_path)

            # Atomic rename (on Windows, os.rename raises OSError if target exists, os.replace is safer)
            os.replace(tmp_path, self.manifest_path)
            logger.info(f"Manifest saved successfully to {self.manifest_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise e

    def sync_with_disk(self, on_progress: Optional[Callable[[int, int], None]] = None) -> Tuple[List[str], List[str]]:
        """
        Scans self.input_dir for image files and updates the manifest.
        Uses PIL lazy-loading to read image dimensions without high memory utilization.
        
        Returns:
            A tuple of (added_files, missing_files).
            added_files: Names of files newly registered to the manifest.
            missing_files: Names of files in the manifest that are missing from the disk.
        """
        if not self.manifest:
            raise ValueError("No manifest loaded. Call load_or_create() first.")

        supported_exts = {".png", ".jpg", ".jpeg", ".webp"}

        disk_files = []
        if os.path.exists(self.input_dir):
            for entry in os.scandir(self.input_dir):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in supported_exts:
                        disk_files.append(entry.name)

        disk_files_set = set(disk_files)
        manifest_files_set = set(self.manifest.pages.keys())

        added_files: List[str] = []
        missing_files: List[str] = []

        total_steps = len(disk_files) + len(manifest_files_set)
        current_step = 0

        # Find and register new files
        for filename in disk_files:
            if filename not in manifest_files_set:
                filepath = os.path.join(self.input_dir, filename)
                try:
                    # Pillow reads only the metadata header to get size, which is memory-efficient
                    with Image.open(filepath) as img:
                        width, height = img.size

                    page_manifest = PageManifest(
                        image_filename=filename,
                        width=width,
                        height=height,
                        text_regions=[],
                        phase_a_completed=False,
                        phase_b_completed=False
                    )
                    self.manifest.pages[filename] = page_manifest
                    added_files.append(filename)
                    logger.info(f"Discovered new page image: {filename} ({width}x{height})")
                except Exception as e:
                    logger.error(f"Error reading image dimensions for {filename}: {e}")
            current_step += 1
            if on_progress:
                on_progress(current_step, total_steps)

        # Identify missing files
        for filename in manifest_files_set:
            if filename not in disk_files_set:
                missing_files.append(filename)
                logger.warning(f"Manifest page {filename} is missing from input directory.")
                # Flag text regions as needing review due to missing source image
                for region in self.manifest.pages[filename].text_regions:
                    region.needs_review = True
            current_step += 1
            if on_progress:
                on_progress(current_step, total_steps)

        # Assign natural page numbers to all pages
        try:
            from finetuning.pipeline_integration import derive_page_order
            page_order_map = derive_page_order(self.manifest)
            for filename, page_num in page_order_map.items():
                self.manifest.pages[filename].page_number = page_num
        except Exception as exc:
            logger.warning(f"Failed to assign natural page numbers: {exc}")

        return added_files, missing_files

    def get_page(self, image_filename: str) -> Optional[PageManifest]:
        """Retrieves the page manifest for a given image filename."""
        if not self.manifest:
            raise ValueError("No manifest loaded.")
        return self.manifest.pages.get(image_filename)

    def update_page(self, image_filename: str, page_manifest: PageManifest) -> None:
        """Updates the page manifest entry for the given image filename."""
        if not self.manifest:
            raise ValueError("No manifest loaded.")
        self.manifest.pages[image_filename] = page_manifest

    def add_text_region(self, image_filename: str, region: TextRegion) -> None:
        """Adds a new text region to the specified page manifest."""
        if not self.manifest:
            raise ValueError("No manifest loaded.")
        page = self.get_page(image_filename)
        if not page:
            raise KeyError(f"Page {image_filename} not found in manifest.")
        if any(r.id == region.id for r in page.text_regions):
            raise ValueError(f"Text region with ID '{region.id}' already exists in page {image_filename}.")
        page.text_regions.append(region)

    def update_text_region(self, image_filename: str, region_id: str, updated_region: TextRegion) -> None:
        """Updates an existing text region by ID in the specified page manifest."""
        if not self.manifest:
            raise ValueError("No manifest loaded.")
        page = self.get_page(image_filename)
        if not page:
            raise KeyError(f"Page {image_filename} not found in manifest.")
        for idx, r in enumerate(page.text_regions):
            if r.id == region_id:
                page.text_regions[idx] = updated_region
                return
        raise KeyError(f"Text region '{region_id}' not found in page {image_filename}.")

    def delete_text_region(self, image_filename: str, region_id: str) -> None:
        """Removes a text region by ID from the specified page manifest."""
        if not self.manifest:
            raise ValueError("No manifest loaded.")
        page = self.get_page(image_filename)
        if not page:
            raise KeyError(f"Page {image_filename} not found in manifest.")
        for idx, r in enumerate(page.text_regions):
            if r.id == region_id:
                page.text_regions.pop(idx)
                return
        raise KeyError(f"Text region '{region_id}' not found in page {image_filename}.")
