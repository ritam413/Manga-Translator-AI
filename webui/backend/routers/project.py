"""
Project router — exposes project manifest and directory information.
"""

import os
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

# ── Response models ────────────────────────────────────────────────────────────

class PageStatus(BaseModel):
    filename: str
    phase_a_completed: bool
    phase_b_completed: bool
    region_count: int
    needs_review_count: int

class DirectoryInfo(BaseModel):
    path: str
    exists: bool
    image_count: int
    total_size_mb: float
    images: list[str]

class ProjectInfo(BaseModel):
    project_name: str
    project_dir: str
    manifest_exists: bool
    total_pages: int
    phase_a_done: int
    phase_b_done: int
    needs_review: int
    source_language: str
    target_language: str
    input_dir: DirectoryInfo
    output_dir: DirectoryInfo
    pages: list[PageStatus]

# ── Import additions for integration ──────────────────────────────────────────
import shutil
import json
import httpx
import logging
from datetime import datetime
from fastapi import Request

logger = logging.getLogger("routers.project")
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8001")
ENGINE_DIR = r"e:\u torrentz\MangaTranslator FUll STACK\Manga Translator"

# ── Request / Response models for editing ──────────────────────────────────────

class BackupInfo(BaseModel):
    filename: str
    timestamp: str
    display_name: str

class BBoxIn(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class TextRegionIn(BaseModel):
    id: str
    box: BBoxIn
    raw_text: str = ""
    translated_text: str = ""
    ocr_confidence: float = 0.0
    is_vertical: bool = True
    needs_review: bool = False
    region_index: int = 0
    speaker_id: str = ""

class PageDetail(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    text_regions: list[TextRegionIn] = []
    phase_a_completed: bool = False
    phase_b_completed: bool = False

class SaveRegionsRequest(BaseModel):
    project_dir: str
    filename: str
    regions: list[TextRegionIn]

class RestoreBackupRequest(BaseModel):
    project_dir: str
    backup_filename: str

# ── Helpers ────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def _scan_dir(path: str) -> DirectoryInfo:
    p = Path(path)
    if not p.exists():
        return DirectoryInfo(path=path, exists=False, image_count=0, total_size_mb=0.0, images=[])
    images = [f.name for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
    total_bytes = sum((p / img).stat().st_size for img in images)
    return DirectoryInfo(
        path=str(p),
        exists=True,
        image_count=len(images),
        total_size_mb=round(total_bytes / (1024 * 1024), 2),
        images=images,
    )

def _load_manifest(project_dir: str) -> Optional[dict]:
    manifest_path = Path(project_dir) / "project_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/info", response_model=ProjectInfo)
async def get_project_info(
    project_dir: str = Query(..., description="Absolute path to the manga project directory"),
    input_subdir: str = Query("input", description="Input subdirectory name"),
):
    """Return full project info: manifest stats + input/output directory scan."""
    project_dir = os.path.normpath(project_dir)
    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail=f"Project directory not found: {project_dir}")

    input_path = os.path.join(project_dir, input_subdir)
    output_path = os.path.join(project_dir, "output")

    input_info = _scan_dir(input_path)
    output_info = _scan_dir(output_path)

    manifest = _load_manifest(project_dir)

    if manifest is None:
        return ProjectInfo(
            project_name=os.path.basename(project_dir),
            project_dir=project_dir,
            manifest_exists=False,
            total_pages=0,
            phase_a_done=0,
            phase_b_done=0,
            needs_review=0,
            source_language="ja",
            target_language="en",
            input_dir=input_info,
            output_dir=output_info,
            pages=[],
        )

    pages_data = manifest.get("pages", {})
    page_statuses = []
    phase_a_done = 0
    phase_b_done = 0
    total_needs_review = 0

    for filename, page in pages_data.items():
        regions = page.get("text_regions", [])
        nr_count = sum(1 for r in regions if r.get("needs_review", False))
        total_needs_review += nr_count
        a_done = page.get("phase_a_completed", False)
        b_done = page.get("phase_b_completed", False)
        if a_done:
            phase_a_done += 1
        if b_done:
            phase_b_done += 1
        page_statuses.append(PageStatus(
            filename=filename,
            phase_a_completed=a_done,
            phase_b_completed=b_done,
            region_count=len(regions),
            needs_review_count=nr_count,
        ))

    return ProjectInfo(
        project_name=manifest.get("project_name", os.path.basename(project_dir)),
        project_dir=project_dir,
        manifest_exists=True,
        total_pages=len(pages_data),
        phase_a_done=phase_a_done,
        phase_b_done=phase_b_done,
        needs_review=total_needs_review,
        source_language=manifest.get("source_language", "ja"),
        target_language=manifest.get("target_language", "en"),
        input_dir=input_info,
        output_dir=output_info,
        pages=page_statuses,
    )


@router.get("/page-detail", response_model=PageDetail)
async def get_page_detail(
    project_dir: str = Query(..., description="Absolute path to the manga project directory"),
    filename: str = Query(..., description="Manga page filename"),
):
    """Return the detailed manifest entry for a single page."""
    project_dir = os.path.normpath(project_dir)
    manifest = _load_manifest(project_dir)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Project manifest not found")
        
    pages_data = manifest.get("pages", {})
    if filename not in pages_data:
        raise HTTPException(status_code=404, detail=f"Page not found in manifest: {filename}")
        
    page = pages_data[filename]
    return PageDetail(
        width=page.get("width"),
        height=page.get("height"),
        text_regions=page.get("text_regions", []),
        phase_a_completed=page.get("phase_a_completed", False),
        phase_b_completed=page.get("phase_b_completed", False),
    )


@router.get("/stages")
async def get_stages():
    """Return the fixed 5-stage pipeline definition."""
    return {
        "stages": [
            {"id": 1, "name": "detect-text",  "label": "Text Detection",  "description": "Detect speech bubbles via EasyOCR", "phase": "a"},
            {"id": 2, "name": "ocr",           "label": "OCR",             "description": "Re-OCR crops via MangaOCR",           "phase": "a"},
            {"id": 3, "name": "translate",     "label": "Translation",     "description": "Translate Japanese → English",         "phase": "a"},
            {"id": 4, "name": "inpaint",       "label": "Inpainting",      "description": "Erase original Japanese text",         "phase": "b"},
            {"id": 5, "name": "typeset",       "label": "Typesetting",     "description": "Render English text on clean pages",   "phase": "b"},
        ]
    }


@router.put("/save-page-regions")
async def save_page_regions(req: SaveRegionsRequest, request: Request):
    """Save user modified text regions to project manifest, create a single backup, and log corrections."""
    project_dir = os.path.normpath(req.project_dir)
    filename = req.filename
    manifest_path = Path(project_dir) / "project_manifest.json"
    
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Project manifest not found")
        
    # Load manifest data directly as JSON
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read project manifest: {e}")
        
    pages = manifest_data.get("pages", {})
    if filename not in pages:
        raise HTTPException(status_code=404, detail=f"Page not found in manifest: {filename}")
        
    page_data = pages[filename]
    
    # 1. Ask engine container to fit fonts for the updated regions
    payload = {
        "regions": [r.model_dump() for r in req.regions],
        "page_width": page_data.get("width", 9999),
        "page_height": page_data.get("height", 9999)
    }
    
    font_sizes = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(f"{ENGINE_URL}/api/engine/fit-font", json=payload)
            if res.status_code == 200:
                for r_res in res.json().get("regions", []):
                    font_sizes[r_res["id"]] = r_res["font_size"]
            else:
                logger.warning(f"Engine fit-font returned status {res.status_code}: {res.text}")
    except Exception as e:
        logger.error(f"Failed to call engine fit-font API: {e}")
        
    # 2. Rebuild regions list and corrections rows
    new_regions = []
    corrections = []
    timestamp_str = datetime.now().isoformat() + "Z"
    
    input_dir = os.path.join(project_dir, "input")
    output_dir = os.path.join(project_dir, "output")
    
    for r in req.regions:
        fitted_size = font_sizes.get(r.id, 12.0)
        
        tr_dict = {
            "id": r.id,
            "box": {
                "x_min": r.box.x_min,
                "y_min": r.box.y_min,
                "x_max": r.box.x_max,
                "y_max": r.box.y_max
            },
            "raw_text": r.raw_text,
            "translated_text": r.translated_text,
            "ocr_confidence": r.ocr_confidence,
            "is_vertical": r.is_vertical,
            "needs_review": r.needs_review,
            "region_index": r.region_index,
            "speaker_id": r.speaker_id
        }
        new_regions.append(tr_dict)
        
        # Append correction row
        corr_row = {
            "pair_id": f"default/{filename}",
            "manga_id": "default",
            "page": filename,
            "region_index": r.region_index,
            "status": "matched",
            "iou": 1.0,
            "jp_box": {
                "x_min": r.box.x_min,
                "y_min": r.box.y_min,
                "x_max": r.box.x_max,
                "y_max": r.box.y_max
            },
            "en_box": {
                "x_min": r.box.x_min,
                "y_min": r.box.y_min,
                "x_max": r.box.x_max,
                "y_max": r.box.y_max
            },
            "jp_ocr_text": r.raw_text,
            "jp_text_detected": r.raw_text,
            "en_text_detected": r.translated_text,
            "raw_path": os.path.join(input_dir, filename),
            "eng_path": os.path.join(output_dir, filename),
            "font_size": fitted_size,
            "padding": 4.0,
            "line_spacing": 1.2,
            "typesetting_confidence": 1.0,
            "corrected_by": "user",
            "correction_timestamp": timestamp_str
        }
        corrections.append(corr_row)
        
    page_data["text_regions"] = new_regions
    
    # 3. Save manifest with single-backup scheme
    # Create single backup file
    backup_path = manifest_path.with_suffix(".json.bak")
    try:
        if manifest_path.exists():
            shutil.copy2(manifest_path, backup_path)
    except Exception as e:
        logger.warning(f"Failed to create manifest backup: {e}")
        
    # Write to temporary file, then atomically replace
    temp_path = manifest_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, manifest_path)
    except Exception as e:
        logger.error(f"Failed to save manifest file: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to save manifest: {e}")
        
    # 4. Log corrections to user_corrections.jsonl in the projects training data directory
    try:
        corrections_file = Path(project_dir) / "training_data" / "user_corrections.jsonl"
        corrections_file.parent.mkdir(parents=True, exist_ok=True)
        with open(corrections_file, "a", encoding="utf-8") as f:
            for row in corrections:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to save corrections: {e}")
        
    # 5. Trigger auto-inpainting and typesetting for this page in real-time via task queue
    try:
        queue_manager = request.app.state.queue_manager
        exec_data = {
            "project_dir": project_dir,
            "input_subdir": "input",
            "from_stage": "inpaint",
            "to_stage": "typeset",
            "pages": [filename],
            "force": True,
            "model_version": "v2",
            "action": "execute"
        }
        
        task_name = f"Auto Inpaint/Typeset ({filename}): {os.path.basename(project_dir)}"
        async def run_fn(task):
            from routers.pipeline import run_pipeline_task
            await run_pipeline_task(task, exec_data)
            
        queue_manager.submit(task_name, run_fn)
        logger.info(f"Queued auto inpaint/typeset for page {filename}")
    except Exception as e:
        logger.error(f"Failed to initiate auto-run inpaint and typeset: {e}")
        
    return {"success": True}


@router.get("/list-backups", response_model=list[BackupInfo])
async def list_backups(project_dir: str = Query(...)):
    """List project_manifest.json.bak and legacy backups in the project directory."""
    project_dir = os.path.normpath(project_dir)
    p = Path(project_dir)
    if not p.is_dir():
        raise HTTPException(status_code=404, detail="Project directory not found")
        
    backups = []
    
    # 1. Include the single .bak file
    bak_file = p / "project_manifest.json.bak"
    if bak_file.exists():
        dt = datetime.fromtimestamp(bak_file.stat().st_mtime)
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        backups.append(BackupInfo(
            filename=bak_file.name,
            timestamp=timestamp_str,
            display_name=f"Backup (Single) — {timestamp_str}"
        ))
        
    # 2. Include any legacy timestamped backups
    for f in p.glob("project_manifest.backup_*.json"):
        parts = f.stem.split("_")
        timestamp_str = ""
        if len(parts) >= 3:
            ts_part = parts[2]
            if len(ts_part) == 8 and len(parts) >= 4:
                time_part = parts[3]
                try:
                    dt = datetime.strptime(f"{ts_part}_{time_part}", "%Y%m%d_%H%M%S")
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
        if not timestamp_str:
            dt = datetime.fromtimestamp(f.stat().st_mtime)
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            
        backups.append(BackupInfo(
            filename=f.name,
            timestamp=timestamp_str,
            display_name=f"Backup — {timestamp_str}"
        ))
        
    backups.sort(key=lambda b: b.timestamp, reverse=True)
    return backups


@router.post("/restore-backup")
async def restore_backup(req: RestoreBackupRequest):
    """Restore project_manifest.json from a selected backup file."""
    project_dir = os.path.normpath(req.project_dir)
    backup_path = Path(project_dir) / req.backup_filename
    manifest_path = Path(project_dir) / "project_manifest.json"
    
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail=f"Backup file not found: {req.backup_filename}")
        
    try:
        shutil.copy2(backup_path, manifest_path)
        return {"success": True, "detail": f"Restored manifest from {req.backup_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {e}")


