"""
Images router — serves manga page images directly from the project directory.
Endpoints:
  GET /api/images/list   → sorted list of filenames for input/output dir
  GET /api/images/file   → raw image bytes (FileResponse)
  GET /api/images/thumb  → scaled-down thumbnail (max 400px wide)
"""

import os
import io
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

router = APIRouter()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def _resolve_dir(project_dir: str, img_type: str, input_subdir: str) -> Path:
    root = Path(project_dir)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Project directory not found: {project_dir}")

    subdir = input_subdir if img_type == "input" else "output"
    target = root / subdir
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {target}")
    return target


def _safe_image_path(directory: Path, filename: str) -> Path:
    """Return resolved path, ensuring it stays within directory (no path traversal)."""
    resolved = (directory / filename).resolve()
    if not str(resolved).startswith(str(directory.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not an image file")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    return resolved


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/list")
async def list_images(
    project_dir: str = Query(...),
    type: Literal["input", "output"] = Query("input"),
    input_subdir: str = Query("input"),
):
    """Return sorted list of image filenames in input or output directory."""
    target = _resolve_dir(project_dir, type, input_subdir)
    files = sorted(
        [f.name for f in target.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda n: (
            int("".join(c for c in n.split(".")[0] if c.isdigit()) or "0"),
            n,
        ),
    )
    return {"type": type, "directory": str(target), "count": len(files), "files": files}


# ── Full image ────────────────────────────────────────────────────────────────

@router.get("/file")
async def get_image(
    project_dir: str = Query(...),
    type: Literal["input", "output"] = Query("input"),
    filename: str = Query(...),
    input_subdir: str = Query("input"),
):
    """Serve the raw image file."""
    target_dir = _resolve_dir(project_dir, type, input_subdir)
    img_path = _safe_image_path(target_dir, filename)

    # Pick MIME type
    suffix = img_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".gif": "image/gif", ".bmp": "image/bmp",
    }
    media_type = media_types.get(suffix, "image/jpeg")
    return FileResponse(str(img_path), media_type=media_type,
                        headers={"Cache-Control": "public, max-age=3600"})


# ── Thumbnail ─────────────────────────────────────────────────────────────────

@router.get("/thumb")
async def get_thumbnail(
    project_dir: str = Query(...),
    type: Literal["input", "output"] = Query("input"),
    filename: str = Query(...),
    input_subdir: str = Query("input"),
    max_width: int = Query(400, ge=32, le=1200),
):
    """
    Serve a downscaled thumbnail.  Uses Pillow if available; falls back to
    sending the full image so the frontend can handle scaling via CSS.
    """
    target_dir = _resolve_dir(project_dir, type, input_subdir)
    img_path = _safe_image_path(target_dir, filename)

    try:
        from PIL import Image as PILImage

        with PILImage.open(str(img_path)) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w > max_width:
                ratio = max_width / w
                img = img.resize((max_width, int(h * ratio)), PILImage.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
            buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    except ImportError:
        # Pillow not available — serve original with cache header
        return FileResponse(
            str(img_path), media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
