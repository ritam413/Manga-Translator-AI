import os
import sys
import logging
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("engine_api")

app = FastAPI(
    title="Manga Translator Engine API",
    description="REST API wrapper for the ML translation engine",
    version="1.0.0",
)

# Add engine root to Python path if needed
ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from core.models import BoundingBox
from workers.typesetter import TypesetterWorker

# ── Models ────────────────────────────────────────────────────────────────────

class BBoxIn(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class RegionIn(BaseModel):
    id: str
    box: BBoxIn
    raw_text: str = ""
    translated_text: str = ""
    ocr_confidence: float = 0.0
    is_vertical: bool = True
    needs_review: bool = False
    region_index: int = 0
    speaker_id: str = ""

class FitFontRequest(BaseModel):
    regions: List[RegionIn]
    page_width: int
    page_height: int

class ExecuteRequest(BaseModel):
    project_dir: str
    input_subdir: str = "input"
    stage: Optional[str] = None
    from_stage: Optional[str] = None
    to_stage: Optional[str] = None
    all_stages: bool = False
    force: bool = False
    model_version: str = "v2"
    limit: Optional[int] = None
    pages: Optional[List[str]] = None

class TrainRequest(BaseModel):
    project_dir: str
    model_version: str = "v2"
    use_corrections: str = "training_data/user_corrections.jsonl"

class PlanRequest(BaseModel):
    project_dir: str
    input_subdir: str = "input"

# ── Typesetter Cache ──────────────────────────────────────────────────────────

_typesetter: Optional[TypesetterWorker] = None

def get_typesetter() -> TypesetterWorker:
    global _typesetter
    if _typesetter is None:
        logger.info("Initializing TypesetterWorker for font size fitting...")
        _typesetter = TypesetterWorker(
            input_dir="",
            output_dir="",
            tone_engine_enabled=False
        )
        _typesetter.load()
    return _typesetter

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/engine/health")
async def health_check():
    return {"status": "ok", "service": "manga-translator-engine"}

@app.post("/api/engine/plan")
async def plan_project(req: PlanRequest):
    """
    Spawns main.py plan in a subprocess to scan the input folder and update the manifest.
    """
    main_py = os.path.join(ENGINE_ROOT, "main.py")
    cmd = [
        sys.executable,
        "-u",
        main_py,
        "--project-dir", req.project_dir,
        "--input-subdir", req.input_subdir,
        "plan",
    ]
    logger.info(f"Running plan command: {' '.join(cmd)}")
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=ENGINE_ROOT,
        )
        stdout_bytes, _ = await proc.communicate()
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        
        return {
            "success": proc.returncode == 0,
            "stdout": stdout_text,
            "stderr": "",
            "returncode": proc.returncode,
        }
    except Exception as e:
        logger.error(f"Plan subprocess execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/engine/fit-font")
async def fit_font(req: FitFontRequest):
    """
    Given a list of regions and page dimensions, calculate the optimal fitted font size
    for each region using TypesetterWorker's binary search logic.
    """
    typesetter = get_typesetter()
    results = []
    
    for r in req.regions:
        fitted_size = 12.0
        try:
            box = BoundingBox(
                x_min=r.box.x_min,
                y_min=r.box.y_min,
                x_max=r.box.x_max,
                y_max=r.box.y_max
            )
            cw, ch = req.page_width or 9999, req.page_height or 9999
            box_x_min, box_y_min, box_x_max, box_y_max = typesetter._get_expanded_box(box, cw, ch)
            pad = 4
            x_min = box_x_min + pad
            y_min = box_y_min + pad
            x_max = box_x_max - pad
            y_max = box_y_max - pad
            avail_w = max(1, x_max - x_min)
            avail_h = max(1, y_max - y_min)
            
            font_obj, _, _ = typesetter._fit_font(
                text=r.translated_text.upper().strip(),
                avail_w=avail_w,
                avail_h=avail_h,
            )
            if hasattr(font_obj, "size"):
                fitted_size = float(font_obj.size)
        except Exception as e:
            logger.warning(f"Failed to fit font size for region {r.id}: {e}")
            
        results.append({
            "id": r.id,
            "font_size": fitted_size
        })
        
    return {"regions": results}

@app.post("/api/engine/execute")
async def execute_pipeline(req: ExecuteRequest):
    """
    Spawns main.py execute in a subprocess and streams back output logs as SSE events.
    Handles client disconnection by terminating the subprocess immediately.
    """
    main_py = os.path.join(ENGINE_ROOT, "main.py")
    cmd = [
        sys.executable,
        "-u",
        main_py,
        "--project-dir", req.project_dir,
        "--input-subdir", req.input_subdir,
        "execute",
        "--model-version", req.model_version,
    ]
    if req.force:
        cmd.append("--force")
    if req.limit is not None:
        cmd += ["--limit", str(req.limit)]
    if req.all_stages:
        cmd.append("all")
    elif req.stage:
        cmd += ["--stage", req.stage]
    elif req.from_stage and req.to_stage:
        cmd += ["--from-stage", req.from_stage, "--to-stage", req.to_stage]
        
    if req.pages:
        cmd += ["--pages"] + req.pages

    logger.info(f"Running pipeline command: {' '.join(cmd)}")

    async def event_generator():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=ENGINE_ROOT,
        )
        yield "event: start\ndata: Engine pipeline started\n\n"
        
        try:
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    yield f"data: {text}\n\n"
        except asyncio.CancelledError:
            logger.info("Connection cancelled. Terminating subprocess...")
            proc.terminate()
            await proc.wait()
            yield "event: error\ndata: Pipeline cancelled\n\n"
            raise
            
        await proc.wait()
        if proc.returncode == 0:
            yield "event: done\ndata: Pipeline completed successfully\n\n"
        else:
            yield f"event: error\ndata: Pipeline failed with code {proc.returncode}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/api/engine/train")
async def train_layout(req: TrainRequest):
    """
    Spawns main.py finetune in a subprocess and streams back output logs as SSE events.
    """
    main_py = os.path.join(ENGINE_ROOT, "main.py")
    cmd = [
        sys.executable,
        "-u",
        main_py,
        "--project-dir", req.project_dir,
        "finetune",
        "--use-corrections", req.use_corrections,
        "--model-version", req.model_version,
    ]
    
    logger.info(f"Running training command: {' '.join(cmd)}")

    async def event_generator():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=ENGINE_ROOT,
        )
        yield "event: start\ndata: Layout model training started\n\n"
        
        try:
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    yield f"data: {text}\n\n"
        except asyncio.CancelledError:
            logger.info("Connection cancelled. Terminating training subprocess...")
            proc.terminate()
            await proc.wait()
            yield "event: error\ndata: Training cancelled\n\n"
            raise
            
        await proc.wait()
        if proc.returncode == 0:
            yield "event: done\ndata: Layout model retrained successfully!\n\n"
        else:
            yield f"event: error\ndata: Training failed with code {proc.returncode}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("engine_api:app", host="0.0.0.0", port=8001)
