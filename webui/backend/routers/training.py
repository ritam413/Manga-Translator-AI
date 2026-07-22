"""
Training router — run / stream model finetuning via REST API and task queue.
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("routers.training")
router = APIRouter()

ENGINE_DIR = Path(r"e:\u torrentz\MangaTranslator FUll STACK\Manga Translator")


# ── Request / Response models ──────────────────────────────────────────────────

class ClearCorrectionsRequest(BaseModel):
    project_dir: str

class TrainRequest(BaseModel):
    project_dir: str
    model_version: str = "v2"
    use_corrections: str = "training_data/user_corrections.jsonl"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/correction-count")
async def get_correction_count(
    project_dir: str = Query(..., description="Absolute path to the manga project directory")
):
    """Return the count of unsaved manual annotations from user_corrections.jsonl."""
    corrections_file = ENGINE_DIR / "training_data" / "user_corrections.jsonl"
    if not corrections_file.exists():
        return {"count": 0}
        
    try:
        count = 0
        with open(corrections_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read corrections count: {e}")


@router.post("/clear-corrections")
async def clear_corrections(req: ClearCorrectionsRequest):
    """Delete the accumulated user_corrections.jsonl file."""
    corrections_file = ENGINE_DIR / "training_data" / "user_corrections.jsonl"
    if corrections_file.exists():
        try:
            corrections_file.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clear corrections: {e}")
    return {"success": True}


@router.post("/train-detection")
async def train_detection(req: TrainRequest, request: Request):
    """
    Run finetuning of layout model with user corrections.
    Streams execution output back via SSE.
    """
    queue_manager = request.app.state.queue_manager
    task_name = f"Training: {os.path.basename(req.project_dir)}"
    
    # Pack parameters
    req_dict = req.model_dump()
    req_dict["action"] = "train"

    async def run_fn(task):
        from routers.pipeline import run_pipeline_task
        await run_pipeline_task(task, req_dict)

    task_id = queue_manager.submit(task_name, run_fn)
    task = queue_manager.tasks[task_id]

    async def event_generator():
        yield "event: start\ndata: Layout model training queued\n\n"
        try:
            while True:
                log_line = await task.log_queue.get()
                yield f"data: {log_line}\n\n"
                task.log_queue.task_done()
                
                if task.status == "completed":
                    yield "event: done\ndata: Layout model retrained successfully!\n\n"
                    break
                elif task.status == "failed":
                    yield "event: error\ndata: Training failed\n\n"
                    break
                elif task.status == "cancelled":
                    yield "event: error\ndata: Training cancelled\n\n"
                    break
        except asyncio.CancelledError:
            logger.info(f"SSE client disconnected. Cancelling training task {task_id}.")
            queue_manager.cancel_task(task_id)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
