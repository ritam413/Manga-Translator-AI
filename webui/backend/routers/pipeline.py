"""
Pipeline router — run / stream pipeline stages via REST API and task queue.
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Optional, List

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from utils.queue_manager import QueueTask

logger = logging.getLogger("routers.pipeline")
router = APIRouter()

ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8001")


# ── Request models ─────────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    project_dir: str
    input_subdir: str = "input"

class ExecuteRequest(BaseModel):
    project_dir: str
    input_subdir: str = "input"
    stage: Optional[str] = None          # single stage name
    from_stage: Optional[str] = None
    to_stage: Optional[str] = None
    all_stages: bool = False
    force: bool = False
    model_version: str = "v2"
    limit: Optional[int] = None
    pages: Optional[List[str]] = None


# ── Task Runner ────────────────────────────────────────────────────────────────

async def run_pipeline_task(task: QueueTask, req_data: dict):
    """
    Submits execute/train request to the engine container API and streams logs.
    Proxying SSE log streams into the queue task logs queue.
    """
    action = req_data.get("action", "execute")
    if action == "train":
        url = f"{ENGINE_URL}/api/engine/train"
        payload = {
            "project_dir": req_data.get("project_dir"),
            "model_version": req_data.get("model_version", "v2"),
            "use_corrections": req_data.get("use_corrections", "training_data/user_corrections.jsonl"),
        }
    else:
        url = f"{ENGINE_URL}/api/engine/execute"
        payload = {
            "project_dir": req_data.get("project_dir"),
            "input_subdir": req_data.get("input_subdir", "input"),
            "stage": req_data.get("stage"),
            "from_stage": req_data.get("from_stage"),
            "to_stage": req_data.get("to_stage"),
            "all_stages": req_data.get("all_stages", False),
            "force": req_data.get("force", False),
            "model_version": req_data.get("model_version", "v2"),
            "limit": req_data.get("limit"),
            "pages": req_data.get("pages"),
        }

    await task.put_log(f"Connecting to engine REST API at {url}...")
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_content = await response.aread()
                    raise RuntimeError(f"Engine REST API returned code {response.status_code}: {err_content.decode('utf-8', errors='replace')}")

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        await task.put_log(line[6:])
                    elif line.startswith("event: "):
                        event_type = line[7:]
                        if event_type == "done":
                            await task.put_log("✓ Engine execution completed.")
                        elif event_type == "error":
                            await task.put_log("✗ Engine execution failed.")
                    else:
                        await task.put_log(line)
    except asyncio.CancelledError:
        logger.info(f"Task {task.task_id} was cancelled. Aborting HTTP stream.")
        raise
    except Exception as e:
        logger.error(f"Task {task.task_id} failed: {e}")
        raise


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/plan")
async def plan_project(req: PlanRequest):
    """Dispatches `plan` request to the engine container API."""
    url = f"{ENGINE_URL}/api/engine/plan"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(url, json={"project_dir": req.project_dir, "input_subdir": req.input_subdir})
            if res.status_code != 200:
                raise HTTPException(status_code=res.status_code, detail=f"Engine error: {res.text}")
            return res.json()
    except Exception as e:
        logger.error(f"Failed to communicate with engine plan endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to contact engine: {e}")


@router.post("/execute/stream")
async def execute_stream(req: ExecuteRequest, request: Request):
    """
    Saves execute request to the sequential queue, streams stdout logs via SSE.
    Enables backward compatibility with older Next.js clients.
    """
    queue_manager = request.app.state.queue_manager
    stage_name = req.stage or req.from_stage or "all"
    task_name = f"Pipeline ({stage_name}): {os.path.basename(req.project_dir)}"
    
    # Pack parameters
    req_dict = req.model_dump()
    req_dict["action"] = "execute"

    async def run_fn(task: QueueTask):
        await run_pipeline_task(task, req_dict)

    task_id = queue_manager.submit(task_name, run_fn)
    task = queue_manager.tasks[task_id]

    async def event_generator():
        yield "event: start\ndata: Pipeline execution queued\n\n"
        try:
            while True:
                log_line = await task.log_queue.get()
                yield f"data: {log_line}\n\n"
                task.log_queue.task_done()
                
                if task.status == "completed":
                    yield "event: done\ndata: Pipeline completed successfully\n\n"
                    break
                elif task.status == "failed":
                    yield "event: error\ndata: Pipeline failed\n\n"
                    break
                elif task.status == "cancelled":
                    yield "event: error\ndata: Pipeline cancelled\n\n"
                    break
        except asyncio.CancelledError:
            logger.info(f"SSE client disconnected. Cancelling task {task_id}.")
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


@router.get("/status")
async def pipeline_status(project_dir: str = Query(...)):
    """Quick snapshot of how many pages have completed each phase."""
    import json
    manifest_path = Path(project_dir) / "project_manifest.json"
    if not manifest_path.exists():
        return {"manifest_exists": False}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pages = manifest.get("pages", {})
    total = len(pages)
    phase_a = sum(1 for p in pages.values() if p.get("phase_a_completed"))
    phase_b = sum(1 for p in pages.values() if p.get("phase_b_completed"))

    return {
        "manifest_exists": True,
        "total": total,
        "phase_a_completed": phase_a,
        "phase_b_completed": phase_b,
        "phase_a_pct": round(phase_a / total * 100) if total else 0,
        "phase_b_pct": round(phase_b / total * 100) if total else 0,
    }
