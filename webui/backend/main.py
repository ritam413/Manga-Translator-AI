"""
WebUI Backend — FastAPI bridge for Manga Translator Pipeline
Phase 1: Project manifest & directory data API
"""

import os
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import project, pipeline, images, training
from utils.queue_manager import QueueManager
from utils.ws_manager import ws_manager

logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Manga Translator WebUI Backend starting...")
    # Initialize and start global QueueManager
    app.state.queue_manager = QueueManager()
    app.state.queue_manager.start()
    yield
    logger.info("Manga Translator WebUI Backend shutting down...")
    await app.state.queue_manager.stop()


app = FastAPI(
    title="Manga Translator WebUI API",
    description="FastAPI bridge for the Manga Translator CLI pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(project.router, prefix="/api/project", tags=["Project"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(images.router, prefix="/api/images", tags=["Images"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])


@app.websocket("/api/ws/pipeline/execute")
@app.websocket("/api/ws/pipeline/run")
async def websocket_pipeline_execute(websocket: WebSocket):
    """Bidirectional WebSocket to run pipeline stages or training and stream logs."""
    from routers.pipeline import run_pipeline_task
    
    await ws_manager.connect(websocket)
    try:
        req_data = await websocket.receive_json()
        project_dir = req_data.get("project_dir")
        if not project_dir:
            await websocket.send_json({"type": "error", "data": "project_dir is required"})
            await websocket.close()
            ws_manager.disconnect(websocket)
            return
            
        queue_manager = websocket.app.state.queue_manager
        action = req_data.get("action", "execute")
        if action == "train":
            task_name = f"Training: {os.path.basename(project_dir)}"
        else:
            stage_name = req_data.get("stage") or req_data.get("from_stage") or "all"
            task_name = f"Pipeline ({stage_name}): {os.path.basename(project_dir)}"
            
        async def run_fn(task):
            await run_pipeline_task(task, req_data)
            
        task_id = queue_manager.submit(task_name, run_fn)
        task = queue_manager.tasks[task_id]
        
        await websocket.send_json({"type": "queued", "task_id": task_id})
        await ws_manager.stream_task_logs(websocket, task, queue_manager)
    except Exception as e:
        logger.error(f"Error in WebSocket pipeline handler: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        ws_manager.disconnect(websocket)


@app.websocket("/api/ws/health")
async def websocket_health(websocket: WebSocket):
    """Provide real-time server health status over a WebSocket connection."""
    import httpx
    from routers.project import ENGINE_URL
    await websocket.accept()
    try:
        while True:
            engine_connected = False
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    res = await client.get(f"{ENGINE_URL}/api/engine/health")
                    if res.status_code == 200:
                        engine_connected = True
            except Exception:
                pass
            await websocket.send_json({
                "status": "healthy",
                "engine_connected": engine_connected
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


@app.get("/api/health")
async def health_check():
    import httpx
    from routers.project import ENGINE_URL
    engine_connected = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            res = await client.get(f"{ENGINE_URL}/api/engine/health")
            if res.status_code == 200:
                engine_connected = True
    except Exception:
        pass
    return {
        "status": "ok",
        "service": "manga-translator-webui",
        "engine_connected": engine_connected
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
