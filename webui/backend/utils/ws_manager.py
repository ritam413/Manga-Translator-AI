import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
from utils.queue_manager import QueueManager, QueueTask

logger = logging.getLogger("ws_manager")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket connection accepted")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def stream_task_logs(self, websocket: WebSocket, task: QueueTask, queue_manager: QueueManager):
        """
        Streams logs from the given queue task to the WebSocket.
        Listens concurrently for client cancellations.
        """
        task_id = task.task_id

        # 1. Send all past logs in case we connected late or there were queued logs
        for log in list(task.logs_history):
            await websocket.send_json({"type": "log", "data": log})

        async def read_logs():
            try:
                while True:
                    log_line = await task.log_queue.get()
                    await websocket.send_json({"type": "log", "data": log_line})
                    task.log_queue.task_done()
                    # Check terminal conditions
                    if task.status in ["completed", "failed", "cancelled"]:
                        await websocket.send_json({"type": "status", "status": task.status})
                        break
            except Exception as e:
                logger.error(f"Error in read_logs task: {e}")

        async def listen_cancel():
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("action") == "cancel":
                        logger.info(f"Client requested cancellation for task {task_id}")
                        queue_manager.cancel_task(task_id)
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for task {task_id}. Cancelling task.")
                queue_manager.cancel_task(task_id)
            except Exception as e:
                logger.warning(f"Error or closed connection in listen_cancel: {e}")
                queue_manager.cancel_task(task_id)

        # Run both reading from task log queue and listening to client cancel requests concurrently
        await asyncio.gather(read_logs(), listen_cancel())

ws_manager = ConnectionManager()
