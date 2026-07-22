import asyncio
import uuid
import logging
from typing import Dict, Any, Callable, Awaitable, Optional, List

logger = logging.getLogger("queue_manager")

class QueueTask:
    def __init__(self, task_id: str, name: str, run_fn: Callable[['QueueTask'], Awaitable[None]]):
        self.task_id = task_id
        self.name = name
        self.run_fn = run_fn
        self.status = "queued"  # queued, running, completed, failed, cancelled
        self.asyncio_task: Optional[asyncio.Task] = None
        self.log_queue: asyncio.Queue = asyncio.Queue()
        self.logs_history: List[str] = []

    async def put_log(self, text: str):
        self.logs_history.append(text)
        await self.log_queue.put(text)

class QueueManager:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, QueueTask] = {}
        self.current_task: Optional[QueueTask] = None
        self.worker_loop_task: Optional[asyncio.Task] = None

    def start(self):
        if self.worker_loop_task is None:
            self.worker_loop_task = asyncio.create_task(self._worker_loop())
            logger.info("QueueManager worker loop started")

    async def stop(self):
        if self.worker_loop_task:
            self.worker_loop_task.cancel()
            try:
                await self.worker_loop_task
            except asyncio.CancelledError:
                pass
            self.worker_loop_task = None
            logger.info("QueueManager worker loop stopped")

    async def _worker_loop(self):
        while True:
            try:
                task: QueueTask = await self.queue.get()
                if task.status == "cancelled":
                    self.queue.task_done()
                    continue

                self.current_task = task
                task.status = "running"
                logger.info(f"Running task {task.task_id} ({task.name})")

                # Create asyncio Task to run the function
                task.asyncio_task = asyncio.create_task(task.run_fn(task))
                try:
                    await task.asyncio_task
                    task.status = "completed"
                    logger.info(f"Task {task.task_id} completed successfully")
                except asyncio.CancelledError:
                    task.status = "cancelled"
                    logger.info(f"Task {task.task_id} was cancelled")
                    await task.put_log("⏹ Task cancelled by user or system.")
                except Exception as e:
                    task.status = "failed"
                    logger.error(f"Task {task.task_id} failed: {e}")
                    await task.put_log(f"✗ Task failed: {e}")
                finally:
                    self.current_task = None
                    self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}")
                await asyncio.sleep(1)

    def submit(self, name: str, run_fn: Callable[[QueueTask], Awaitable[None]]) -> str:
        task_id = str(uuid.uuid4())
        task = QueueTask(task_id, name, run_fn)
        self.tasks[task_id] = task
        self.queue.put_nowait(task)
        logger.info(f"Submitted task {task_id} ({name}) to queue")
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        if task_id not in self.tasks:
            return False
        task = self.tasks[task_id]
        if task.status in ["completed", "failed", "cancelled"]:
            return False

        if task.status == "queued":
            task.status = "cancelled"
            logger.info(f"Cancelled queued task {task_id}")
            return True

        if task.status == "running" and task.asyncio_task:
            task.status = "cancelled"
            task.asyncio_task.cancel()
            logger.info(f"Cancelled active task {task_id}")
            return True

        return False

    def cancel_active(self) -> bool:
        if self.current_task:
            return self.cancel_task(self.current_task.task_id)
        return False

    def get_status(self, task_id: str) -> Optional[str]:
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
