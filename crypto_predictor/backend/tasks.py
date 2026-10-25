"""
Background task system for CryptoPredictor.
Provides async task execution with optional Celery support.
"""

import logging
import threading
import queue
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    progress_message: str = ""
    
    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'progress_message': self.progress_message
        }


@dataclass
class Task:
    """A background task to be executed."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class TaskWorker(threading.Thread):
    """Worker thread that executes tasks from a queue."""
    
    def __init__(
        self,
        task_queue: queue.Queue,
        results: Dict[str, TaskResult],
        results_lock: threading.Lock
    ):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.results = results
        self.results_lock = results_lock
        self._running = True
    
    def run(self):
        while self._running:
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if task is None:
                break
            
            self._execute_task(task)
            self.task_queue.task_done()
    
    def _execute_task(self, task: Task):
        """Execute a single task."""
        with self.results_lock:
            if task.id in self.results:
                if self.results[task.id].status == TaskStatus.CANCELLED:
                    return
                self.results[task.id].status = TaskStatus.RUNNING
                self.results[task.id].started_at = datetime.now()
        
        try:
            # Create a progress reporter for the task
            def report_progress(progress: float, message: str = ""):
                with self.results_lock:
                    if task.id in self.results:
                        self.results[task.id].progress = progress
                        self.results[task.id].progress_message = message
            
            # Add progress reporter to kwargs if function accepts it
            import inspect
            sig = inspect.signature(task.func)
            if 'progress_callback' in sig.parameters:
                task.kwargs['progress_callback'] = report_progress
            if 'cancel_event' in sig.parameters:
                task.kwargs['cancel_event'] = task.cancel_event
            
            # Execute the task
            result = task.func(*task.args, **task.kwargs)
            
            # Update result
            with self.results_lock:
                self.results[task.id].status = TaskStatus.COMPLETED
                self.results[task.id].result = result
                self.results[task.id].completed_at = datetime.now()
                self.results[task.id].progress = 100.0
            
            # Call success callback
            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    logger.error(f"Task callback error: {e}")
        
        except Exception as e:
            if task.cancel_event.is_set():
                logger.info(f"Task {task.id} cancelled")
                with self.results_lock:
                    self.results[task.id].status = TaskStatus.CANCELLED
                    self.results[task.id].completed_at = datetime.now()
                return

            logger.error(f"Task {task.id} failed: {e}")
            
            with self.results_lock:
                self.results[task.id].status = TaskStatus.FAILED
                self.results[task.id].error = str(e)
                self.results[task.id].completed_at = datetime.now()
            
            # Call error callback
            if task.error_callback:
                try:
                    task.error_callback(e)
                except Exception as cb_error:
                    logger.error(f"Task error callback failed: {cb_error}")
    
    def stop(self):
        self._running = False


class TaskManager:
    """
    Manages background task execution.
    Provides a simple thread-pool based task execution system.
    """
    
    def __init__(self, num_workers: int = 2):
        self.task_queue: queue.Queue = queue.Queue()
        self.results: Dict[str, TaskResult] = {}
        self.results_lock = threading.Lock()
        self.workers: List[TaskWorker] = []
        self.num_workers = num_workers
        self._started = False
        self._cancelled_tasks: set = set()
        self._tasks: Dict[str, Task] = {}
    
    def start(self):
        """Start the worker threads."""
        if self._started:
            return
        
        for i in range(self.num_workers):
            worker = TaskWorker(
                self.task_queue,
                self.results,
                self.results_lock
            )
            worker.start()
            self.workers.append(worker)
        
        self._started = True
        logger.info(f"Task manager started with {self.num_workers} workers")
    
    def stop(self):
        """Stop all worker threads."""
        for worker in self.workers:
            worker.stop()
        
        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.workers.clear()
        self._started = False
        logger.info("Task manager stopped")
    
    def submit(
        self,
        func: Callable,
        *args,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        if not self._started:
            self.start()
        
        task_id = str(uuid.uuid4())
        
        # Create result entry
        with self.results_lock:
            self.results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING
            )
        
        # Create and queue task
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            callback=callback,
            error_callback=error_callback
        )
        
        with self.results_lock:
            self._tasks[task_id] = task
        self.task_queue.put(task)
        logger.info(f"Task {task_id} submitted")
        
        return task_id
    
    def get_status(self, task_id: str) -> Optional[TaskResult]:
        """Get the status of a task."""
        with self.results_lock:
            return self.results.get(task_id)
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a task. Pending tasks are skipped; running tasks are signalled
        through their cancel_event and stop at their next checkpoint.
        
        Returns:
            True if the task was cancelled or asked to stop.
        """
        with self.results_lock:
            result = self.results.get(task_id)
            task = self._tasks.get(task_id)
            if result is None or task is None:
                return False
            if result.status == TaskStatus.PENDING:
                result.status = TaskStatus.CANCELLED
                self._cancelled_tasks.add(task_id)
                task.cancel_event.set()
                return True
            if result.status == TaskStatus.RUNNING:
                task.cancel_event.set()
                return True
        return False
    
    def cleanup_old_results(self, max_age_seconds: int = 3600):
        """Remove old completed/failed results."""
        cutoff = datetime.now()
        
        with self.results_lock:
            to_remove = []
            for task_id, result in self.results.items():
                if result.completed_at:
                    age = (cutoff - result.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.results[task_id]
                self._tasks.pop(task_id, None)
            
            return len(to_remove)


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def submit_task(
    func: Callable,
    *args,
    callback: Optional[Callable] = None,
    error_callback: Optional[Callable] = None,
    **kwargs
) -> str:
    """
    Convenience function to submit a task.
    
    Returns:
        Task ID for tracking.
    """
    return get_task_manager().submit(
        func, *args,
        callback=callback,
        error_callback=error_callback,
        **kwargs
    )


def get_task_status(task_id: str) -> Optional[TaskResult]:
    """Get the status of a task."""
    return get_task_manager().get_status(task_id)


def cancel_task(task_id: str) -> bool:
    """Cancel a pending task."""
    return get_task_manager().cancel(task_id)
