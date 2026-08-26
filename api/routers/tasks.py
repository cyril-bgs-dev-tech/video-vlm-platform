"""
api/routers/tasks.py — Polling et streaming des tâches Celery
Endpoints : GET status, GET résultat, DELETE (révocation), SSE live updates
"""
import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from workers.celery_app import celery_app
from logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])

CELERY_STATES = {"PENDING", "STARTED", "RETRY", "SUCCESS", "FAILURE", "REVOKED"}


class TaskStatus(BaseModel):
    task_id:   str
    status:    str
    result:    Optional[dict] = None
    error:     Optional[str]  = None
    progress:  Optional[int]  = None


@router.get("")
async def list_active_tasks(queue: Optional[str] = None):
    try:
        inspect = celery_app.control.inspect(timeout=3)
        active  = inspect.active() or {}
        reserved = inspect.reserved() or {}

        all_tasks = []
        for worker, tasks in {**active, **reserved}.items():
            for t in (tasks or []):
                if queue and t.get("delivery_info", {}).get("routing_key") != queue:
                    continue
                all_tasks.append({
                    "task_id":   t.get("id"),
                    "name":      t.get("name", "").split(".")[-1],
                    "worker":    worker,
                    "args":      t.get("kwargs", {}),
                    "started":   t.get("time_start"),
                })

        return {"tasks": all_tasks, "count": len(all_tasks)}
    except Exception as e:
        logger.warning("Failed to inspect workers", error=str(e))
        return {"tasks": [], "count": 0, "warning": str(e)}


@router.get("/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    try:
        result = celery_app.AsyncResult(task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Celery : {e}")

    status = result.state
    data   = None
    error  = None

    if status == "SUCCESS":
        data = result.result if isinstance(result.result, dict) else {"value": result.result}
    elif status == "FAILURE":
        error = str(result.result)
    elif status == "PROGRESS":
        meta = result.info or {}
        return TaskStatus(
            task_id=task_id,
            status="STARTED",
            progress=meta.get("progress"),
        )

    return TaskStatus(
        task_id=task_id,
        status=status,
        result=data,
        error=error,
    )


@router.get("/{task_id}/stream")
async def stream_task_updates(
    task_id: str,
    poll_interval: float = Query(0.5, ge=0.1, le=5.0),
    timeout:       float = Query(300.0, le=3600.0),
):
    async def event_generator():
        elapsed = 0.0
        last_state = None
        while elapsed < timeout:
            try:
                result = celery_app.AsyncResult(task_id)
                state  = result.state
                info   = result.info if isinstance(result.info, dict) else {}

                if state != last_state:
                    payload = {"task_id": task_id, "status": state}
                    if state == "SUCCESS":
                        payload["result"] = (
                            result.result
                            if isinstance(result.result, dict)
                            else {"value": result.result}
                        )
                    elif state == "FAILURE":
                        payload["error"] = str(result.result)
                    elif state in ("STARTED", "PROGRESS"):
                        payload["progress"] = info.get("progress")

                    yield f"data: {json.dumps(payload)}\n\n"
                    last_state = state

                if state in ("SUCCESS", "FAILURE", "REVOKED"):
                    yield "event: done\ndata: {}\n\n"
                    return

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        yield f"data: {json.dumps({'status': 'TIMEOUT', 'task_id': task_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{task_id}")
async def revoke_task(
    task_id:   str,
    terminate: bool = Query(False, description="Tuer le process worker en cours"),
):
    try:
        celery_app.control.revoke(task_id, terminate=terminate, signal="SIGTERM")
        logger.info("Task revoked", task_id=task_id, terminate=terminate)
        return {"revoked": task_id, "terminate": terminate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur révocation : {e}")
