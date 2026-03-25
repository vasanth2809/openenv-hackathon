from fastapi import APIRouter
from openenv.core.env_server import create_fastapi_app

from ..models import CleaningAction, CleaningObservation
from .environment import DataCleaningEnvironment, TASKS, grade

app = create_fastapi_app(DataCleaningEnvironment, CleaningAction, CleaningObservation)

router = APIRouter()


@router.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": task_id, "description": t["description"], "sample_rows": t["data"][:2]}
            for task_id, t in TASKS.items()
        ]
    }


@router.get("/grader")
def latest_grader():
    # Note: create_fastapi_app binds env instance per-connection; we expose static task grading helper instead
    return {"detail": "Call /state to see current score; graders run per step internally."}


@app.get("/baseline")
def run_baseline():
    # Lazy import to avoid startup overhead
    from ..baseline import run_baseline  # type: ignore

    return run_baseline()


app.include_router(router)
