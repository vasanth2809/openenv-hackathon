from typing import Any, Dict, List, Literal, Optional
from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class CleaningAction(Action):
    """Agent intent for a cleaning operation."""

    operation: Literal[
        "normalize_text",
        "fill_missing",
        "standardize_date",
        "dedupe",
        "split_column",
        "merge_columns",
        "clip_outliers",
        "map_values",
    ]
    target_columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class CleaningObservation(Observation):
    """Environment feedback after an action."""

    preview_rows: List[Dict[str, Any]]
    schema: List[Dict[str, Any]]
    issues_detected: List[str]
    applied_ops: List[str]
    messages: List[str]


class CleaningState(State):
    """Episode metadata and progress."""

    task_id: str
    step_count: int = 0
    score: float = 0.0
    remaining_issues: int = 0
    invalid_ops: int = 0
