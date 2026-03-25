from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import CleaningAction, CleaningObservation, CleaningState


class DataCleaningEnv(EnvClient[CleaningAction, CleaningObservation, CleaningState]):
    def _step_payload(self, action: CleaningAction) -> dict:
        return {
            "operation": action.operation,
            "target_columns": action.target_columns,
            "parameters": action.parameters,
            "notes": action.notes,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=CleaningObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                preview_rows=obs_data.get("preview_rows", []),
                schema=obs_data.get("schema", []),
                issues_detected=obs_data.get("issues_detected", []),
                applied_ops=obs_data.get("applied_ops", []),
                messages=obs_data.get("messages", []),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CleaningState:
        return CleaningState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "unknown"),
            score=payload.get("score", 0.0),
            remaining_issues=payload.get("remaining_issues", 0),
            invalid_ops=payload.get("invalid_ops", 0),
        )
