# Data Cleaning Env (OpenEnv)

Real-world tabular data-cleaning environment with three graded tasks (easy → medium → hard). Agents interact via the standard OpenEnv `reset / step / state` API and receive shaped rewards for incremental progress toward clean data.

## Tasks
- **easy:** Normalize customer names/cities, fill missing tiers.
- **medium:** Standardize order dates to ISO, normalize currency to numeric, remove duplicate order_ids, fill missing status.
- **hard:** Standardize dates, validate/lowercase emails, clip sentiment scores to [-1, 1], fix common typos in summaries.

Each task has a deterministic grader that returns per-criterion scores and an aggregate in [0.0, 1.0].

## Action Space
`CleaningAction`
- `operation`: one of `normalize_text`, `fill_missing`, `standardize_date`, `dedupe`, `split_column`, `merge_columns`, `clip_outliers`, `map_values`
- `target_columns`: list of columns to apply the op
- `parameters`: op-specific parameters (e.g., `{ "case": "lower" }`, `{ "value": "basic" }`, `{ "min": -1, "max": 1 }`)
- `notes`: optional freeform rationale

## Observation Space
`CleaningObservation`
- `preview_rows`: sample of the table after the step
- `schema`: column names and inferred dtypes
- `issues_detected`: list of remaining failing checks
- `applied_ops`: recent operations/messages
- `messages`: running log
- `done`, `reward`: standard OpenEnv fields

## Reward Shaping
- +Δ in aggregate grader score since last step
- -0.2 for invalid/no-op, -0.3 on exceptions
- Episode ends when aggregate ≥ 0.95 or after 12 steps

## Files
- `openenv.yaml` — metadata and task list
- `models.py` — typed Action/Observation/State
- `server/environment.py` — task payloads, graders, reward shaping
- `server/app.py` — FastAPI app with `/tasks`, `/baseline` endpoints
- `client.py` — EnvClient wrapper
- `baseline.py` — deterministic baseline; uses OpenAI if key is set, otherwise heuristics
- `Dockerfile` — container entrypoint (uvicorn)

## Usage
```bash
pip install -r ../requirements.txt  # ensure openenv-core & fastapi/uvicorn
cd data-cleaning-env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Example (sync client):
```python
from client import DataCleaningEnv
from models import CleaningAction

with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task_id="easy")
    obs = env.step(CleaningAction(operation="normalize_text", target_columns=["name"], parameters={"case": "lower"}))
    print(obs.reward, obs.issues_detected)
```

Run baseline:
```bash
python -m data-cleaning-env.baseline
# or GET /baseline once the server is running
```

## Deployment
- Build locally: `docker build -t data-cleaning-env .`
- Run: `docker run -p 8000:8000 data-cleaning-env`
- Deploy to HF Spaces: set entrypoint to `server.app:app` and tag with `openenv`

## Validation
- `openenv validate` (when available) should pass typed models and endpoints.
- `/tasks` lists task specs; `/baseline` returns baseline scores.
