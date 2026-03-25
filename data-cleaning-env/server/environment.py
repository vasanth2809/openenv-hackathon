import copy
import datetime as dt
import math
import random
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from ..models import CleaningAction, CleaningObservation, CleaningState

# -----------------------------------------------------------------------------
# Task definitions (small tabular payloads) and deterministic graders
# -----------------------------------------------------------------------------


def _strip_lower(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().split()).lower()
    return value


def _parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    value = value.strip()
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return dt.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _is_valid_email(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip()
    return "@" in value and "." in value.split("@")[-1]


TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Normalize customer names/cities and fill missing tiers",
        "data": [
            {"name": " Alice ", "city": "new york", "tier": None},
            {"name": "bob", "city": " San Francisco ", "tier": "pro"},
            {"name": "alice", "city": "NEW YORK", "tier": ""},
        ],
        "required": {
            "name": "normalize_lower",
            "city": "title_case",
            "tier": {"fill": "basic"},
        },
    },
    "medium": {
        "description": "Standardize dates, normalize currency, remove duplicates",
        "data": [
            {"order_id": "A-1", "date": "2024/01/05", "amount_usd": "$120.50", "status": None},
            {"order_id": "A-1", "date": "05-01-2024", "amount_usd": "$120.50", "status": ""},
            {"order_id": "B-2", "date": "2024-02-12", "amount_usd": "99", "status": "paid"},
        ],
        "required": {
            "date": "iso",
            "amount_usd": "float",
            "dedupe_on": ["order_id"],
            "status": {"fill": "pending"},
        },
    },
    "hard": {
        "description": "Multi-issue cleanup: dates, emails, sentiment clipping, typo fixes",
        "data": [
            {
                "ticket_id": "T-1",
                "created": "1/3/2024",
                "email": "User@Example.Com ",
                "summary": "User cant recieve email",
                "sentiment": 1.8,
            },
            {
                "ticket_id": "T-2",
                "created": "2024-03-14",
                "email": "bad-email",
                "summary": "Custmer recieveed late response",
                "sentiment": -2.5,
            },
        ],
        "required": {
            "created": "iso",
            "email": "valid_lower",
            "sentiment": {"clip": [-1.0, 1.0]},
            "summary": {"typo_map": {"recieve": "receive", "recieveed": "received", "cant": "can't", "custmer": "customer"}},
        },
    },
}


def grade(task_id: str, rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return per-criterion scores and aggregate for the given task."""

    task = TASKS[task_id]
    required = task["required"]
    scores: Dict[str, float] = {}

    if task_id == "easy":
        name_ok = all(_strip_lower(r.get("name")) == _strip_lower(rows[0].get("name")) for r in rows)
        city_ok = all(r.get("city", "").strip().title() == r.get("city", "").strip().title() for r in rows)
        tier_default = required["tier"]["fill"]
        tier_ok = all((r.get("tier") or tier_default) == tier_default or (r.get("tier") or "") in {"basic", "pro"} for r in rows)
        scores.update({
            "name_normalized": 1.0 if name_ok else 0.0,
            "city_normalized": 1.0 if city_ok else 0.0,
            "tier_filled": 1.0 if tier_ok else 0.0,
        })

    elif task_id == "medium":
        dates = [_parse_date(r.get("date")) for r in rows]
        date_ok = all(d is not None for d in dates)
        amounts = [_to_float(r.get("amount_usd")) for r in rows]
        amt_ok = all(a is not None for a in amounts)
        deduped = len(rows) == len({r.get("order_id") for r in rows})
        status_default = required["status"]["fill"]
        status_ok = all((r.get("status") or status_default) != "" for r in rows)
        scores.update({
            "date_iso": 1.0 if date_ok else 0.0,
            "amount_numeric": 1.0 if amt_ok else 0.0,
            "deduped": 1.0 if deduped else 0.0,
            "status_filled": 1.0 if status_ok else 0.0,
        })

    elif task_id == "hard":
        date_ok = all(_parse_date(r.get("created")) is not None for r in rows)
        email_ok = all(_is_valid_email((r.get("email") or "").lower().strip()) for r in rows)
        sentiment_ok = all(-1.0 <= (r.get("sentiment") or 0) <= 1.0 for r in rows)
        typo_map = required["summary"].get("typo_map", {})
        summary_ok = True
        for r in rows:
            s = (r.get("summary") or "").lower()
            for wrong, correct in typo_map.items():
                if wrong in s and correct not in s:
                    summary_ok = False
                    break
        scores.update({
            "date_iso": 1.0 if date_ok else 0.0,
            "email_valid": 1.0 if email_ok else 0.0,
            "sentiment_clipped": 1.0 if sentiment_ok else 0.0,
            "summary_clean": 1.0 if summary_ok else 0.0,
        })

    scores["aggregate"] = sum(scores.values()) / max(len(scores), 1)
    return scores


# -----------------------------------------------------------------------------
# Environment implementation
# -----------------------------------------------------------------------------


class DataCleaningEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 12

    def __init__(self):
        self._task_id: str = "easy"
        self._rows: List[Dict[str, Any]] = []
        self._state = CleaningState(task_id="easy")
        self._last_scores: Dict[str, float] = {}
        self._messages: List[str] = []

    # ------------------------- core interface ------------------------------
    def reset(self, seed: Optional[int] = None, task_id: Optional[str] = None, **kwargs) -> CleaningObservation:
        if seed is not None:
            random.seed(seed)
        self._task_id = task_id or random.choice(list(TASKS.keys()))
        self._rows = copy.deepcopy(TASKS[self._task_id]["data"])
        self._state = CleaningState(task_id=self._task_id, step_count=0, score=0.0, remaining_issues=0, invalid_ops=0)
        self._messages = [f"Task: {TASKS[self._task_id]['description']}"]
        self._last_scores = grade(self._task_id, self._rows)
        self._state.score = self._last_scores["aggregate"]
        return self._make_observation()

    def step(self, action: CleaningAction, timeout_s: Optional[float] = None, **kwargs) -> CleaningObservation:
        self._state.step_count += 1
        penalty = 0.0
        applied = False

        try:
            applied = self._apply_action(action)
        except Exception as exc:  # noqa: BLE001
            penalty -= 0.3
            self._state.invalid_ops += 1
            self._messages.append(f"Error applying action: {exc}")

        if not applied:
            penalty -= 0.2
            self._state.invalid_ops += 1
            self._messages.append("Action was invalid or no-op.")

        new_scores = grade(self._task_id, self._rows)
        delta = new_scores["aggregate"] - self._last_scores.get("aggregate", 0.0)
        shaped_reward = delta + penalty
        self._last_scores = new_scores
        self._state.score = new_scores["aggregate"]

        done = self._state.step_count >= self.MAX_STEPS or new_scores["aggregate"] >= 0.95
        obs = self._make_observation(reward=shaped_reward, done=done)
        return obs

    @property
    def state(self) -> CleaningState:
        return self._state

    # ------------------------- helpers -------------------------------------
    def _make_observation(self, reward: Optional[float] = None, done: bool = False) -> CleaningObservation:
        preview = copy.deepcopy(self._rows[:5])
        schema = []
        if self._rows:
            for key in self._rows[0].keys():
                dtype = type(self._rows[0][key]).__name__
                schema.append({"name": key, "dtype": dtype})
        issues = self._issues_summary()
        applied_ops = self._messages[-3:]
        return CleaningObservation(
            done=done,
            reward=reward,
            preview_rows=preview,
            schema=schema,
            issues_detected=issues,
            applied_ops=applied_ops,
            messages=list(self._messages),
        )

    def _issues_summary(self) -> List[str]:
        issues = []
        scores = grade(self._task_id, self._rows)
        for k, v in scores.items():
            if k == "aggregate":
                continue
            if v < 1.0:
                issues.append(f"{k}: {v:.2f}")
        if not issues:
            issues.append("all_clean")
        return issues

    def _apply_action(self, action: CleaningAction) -> bool:
        op = action.operation
        cols = action.target_columns
        params = action.parameters or {}

        if op == "normalize_text":
            return self._op_normalize_text(cols, params)
        if op == "fill_missing":
            return self._op_fill_missing(cols, params)
        if op == "standardize_date":
            return self._op_standardize_date(cols)
        if op == "dedupe":
            return self._op_dedupe(cols)
        if op == "split_column":
            return self._op_split_column(cols, params)
        if op == "merge_columns":
            return self._op_merge_columns(cols, params)
        if op == "clip_outliers":
            return self._op_clip(cols, params)
        if op == "map_values":
            return self._op_map_values(cols, params)
        return False

    # ------------------------- operations ----------------------------------
    def _op_normalize_text(self, cols: List[str], params: Dict[str, Any]) -> bool:
        if not cols:
            return False
        mode = params.get("case", "lower")
        changed = False
        for row in self._rows:
            for col in cols:
                if col not in row or row[col] is None:
                    continue
                if isinstance(row[col], str):
                    original = row[col]
                    val = " ".join(row[col].strip().split())
                    if mode == "lower":
                        val = val.lower()
                    elif mode == "title":
                        val = val.title()
                    if val != original:
                        row[col] = val
                        changed = True
        if changed:
            self._messages.append(f"normalize_text on {cols} ({mode})")
        return changed

    def _op_fill_missing(self, cols: List[str], params: Dict[str, Any]) -> bool:
        if not cols:
            return False
        default = params.get("value")
        changed = False
        for row in self._rows:
            for col in cols:
                if col not in row:
                    continue
                if row[col] in (None, ""):
                    row[col] = default
                    changed = True
        if changed:
            self._messages.append(f"fill_missing on {cols} -> {default}")
        return changed

    def _op_standardize_date(self, cols: List[str]) -> bool:
        changed = False
        for row in self._rows:
            for col in cols:
                if col not in row:
                    continue
                parsed = _parse_date(row[col])
                if parsed and row[col] != parsed:
                    row[col] = parsed
                    changed = True
        if changed:
            self._messages.append(f"standardize_date on {cols}")
        return changed

    def _op_dedupe(self, cols: List[str]) -> bool:
        if not cols:
            return False
        seen = set()
        deduped = []
        changed = False
        for row in self._rows:
            key = tuple(row.get(c) for c in cols)
            if key in seen:
                changed = True
                continue
            seen.add(key)
            deduped.append(row)
        if changed:
            self._rows = deduped
            self._messages.append(f"dedupe on {cols}")
        return changed

    def _op_split_column(self, cols: List[str], params: Dict[str, Any]) -> bool:
        if not cols:
            return False
        delimiter = params.get("delimiter", " ")
        new_cols = params.get("new_columns") or [f"{cols[0]}_part1", f"{cols[0]}_part2"]
        changed = False
        for row in self._rows:
            src = cols[0]
            if src not in row or not isinstance(row[src], str):
                continue
            parts = row[src].split(delimiter)
            if len(parts) >= 2:
                row[new_cols[0]] = parts[0]
                row[new_cols[1]] = delimiter.join(parts[1:])
                changed = True
        if changed:
            self._messages.append(f"split_column {cols[0]} -> {new_cols}")
        return changed

    def _op_merge_columns(self, cols: List[str], params: Dict[str, Any]) -> bool:
        if len(cols) < 2:
            return False
        dest = params.get("dest") or "merged"
        sep = params.get("separator", " ")
        changed = False
        for row in self._rows:
            vals = [str(row.get(c, "")) for c in cols]
            merged = sep.join(vals).strip()
            row[dest] = merged
            changed = True
        if changed:
            self._messages.append(f"merge_columns {cols} -> {dest}")
        return changed

    def _op_clip(self, cols: List[str], params: Dict[str, Any]) -> bool:
        if not cols:
            return False
        lo, hi = params.get("min", -math.inf), params.get("max", math.inf)
        changed = False
        for row in self._rows:
            for col in cols:
                val = _to_float(row.get(col))
                if val is None:
                    continue
                clipped = min(max(val, lo), hi)
                if clipped != val:
                    row[col] = clipped
                    changed = True
        if changed:
            self._messages.append(f"clip_outliers on {cols} to [{lo}, {hi}]")
        return changed

    def _op_map_values(self, cols: List[str], params: Dict[str, Any]) -> bool:
        if not cols:
            return False
        mapping: Dict[str, str] = params.get("mapping", {})
        changed = False
        for row in self._rows:
            for col in cols:
                if col not in row:
                    continue
                val = row[col]
                if isinstance(val, str):
                    key = val.lower().strip()
                    if key in mapping:
                        row[col] = mapping[key]
                        changed = True
        if changed:
            self._messages.append(f"map_values on {cols}")
        return changed
