"""Deterministic baseline runner.

Tries to use OpenAI if OPENAI_API_KEY is set; otherwise falls back to heuristic rules.
Returns per-task scores and mean score.
"""

import os
from typing import Dict, List

from .server.environment import TASKS, grade
from .server.environment import DataCleaningEnvironment
from .models import CleaningAction

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _heuristic_policy(task_id: str, env: DataCleaningEnvironment):
    """Simple non-LLM policy to achieve reasonable scores deterministically."""
    if task_id == "easy":
        env._op_normalize_text(["name"], {"case": "lower"})
        env._op_normalize_text(["city"], {"case": "title"})
        env._op_fill_missing(["tier"], {"value": "basic"})
    elif task_id == "medium":
        env._op_standardize_date(["date"])
        env._op_map_values(["amount_usd"], {"mapping": {}})
        env._op_dedupe(["order_id"])
        env._op_fill_missing(["status"], {"value": "pending"})
        # normalize currency to float by clip helper with wide bounds
        env._op_clip(["amount_usd"], {"min": -1e9, "max": 1e9})
    else:
        env._op_standardize_date(["created"])
        env._op_normalize_text(["email"], {"case": "lower"})
        env._op_clip(["sentiment"], {"min": -1.0, "max": 1.0})
        env._op_map_values(["summary"], {"mapping": {"recieve": "receive", "recieveed": "received", "cant": "can't", "custmer": "customer"}})


def _llm_policy(task_id: str, env: DataCleaningEnvironment):
    """Illustrative LLM policy using OpenAI; deterministic prompt, greedy decoding."""
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        _heuristic_policy(task_id, env)
        return

    client = OpenAI()
    system_prompt = (
        "You are a data-cleaning agent. Given a task id, return JSON with a list of operations to clean the data. "
        "Use operations: normalize_text, fill_missing, standardize_date, dedupe, clip_outliers, map_values."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Task: {task_id}. Columns: {list(env._rows[0].keys())}. Return 3-5 actions.",
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=300,
    )
    text = resp.choices[0].message.content
    # Very simple parser: look for known operations in text and apply heuristics
    # to keep deterministic without eval.
    ops = []
    for op in ["standardize_date", "normalize_text", "fill_missing", "dedupe", "clip_outliers", "map_values"]:
        if op in text:
            ops.append(op)
    if not ops:
        _heuristic_policy(task_id, env)
        return
    for op in ops:
        if op == "standardize_date":
            env._op_standardize_date([c for c in env._rows[0].keys() if "date" in c or "created" in c])
        elif op == "normalize_text":
            env._op_normalize_text([c for c in env._rows[0].keys() if isinstance(env._rows[0][c], str)], {"case": "lower"})
        elif op == "fill_missing":
            env._op_fill_missing(list(env._rows[0].keys()), {"value": "unknown"})
        elif op == "dedupe":
            env._op_dedupe([list(env._rows[0].keys())[0]])
        elif op == "clip_outliers":
            env._op_clip([c for c in env._rows[0].keys()], {"min": -1e6, "max": 1e6})
        elif op == "map_values":
            env._op_map_values([c for c in env._rows[0].keys()], {"mapping": {}})


def run_baseline() -> Dict[str, float]:
    results: Dict[str, float] = {}
    env = DataCleaningEnvironment()
    for task_id in TASKS.keys():
        env.reset(task_id=task_id)
        _llm_policy(task_id, env)
        scores = grade(task_id, env._rows)
        results[task_id] = scores["aggregate"]
    results["mean"] = sum(results.values()) / len(TASKS)
    return results


if __name__ == "__main__":
    print(run_baseline())
