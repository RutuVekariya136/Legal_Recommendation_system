"""Serve the cached evaluation report (produced by scripts/evaluate.py)."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app.utils.paths import data_processed_dir

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


def _load_report() -> dict:
    path = data_processed_dir() / "evaluation_report.json"
    if not path.is_file():
        raise HTTPException(
            404,
            "no evaluation report yet - run `python scripts/evaluate.py` to generate "
            "data/processed/evaluation_report.json",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("")
def evaluation_report() -> dict:
    return _load_report()


@router.get("/summary")
def evaluation_summary() -> dict:
    """A compact table for the dashboard: config -> {validation, test} Spearman/MSE."""
    report = _load_report()
    rows = []
    for r in report.get("results", []):
        v = r["splits"]["validation"]["correlation"]
        t = r["splits"]["test"]["correlation"]
        rows.append({
            "config": r["config"], "kind": r["kind"],
            "validation": {"spearman": v["spearman"], "pearson": v["pearson"], "mse": v["mse"]},
            "test": {"spearman": t["spearman"], "pearson": t["pearson"], "mse": t["mse"]},
        })
    return {
        "generated_at": report.get("generated_at"),
        "pairs": report.get("pairs"),
        "protocol": report.get("protocol"),
        "rows": rows,
        "diagnosis": report.get("diagnosis"),
    }
