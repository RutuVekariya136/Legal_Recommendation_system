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
        v, t = r["splits"]["validation"], r["splits"]["test"]
        vc, tc = v["correlation"], t["correlation"]
        vcl, tcl = v.get("classification"), t.get("classification")
        row = {
            "config": r["config"], "kind": r["kind"],
            "validation": {"spearman": vc["spearman"], "pearson": vc["pearson"], "mse": vc["mse"]},
            "test": {"spearman": tc["spearman"], "pearson": tc["pearson"], "mse": tc["mse"]},
        }
        if vcl:
            row["validation"]["classification"] = {
                "accuracy": vcl["accuracy"], "precision": vcl["precision"],
                "recall": vcl["recall"], "f1": vcl["f1"], "roc_auc": vcl["roc_auc"],
                "pred_threshold": vcl["pred_threshold"],
            }
        if tcl:
            row["test"]["classification"] = {
                "accuracy": tcl["accuracy"], "precision": tcl["precision"],
                "recall": tcl["recall"], "f1": tcl["f1"], "roc_auc": tcl["roc_auc"],
                "pred_threshold": tcl["pred_threshold"],
            }
        rows.append(row)
    return {
        "generated_at": report.get("generated_at"),
        "pairs": report.get("pairs"),
        "protocol": report.get("protocol"),
        "rows": rows,
        "diagnosis": report.get("diagnosis"),
    }


@router.get("/parameters")
def parameter_impact() -> dict:
    """Measured effect of supplying each case parameter on the accuracy of the
    true match (produced by scripts/parameter_ablation.py)."""
    path = data_processed_dir() / "parameter_ablation.json"
    if not path.is_file():
        raise HTTPException(404, "no parameter-impact report yet - run `python scripts/parameter_ablation.py`")
    return json.loads(path.read_text(encoding="utf-8"))
