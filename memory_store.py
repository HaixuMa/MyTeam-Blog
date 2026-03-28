from __future__ import annotations

import json
from pathlib import Path


def update_quality_memory(*, data_dir: Path, issues: list[dict]) -> None:
    mem_dir = (data_dir / "memory").resolve()
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = (mem_dir / "quality_feedback.json").resolve()

    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
    else:
        obj = {}

    counts = obj.get("issue_counts")
    if not isinstance(counts, dict):
        counts = {}
        obj["issue_counts"] = counts

    for it in issues:
        if not isinstance(it, dict):
            continue
        cat = str(it.get("category") or "unknown")
        sev = str(it.get("severity") or "unknown")
        key = f"{cat}:{sev}"
        counts[key] = int(counts.get(key, 0)) + 1

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
