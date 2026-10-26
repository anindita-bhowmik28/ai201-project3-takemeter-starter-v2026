"""Fail fast if the rebuilt submission loses required evidence or stretch work."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABELS = {
    "reported_event",
    "perspective_explainer",
    "project_showcase",
    "question_request",
}
BEFORE_COUNTS = {
    "reported_event": 97,
    "perspective_explainer": 52,
    "project_showcase": 35,
    "question_request": 66,
}
AFTER_COUNTS = {**BEFORE_COUNTS, "project_showcase": 65}


def read_csv(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    labels_path = ROOT / "labels.csv"
    rows = read_csv("labels.csv")
    assert len(rows) == 280
    assert set(rows[0]) == {"text", "label", "note"}
    assert len({row["text"] for row in rows}) == 280
    assert set(row["label"] for row in rows) == LABELS
    assert Counter(row["label"] for row in rows) == AFTER_COUNTS
    assert max(AFTER_COUNTS.values()) / len(rows) < 0.70

    before_rows = read_csv("data/labels_part2_before.csv")
    additions = read_csv("data/added_project_showcases.csv")
    assert len(before_rows) == 250
    assert Counter(row["label"] for row in before_rows) == BEFORE_COUNTS
    assert len(additions) == 30
    assert all(row["label"] == "project_showcase" for row in additions)
    assert all("objectID" in row["note"] for row in additions)
    assert len(rows) - len(before_rows) == 30

    digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    before = json.loads((ROOT / "results_three_seeds_before.json").read_text())
    after = json.loads((ROOT / "results_three_seeds_after.json").read_text())
    stretch = json.loads((ROOT / "results_stretch_epochs4.json").read_text())
    seed_42 = json.loads((ROOT / "results.json").read_text())
    assert [run["seed"] for run in before] == [42, 7, 2024]
    assert [run["seed"] for run in after] == [42, 7, 2024]
    assert [run["seed"] for run in stretch] == [42, 7, 2024]
    assert seed_42 == before[0]
    for run in before:
        meta = run["provenance"]
        assert meta["row_count"] == 250
        assert meta["epochs"] == 3
        assert set(meta["labels"]) == LABELS
        assert run["n_test"] == 38
        assert run["confidence"]["n_most_confident_third"] == 12
    for run in after:
        meta = run["provenance"]
        assert meta["dataset_sha256"] == digest
        assert meta["row_count"] == 280
        assert meta["epochs"] == 3
        assert set(meta["labels"]) == LABELS
        assert run["n_test"] == 42
        assert run["confidence"]["n_most_confident_third"] == 14
        assert "accuracy_least_confident_third" in run["confidence"]
    for run in stretch:
        stretch_meta = run["provenance"]
        assert stretch_meta["dataset_sha256"] == digest
        assert stretch_meta["epochs"] == 4
        for key in ("learning_rate", "batch_size", "max_length", "base_model"):
            assert stretch_meta[key] == after[0]["provenance"][key]

    notebook = json.loads((ROOT / "takemeter.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert all(label in source for label in LABELS)
    assert 'RUN_LABEL = "after"' in source
    assert all(not cell.get("outputs") for cell in notebook["cells"])

    readme = (ROOT / "README.md").read_text()
    assert "<!--" not in readme
    for required in (
        "results_three_seeds_after.json",
        "results_stretch_epochs4.json",
        "What's Still Broken",
        "Agreement Report",
        "Confidence check",
        "Second improvement",
        "Worst-label error write-up",
        "Part 2 plan — declared before experiments",
        "Device beside the seeds",
    ):
        assert required in readme

    print("Submission validation passed")
    print(f"  before rows: {len(before_rows)} ({dict(BEFORE_COUNTS)})")
    print(f"  after rows: {len(rows)} ({dict(AFTER_COUNTS)})")
    print(f"  dataset sha256: {digest}")
    print("  before, after, and stretch seeds: 42, 7, 2024")


if __name__ == "__main__":
    main()
