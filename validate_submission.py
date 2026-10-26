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
EXPECTED_COUNTS = {
    "reported_event": 97,
    "perspective_explainer": 52,
    "project_showcase": 35,
    "question_request": 66,
}


def read_csv(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    labels_path = ROOT / "labels.csv"
    rows = read_csv("labels.csv")
    assert len(rows) == 250
    assert set(rows[0]) == {"text", "label", "note"}
    assert len({row["text"] for row in rows}) == 250
    assert set(row["label"] for row in rows) == LABELS
    assert Counter(row["label"] for row in rows) == EXPECTED_COUNTS
    assert max(EXPECTED_COUNTS.values()) / len(rows) < 0.70

    originals = read_csv("data/labels_original_200.csv")
    additions = read_csv("data/added_question_requests.csv")
    assert len(originals) == 200
    assert len(additions) == 61
    assert all(row["label"] == "question_request" for row in additions)
    assert all("objectID" in row["note"] for row in additions)
    assert len(rows) - len(originals) == 50

    digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    after = json.loads((ROOT / "results_three_seeds_after.json").read_text())
    stretch = json.loads((ROOT / "results_stretch_epochs4.json").read_text())
    seed_42 = json.loads((ROOT / "results.json").read_text())
    assert [run["seed"] for run in after] == [42, 7, 2024]
    assert len(stretch) == 1 and stretch[0]["seed"] == 42
    assert seed_42 == after[0]
    for run in after:
        meta = run["provenance"]
        assert meta["dataset_sha256"] == digest
        assert meta["row_count"] == 250
        assert meta["epochs"] == 3
        assert set(meta["labels"]) == LABELS
        assert run["n_test"] == 38
        assert run["confidence"]["n_most_confident_third"] == 12
    stretch_meta = stretch[0]["provenance"]
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
        "What's still broken",
        "Fourth label",
        "Fifty additional examples",
        "one hyperparameter changed",
    ):
        assert required in readme

    print("Submission validation passed")
    print(f"  rows: {len(rows)} ({dict(EXPECTED_COUNTS)})")
    print(f"  dataset sha256: {digest}")
    print("  after seeds: 42, 7, 2024; stretch: seed 42 at 4 epochs")


if __name__ == "__main__":
    main()
