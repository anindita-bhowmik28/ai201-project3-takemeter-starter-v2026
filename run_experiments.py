"""Run reproducible TakeMeter experiments and save auditable JSON artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
LABELS = [
    "reported_event",
    "perspective_explainer",
    "project_showcase",
    "question_request",
]
BASE_MODEL = "distilbert-base-uncased"
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 128
SEEDS = [42, 7, 2024]


def split_dataset(df: pd.DataFrame, seed: int):
    rest, test = train_test_split(
        df, test_size=0.15, random_state=seed, stratify=df["label"]
    )
    train, val = train_test_split(
        rest,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=rest["label"],
    )
    return tuple(part.reset_index(drop=True) for part in (train, val, test))


def to_dataset(frame: pd.DataFrame, tokenizer) -> Dataset:
    lookup = {label: index for index, label in enumerate(LABELS)}
    data = frame[["text", "label"]].copy()
    data["labels"] = data.pop("label").map(lookup)
    dataset = Dataset.from_pandas(data[["text", "labels"]], preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        ),
        batched=True,
    )


def confusion_counts(gold, predicted):
    matrix = {actual: {guess: 0 for guess in LABELS} for actual in LABELS}
    for actual, guess in zip(gold, predicted):
        matrix[LABELS[int(actual)]][LABELS[int(guess)]] += 1
    return matrix


def evaluate(trainer, dataset, test_frame, seed):
    output = trainer.predict(dataset)
    predicted = np.argmax(output.predictions, axis=-1)
    gold = np.asarray(output.label_ids)
    exp = np.exp(output.predictions - output.predictions.max(axis=-1, keepdims=True))
    confidence = (exp / exp.sum(axis=-1, keepdims=True)).max(axis=-1)
    right = gold == predicted
    ranked = np.argsort(confidence, kind="stable")
    third = max(1, len(gold) // 3)
    least, most = ranked[:third], ranked[-third:]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=range(len(LABELS)), zero_division=0
    )
    return {
        "seed": seed,
        "n_test": int(len(gold)),
        "test_distribution": dict(Counter(test_frame["label"])),
        "accuracy": float(accuracy_score(gold, predicted)),
        "f1_macro": float(f1.mean()),
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion": confusion_counts(gold, predicted),
        "confidence": {
            "definition": "winning softmax probability; highest-ranked third",
            "mean_when_right": float(confidence[right].mean()) if right.any() else None,
            "mean_when_wrong": float(confidence[~right].mean()) if (~right).any() else None,
            "accuracy_most_confident_third": float(right[most].mean()),
            "accuracy_least_confident_third": float(right[least].mean()),
            "n_most_confident_third": int(third),
        },
    }


def run_one(df, seed, epochs, quiet=False):
    set_seed(seed)
    train, val, test = split_dataset(df, seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    train_dataset = to_dataset(train, tokenizer)
    val_dataset = to_dataset(val, tokenizer)
    test_dataset = to_dataset(test, tokenizer)
    args = TrainingArguments(
        output_dir=str(ROOT / "takemeter_out"),
        num_train_epochs=epochs,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        seed=seed,
        report_to="none",
        disable_tqdm=quiet,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    result = evaluate(trainer, test_dataset, test, seed)
    del trainer, model, tokenizer, train_dataset, val_dataset, test_dataset
    gc.collect()
    return result, test


def provenance(data_path: Path, df: pd.DataFrame, epochs: int):
    return {
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data_path.name,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "row_count": len(df),
        "label_distribution": dict(Counter(df["label"])),
        "labels": LABELS,
        "base_model": BASE_MODEL,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "torch_version": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after", "stretch"], required=True)
    args = parser.parse_args()
    data_path = (
        ROOT / "data" / "labels_part2_before.csv"
        if args.mode == "before"
        else ROOT / "labels.csv"
    )
    df = pd.read_csv(data_path)
    found = set(df["label"])
    if found != set(LABELS):
        raise SystemExit(f"Expected labels {LABELS}; found {sorted(found)}")

    epochs = 4 if args.mode == "stretch" else 3
    seeds = SEEDS
    metadata = provenance(data_path, df, epochs)
    print(json.dumps(metadata, indent=2))
    results = []
    seed_42_test = None
    for seed in seeds:
        print(f"\nRunning {args.mode}: seed={seed}, epochs={epochs}")
        result, test = run_one(df, seed, epochs, quiet=False)
        result["provenance"] = metadata
        results.append(result)
        if seed == 42:
            seed_42_test = test
        print(f"accuracy={result['accuracy']:.3f} macro_f1={result['f1_macro']:.3f}")

    if args.mode == "before":
        output = ROOT / "results_three_seeds_before.json"
        (ROOT / "results.json").write_text(json.dumps(results[0], indent=2) + "\n")
        seed_42_test.to_csv(ROOT / "test_split.csv", index=False)
    elif args.mode == "after":
        output = ROOT / "results_three_seeds_after.json"
    else:
        output = ROOT / "results_stretch_epochs4.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {output.name}")


if __name__ == "__main__":
    main()
