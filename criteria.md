# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written before any model was trained.

## 1. Overall held-out accuracy is at least 0.68.

**Why this target:** A model that is right on roughly two out of every three posts is useful in a noisy discussion setting, while still leaving room for edge cases and label ambiguity. With three labels, a random guesser would land near 0.33, so this is a meaningful improvement rather than a loose bar.

## 2. Macro F1 is at least 0.62 on the held-out set.

**Why this target:** Macro F1 is the right summary here because the dataset is not perfectly balanced and the smaller labels matter as much as the larger ones. This target keeps the model from being judged only on the most common class while still being realistic for a small fine-tuning run.

## 3. Every label has an F1 of at least 0.50.

**Why this target:** My smallest class is `reaction`, which is only about 17 examples in the final file. That label will be the noisiest, so the target is looser than the overall accuracy bar, but it still forces the model to learn more than just the majority classes.

## 4. The `analysis` label has an F1 of at least 0.60.

**Why this target:** `analysis` is the most important class for the task because it is the one that distinguishes factual argument from opinion. It also has the clearest boundary against `hot_take`, which makes it the label most worth measuring explicitly.

## 5. At least 70% of high-confidence predictions are correct on the held-out set.

**Why this target:** A usable classifier should be more reliable when it is confident. This target checks that model confidence tracks correctness rather than simply reporting a prediction without useful calibration.

---

These were written before training. They are testable, specific, and tied to the actual label distribution and taxonomy in this project rather than to a generic idea of "good performance."
