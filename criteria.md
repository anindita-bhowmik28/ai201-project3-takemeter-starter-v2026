# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for the rebuilt classifier. The
numeric thresholds are unchanged from the original pre-training criteria; only
the label name and exact integer test-set interpretation were updated for the
four-label, 250-row dataset.

## Measurement rules

Unless a criterion says otherwise, measure it on the 38-row held-out test split
created by the notebook's stratified 70/15/15 split with seed 42. Do not use the
training or validation rows when deciding whether a target was met. For the
three-seed stability check, apply the same target separately to each seed; a
target is stable only if all three runs meet it.

## 1. Overall held-out accuracy is at least 0.68.

**Why this target:** `reported_event` is the largest class at 97 of 250 rows
(38.8%), so a classifier that always predicts the largest class would be only
about 0.39 accurate. On the 38-row test split, this target requires at least 26
correct predictions (26/38 = 0.684), a clear improvement over that
project-specific majority baseline while allowing for genuine boundary cases.

## 2. Macro F1 is at least 0.62 on the held-out set.

**Why this target:** The rebuilt labels are uneven at 97 `reported_event`, 66
`question_request`, 52 `perspective_explainer`, and 35 `project_showcase` rows.
Macro F1 gives the 35-row showcase class the same weight as the largest class,
so a 0.62 target prevents a majority-class strategy from looking successful.

## 3. Every label has an F1 of at least 0.50.

**Why this target:** The smallest final class is `project_showcase`, with 35
examples and about 5 in a 15% test split. Its score will be the noisiest, so
0.50 is looser than the overall accuracy target, but it still requires the
model to recognize every title intent instead of relying on larger classes.

## 4. The `reported_event` label has an F1 of at least 0.60.

**Why this target:** `reported_event` is the central and largest class: it
separates checkable developments such as outages, acquisitions, findings, and
deaths from perspectives, requests, and showcases. Because it has 97 examples,
an F1 below 0.60 would show that the model had not learned even the
best-supported title-level intent.

## 5. At least 70% of the most-confident third of predictions are correct on the held-out set.

**How to test it:** For every held-out row, apply softmax to the model logits and
use the winning label's probability as confidence. Rank all 38 rows by that
value, take the 12 highest-confidence rows (the notebook's integer top third),
and calculate plain accuracy on those rows. The criterion is met when at least
9 of the 12 are correct (0.75; 8/12 is only 0.667). This is the same
calculation returned by the notebook as
`accuracy_most_confident_third`; ties keep their existing test-set order.

**Why this target:** TakeMeter would be most useful as a triage tool that sends
only its surest predictions forward for automatic handling. The 38-row test
split makes the top third a concrete 12-post review queue, and requiring 9
correct posts asks that this selective queue decisively beat the 38.8% accuracy
of always choosing this dataset's 97-row majority class.

---

These tests are specific, reproducible, and tied to this project's actual label
distribution and taxonomy rather than to a generic idea of good performance.
