# TakeMeter

TakeMeter classifies Hacker News titles by the communicative purpose visible in
the title. The four labels are unchanged from the Part 1 submission.

## Label taxonomy

### `reported_event`

A title that states a concrete event, status, finding, outage, policy change,
acquisition, death, or other checkable development.

- “Stephen Hawking has died”
- “SpaceX successfully launches two humans into orbit”

### `perspective_explainer`

A title that presents a personal view, recommendation, argument, reflection,
interpretation, or explanation rather than primarily announcing a project or
requesting answers.

- “Google Search Is Dying”
- “Why I’m done with Chrome”

### `project_showcase`

A title whose main purpose is to present, demonstrate, launch, or name a
specific tool, model, product, creative work, or Show HN project.

- “Show HN: I made an open-source laptop from scratch”
- “Riffusion – Stable Diffusion fine-tuned to generate music”

### `question_request`

A title that explicitly asks the community for information, advice,
recommendations, help, or action, including Ask HN and direct help requests.

- “Ask HN: Is S3 down?”
- “Ask HN: How do people self host email and maintain deliverability?”

### Decision rules

The hardest boundary is `reported_event` versus `perspective_explainer`. A
checkable development that could be verified from another source is a report;
a recommendation, first-person reflection, judgment, “why/how” explanation, or
broad interpretation is a perspective/explainer. `project_showcase` takes
priority when presenting a named artifact is the title's main purpose, and
`question_request` takes priority when the title explicitly solicits an answer
or action.

## Dataset

Part 2 began with the exact 250-row Part 1 dataset, frozen in
[`data/labels_part2_before.csv`](data/labels_part2_before.csv):

| Label | Before rows |
|---|---:|
| `reported_event` | 97 |
| `question_request` | 66 |
| `perspective_explainer` | 52 |
| `project_showcase` | 35 |
| **Total** | **250** |

The one required improvement added 30 traceable Show HN examples to the
thinnest and worst-performing label, `project_showcase`. The current
[`labels.csv`](labels.csv) therefore contains 280 unique rows:

| Label | After rows |
|---|---:|
| `reported_event` | 97 |
| `question_request` | 66 |
| `project_showcase` | 65 |
| `perspective_explainer` | 52 |
| **Total** | **280** |

The additions and their HN Algolia object IDs are in
[`data/added_project_showcases.csv`](data/added_project_showcases.csv).
[`apply_part2_improvement.py`](apply_part2_improvement.py) applies only this
change to the frozen before-data.

## Acceptance criteria

The five targets remain in [`criteria.md`](criteria.md), unchanged for Part 2:

1. Held-out accuracy is at least 0.68.
2. Held-out macro F1 is at least 0.62.
3. Every label has F1 of at least 0.50.
4. `reported_event` F1 is at least 0.60.
5. At least 70% of the most-confident third are correct, where confidence is
   the winning softmax probability.

## Training and reproducibility

The base model is `distilbert-base-uncased`. Required before and after runs use
3 epochs, learning rate `2e-5`, batch size 16, maximum length 128, stratified
70/15/15 splits, and seeds 42, 7, and 2024. Runs used CPU with PyTorch 2.14.1.

```bash
source .venv/bin/activate
python test.py
python run_experiments.py --mode before
python baseline.py --trained results.json
python apply_part2_improvement.py
python run_experiments.py --mode after
python run_experiments.py --mode stretch
python agreement.py
python validate_submission.py
```

Every new result records its completion time, data hash, row count, label
distribution, settings, and device.

## How I used AI

AI helped check the Part 2 brief against the repository, structure the error
analysis, implement provenance checks, and verify that the tables match the
JSON artifacts. The new titles were not generated: they are exact public Show
HN titles with retained Algolia IDs. AI assistance is disclosed in the data
notes and was not used to change the frozen taxonomy or criteria.

# Part 2 — Testing

## Part 2 plan — declared before experiments

The required improvement will add examples to `project_showcase` if the frozen
before-run confirms that it is the thinnest and worst-performing label. The
three stretch items planned before implementation are:

1. compare accuracy in the most-confident and least-confident thirds;
2. write a per-label error analysis for the label with the worst matrix row;
3. measure a second improvement by changing only epochs from 3 to 4 and running
   the same three seeds in the same five-criterion format.

This declaration fixes the scope in advance: the required change is a data
change, while the optional second change is reported separately and cannot be
used to reinterpret the required before/after result.

## Baseline vs. Trained

Prediction written before viewing the rebuilt baseline: the fine-tuned model
would beat the zero-shot model most clearly on `question_request`, because the
training data contains repeated explicit “Ask HN” language.

The prediction was wrong. Both models below were scored on the same 38 held-out
seed-42 titles from the frozen before-data, and the zero-shot baseline won.

| Measure | Zero-shot baseline | Trained | Difference |
|---|---:|---:|---:|
| Accuracy | 0.711 | 0.605 | -0.105 |
| Macro F1 | 0.644 | 0.380 | -0.264 |
| F1 — `reported_event` | 0.688 | 0.650 | -0.038 |
| F1 — `perspective_explainer` | 0.556 | 0.000 | -0.556 |
| F1 — `project_showcase` | 0.333 | 0.000 | -0.333 |
| F1 — `question_request` | 1.000 | 0.870 | -0.130 |

The baseline could use the full written definitions directly. Fine-tuning with
250 titles instead learned the two largest/easiest surface patterns and
ignored the two less formulaic classes.

## Run Log — Before

These values come from three genuine retrainings in
[`results_three_seeds_before.json`](results_three_seeds_before.json).

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall accuracy | ≥ 0.68 | 0.605 | 0.632 | 0.658 | MISSED |
| 2. Macro F1 | ≥ 0.62 | 0.380 | 0.398 | 0.421 | MISSED |
| 3. Every label F1 | ≥ 0.50 | min 0.000 | min 0.000 | min 0.000 | MISSED |
| 4. `reported_event` F1 | ≥ 0.60 | 0.650 | 0.683 | 0.732 | MET |
| 5. Most-confident-third accuracy | ≥ 0.70 | 0.917 | 1.000 | 1.000 | MET |

**Device beside the seeds:** seed 42 — `cpu`; seed 7 — `cpu`; seed 2024 — `cpu`.

Accuracy spans 0.053, below the ten-point instability warning. The result is
stable enough to conclude that the model misses the 0.68 target rather than
having received one unusually hostile split.

### Before confusion matrix — seed 42

| true \ predicted | reported_event | perspective_explainer | project_showcase | question_request |
|---|---:|---:|---:|---:|
| `reported_event` | 13 | 0 | 0 | 2 |
| `perspective_explainer` | 8 | 0 | 0 | 0 |
| `project_showcase` | 4 | 0 | 0 | 1 |
| `question_request` | 0 | 0 | 0 | 10 |

The largest off-diagonal cell is eight real `perspective_explainer` titles
called `reported_event`. The matrix also shows total failure on
`project_showcase`: all five are sent elsewhere.

## Verdicts and Diagnoses

| # | Verdict and evidence |
|---|---|
| 1 | **MISSED.** All three accuracies are below 0.68. The matrix shows that the model uses only `reported_event` and `question_request`, so two unused output classes—not split variance—cause the miss. |
| 2 | **MISSED.** Macro F1 is 0.380–0.421 because `perspective_explainer` and `project_showcase` both have 0.000 F1 on all three seeds. Macro averaging exposes the collapse that accuracy hides. |
| 3 | **MISSED.** The minimum label F1 is 0.000 in every trial. `project_showcase` is also the thinnest class at 35 rows, which makes data scarcity a plausible cause; this directly motivates the improvement. |
| 4 | **MET.** `reported_event` F1 is 0.650, 0.683, and 0.732, above 0.60 in every trial. |
| 5 | **MET.** The top confidence third is 0.917, 1.000, and 1.000 accurate, above 0.70 in every trial. |

The agreement exercise reached 83%, so it provides evidence that written-rule
application can drift on subtle argument/reaction boundaries. It used a
separate staff taxonomy, however, so it cannot by itself prove that the HN
showcase boundary was mislabeled. For this task the direct evidence is the
zero F1, the five seed-42 showcase errors, and the class's 35-row size.

## Agreement Report

Agreement was 25/30 = **83.3%**. All 30 of my labels matched a staff post; five
labels disagreed. These calls use only the staff taxonomy in
[`data/staff_taxonomy.md`](data/staff_taxonomy.md).

### 1. Same session, different shoes

> “Three of us ran the same session on the same day, different shoes, within
> 10 seconds of each other over 5 miles. I'm not saying shoes do nothing, but
> whatever they do was smaller than the noise between three people.”

- Staff: `hot_take`; mine: `analysis`.
- **My label is defensible.** The measured ten-second spread across three
  runners is the stated reason for the limited conclusion about effect size;
  the fact is load-bearing, not decorative. Rule 1 does not specify how many
  observations make a comparison strong enough, so this belongs in pile (c):
  the rule is ambiguous here and my reading remains defensible.

### 2. Times down about 3%

> “Times are down about 3% since the shoe rules changed, so obviously the
> shoes are the whole story and the sport has a problem.”

- Staff: `hot_take`; mine: `analysis`.
- **Staff is right.** “Obviously” and “the whole story” leap past the statistic;
  the 3% decorates a conclusion instead of establishing causation. Rule 1
  covers this, so this is pile (a): I applied the rule loosely.

### 3. Gel before 90 minutes

> “If you need a gel before 90 minutes you have a fuelling problem, not an
> energy problem.”

- Staff: `hot_take`; mine: `analysis`.
- **Staff is right.** Ninety minutes is a threshold, but no supporting
  observation connects it to the general claim. This is pile (a): the rule
  covered it, and I mistook a number for evidence.

### 4. Watching the last 400

> “Watching the last 400 of that and I had to stand up. Still standing up
> actually.”

- Staff: `reaction`; mine: `hot_take`.
- **Staff is right.** The post reports the writer's immediate physical response
  and makes no general claim. Rule 2 resolves it directly, so this is pile (a).

### 5. Personal-best shoes were the wrong size

> “Just realised the shoes I set every PB in this year are the ones I bought by
> accident in the wrong size.”

- Staff: `reaction`; mine: `hot_take`.
- **Staff is right.** This describes a recent personal realization rather than
  asserting a general rule about shoes. Under rule 2 it is a reaction, making
  this another pile (a) application error.

The disagreement pattern shows that I sometimes treated any number as
analysis and underused reaction for first-person immediacy. Four cases were
covered by the written rules; one comparison exposed a genuinely unspecified
evidence-strength boundary.

## The Improvement

The single required change was adding 30 `project_showcase` examples. This was
chosen because that label was both the thinnest (35 rows) and tied for worst F1
(0.000 on all seeds). No label definition, criterion, model, split seed,
learning rate, batch size, maximum length, or epoch count changed.

### Run Log — After

The results are in
[`results_three_seeds_after.json`](results_three_seeds_after.json).

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall accuracy | ≥ 0.68 | 0.690 | 0.643 | 0.619 | MISSED |
| 2. Macro F1 | ≥ 0.62 | 0.587 | 0.520 | 0.501 | MISSED |
| 3. Every label F1 | ≥ 0.50 | min 0.000 | min 0.000 | min 0.000 | MISSED |
| 4. `reported_event` F1 | ≥ 0.60 | 0.683 | 0.667 | 0.650 | MET |
| 5. Most-confident-third accuracy | ≥ 0.70 | 0.929 | 1.000 | 1.000 | MET |

**Device beside the seeds:** seed 42 — `cpu`; seed 7 — `cpu`; seed 2024 — `cpu`.

The improvement helped its intended label: `project_showcase` F1 moved from
0.000/0.000/0.000 to 0.667/0.462/0.400, and macro F1 increased on every seed.
Accuracy improved on seeds 42 and 7 but fell on seed 2024, so the accuracy
criterion still misses. The change worked locally without fixing the whole
classifier.

### After confusion matrix — seed 42

| true \ predicted | reported_event | perspective_explainer | project_showcase | question_request |
|---|---:|---:|---:|---:|
| `reported_event` | 14 | 0 | 0 | 0 |
| `perspective_explainer` | 8 | 0 | 0 | 0 |
| `project_showcase` | 5 | 0 | 5 | 0 |
| `question_request` | 0 | 0 | 0 | 10 |

## Stretch Features

### Confidence check

Confidence is the winning softmax probability. On both datasets, the highest
confidence third is much more accurate than the lowest third:

| Run | Seed | Most-confident third | Least-confident third |
|---|---:|---:|---:|
| Before | 42 | 0.917 | 0.083 |
| Before | 7 | 1.000 | 0.167 |
| Before | 2024 | 1.000 | 0.167 |
| After | 42 | 0.929 | 0.500 |
| After | 7 | 1.000 | 0.357 |
| After | 2024 | 1.000 | 0.214 |

The ranking is informative even though the full classifier misses three
criteria: high-confidence predictions form a much safer review queue.

### Second improvement

The optional second improvement changed only epochs from 3 to 4 on the same
280 rows and the same three seeds. Its results are in
[`results_stretch_epochs4.json`](results_stretch_epochs4.json).

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall accuracy | ≥ 0.68 | 0.714 | 0.667 | 0.690 | MISSED |
| 2. Macro F1 | ≥ 0.62 | 0.613 | 0.560 | 0.581 | MISSED |
| 3. Every label F1 | ≥ 0.50 | min 0.000 | min 0.000 | min 0.000 | MISSED |
| 4. `reported_event` F1 | ≥ 0.60 | 0.700 | 0.667 | 0.700 | MET |
| 5. Most-confident-third accuracy | ≥ 0.70 | 0.929 | 1.000 | 1.000 | MET |

**Device beside the seeds:** seed 42 — `cpu`; seed 7 — `cpu`; seed 2024 — `cpu`.

The extra epoch improved accuracy, macro F1, and showcase F1 on all three
seeds. It still did not make `perspective_explainer` learnable.

### Worst-label error write-up: `perspective_explainer`

`perspective_explainer` has 0.000 F1 across every before, after, and four-epoch
run. In the after seed-42 matrix, all eight true perspective titles are called
`reported_event`. Unlike “Ask HN” and “Show HN,” perspective titles have no
consistent prefix; many contain names, numbers, or factual wording also common
in event reports. Adding showcase rows therefore could not address this
separate boundary. The next targeted test should collect varied perspectives
paired with lexically similar event reports, then double-label that boundary
before retraining.

## What's Still Broken

After the required improvement, criteria 1–3 still miss. For accuracy and
macro F1, I would next address the total failure on `perspective_explainer`
with boundary-paired examples and class weighting, validated on a new held-out
set. For the every-label criterion, the same zero-F1 class is the direct cause.
I stopped because another data or training change would violate the required
one-change before/after experiment; the four-epoch result is reported
separately as stretch work.

I meant the labels to separate checkable developments from interpretation,
artifact presentation, and requests. The model actually learned explicit
request/showcase cues and a broad event default: the after matrix sends all
eight perspectives to `reported_event`, which is the remaining gap between the
taxonomy and the learned decision rule.

## Artifacts

- [`criteria.md`](criteria.md): unchanged Part 1 acceptance criteria
- [`data/labels_part2_before.csv`](data/labels_part2_before.csv): frozen before-data
- [`labels.csv`](labels.csv): 280-row data after the one required improvement
- [`data/added_project_showcases.csv`](data/added_project_showcases.csv): 30 traceable additions
- [`results_three_seeds_before.json`](results_three_seeds_before.json): required before run
- [`results_three_seeds_after.json`](results_three_seeds_after.json): required after run
- [`results_stretch_epochs4.json`](results_stretch_epochs4.json): second improvement
- [`baseline_results.json`](baseline_results.json): zero-shot baseline
- [`agreement_results.json`](agreement_results.json): 25/30 agreement and disagreements
- [`my_staff_labels.csv`](my_staff_labels.csv): my 30 staff-taxonomy labels
- [`results.json`](results.json) and [`test_split.csv`](test_split.csv): frozen seed-42 before evidence
- [`run_experiments.py`](run_experiments.py): reproducible three-seed runner
- [`validate_submission.py`](validate_submission.py): automated evidence checks

See [`RUNNING.md`](RUNNING.md) for environment details.
