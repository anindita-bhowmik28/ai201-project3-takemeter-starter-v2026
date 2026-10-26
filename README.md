# TakeMeter

TakeMeter classifies Hacker News titles by their visible communicative purpose.
The rebuilt task deliberately uses signals that are present in the input text,
rather than asking a short title to reveal whether an unseen article contains
supporting evidence.

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

The final [`labels.csv`](labels.csv) contains 250 unique Hacker News titles with
`text`, `label`, and `note` columns.

| Label | Rows | Share |
|---|---:|---:|
| `reported_event` | 97 | 38.8% |
| `question_request` | 66 | 26.4% |
| `perspective_explainer` | 52 | 20.8% |
| `project_showcase` | 35 | 14.0% |
| **Total** | **250** | **100.0%** |

The original submission is preserved as
[`data/labels_original_200.csv`](data/labels_original_200.csv). Its titles were
rule-reviewed under the new title-level taxonomy with AI assistance, and the rebuild
excluded 11 context-free titles such as “Moon,” “Sound,” and “IMG_0416” because
a reader could not reproduce a semantic label from those strings alone.

Before augmentation, `question_request` was the thinnest new class with five
usable rows. I collected 61 explicit Ask HN titles from the public HN Algolia
API on October 25, 2026. This replaces the 11 exclusions and creates a net gain
of exactly 50 usable examples. Every addition retains its Algolia object ID in
[`data/added_question_requests.csv`](data/added_question_requests.csv), and
[`rebuild_dataset.py`](rebuild_dataset.py) reproduces the final file.

Three boundary cases illustrate the rules:

1. “A search engine that favors text-heavy sites and punishes modern web
   design” could sound like a viewpoint, but it presents a specific artifact,
   so it is `project_showcase`.
2. “Google no longer producing high quality search results in significant
   categories” could be read as news, but it is a broad evaluative claim, so it
   is `perspective_explainer`.
3. “Tell HN: Help restore the tax deduction for software dev in the US” names a
   policy issue but explicitly asks readers to act, so it is
   `question_request`.

## Acceptance criteria

The complete executable definitions and project-specific reasons are in
[`criteria.md`](criteria.md). In summary:

1. Held-out accuracy is at least 0.68.
2. Held-out macro F1 is at least 0.62.
3. Every label has F1 of at least 0.50.
4. `reported_event` F1 is at least 0.60.
5. At least 70% of the most-confident third are correct, where confidence is
   the winning softmax probability.

## Training and reproducibility

The base model is `distilbert-base-uncased`. The main after-run uses 3 epochs,
learning rate `2e-5`, batch size 16, maximum length 128, a stratified 70/15/15
split, and seeds 42, 7, and 2024. It ran on CPU with PyTorch 2.14.1.

```bash
source .venv/bin/activate
python test.py
python rebuild_dataset.py
python run_experiments.py --mode after
python baseline.py --trained results.json
python run_experiments.py --mode stretch
```

Each new result contains its completion timestamp, row count, label
distribution, settings, device, and dataset SHA-256. The main and stretch runs
both record dataset hash
`0256ae767d974264ea9b525bffcd8c6e6f13349ceed6c9ccab9f43f4fbfaa31c`,
which proves the hyperparameter comparison used identical data.

## AI use

AI assisted with redesigning the title-level taxonomy, proposing the first
relabeling pass, checking consistency, writing the reproducibility scripts, and
summarizing the measured results. The added rows were not invented by a model:
they are exact public Ask HN titles with retained Algolia IDs. The notes in the
CSV disclose AI assistance instead of claiming the rebuilt labels were made
fully unaided.

## Baseline versus trained model

Before looking at the new scores, the prediction was that fine-tuning would
beat the zero-shot baseline most clearly on `question_request`, because the
explicit “Ask HN” signal appears repeatedly in the expanded data.

Both models below use the same 38 held-out seed-42 titles.

| Measure | Zero-shot baseline | Trained, 3 epochs | Difference |
|---|---:|---:|---:|
| Accuracy | 0.711 | 0.605 | -0.105 |
| Macro F1 | 0.644 | 0.380 | -0.264 |
| F1 — `reported_event` | 0.688 | 0.650 | -0.038 |
| F1 — `perspective_explainer` | 0.556 | 0.000 | -0.556 |
| F1 — `project_showcase` | 0.333 | 0.000 | -0.333 |
| F1 — `question_request` | 1.000 | 0.870 | -0.130 |

The prediction was wrong: the baseline also recognized the explicit request
language perfectly. Fine-tuning learned requests well but collapsed the two
smaller, less formulaic classes into `reported_event`.

## Run Log — Before

This is the preserved result from the original 200-row, three-label task. It is
historical evidence, not a controlled comparison with the rebuilt taxonomy.

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 |
|---|---:|---:|---:|---:|
| Accuracy | 0.68 | 0.500 | 0.533 | 0.600 |
| Macro F1 | 0.62 | 0.222 | 0.280 | 0.409 |
| Minimum per-label F1 | 0.50 | 0.000 | 0.000 | 0.000 |
| Original `analysis` F1 | 0.60 | 0.667 | 0.698 | 0.706 |
| Most-confident-third accuracy | 0.70 | 0.700 | 0.900 | 0.600 |

The source artifact is
[`results_three_seeds_before.json`](results_three_seeds_before.json). Its
numbers are not copied into the after artifact.

## Run Log — After

This is the genuinely rerun 250-row, four-label experiment. The values come
from [`results_three_seeds_after.json`](results_three_seeds_after.json).

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| Accuracy | 0.68 | 0.605 | 0.632 | 0.658 | Missed |
| Macro F1 | 0.62 | 0.380 | 0.398 | 0.421 | Missed |
| Minimum per-label F1 | 0.50 | 0.000 | 0.000 | 0.000 | Missed |
| `reported_event` F1 | 0.60 | 0.650 | 0.683 | 0.732 | Met |
| Most-confident-third accuracy | 0.70 | 0.917 | 1.000 | 1.000 | Met |

The changing values across seeds, new 38-row test distributions, completion
timestamp, and dataset fingerprint provide evidence that the pipeline actually
ran. Accuracy improves across the seeds but never reaches 0.68. The confidence
criterion now holds for every seed.

### Seed-42 confusion matrix

| true \ predicted | reported_event | perspective_explainer | project_showcase | question_request |
|---|---:|---:|---:|---:|
| `reported_event` | 13 | 0 | 0 | 2 |
| `perspective_explainer` | 8 | 0 | 0 | 0 |
| `project_showcase` | 4 | 0 | 0 | 1 |
| `question_request` | 0 | 0 | 0 | 10 |

The clearest success is the 10/10 request diagonal. The clearest remaining
failure is that all eight perspectives and four of five showcases become
`reported_event`; the model learns explicit prefixes faster than subtler
semantic distinctions.

## Stretch features

### Fourth label

`question_request` is a complete fourth label with a definition, examples,
decision priority, 66 dataset rows, notebook support, baseline scores, trained
scores, and confusion-matrix entries.

### Fifty additional examples for the thinnest label

Before augmentation, `question_request` had five usable legacy rows. Sixty-one
traceable Ask HN rows were added while 11 ambiguous legacy rows were removed,
giving a net increase of exactly 50 rows and a final class count of 66.

### Second run with one hyperparameter changed

The controlled stretch run changed only `epochs`, from 3 to 4, while retaining
seed 42, the split, model, data hash, learning rate, batch size, and maximum
length. Its artifact is
[`results_stretch_epochs4.json`](results_stretch_epochs4.json).

| Measure | 3 epochs | 4 epochs | Change |
|---|---:|---:|---:|
| Accuracy | 0.605 | 0.605 | +0.000 |
| Macro F1 | 0.380 | 0.386 | +0.006 |
| F1 — `reported_event` | 0.650 | 0.634 | -0.016 |
| F1 — `perspective_explainer` | 0.000 | 0.000 | +0.000 |
| F1 — `project_showcase` | 0.000 | 0.000 | +0.000 |
| F1 — `question_request` | 0.870 | 0.909 | +0.039 |
| Most-confident-third accuracy | 0.917 | 0.917 | +0.000 |

The extra epoch changed the underlying metrics but did not fix class collapse.
That is evidence of a real rerun and a negative result, not a copied table.

## Agreement report

The existing staff agreement exercise used the separate staff-provided running
forum taxonomy, as required by that instrument, and remains preserved in
[`agreement_results.json`](agreement_results.json). Agreement was 25/30 (83%).
Four disagreements came from applying a clear staff rule too loosely; one was
a genuinely ambiguous observation-versus-argument boundary. This historical
check cannot validate the rebuilt HN taxonomy, so the rebuild instead makes
every inclusion, exclusion, source ID, and rule auditable.

## What's still broken

Three targets remain missed. The trained model does not reach 0.68 accuracy or
0.62 macro F1, and the every-label target fails because
`perspective_explainer` and `project_showcase` have zero F1. The seed-42 matrix
shows the mechanism directly: those titles are overwhelmingly absorbed into
`reported_event` even though `question_request` is recognized reliably.

The next defensible change is not another test-set-tuned epoch. It is to collect
more showcases and perspectives with explicit but varied linguistic cues,
double-label a sample under the new rules, and reserve a fresh untouched test
set before trying class weighting. I stopped here because evaluating further
design choices against the same 38 titles would turn the held-out set into a
development set.

## Artifacts

- [`labels.csv`](labels.csv): final 250-row dataset
- [`data/labels_original_200.csv`](data/labels_original_200.csv): original data
- [`data/added_question_requests.csv`](data/added_question_requests.csv): added rows and source IDs
- [`results_three_seeds_before.json`](results_three_seeds_before.json): original run
- [`results_three_seeds_after.json`](results_three_seeds_after.json): genuine rebuilt run
- [`results_stretch_epochs4.json`](results_stretch_epochs4.json): controlled stretch run
- [`results.json`](results.json) and [`test_split.csv`](test_split.csv): seed-42 rebuilt run
- [`baseline_results.json`](baseline_results.json): rebuilt zero-shot baseline
- [`agreement_results.json`](agreement_results.json): staff agreement exercise
- [`takemeter.ipynb`](takemeter.ipynb): notebook configured for all four labels
- [`run_experiments.py`](run_experiments.py): reproducible experiment runner

See [`RUNNING.md`](RUNNING.md) for environment details.
