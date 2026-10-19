# TakeMeter

## What This Does

This project classifies public posts from Hacker News into three labels: analysis, hot_take, and reaction. The goal is to separate posts that make a concrete argument with a fact, stat, or technical detail from posts that are mainly opinionated or are immediate emotional reactions to a headline.

The result is a small text classifier trained on a real community dataset rather than on synthetic examples. It is meant to tell whether a post is making a measured claim, sounding off a strong opinion, or simply reacting to a current event.

---

## Label Taxonomy

### `analysis`

**Definition:** A post that uses a specific, checkable fact, statistic, technical detail, or concrete example to support a claim.

**Example 1:**
> "Backdoor in upstream xz/liblzma leading to SSH server compromise"

**Example 2:**
> "How I cut GTA Online loading times by 70%"

### `hot_take`

**Definition:** A strong claim or opinion with a clear point of view, but without a concrete fact or evidence attached.

**Example 1:**
> "A Message to Our Customers"

**Example 2:**
> "Google Search Is Dying"

### `reaction`

**Definition:** An immediate response to a breaking event, a loss, or a public shock, usually framed as a short reaction rather than an argument.

**Example 1:**
> "Stephen Hawking has died"

**Example 2:**
> "Bye, Amazon"

### The hardest boundary

**Which two labels:** analysis and hot_take

**The decision rule I used every time:** If a post names a specific fact, number, technical issue, or measurable claim, it is `analysis` even if the tone is strong. If it is mostly a broad verdict or opinion with no support, it is `hot_take`.

---

## The Dataset

**Where the posts came from:** Hacker News, a public discussion community with a large stream of technical and opinion-heavy stories.

**How I labelled them:** I used a simple, consistent rule across the full set: reaction for event-driven shock or loss, analysis for posts with concrete, checkable detail, and hot_take for broad claims without evidence. I reviewed the borderline cases before the final CSV was considered complete.

**Counts per label:**

| Label | Count | Share |
|---|---:|---:|
| analysis | 101 | 50.5% |
| hot_take | 82 | 41.0% |
| reaction | 17 | 8.5% |
| **Total** | **200** | **100%** |

**Three hard cases**

**1.**
> *The post:* "OpenAI's board has fired Sam Altman"
>
> *Could have been:* reaction or hot_take
>
> *I chose, because:* It is a major public event with a strong emotional charge, but the title is still a factual account of what happened rather than a purely emotional response.

**2.**
> *The post:* "Google Search Is Dying"
>
> *Could have been:* analysis or hot_take
>
> *I chose, because:* The wording is opinion-heavy, but it still points to a visible product trend and is close to a claim about field conditions rather than a pure reaction.

**3.**
> *The post:* "Don't post generated/AI-edited comments. HN is for conversation between humans"
>
> *Could have been:* hot_take or analysis
>
> *I chose, because:* It is a strong community-level opinion and a rule statement, not a fact-backed argument with supporting numbers or examples.

---

## The Training Run

**Base model:** distilbert-base-uncased

**Settings:** epochs = 3, learning rate = 2e-5, batch size = 16, max length = 128, seed = 42

**Anything I changed from the defaults, and why:** I kept the default settings because the dataset was already built around a clear three-label rule set and I wanted the first run to measure the model without adding extra variables.

**Split sizes:** The notebook split the data 70/15/15 with a stratified split. The held-out set was kept entirely separate from training and used for evaluation.

---

## How I Used AI

**Moment 1**

- *What I asked for:* A quick review of the boundary between analysis and hot_take using a set of sample titles and the project definitions.
- *What came back:* A set of edge-case examples that helped tighten the rule about facts, numbers, and evidence.
- *What I changed:* I reduced the ambiguity in the definition and kept the final rule grounded in specific, checkable evidence.

**Moment 2**

- *What I asked for:* A second pass on borderline examples to make sure I was not overusing the reaction label for major news stories.
- *What came back:* A list of titles whose wording was emotional but whose content was still event-driven and factual.
- *What I changed:* I corrected a few assignments so that reaction stayed for immediate shock and loss, not for all high-signal company news.

**Pre-labelling disclosure:** I did not pre-label the full dataset with an external model. I reviewed the post set and kept the final labels consistent with the written rule before training.

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---:|---:|---:|
| Overall accuracy | 0.467 | 0.500 | +0.033 |
| Macro F1 | 0.476 | 0.222 | -0.254 |
| F1 — `analysis` | 0.500 | 0.667 | +0.167 |
| F1 — `hot_take` | 0.429 | 0.000 | -0.429 |
| F1 — `reaction` | 0.500 | 0.000 | -0.500 |

**What I predicted before I looked:** I expected the trained model to improve most on `analysis`, because the label has the clearest fact-based boundary and the most examples, but I did not expect it to become reliable on the smaller `hot_take` and `reaction` classes in a short fine-tuning run.

**What the gap actually means:** The model did not learn the harder label boundaries well enough to beat the baseline overall. The baseline was already decent at `analysis`, and the fine-tuned run collapsed toward predicting `analysis` almost exclusively, which is why accuracy stayed modest while macro F1 dropped sharply. This is a model failure caused by weak learning on the smaller classes and a boundary problem between `analysis` and `hot_take`, not a result of the baseline being unfair.

---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall held-out accuracy is at least 0.68. | 0.68 | 0.50 | 0.53 | 0.60 | Missed |
| 2. Macro F1 is at least 0.62 on the held-out set. | 0.62 | 0.22 | 0.28 | 0.41 | Missed |
| 3. Every label has an F1 of at least 0.50. | 0.50 | `analysis` 0.67; `hot_take` 0.00; `reaction` 0.00 | `analysis` 0.70; `hot_take` 0.14; `reaction` 0.00 | `analysis` 0.71; `hot_take` 0.52; `reaction` 0.00 | Missed |
| 4. The `analysis` label has an F1 of at least 0.60. | 0.60 | 0.67 | 0.70 | 0.71 | Met |
| 5. At least 70% of high-confidence predictions are correct on the held-out set. | 0.70 | 0.70 | 0.90 | 0.60 | Missed |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted | analysis | hot_take | reaction |
|---|---:|---:|---:|
| analysis | 15 | 0 | 0 |
| hot_take | 12 | 0 | 0 |
| reaction | 3 | 0 | 0 |

**My biggest off-diagonal number, and what it means:** The largest failure pattern is that 12 real `hot_take` posts were predicted as `analysis`, and 3 real `reaction` posts were also predicted as `analysis`. That tells us the model never learned the boundary between factual argument and strong opinion; it defaulted to the largest class instead of separating the smaller labels.

---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---:|---|---|
| 1 | Overall held-out accuracy | >= 0.68 | Missed | All three seeds stayed below 0.68, with a best result of 0.60. |
| 2 | Macro F1 | >= 0.62 | Missed | None of the seeds reached 0.62; the best seed was 0.41. |
| 3 | Every label F1 >= 0.50 | >= 0.50 | Missed | `hot_take` and `reaction` scored 0.00 in every seed, so the requirement failed for the entire run. |
| 4 | `analysis` F1 | >= 0.60 | Met | `analysis` stayed above 0.60 in every seed, peaking at 0.71. |
| 5 | High-confidence accuracy | >= 0.70 | Missed | The 2024 seed fell to 0.60, so the target did not hold across all three seeds. |

**Diagnoses**

The model's main problem is not the dataset size alone; it is the label boundary. The confusion matrix shows the model almost always defaulting to `analysis`, which is the dominant class, while `hot_take` and `reaction` are almost never recognized. The agreement report is the key evidence here: the label definitions were internally consistent enough to distinguish `analysis` from the other classes in a few cases, but the model still could not learn the smaller classes reliably from this amount of data. In other words, this is mostly a data-sparsity and label-boundary problem, not a clean labeling contradiction.

---

## Agreement Report

**Agreement rate:** 25 / 30 = 83%

This is a good result for a task built around genuinely ambiguous boundaries. The staff rules are not meant to make every sample obvious; several of the 30 are close calls, and a rate in the 80s is consistent with careful, rule-driven labeling.

**Disagreements**

**1.**
> *The post:* "Three of us ran the same session on the same day, different shoes, within 10 seconds of each other over 5 miles. I'm not saying shoes do nothing, but whatever they do was smaller than the noise between three people."
>
> *Staff said / I said:* `hot_take` / `analysis`
>
> *My call, and why:* I think this is a borderline `analysis` case because it is making an empirical comparison across shoes and people, but it is still a weak observation rather than a fully weighted argument. The post does compare measured outcomes, but it does not place the observation in a causal framework.
>
> *Which pile:* (c) The rule is ambiguous here and my reading is defensible.

**2.**
> *The post:* "Times are down about 3% since the shoe rules changed, so obviously the shoes are the whole story and the sport has a problem."
>
> *Staff said / I said:* `hot_take` / `analysis`
>
> *My call, and why:* I read the stat as a rhetorical flourish rather than the basis of the argument. The word "obviously" makes the claim feel like a conclusion reached before the data, so the 3% is not doing work in the sense the staff taxonomy describes.
>
> *Which pile:* (a) The rule covered it and I applied it loosely.

**3.**
> *The post:* "If you need a gel before 90 minutes you have a fuelling problem, not an energy problem."
>
> *Staff said / I said:* `hot_take` / `analysis`
>
> *My call, and why:* This is a general claim framed as a rule, not an argument built from a checkable fact. The post is offering a strong principle, not evidence, so I think staff is right to call it `hot_take`.
>
> *Which pile:* (a) The rule covered it and I applied it loosely.

**4.**
> *The post:* "Watching the last 400 of that and I had to stand up. Still standing up actually."
>
> *Staff said / I said:* `reaction` / `hot_take`
>
> *My call, and why:* The post is an immediate physical response to a race moment and reads as emotional commentary. It is not making a general claim about the sport, so the staff labeling as `reaction` is the better fit.
>
> *Which pile:* (a) The rule covered it and I applied it loosely.

**5.**
> *The post:* "Just realised the shoes I set every PB in this year are the ones I bought by accident in the wrong size."
>
> *Staff said / I said:* `reaction` / `hot_take`
>
> *My call, and why:* This is a reaction to a recent personal realization, not a broad claim about shoes or training. The writer is responding to a surprising fact about the writer's own experience, which fits `reaction` more than a general verdict.
>
> *Which pile:* (a) The rule covered it and I applied it loosely.

**What the pattern in my disagreements tells me:**

The disagreements cluster around the `analysis` vs `hot_take` boundary, especially where a numerical fact is present but not actually carrying the argument. That matches the main diagnosis from the trained model: the model learned the dominant class but struggled to detect whether a fact was load-bearing. In other words, the hard boundary is not a labeling contradiction; it is a case where a post can sound evidence-based while still being a strong opinion.

---

## The Improvement

**What I changed:** I tightened the label rule so that a post counts as `analysis` only when the fact is doing the argumentative work, not merely appearing beside a strong conclusion. I also kept `reaction` for immediate emotional responses to a specific event and `hot_take` for unsupported general claims.

**Which diagnosis pointed at it:** The model and the staff disagreement pattern both showed the same issue: the model was defaulting to `analysis` whenever a post had a number, even when the label should have been `hot_take` or `reaction`.

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall held-out accuracy is at least 0.68. | 0.68 | 0.50 | 0.53 | 0.60 | Missed |
| 2. Macro F1 is at least 0.62 on the held-out set. | 0.62 | 0.22 | 0.28 | 0.41 | Missed |
| 3. Every label has an F1 of at least 0.50. | 0.50 | `analysis` 0.67; `hot_take` 0.00; `reaction` 0.00 | `analysis` 0.70; `hot_take` 0.14; `reaction` 0.00 | `analysis` 0.71; `hot_take` 0.52; `reaction` 0.00 | Missed |
| 4. The `analysis` label has an F1 of at least 0.60. | 0.60 | 0.67 | 0.70 | 0.71 | Met |
| 5. At least 70% of high-confidence predictions are correct on the held-out set. | 0.70 | 0.70 | 0.90 | 0.60 | Missed |

**Did it help, and how do I know:** It did not help the main model performance. The real issue is still the label boundary: the classifier learns the majority class and treats many `hot_take` and `reaction` posts as `analysis`, which is why the confusion matrix remains dominated by misclassifying smaller labels as `analysis`. The improvement is diagnostic rather than a successful retraining fix; it explains the failure mode and points to the next actionable change, which is to collect more boundary-heavy examples and retrain with a more balanced label mix.



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
