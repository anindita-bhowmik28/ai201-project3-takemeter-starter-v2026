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

<!-- Unit 6 content goes here. -->

---

## Run Log — Before

<!-- Unit 6 content goes here. -->

---

## Verdicts and Diagnoses

<!-- Unit 6 content goes here. -->

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy |  |  |  |
| Macro F1 |  |  |  |
| F1 — `label_one` |  |  |  |
| F1 — `label_two` |  |  |  |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



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
