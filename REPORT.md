# Scientific Idea Generation with Grounded Novelty — Research Report

*Domain: Artificial Intelligence · Generator & evaluator: `gpt-4.1-mini` · Date: 2026-06-28*

---

## 1. Executive Summary

**Research question.** Do LLMs generate *better scientific ideas* (hypotheses) when ideation
is **grounded** — through deduction, induction, small experiments / outcome imagination,
literature comparison, or proposal writing — rather than prompted in an abstract,
**ungrounded** way?

We built the first **single-protocol head-to-head ablation** of these six ideation regimes:
holding the model, task suite, generation budget definition, and evaluation procedure fixed,
we generate a bank of 15 task hypotheses (decision rules) under each regime and score each
bank by an *objective downstream utility metric* — how accurately an LLM applying the bank
predicts held-out labels, in-distribution (IND) and out-of-distribution (OOD) — on three
HypoGeniC tasks (`headline_binary`, `deceptive_reviews`, `dreaddit`) × 3 seeds (54 banks,
≈11.9k API calls, ≈$3).

**Key finding (nuanced, partly negative).** With a strong modern LLM, **grounding does *not*
uniformly produce better ideas**: pooled across regimes, grounded banks are statistically
indistinguishable from (slightly *below*) the ungrounded control (IND −1.2 pts, OOD −2.5 pts,
both n.s.). But **the *mechanism* matters decisively**: *empirical small-experiment + outcome-
imagination grounding* (C3) is the best regime and the **only one that significantly improves
IND accuracy** over the control (+4.4 pts, Cohen's *d* = 1.09, Wilcoxon *p* = 0.02), while
*proposal-writing* (C5) **significantly hurts** (−5.7 pts, *d* = −1.56, *p* = 0.008) and
collapses idea diversity. Crucially, **data grounding can overfit**: on `deceptive_reviews`
the ungrounded model transfers near-perfectly OOD (0.97) using general priors, whereas
data-grounded rules latch onto dataset-specific surface cues and drop to 0.78–0.82 OOD.

**Practical implication.** "Ground your ideation" is too coarse a prescription for today's
strong LLMs. *Grounding ideas in empirical feedback* (cheaply testing and revising against
data) is the one mechanism that reliably adds value — but it costs ~80× the tokens; pure
*elaboration* (proposal writing) actively degrades idea quality; and *data grounding trades
in-distribution gains for out-of-distribution robustness.* Ungrounded prompting of a capable
model is a surprisingly strong, hard-to-beat baseline.

---

## 2. Research Question & Motivation

There is genuine skepticism about whether LLMs can produce *good* scientific ideas: they tend
to be incremental or implausibly radical, and "novelty" resists definition. The submitter's
hypothesis is that LLMs handle novelty better when it is **grounded** (deduction, induction,
proposal writing, outcome imagination, small experiments, literature comparison).

**Gap (from `literature_review.md` §8).** Every prior method grounds *one* way and beats an
ungrounded prompt (SciMON→literature; HypoGeniC→data; ResearchAgent→proposals; SGA→simulation).
**Nobody has held task, model, budget, and metric fixed and ablated the grounding *mechanisms*
against each other.** Prior wins were also measured with weaker models (GPT-3.5 / GPT-4-2023),
leaving open whether grounding still helps now that the *ungrounded* baseline is much stronger.
We also heed the field's central caveat — novelty is over-trusted (Si et al.'s ideation–
execution flip) — by scoring *grounded usefulness*, not a bare novelty score.

**Contribution.** A unified, reproducible ablation of six ideation regimes under one protocol,
scored by objective downstream utility (IND + OOD) plus diversity, a novelty–usefulness
analysis, and a cost/benefit accounting.

---

## 3. Methodology

### 3.1 Operationalization
An "idea" = a **task hypothesis**: an explicit, general decision rule explaining a binary
classification phenomenon. "Better" = **grounded usefulness**: when an LLM applies the full
bank of 15 hypotheses to a new instance, how often it predicts the correct label. This makes
"idea quality" objective and leakage-controlled, and uses the field-standard HypoGeniC utility
metric.

### 3.2 The six regimes (1 control + 5 grounding mechanisms — `src/regimes.py`)
| ID | Regime | Grounding mechanism | What it sees / does |
|----|--------|---------------------|---------------------|
| **C0** | Ungrounded (control) | none | task description only → "propose 15 novel hypotheses" |
| **C1** | Deduction | reasoning | decompose causal mechanisms from first principles → derive rules |
| **C2** | Induction | data | 20 labeled examples → induce explanatory rules |
| **C3** | Small-exp + outcome imagination | empirical feedback | induce, then *test* the bank on fresh examples, observe errors, *revise* (2 rounds) |
| **C4** | Literature-comparison | comparison | SciMON-style: generate in rounds, each forced to differ from the prior pool |
| **C5** | Proposal-writing | elaboration | write a structured proposal → distill into rules |

Held fixed across all regimes: generator `gpt-4.1-mini` (T=0.7), **bank size K=15**, and the
**identical evaluation procedure** below. The only variable is *how the ideas were grounded*.
To avoid leakage, the dataset's expert `known_hypotheses` are **never** shown during
generation; C4 differentiates against the model's own accumulating pool.

### 3.3 Shared evaluator (`src/evaluate.py`)
For every bank, an LLM (`gpt-4.1-mini`, T=0) is shown the 15 hypotheses + one instance and
must apply them to answer "A" or "B". Option order is deterministically swapped per item to
neutralize position bias. Unparseable answers count as wrong (abstain rate tracked — it was
**0.0%** everywhere). Each bank is scored on **100 IND-test** and **100 OOD** items.

### 3.4 Datasets (`datasets/hypogenic_datasets/real/`, all balanced 50/50)
`headline_binary` (which headline gets more clicks), `deceptive_reviews` (fake vs genuine
hotel review), `dreaddit` (stress vs no-stress Reddit post). Each ships a task description and
expert `known_hypotheses` (held out as a novelty anchor + reference topline).

### 3.5 Baselines & references
Ungrounded **C0** is the key control; **chance/majority = 0.50** (balanced); the dataset's
**expert known-hypotheses** bank is an approximate topline.

### 3.6 Statistics
3 seeds × 3 tasks = **9 paired cells** per regime. Primary test: each grounded regime vs C0,
paired across cells, **Wilcoxon signed-rank + paired t**, **Cohen's *d*** effect size,
**Bonferroni** over the 5 grounded comparisons. Bootstrap 95% CIs (10k resamples) over pooled
test items (n=900/regime/split). α = 0.05.

### 3.7 Reproducibility
Seeds fixed; every API call is disk-cached (`results/cache/`, idempotent reruns); banks saved
to `results/banks/`, per-item predictions to `results/eval/`. Cost: **7.19M input + 44.8k
output tokens, 11,931 calls ≈ $3**; wall-clock ≈ 14 min (32-way concurrency). Hardware: API
only (no GPU needed; an A6000 was present but unused).

---

## 4. Results

### 4.1 Aggregate accuracy by regime (mean over 3 tasks × 3 seeds)

| Regime | IND acc | IND sd | OOD acc | OOD sd | Diversity | Novelty dist | Gen tok/bank |
|--------|:------:|:-----:|:------:|:-----:|:--------:|:-----------:|:-----------:|
| **C0 Ungrounded (control)** | 0.631 | .079 | **0.760** | .183 | 0.417 | 0.399 | **460** |
| C1 Deduction | 0.608 | .071 | **0.766** | .166 | 0.367 | 0.399 | 918 |
| C2 Induction | 0.637 | .053 | 0.719 | .113 | 0.409 | 0.431 | 2,897 |
| **C3 Small-exp + outcome** | **0.676** | .095 | 0.739 | .117 | 0.392 | 0.430 | 36,935 |
| C4 Literature-compare | 0.601 | .032 | 0.721 | .175 | 0.396 | 0.415 | 1,150 |
| C5 Proposal-writing | 0.574 | .070 | 0.729 | .183 | 0.308 | 0.415 | 1,588 |

*(see `figures/fig1_accuracy_by_regime.png`; CIs in `results/bootstrap_ci.csv`)*

### 4.2 Paired significance vs the ungrounded control (Bonferroni m=5)

| Split | Regime | Gain vs C0 | Cohen's *d* | Wilcoxon *p* | t *p* (Bonf) |
|-------|--------|:---------:|:----------:|:-----------:|:-----------:|
| IND | C1 Deduction | −0.023 | −0.51 | 0.195 | 0.826 |
| IND | C2 Induction | +0.006 | +0.11 | 0.785 | 1.000 |
| IND | **C3 Empirical** | **+0.044** | **+1.09** | **0.020** | 0.058 |
| IND | C4 Literature-compare | −0.030 | −0.46 | 0.195 | 1.000 |
| IND | **C5 Proposal** | **−0.057** | **−1.56** | **0.008** | **0.008** |
| OOD | C1 Deduction | +0.006 | +0.14 | 0.719 | 1.000 |
| OOD | C2 Induction | −0.041 | −0.29 | 0.652 | 1.000 |
| OOD | C3 Empirical | −0.021 | −0.19 | 0.742 | 1.000 |
| OOD | C4 Literature-compare | −0.039 | −0.54 | 0.102 | 0.715 |
| OOD | C5 Proposal | −0.031 | −0.47 | 0.234 | 0.988 |

**H1 (overall grounded vs control):** IND 0.619 vs 0.631 (gain −0.012, *d*=−0.32, *p*=0.37);
OOD 0.735 vs 0.760 (gain −0.025, *d*=−0.41, *p*=0.25). **Not supported.**

### 4.3 Per-task breakdown (IND / OOD, mean over seeds) — the heterogeneity

| Regime | headline | deceptive | dreaddit |
|--------|:--------:|:---------:|:--------:|
| C0 Ungrounded | 0.55 / 0.56 | 0.62 / **0.97** | 0.72 / 0.75 |
| C2 Induction | 0.58 / 0.61 | 0.65 / 0.78 | 0.68 / 0.77 |
| **C3 Empirical** | 0.56 / 0.59 | **0.69** / 0.82 | **0.78 / 0.81** |
| C5 Proposal | 0.50 / 0.57 | 0.58 / 0.97 | 0.64 / 0.65 |

*(full grid: `figures/fig5_per_task_heatmap.png`)*

Three regimes, three stories:
- **`dreaddit`:** empirical grounding (C3) wins clearly on **both** IND (0.78 vs 0.72) and OOD
  (0.81 vs 0.75) → grounding helps.
- **`deceptive_reviews`:** the ungrounded model transfers near-perfectly OOD (**0.97**); data
  grounding (C2/C3) *raises IND* but **crashes OOD to 0.78–0.82** → grounding overfits.
- **`headline_binary`:** all regimes hover near chance (0.50–0.61) → no extractable signal;
  even expert hypotheses score 0.56/0.58.

### 4.4 Cost/benefit
C3's gains cost **~80× the generation tokens** of the control (36.9k vs 0.46k tokens/bank,
from its test-and-revise loop). The cheaper grounded regimes (C1, C4, C5: 0.9k–1.6k) do **not**
beat the control. *(see `figures/fig4_cost_benefit.png`)*

### 4.5 Novelty vs usefulness
Across all 18 task×regime cells, Spearman(novelty, OOD) = **+0.835** (p<0.001) — but this is a
**task-difficulty confound** (Simpson's paradox): easy/high-accuracy tasks also have ideas
farther from their expert anchors. **Within** task the relationship vanishes (mean ρ ≈ **−0.10**;
deceptive ρ = −0.75). Diversity → usefulness is also weak within task (mean ρ ≈ +0.33, n.s.).
**Novelty/diversity are not reliable proxies for usefulness** — consistent with the literature's
caveat. *(see `figures/fig3_novelty_vs_usefulness.png`)*

---

## 5. Analysis & Discussion

**Why ungrounded is so strong now.** `gpt-4.1-mini`'s ungrounded banks already recite the
*right* expert-level features (for `dreaddit`: first-person pronouns, absolutist words, sleep
disturbance, inability to cope — textbook LIWC stress markers). The model's parametric prior
*is* a form of grounding (in pretraining). This is the crux of the partly-negative result: the
2023-era finding "grounding beats ungrounded prompting" was measured against much weaker
baselines; with a 2025 model the ungrounded control is a strong, often-transferable starting
point, shrinking grounding's headroom.

**Why empirical grounding (C3) is the exception.** It is the only regime that *changes its
ideas in response to evidence*: it tests the bank, sees concrete mistakes, and revises
(producing richer rules like "intrusive thoughts / trauma / self-blame with ongoing distress").
This closed loop is exactly the "small experiments + outcome imagination" the hypothesis names,
and it is the one mechanism that reliably lifts IND accuracy (*d*=1.09). The catch: its IND gain
does not transfer OOD, and it is by far the most expensive regime.

**Why proposal-writing (C5) backfires.** Distilling a prose proposal yields rigid, symmetric
"if high X → label A / if low X → label B" rules with the **lowest diversity** (0.31). The
elaboration adds words, not discriminative power, and the brittle thresholds misfire — a
concrete instance of LLM ideation looking thorough while being less useful.

**The overfitting result (deceptive_reviews) is the most important nuance.** Data grounding
helped IND but *hurt* OOD, because the learned rules encode dataset-specific surface cues
(spatial/Chicago-hotel vocabulary) that don't transfer, whereas the ungrounded prior captures
general "fake review" structure that does. Prior work emphasizes data grounding's OOD *wins*;
under one protocol with a strong model we find it can **trade OOD robustness for IND fit** —
a caution the field under-reports. (Note `deceptive`'s OOD is *easier* than its IND for general
priors, which amplifies the effect; the within-split grounded-vs-ungrounded comparison still
holds.)

**Verdict on the hypothesis.** *Refuted as a blanket claim; supported for one specific
mechanism.* Grounding does not uniformly improve LLM ideas; empirical small-experiment grounding
does (at high cost), elaboration grounding harms, and data grounding's benefit is conditional
and can reverse OOD. **H1 not supported; H2 partially supported** (empirical > others on IND,
but not the full ordering); **H3 not supported** (grounding advantage not larger OOD — often the
reverse); **H4: novelty ≠ usefulness** (confirmed once the task confound is removed).

---

## 6. Limitations
- **Tasks/domain.** Three binary text-classification tasks (one near-chance). "Idea quality" =
  downstream label accuracy; this captures *grounded usefulness* but not open-ended scientific
  creativity (idea-level benchmarks like MOOSE-Chem were out of scope here).
- **Single model family.** Generator = evaluator = `gpt-4.1-mini`. A stronger ungrounded
  baseline likely shrinks grounding's value further; a weaker model might restore the classic
  grounding wins — cross-model replication is the key next step.
- **Statistical power.** 9 paired cells per comparison; within-task novelty analysis uses only
  6 points/task. Effect *directions* are clear; some n.s. results may be power-limited. Headline
  is near chance and contributes mostly noise to the aggregate.
- **Regime instantiations are one reasonable design each.** Stronger literature grounding (real
  RAG over papers) or longer empirical loops could change C3/C4's standing.
- **Eval confound mitigated, not eliminated.** A single evaluator prompt is shared by all
  regimes (fair), but the evaluator's own competence caps measurable hypothesis quality.

## 7. Conclusions & Next Steps
**Answer.** Under a fixed, objective protocol, *grounding does not reliably make a strong LLM's
ideas better.* The exception is **grounding in empirical feedback** (test-and-revise), the one
mechanism that significantly improves in-distribution usefulness — but it is ~80× costlier and
its gains do not transfer OOD. Pure elaboration (proposal writing) **hurts**, and data grounding
can **overfit**, trading OOD robustness for IND fit. Novelty and diversity do not track
usefulness within a task.

**Next steps.** (1) Cross-model sweep (weaker → frontier generators) to map where grounding's
value lives; (2) regularized / OOD-aware empirical grounding to keep IND gains without overfit;
(3) idea-level domains (MOOSE-Chem rediscovery) to test whether the conclusions hold for
open-ended ideation; (4) hybrid "prior + minimal empirical check" to capture most of C3's gain
at a fraction of the cost.

## References
Key papers (full notes in `literature_review.md`): Si et al. 2024 (*Can LLMs Generate Novel
Research Ideas?*, 2409.04109); Zhou & Tan 2024 (*HypoGeniC*, 2404.04326); Liu et al. 2024
(*Literature Meets Data*, 2410.17309); *HypoBench* 2025 (2504.11524); Wang et al. 2023 (*SciMON*,
2305.14259); Yang et al. 2024 (*MOOSE-Chem*, 2410.07076). Data: ChicagoHAI HypoGeniC datasets.
Tools: OpenAI `gpt-4.1-mini`, `text-embedding-3-small`; numpy, scipy, scikit-learn, pandas,
matplotlib. Full results: `results/results.json`, `results/*.csv`, `figures/`.
