# Planning — Scientific Idea Generation with Grounded Novelty

## Research Question

Do large language models generate **better scientific ideas (hypotheses)** when the
ideation process is **grounded** — through deduction, induction, empirical "small
experiments" / outcome imagination, literature comparison, or proposal writing — than
when prompted in an abstract, **ungrounded** way?

We operationalize "idea" as a *task hypothesis* (an explicit rule explaining a
classification phenomenon), and "better" as **grounded usefulness**: how well a bank of
generated hypotheses, when applied by an LLM, predicts held-out labels in-distribution
(IND) and out-of-distribution (OOD). We complement this with **diversity** and a
**novelty-vs-usefulness** analysis (the Si et al. caveat: novelty ≠ value).

---

## Motivation & Novelty Assessment

### Why This Research Matters
Automated scientific ideation is advancing fast, but there is real skepticism about
whether LLMs produce *good* ideas: they tend to be either incremental or wildly
implausible, and "novelty" is notoriously hard to define. If we can show *which kinds of
grounding* turn raw LLM ideation into useful, verifiable ideas — and quantify the
cost/benefit — we give practitioners a concrete recipe and move the field past
single-score "novelty" leaderboards toward grounded usefulness.

### Gap in Existing Work (from `literature_review.md` §8)
Every prior method grounds in **one** way and compares to ungrounded prompting (SciMON →
literature; HypoGeniC → data; ResearchAgent → proposals; SGA → simulation). **Nobody has
held task, model, budget, and metric fixed and ablated the grounding *mechanisms* against
each other.** This is exactly our opening. We also heed the literature's central caveat —
novelty is over-trusted (Si et al.'s ideation–execution flip) — by measuring *grounded
usefulness*, not a bare novelty score.

### Our Novel Contribution
A **unified, single-protocol ablation of six ideation regimes** (one ungrounded control +
five grounding mechanisms) with model, task set, generation budget, and evaluation held
fixed, scored by an objective downstream-utility metric (IND + OOD accuracy) plus diversity
and a novelty–usefulness frontier. To our knowledge this head-to-head does not exist.

### Experiment Justification
- **Exp 1 (Main ablation):** Generate hypothesis banks under each regime for several tasks
  and score downstream IND/OOD accuracy. *Needed* to directly test the hypothesis: does
  grounding beat ungrounded, and which grounding wins?
- **Exp 2 (Diversity & novelty–usefulness):** Embedding diversity of each bank, plus
  novelty (distance from held-out expert "known hypotheses") vs usefulness. *Needed* to
  check whether grounding's gains are real usefulness vs mode collapse, and to test the
  novelty≠value caveat.
- **Exp 3 (Cost/benefit):** Accuracy-per-1k-tokens of generation per regime. *Needed*
  because outcome-imagination/small-experiment grounding cost compute (gap #4); we quantify
  whether the accuracy buys its cost.
- **Robustness:** multiple generation seeds (variance of the *ideation process*), bootstrap
  CIs over test items, and a second generator model on one task (cross-model check).

---

## Background and Motivation
See `literature_review.md`. Repeated quantitative evidence that grounding helps
(SciMON +46–58% novelty wins; HypoGeniC +3.3–31.7% acc; Literature-Meets-Data +8.97% vs
few-shot), tempered by the novelty≠value caveat. We reuse the **HypoGeniC/HypoBench shared
task suite** (downloaded in `datasets/`) precisely because it gives an objective,
leakage-controlled utility metric and a single task vocabulary for a clean ablation.

## Hypothesis Decomposition
- **H1 (grounding helps):** Any evidence/reasoning-grounded regime ≥ ungrounded control on
  downstream accuracy.
- **H2 (empirical grounding is strongest):** Regimes grounded in *data and empirical
  feedback* (induction, small-experiments/outcome-imagination) > regimes grounded only in
  reasoning/elaboration (deduction, proposal-writing) > ungrounded.
- **H3 (grounding helps most OOD):** The grounding advantage is larger OOD than IND
  (grounded ideas capture transferable structure, not surface cues).
- **H4 (novelty ≠ usefulness):** The *most novel* bank (farthest from expert hypotheses /
  most internally diverse) is **not** the most useful; grounding improves usefulness without
  requiring maximal novelty.

- **Independent variable:** ideation regime (6 levels).
- **Dependent variables:** IND accuracy, OOD accuracy, bank diversity, novelty distance,
  generation token cost.
- **Controlled:** model, decoding params, #hypotheses per bank, evaluation prompt &
  test items, seeds.

## Proposed Methodology

### Approach
Build one harness that (a) generates a **hypothesis bank** of fixed size K per (task,
regime, seed) under a regime-specific prompt, then (b) evaluates *every* bank with an
**identical** inference procedure: an LLM applies the bank to each test item and predicts
the label. Holding (b) fixed makes regimes comparable apples-to-apples; the only thing that
varies is *how the ideas were grounded*.

### The six regimes (control + five grounding mechanisms; map to hypothesis's list)
- **C0 Ungrounded (control):** task description only — "propose K novel hypotheses." (the
  zero-shot hypothesis-proposer baseline.)
- **C1 Deduction:** reason from first principles / decompose causal factors → derive
  hypotheses. No data. (grounding in *reasoning*.)
- **C2 Induction:** shown K_ex=20 labeled training examples → hypotheses explaining the
  labels. (grounding in *data*; HypoGeniC core.)
- **C3 Small-experiments + outcome-imagination:** induction, then iteratively *test* the
  current hypotheses on a fresh training batch, observe mistakes, and refine. (grounding in
  *empirical feedback*; HypoGeniC reward/refine loop.)
- **C4 Literature-comparison:** generate in rounds; each round is shown the prior idea pool
  and forced to be substantively different (SciMON-style novelty boosting). (grounding
  novelty by *comparison*.) NOTE: to avoid leakage we differentiate against the model's own
  growing pool, **not** the dataset's expert hypotheses (those are held out for evaluation
  only).
- **C5 Proposal-writing:** write a structured mini-proposal (motivation → mechanisms →
  predictions) then distill it into hypotheses. (grounding in *elaboration*.)

### Baselines
C0 ungrounded is the key control. We also report a **random** classifier and a
**majority-class** floor, and the dataset's **expert known-hypotheses** bank as an
(approximate) topline reference (used as a reference only; it is also the novelty anchor).

### Datasets
HypoGeniC *real* tasks (binary classification, 200 train / 500 test / OOD each):
`headline_binary`, `deceptive_reviews`, `dreaddit`. Each ships a task description and a set
of expert `known_hypotheses` we hold out for the novelty analysis. (Add a 4th task if time.)

### Evaluation Metrics
- **Primary:** downstream accuracy on IND test and OOD (N=100 each, fixed subset), via the
  shared bank-application inference prompt. Justified: objective, leakage-controlled, the
  field-standard utility metric for these tasks.
- **Secondary:** bank diversity (mean pairwise cosine distance of hypothesis embeddings);
  novelty distance (mean cosine distance of bank to expert known-hypotheses embeddings);
  accuracy-per-1k generation tokens.

### Statistical Analysis Plan
- 3 generation seeds per (task, regime) → regime means ± SD across seeds.
- Bootstrap 95% CIs over test items for each accuracy estimate.
- Primary test: paired comparison of each grounded regime vs C0, **paired across (task×seed)
  cells**, Wilcoxon signed-rank (non-parametric, robust to n) + paired t; report Cohen's d
  effect size. Bonferroni correction across the 5 grounded-vs-control comparisons.
- H3: paired test on (OOD−IND) grounding advantage. α = 0.05.

## Expected Outcomes
- **Supports H1/H2** if grounded regimes (esp. C2/C3) beat C0 with positive effect sizes.
- **Supports H3** if grounding advantage is larger OOD.
- **Supports H4** if the most diverse/novel bank is not the most accurate.
- **Refutes** if C0 ≈ grounded regimes, or grounding only helps IND (surface fitting).

## Timeline and Milestones
1. Setup + data loaders + API client w/ caching & concurrency — 20 min
2. Implement 6 regimes + shared evaluator — 45 min
3. Smoke test on 1 task, tiny N — 15 min
4. Full run (3 tasks × 6 regimes × 3 seeds; eval N=100 IND+100 OOD) — 60–90 min
5. Analysis + figures + stats — 40 min
6. REPORT.md + README.md — 30 min

## Potential Challenges
- **Cost/throughput:** ~9–11k mini calls. Mitigate: `gpt-4.1-mini`, async concurrency,
  on-disk response cache (idempotent reruns), capped test N.
- **Parsing LLM output:** enforce strict JSON / regex fallback; log parse failures; count
  as abstain (scored wrong, not dropped) to avoid silent bias.
- **Leakage:** never feed expert `known_hypotheses` into generation; use them only as the
  novelty anchor and reference topline.
- **Eval confound:** identical evaluator model/prompt for all regimes; randomize label
  order in the eval prompt to avoid position bias.
- **Mode collapse:** measured directly via diversity.

## Success Criteria
A complete, reproducible single-protocol ablation with: per-regime IND/OOD accuracy with
CIs, significance tests vs control with effect sizes, diversity + novelty–usefulness
analysis, cost/benefit, honest error analysis, and a clear verdict on H1–H4 — even if the
result is null or mixed.
