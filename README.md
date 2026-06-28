# Scientific Idea Generation with Grounded Novelty

A **single-protocol head-to-head ablation** of six LLM ideation regimes — one ungrounded
control plus five grounding mechanisms (deduction, induction, small-experiments/outcome-
imagination, literature-comparison, proposal-writing) — testing whether *grounding novelty*
makes an LLM's scientific ideas (task hypotheses) **more useful**, measured by objective
downstream classification accuracy on the HypoGeniC tasks.

## Key findings

- **Grounding is *not* a uniform win.** Pooled across mechanisms, grounded hypothesis banks are
  statistically indistinguishable from (slightly below) the ungrounded control (IND −1.2 pts,
  OOD −2.5 pts, n.s.). A strong 2025 LLM is a hard-to-beat ungrounded baseline.
- **Mechanism is decisive.** *Empirical small-experiment + outcome-imagination* grounding (C3)
  is the only regime that significantly improves in-distribution accuracy (+4.4 pts, *d*=1.09,
  Wilcoxon *p*=0.02) — but costs ~80× the generation tokens.
- **Elaboration backfires.** *Proposal-writing* (C5) significantly *hurts* (−5.7 pts, *d*=−1.56,
  *p*=0.008) and collapses idea diversity.
- **Data grounding can overfit.** On `deceptive_reviews` the ungrounded model transfers near-
  perfectly OOD (0.97); data-grounded rules latch onto dataset-specific cues and drop to
  0.78–0.82 OOD.
- **Novelty ≠ usefulness.** The cross-task novelty→usefulness correlation (+0.84) is a task-
  difficulty confound; within task it vanishes (ρ≈−0.10).

Full write-up: **[REPORT.md](REPORT.md)**.

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv add openai numpy scikit-learn matplotlib scipy pandas tenacity
export OPENAI_API_KEY=...                 # required
cd src
python run_experiment.py                  # ~14 min, ~$3, ~11.9k cached API calls
python analyze.py                         # stats + figures
```
All API calls are disk-cached (`results/cache/`), so reruns are free and resumable.

## Structure
```
src/
  llm.py            cached/concurrent/retrying OpenAI client + embeddings + token tracking
  data.py           loads & normalizes the 3 HypoGeniC tasks
  regimes.py        the 6 ideation regimes (generation)
  evaluate.py       shared regime-agnostic bank->prediction classifier
  run_experiment.py orchestrates generation + evaluation (54 banks)
  analyze.py        aggregation, paired stats, bootstrap CIs, figures
planning.md         Phase 0/1 plan (motivation, hypotheses H1-H4, design)
REPORT.md           full research report
results/            results.json, *.csv stats, banks/, eval/ (per-item), cache/
figures/            fig1-fig5 (.png)
datasets/, code/, papers/, literature_review.md, resources.md   pre-gathered resources
```

## Conditions tested
`C0` ungrounded (control) · `C1` deduction · `C2` induction · `C3` small-exp + outcome ·
`C4` literature-comparison · `C5` proposal-writing. Generator & evaluator = `gpt-4.1-mini`;
bank size K=15; 3 tasks × 3 seeds; 100 IND + 100 OOD eval items per bank.
