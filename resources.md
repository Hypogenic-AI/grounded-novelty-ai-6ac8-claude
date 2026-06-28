# Resources Catalog — Scientific Idea Generation with Grounded Novelty

## Summary

All resources gathered for testing the hypothesis that **LLMs generate better scientific ideas when
novelty is grounded** (deduction, induction, proposal writing, outcome imagination, small experiments,
literature comparison) rather than via abstract/ungrounded prompts.

- **Papers downloaded**: 20 PDFs (`papers/`), curated from **218** unique candidates
  (`paper_search_results/all_papers_consolidated.json`).
- **Datasets**: 3 cloned repos (`datasets/`) covering the shared HypoGeniC / HypoBench task suite.
- **Code repositories**: 6 cloned (`code/`), incl. the ChicagoHAI framework that implements the core
  grounded vs. ungrounded conditions.
- Full analysis: `literature_review.md`. Per-directory detail: `papers/README.md`,
  `datasets/README.md`, `code/README.md`.

---

## Papers (20)

| # | Title | Year | arXiv | Grounding angle |
|---|-------|------|-------|-----------------|
| 1 | Can LLMs Generate Novel Research Ideas? (100+ NLP researchers) | 2024 | 2409.04109 | Novelty **evaluation** (human study) |
| 2 | SciMON: Scientific Inspiration Machines Optimized for Novelty | 2023 | 2305.14259 | Literature comparison (novelty loop) |
| 3 | Hypothesis Generation with LLMs (HypoGeniC) | 2024 | 2404.04326 | Induction / data-driven |
| 4 | LLMs for Automated Open-domain Scientific Hypotheses Discovery (MOOSE) | 2023 | 2309.02726 | Induction + feedback |
| 5 | Literature Meets Data (HypoRefine / Union) | 2024 | 2410.17309 | Literature + induction + small-exp |
| 6 | HypoBench | 2025 | 2504.11524 | Benchmark (grounded) |
| 7 | Literature-Grounded Novelty Assessment of Scientific Ideas | 2025 | 2506.22026 | Grounded novelty **evaluation** |
| 8 | Evaluating & Enhancing LLMs for Novelty Assessment (SchNovel/RAG-Novelty) | 2024 | 2409.16605 | Grounded novelty **evaluation** |
| 9 | MOOSE-Chem (chemistry rediscovery) | 2024 | 2410.07076 | Deduction/decomposition + literature |
| 10 | LLM + Simulation as Bilevel Optimizers (SGA) | 2024 | 2405.09783 | Small experiments + outcome imagination |
| 11 | ResearchAgent | 2024 | 2404.07738 | Proposal writing (agentic) |
| 12 | Chain of Ideas | 2024 | 2410.13185 | Literature chain |
| 13 | Nova: Iterative Planning & Search for Novelty | 2024 | 2410.14255 | Planning/search for novelty |
| 14 | Scideator (facet recombination + novelty eval) | 2024 | 2409.14634 | Recombination + novelty eval |
| 15 | HypER: Literature-grounded Hypothesis Generation | 2025 | 2506.12937 | Literature-grounded + reasoning |
| 16 | LLMs are Zero-Shot Hypothesis Proposers | 2023 | 2311.05965 | **Ungrounded baseline** |
| 17 | Learning to Generate Research Idea with Dynamic Control | 2024 | 2412.14626 | Controlled novelty/feasibility |
| 18 | IdeaBench | 2024 | 2411.02429 | Benchmark (idea generation) |
| 19 | ResearchBench | 2025 | 2503.21248 | Benchmark (inspiration decomposition) |
| 20 | Survey on Hypothesis Generation in the Era of LLMs | 2025 | 2504.05496 | Survey |

See `papers/README.md` for abstracts and `literature_review.md` §10 for full structured notes.

## Datasets (3 repos, ~70 MB)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| HypoGeniC datasets | github.com/ChicagoHAI/HypoGeniC-datasets | ~29 MB | data-rule hypothesis gen → classification | `datasets/hypogenic_datasets/` | 9 real + 8 synthetic tasks; HF-JSON; 200 train each |
| HypoBench datasets | github.com/ChicagoHAI/HypoBench-datasets | ~32 MB | benchmark, controllable difficulty | `datasets/hypobench_datasets/` | same task vocab; ground-truth recovery |
| Hypothesis-agent datasets | github.com/ChicagoHAI/Hypothesis-agent-datasets | ~9 MB | literature for grounded methods | `datasets/hypothesis_agent_datasets/` | per-task PDFs + parsed papers |

Data files are git-ignored; re-clone via `datasets/README.md`. Idea-level benchmarks (MOOSE-Chem,
SchNovel, IdeaBench, ResearchBench) ship with the cloned code or are downloadable on demand.

## Code Repositories (6)

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| hypogenic_chicagohai ⭐ | github.com/ChicagoHAI/hypothesis-generation | **Primary backbone**: HypoGeniC (induction) + HypoRefine/Union (literature+data) + HypoBench; includes ungrounded baseline | `code/hypogenic_chicagohai/` |
| ai_researcher_si2024 ⭐ | github.com/NoviScl/AI-Researcher | Ideation agent + human-study ideas/reviews + Ideation–Execution Gap data | `code/ai_researcher_si2024/` |
| moose_chem | github.com/ZonglinY/MOOSE-Chem | Multi-agent chemistry hypothesis discovery + 51-paper annotated benchmark | `code/moose_chem/` |
| chain_of_ideas | github.com/DAMO-NLP-SG/CoI-Agent | Literature-chain ideation + Idea-Arena eval | `code/chain_of_ideas/` |
| scimon_clbd | github.com/eaglew/clbd | SciMON: inspiration retrieval + iterative novelty-boosting loop | `code/scimon_clbd/` |

(`code/README.md` has install notes, entry points, and the mechanism→repo map.) Bulky data/model
artifacts are git-ignored via `code/.gitignore`.

## Resource Gathering Notes

### Search strategy
Used the local **paper-finder** service (diligent + fast modes) with 4 complementary queries
(grounding-novelty, hypothesis-generation, idea-generation, novelty-assessment), yielding 218 unique
papers after dedup. Ranked by relevance (129 at 3/3) and curated 20 spanning every grounding mechanism +
the ungrounded baseline + evaluation. arXiv IDs resolved via Semantic Scholar (with arXiv title-search
fallback when rate-limited); PDFs pulled from arXiv (20/20 succeeded). Repos discovered via the papers'
code links and the UChicago/ChicagoHAI authorship of the core data-driven line.

### Selection criteria
Coverage of each named grounding mode; the seminal evaluation study (Si et al.); shared-task benchmarks
enabling a clean ablation; code availability; recency (2023–2025) plus the highest-cited foundations.

### Challenges & workarounds
- Semantic Scholar 429 rate-limits → arXiv title-search fallback (all 20 resolved).
- A `head`-truncated pipe initially broke a JSONL save → re-ran with full output capture.
- **Verification catch**: the parallel deep-reader confused *Literature Meets Data* with *SciMON*; the
  main agent re-read both that paper and the SchNovel paper directly and corrected the notes.

### Quality/verification
20/20 PDFs verified as valid (`%PDF` header + size). Core-paper notes spot-checked against the PDFs;
one confusion found and fixed (above). Dataset schemas inspected (HF-JSON, sample records confirmed).

## Recommendations for Experiment Design

1. **Primary dataset(s)**: HypoGeniC real tasks (deceptive_reviews, headline_binary, dreaddit/AIGC) +
   2 HypoBench synthetic controllable tasks + MOOSE-Chem for idea-level novelty.
2. **Baselines / conditions**: ungrounded zero/few-shot → induction (HypoGeniC) → literature-comparison
   (SciMON-style / HypoRefine) → literature+data Union → outcome/small-experiment grounding →
   proposal-writing; model & task held fixed.
3. **Metrics**: downstream IND/OOD accuracy + HypoBench discovery rate / MOOSE-Chem rediscovery (primary);
   grounded novelty (RAG-Novelty) + feasibility, Cohen's κ for judging (secondary). Report idea diversity.
4. **Code to reuse**: `hypogenic_chicagohai` as backbone; SciMON novelty loop; MOOSE-Chem benchmark;
   `ai_researcher_si2024` review data/agent for human-aligned evaluation.
