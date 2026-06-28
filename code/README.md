# Cloned Code Repositories

Code resources for **"Scientific Idea Generation with Grounded Novelty."** Each repo implements one or
more of the grounding mechanisms in the hypothesis (deduction, induction, proposal writing, outcome
imagination, small experiments, literature comparison) or provides idea/novelty **evaluation**.

> **Git note:** these are third-party clones. Source is kept for reference; bulky `data/`, model and
> result artifacts are excluded via `code/.gitignore`. Nested `.git` directories were removed — re-clone
> from the URLs below to get full history/data.

---

## ⭐ 1. `hypogenic_chicagohai/` — HypoGeniC + HypoRefine + Union (PRIMARY backbone)

- **URL**: https://github.com/ChicagoHAI/hypothesis-generation  (ChicagoHAI / Chenhao Tan's group)
- **Implements three papers in one framework**:
  - *Hypothesis Generation with LLMs* (Zhou & Tan 2024) → **HypoGeniC**: data-driven, in-context
    hypothesis generation with a multi-armed-bandit reward (**induction** grounding).
  - *Literature Meets Data* (2024) → **HypoRefine** (agentic literature+data) and **Union** methods
    (`Literature∪HypoGeniC`, `Literature∪HypoRefine`) → **literature-comparison + induction** grounding.
  - *HypoBench* (2025) → principled benchmarking.
- **Install**: `pip install hypogenic` (API models) or `pip install -e .` from the clone. Literature
  feature needs `pip install git+https://github.com/allenai/s2orc-doc2json`. Optional Redis cache.
- **Key entry points**: `pipeline.py`, `run_pipeline.sh`, `run_pipeline_synth.sh`,
  `run_cross_model_ind.sh`/`_ood.sh`; library in `hypogenic/`; CLI in `hypogenic_cmd/`; worked
  scripts in `examples/` (`generation.py`, `inference.py`, `union_generation.py`,
  `baseline_no_hypothesis_inference.py`, `pdf_preprocess.py`).
- **Bundled literature** in `literature/<task>/{raw,processed}/` for the literature-grounded methods.
- **Datasets**: in sibling `datasets/hypogenic_datasets`, `datasets/hypobench_datasets`,
  `datasets/hypothesis_agent_datasets` (cloned separately).
- **Why central**: it already implements *ungrounded baseline* (`baseline_no_hypothesis_inference`),
  *induction* (HypoGeniC), and *literature-comparison* (HypoRefine/Union) on a **shared task set with a
  shared metric** (downstream accuracy + ground-truth recovery) — exactly the ablation the hypothesis
  needs. Recommended as the experimental skeleton.

## ⭐ 2. `ai_researcher_si2024/` — Stanford research-ideation agent + human-study data

- **URL**: https://github.com/NoviScl/AI-Researcher  (Si, Yang, Hashimoto 2024, arXiv:2409.04109)
- **Provides**: (a) the **research-ideation agent** (retrieval-augmented idea generation with novelty
  re-ranking ≈ **literature-comparison** + **proposal-writing**); (b) **all ideas + reviews** from the
  100+ NLP-researcher blind study; (c) the **Ideation–Execution Gap** data (43 executed ideas, reviews
  before vs. after execution).
- **Dirs**: `ai_researcher/` (agent), `reviews_ideation/`, `reviews_execution/`, `figures/`.
- **Why central**: the canonical **human evaluation** of LLM idea novelty. Headline finding — LLM ideas
  judged *more novel* but *less feasible* than experts'; after execution the novelty advantage erodes.
  This is the strongest evidence on *how* to measure "better ideas," and a caution that novelty alone is
  not enough (motivates **outcome-imagination / small-experiments** grounding).

## 3. `moose_chem/` — multi-agent chemistry hypothesis discovery + annotated benchmark

- **URL**: https://github.com/ZonglinY/MOOSE-Chem  (Yang et al. 2024, arXiv:2410.07076)
- **Implements**: decomposition `hypothesis = background + inspirations`; subtasks = inspiration
  retrieval → hypothesis composition → ranking (**literature-comparison + deduction**).
- **Benchmark**: 51 high-impact 2024 chemistry papers annotated by PhD chemists (background,
  inspirations, ground-truth hypothesis) in `Data/`; pipeline scripts `main.sh`,
  `Assumption1-RetrieveUnseen.sh`, `Assumption2-Reason2Unknown.sh`.
- **Why relevant**: gives an idea-level **ground-truth novelty** signal (rediscovery similarity to real
  unseen hypotheses), complementary to the data-rule tasks of HypoGeniC. Related: MOOSE
  (open-domain, arXiv:2309.02726) at https://github.com/ZonglinY/MOOSE.

## 4. `chain_of_ideas/` — Chain-of-Ideas (CoI) agent

- **URL**: https://github.com/DAMO-NLP-SG/CoI-Agent  (arXiv:2410.13185)
- **Implements**: organizes retrieved literature into a **chain** of progressing ideas to expose research
  trends, then generates the next idea (**literature-comparison** structured as a temporal chain).
  Includes an "Idea Arena" LLM-judge evaluation protocol.
- **Entry points**: `main.py`, `agents.py`, `searcher/`, `prompts/`, `config.yaml`. Live demo on HF Spaces.

## 5. `scimon_clbd/` — SciMON

- **URL**: https://github.com/eaglew/clbd  (Wang, Downey, Ji, Hope 2023, arXiv:2305.14259)
- **Implements**: retrieval of "inspirations" from prior papers + an explicit **novelty-optimization
  loop** that iteratively compares a candidate idea to prior work and revises until novel enough — the
  reference implementation of **literature-comparison grounding**. Includes NLP and biomedical variants,
  preprocessing, and evaluation.
- **Note**: large repo (data-heavy). The novelty-iteration logic is the part most relevant to reuse.

---

## How the repos map to the six grounding mechanisms

| Grounding mechanism | Primary code |
|---------------------|--------------|
| Induction (data-driven) | `hypogenic_chicagohai` (HypoGeniC) |
| Literature comparison | `scimon_clbd`, `chain_of_ideas`, `hypogenic_chicagohai` (HypoRefine/Union), `ai_researcher_si2024` |
| Proposal writing | `ai_researcher_si2024`, (ResearchAgent — see papers) |
| Deduction / decomposition | `moose_chem` |
| Small experiments / outcome imagination | `hypogenic_chicagohai` (downstream-accuracy reward), SGA (paper arXiv:2405.09783) |
| Novelty **evaluation** | `ai_researcher_si2024` (human study), `chain_of_ideas` (Idea Arena), `hypogenic_chicagohai`/HypoBench (ground-truth recovery) |

## Recommendation
Use **`hypogenic_chicagohai`** as the experimental backbone (it already contains the ungrounded baseline,
induction, and literature-comparison conditions on a shared task set with shared metrics). Borrow
**SciMON's** novelty-iteration loop and **MOOSE-Chem's** ground-truth rediscovery benchmark to cover
idea-level novelty, and reuse **`ai_researcher_si2024`** review data / agent for human-aligned novelty
evaluation. See `../literature_review.md` for the methodological synthesis and `../resources.md` for the
full catalog.
