# Datasets

Datasets gathered for the project **"Scientific Idea Generation with Grounded Novelty."**
The research hypothesis is that LLMs generate *better* scientific ideas when novelty is **grounded**
(through deduction, induction, proposal writing, outcome imagination, small experiments, or literature
comparison) rather than prompted in the abstract. Testing this needs (a) tasks where a *grounded*
hypothesis can be checked against data/literature, and (b) benchmarks of idea/hypothesis novelty.

> **Git note:** data files are **not** committed to git (see `.gitignore`). Re-create this directory by
> running the clone commands below. Each dataset is small enough (< 35 MB/repo) to re-clone quickly.

---

## Dataset 1 — HypoGeniC datasets (data-driven / *induction* grounding)

- **Source**: https://github.com/ChicagoHAI/HypoGeniC-datasets  → cloned to `datasets/hypogenic_datasets/`
- **Used by**: *Hypothesis Generation with Large Language Models* (Zhou & Tan, 2024, arXiv:2404.04326)
- **Local size**: ~29 MB
- **Format**: HuggingFace-style JSON. Each task `<T>` has `<T>_train.json`, `<T>_test.json`,
  (optional `<T>_val.json`), `<T>_ood.json`, plus `config.yaml` (prompt templates) and `metadata.json`
  (task/feature/label descriptions). A JSON file is a dict mapping each text feature and `label`
  to a parallel list of strings.
- **Splits**: typically `train = 200` examples per task, with separate `test` and out-of-distribution
  (`ood`) splits.

### Tasks
- **real/** (9 tasks): `deceptive_reviews`, `dreaddit` (stress detection), `gptgc_detect` /
  `llamagc_detect` (AI-generated-text detection), `headline_binary` (which headline gets more clicks),
  `journal_cross`, `journal_same`, `persuasive_pairs`, `retweet` (which tweet gets more retweets).
- **synthetic/** (8 tasks, controllable difficulty): `admission`, `admission_adv`, `election`,
  `election_controlled`, `preference`, `preference_controlled`, `shoe`, `shoe_two_level`.

### Why relevant
These are classification tasks where an LLM must **induce** explanatory hypotheses (rules/patterns) from
labeled examples, then use them to predict held-out labels. They operationalize the **induction** and
**small-experiments** grounding modes: a hypothesis's quality is measured by the downstream predictive
accuracy it yields — a *grounded* signal of usefulness — instead of an abstract "is this novel?" prompt.

### Sample record (`real/headline_binary/headline_binary_train.json`)
```json
{
  "headline_1": "I Get It. You Don't Think Climate Change Is Man Made. That's Sort Of Irrelevant Because Of This.",
  "headline_2": "Is It Getting Hot In Here Or Is It Just Me? Oh, Damn. It's Actually Getting Hot In Here.",
  "label": "Headline 1 has more clicks than Headline 2."
}
```

### Download
```bash
git clone https://github.com/ChicagoHAI/HypoGeniC-datasets datasets/hypogenic_datasets
```

### Loading
```python
import json
d = json.load(open("datasets/hypogenic_datasets/real/headline_binary/headline_binary_train.json"))
# d is {"headline_1": [...], "headline_2": [...], "label": [...]} — parallel lists
```

---

## Dataset 2 — HypoBench datasets (*literature + data* grounding, controllable difficulty)

- **Source**: https://github.com/ChicagoHAI/HypoBench-datasets → cloned to `datasets/hypobench_datasets/`
- **Used by**: *HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation*
  (2025, arXiv:2504.11524)
- **Local size**: ~32 MB
- **Format / splits**: same HuggingFace-style JSON layout as Dataset 1 (`real/` + `synthetic/` with the
  same task names). Synthetic tasks expose *controlled* variants (e.g. `election_controlled`,
  `preference_controlled`, `shoe_two_level`) for systematically varying hypothesis-discovery difficulty
  and measuring how close generated hypotheses get to a known ground-truth rule.

### Why relevant
HypoBench is the principled benchmark for hypothesis generation. It evaluates hypotheses on
**ground-truth recovery** and downstream utility, giving a *grounded* yardstick for novelty/quality —
directly useful for testing "grounded vs. ungrounded" generation under controlled conditions.

### Download
```bash
git clone https://github.com/ChicagoHAI/HypoBench-datasets datasets/hypobench_datasets
```

---

## Dataset 3 — Hypothesis-agent datasets (*literature-comparison* grounding: PDFs + processed papers)

- **Source**: https://github.com/ChicagoHAI/Hypothesis-agent-datasets → cloned to
  `datasets/hypothesis_agent_datasets/`
- **Used by**: HypoRefine / Literature∪HypoGeniC in *Literature Meets Data* (2024, arXiv:2410.17309)
- **Local size**: ~9 MB
- **Format**: per-task folders (`deceptive_reviews`, `dreaddit`, `gptgc_detect`, `llamagc_detect`,
  `persuasive_pairs`) each containing `raw/` (literature **PDFs**) and `processed/` (S2ORC-style JSON of
  parsed papers, plus `papers_info.jsonl`). The same per-task literature also ships inside the framework
  repo at `code/hypogenic_chicagohai/literature/`.

### Why relevant
This is the **literature-comparison** grounding ingredient: relevant papers per task that HypoRefine /
Union methods read to ground hypotheses in prior work (and to check what is already known → novelty).

### Download
```bash
git clone https://github.com/ChicagoHAI/Hypothesis-agent-datasets datasets/hypothesis_agent_datasets
```

---

## Additional / on-demand data sources (not pre-cloned)

These benchmarks target **research-idea generation & novelty evaluation** directly and can be added if the
experiment needs idea-level (rather than data-rule) hypotheses:

| Benchmark | Paper | Where | Note |
|-----------|-------|-------|------|
| **ICLR/AI-Researcher ideation data** | Si et al. 2024 (arXiv:2409.04109) | `code/ai_researcher_si2024/` | prompts, generated ideas & human review scores from the 100+ researcher study |
| **MOOSE-Chem benchmark** | Yang et al. 2024 (arXiv:2410.07076) | `code/moose_chem/Data/` | 51 high-impact 2024 chemistry papers annotated (background, inspirations, ground-truth hypothesis) |
| **IdeaBench** | 2024 (arXiv:2411.02429) | GitHub `IdeaBench` | research-idea generation benchmark + metrics |
| **ResearchBench** | 2025 (arXiv:2503.21248) | GitHub | inspiration-based task decomposition for scientific discovery |
| **SciMON corpus** | Wang et al. 2023 (arXiv:2305.14259) | `code/scimon_clbd/data/` | NLP/biomedical literature-grounded idea generation data |

> The **Semantic Scholar API** (and the included `paper_search_results/all_papers_consolidated.json`,
> 218 papers) can serve as the live literature corpus for the *literature-comparison* grounding condition.

---

## Recommendation for the experiment

Primary: **HypoGeniC datasets** (ungrounded prompt vs. induction-grounded hypotheses, measured by
downstream accuracy) + **Hypothesis-agent literature** (adds literature-comparison grounding) + a few
**HypoBench synthetic** tasks (controlled ground-truth for clean novelty/quality measurement). These
three share a task vocabulary and a single framework (`code/hypogenic_chicagohai`), so all six grounding
conditions can be ablated on the *same* tasks with minimal glue code. For idea-level (not data-rule)
novelty, add the **MOOSE-Chem** annotated benchmark and/or the **Si et al.** ideation data.
