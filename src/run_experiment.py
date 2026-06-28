"""Main experiment runner: single-protocol ablation of six ideation regimes.

For each (task, regime, seed): generate a hypothesis bank, evaluate it on IND-test and OOD
with the shared classifier, and compute diversity + novelty-distance metrics. Everything is
cached on disk (results/cache), so the run is resumable and reruns are free.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

import data
import regimes as R
import evaluate as E
import llm

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BANKS = RES / "banks"
EVAL = RES / "eval"
for d in (BANKS, EVAL):
    d.mkdir(parents=True, exist_ok=True)

TASKS = ["headline_binary", "deceptive_reviews", "dreaddit"]
SEEDS = [0, 1, 2]
N_EVAL = 100  # per split (IND test + OOD)


def cos(u, v):
    u = np.asarray(u); v = np.asarray(v)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def bank_metrics(task, bank: list[str]) -> dict:
    """Diversity (mean pairwise cosine distance) + novelty distance from expert hyps."""
    emb = np.array(llm.embed(bank))
    n = len(emb)
    # mean pairwise cosine distance within bank
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(1 - cos(emb[i], emb[j]))
    diversity = float(np.mean(dists)) if dists else 0.0
    # novelty: mean over bank of (1 - max cosine sim to any expert known hypothesis)
    novelty = None
    if task.known_hypotheses:
        kemb = np.array(llm.embed(task.known_hypotheses))
        per = []
        for e in emb:
            sims = [cos(e, k) for k in kemb]
            per.append(1 - max(sims))
        novelty = float(np.mean(per))
    return {"diversity": diversity, "novelty_distance": novelty}


def get_bank(task, regime_name, seed) -> list[str]:
    fp = BANKS / f"{task.name}__{regime_name}__s{seed}.json"
    if fp.exists():
        return json.loads(fp.read_text())["bank"]
    bank = R.REGIMES[regime_name](task, seed)
    fp.write_text(json.dumps({"task": task.name, "regime": regime_name,
                              "seed": seed, "bank": bank}, indent=2))
    return bank


def run():
    t0 = time.time()
    records = []
    for tname in TASKS:
        task = data.Task(tname)
        print(f"\n##### TASK {tname} | labels={task.label_values}", flush=True)
        for regime in R.REGIMES:
            for seed in SEEDS:
                bank = get_bank(task, regime, seed)
                bm = bank_metrics(task, bank)
                row = {"task": tname, "regime": regime, "seed": seed,
                       "n_hyp": len(bank), **bm}
                for split in ("test", "ood"):
                    res = E.evaluate_bank(task, bank, split, N_EVAL, seed=0,
                                          max_workers=32,
                                          desc=f"{tname[:6]}/{regime}/s{seed}/{split}")
                    row[f"{split}_acc"] = res["accuracy"]
                    row[f"{split}_abstain"] = res["abstain_rate"]
                    row[f"{split}_n"] = res["n"]
                    # persist per-item rows for bootstrap
                    (EVAL / f"{tname}__{regime}__s{seed}__{split}.json").write_text(
                        json.dumps(res["rows"]))
                records.append(row)
                print(f"  {regime} s{seed}: IND={row['test_acc']:.3f} "
                      f"OOD={row['ood_acc']:.3f} div={bm['diversity']:.3f} "
                      f"nov={bm['novelty_distance']}", flush=True)
    # also evaluate the expert known-hypotheses bank as a reference topline
    expert = []
    for tname in TASKS:
        task = data.Task(tname)
        bank = task.known_hypotheses
        row = {"task": tname, "regime": "EXPERT_ref", "seed": 0, "n_hyp": len(bank),
               **bank_metrics(task, bank)}
        for split in ("test", "ood"):
            res = E.evaluate_bank(task, bank, split, N_EVAL, seed=0, max_workers=32,
                                  desc=f"{tname[:6]}/EXPERT/{split}")
            row[f"{split}_acc"] = res["accuracy"]
            row[f"{split}_abstain"] = res["abstain_rate"]
            row[f"{split}_n"] = res["n"]
        expert.append(row)

    out = {"config": {"tasks": TASKS, "seeds": SEEDS, "n_eval": N_EVAL,
                      "K": R.K, "gen_model": R.GEN_MODEL, "eval_model": E.EVAL_MODEL,
                      "regime_labels": R.REGIME_LABELS},
           "records": records, "expert": expert,
           "tokens": llm.token_report(),
           "elapsed_sec": time.time() - t0}
    (RES / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nDONE in {out['elapsed_sec']:.0f}s -> results/results.json")
    print("tokens:", llm.token_report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    if args.tasks:
        TASKS = args.tasks
    if args.seeds:
        SEEDS = args.seeds
    run()
