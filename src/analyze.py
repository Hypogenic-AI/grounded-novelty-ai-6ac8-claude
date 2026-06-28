"""Analysis: aggregation, statistics, figures for the grounding-regime ablation."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import llm
import data
import regimes as R

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
EVAL = RES / "eval"
FIG.mkdir(exist_ok=True)

RESULTS = json.loads((RES / "results.json").read_text())
DF = pd.DataFrame(RESULTS["records"])
EXPERT = pd.DataFrame(RESULTS["expert"])
RLAB = RESULTS["config"]["regime_labels"]
ORDER = list(R.REGIMES.keys())
CONTROL = "C0_ungrounded"


def cohens_d_paired(diff):
    diff = np.asarray(diff)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def boot_ci(vals, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled_item_acc(regime, split):
    """Pool per-item correctness across tasks & seeds for a regime+split."""
    corr = []
    for tname in RESULTS["config"]["tasks"]:
        for seed in RESULTS["config"]["seeds"]:
            fp = EVAL / f"{tname}__{regime}__s{seed}__{split}.json"
            if fp.exists():
                rows = json.loads(fp.read_text())
                corr.extend(r["correct"] for r in rows)
    return np.array(corr, float)


def main():
    print("=" * 70)
    print("AGGREGATE: mean accuracy across 3 tasks x 3 seeds (9 cells per regime)")
    print("=" * 70)
    agg = DF.groupby("regime").agg(
        test_acc=("test_acc", "mean"), test_sd=("test_acc", "std"),
        ood_acc=("ood_acc", "mean"), ood_sd=("ood_acc", "std"),
        diversity=("diversity", "mean"), novelty=("novelty_distance", "mean"),
        abstain=("test_abstain", "mean"),
    ).reindex(ORDER)
    agg["label"] = [RLAB[r] for r in agg.index]
    pd.set_option("display.width", 160)
    print(agg.round(4).to_string())

    # bootstrap CIs (pooled items)
    print("\nBootstrap 95% CI on pooled-item accuracy:")
    ci_rows = []
    for r in ORDER:
        for split in ("test", "ood"):
            v = pooled_item_acc(r, split)
            lo, hi = boot_ci(v)
            ci_rows.append({"regime": r, "split": split, "acc": v.mean(),
                            "lo": lo, "hi": hi, "n": len(v)})
            print(f"  {r:24s} {split:4s} acc={v.mean():.3f} [{lo:.3f},{hi:.3f}] n={len(v)}")
    pd.DataFrame(ci_rows).to_csv(RES / "bootstrap_ci.csv", index=False)

    # paired tests vs control across (task x seed) cells
    print("\n" + "=" * 70)
    print("PAIRED TESTS: each grounded regime vs C0 control (9 paired task x seed cells)")
    print("Bonferroni m=5 grounded regimes; report IND and OOD")
    print("=" * 70)
    key = ["task", "seed"]
    ctrl = DF[DF.regime == CONTROL].set_index(key)
    stat_rows = []
    grounded = [r for r in ORDER if r != CONTROL]
    m = len(grounded)
    for split in ("test", "ood"):
        for r in grounded:
            sub = DF[DF.regime == r].set_index(key)
            j = sub.join(ctrl, lsuffix="_g", rsuffix="_c")
            diff = (j[f"{split}_acc_g"] - j[f"{split}_acc_c"]).values
            t_p = stats.ttest_rel(j[f"{split}_acc_g"], j[f"{split}_acc_c"]).pvalue
            try:
                w_p = stats.wilcoxon(diff).pvalue
            except ValueError:
                w_p = float("nan")
            d = cohens_d_paired(diff)
            stat_rows.append({"split": split, "regime": r, "mean_gain": diff.mean(),
                              "cohens_d": d, "t_p": t_p, "t_p_bonf": min(t_p * m, 1.0),
                              "wilcoxon_p": w_p, "wilcoxon_p_bonf": min(w_p * m, 1.0)})
            print(f"  [{split}] {r:24s} gain={diff.mean():+.3f} d={d:+.2f} "
                  f"t_p={t_p:.4f} (bonf {min(t_p*m,1):.4f})  W_p={w_p:.4f}")
    pd.DataFrame(stat_rows).to_csv(RES / "paired_tests.csv", index=False)

    # H1 overall: all grounded pooled vs control
    print("\n" + "=" * 70)
    print("H1 (overall): grounded regimes (C1-C5) pooled vs C0")
    print("=" * 70)
    for split in ("test", "ood"):
        g = DF[DF.regime != CONTROL].groupby(key)[f"{split}_acc"].mean()
        c = DF[DF.regime == CONTROL].set_index(key)[f"{split}_acc"]
        j = pd.concat([g.rename("g"), c.rename("c")], axis=1).dropna()
        diff = (j.g - j.c).values
        t_p = stats.ttest_rel(j.g, j.c).pvalue
        w_p = stats.wilcoxon(diff).pvalue
        print(f"  [{split}] grounded={j.g.mean():.3f} vs control={j.c.mean():.3f} "
              f"gain={diff.mean():+.3f} d={cohens_d_paired(diff):+.2f} "
              f"t_p={t_p:.4f} W_p={w_p:.4f}")

    # H3: OOD - IND grounding advantage per regime vs control
    print("\n" + "=" * 70)
    print("H3: is the grounding advantage larger OOD than IND? (gain_OOD - gain_IND)")
    print("=" * 70)
    for r in grounded:
        sub = DF[DF.regime == r].set_index(key)
        j = sub.join(ctrl, lsuffix="_g", rsuffix="_c")
        gain_ind = (j["test_acc_g"] - j["test_acc_c"]).values
        gain_ood = (j["ood_acc_g"] - j["ood_acc_c"]).values
        print(f"  {r:24s} gain_IND={gain_ind.mean():+.3f} gain_OOD={gain_ood.mean():+.3f} "
              f"Δ={gain_ood.mean()-gain_ind.mean():+.3f}")

    # H4: novelty vs usefulness (per task x regime mean)
    print("\n" + "=" * 70)
    print("H4: novelty (distance from expert hyps) vs usefulness (OOD acc)")
    print("=" * 70)
    cell = DF.groupby(["task", "regime"]).agg(
        ood_acc=("ood_acc", "mean"), novelty=("novelty_distance", "mean"),
        diversity=("diversity", "mean")).reset_index()
    rho_n, p_n = stats.spearmanr(cell.novelty, cell.ood_acc)
    rho_d, p_d = stats.spearmanr(cell.diversity, cell.ood_acc)
    print(f"  Spearman(novelty, OOD acc) = {rho_n:+.3f} (p={p_n:.3f})")
    print(f"  Spearman(diversity, OOD acc) = {rho_d:+.3f} (p={p_d:.3f})")
    # most novel vs most useful regime per task
    for tname in RESULTS["config"]["tasks"]:
        c = cell[cell.task == tname]
        mn = c.loc[c.novelty.idxmax(), "regime"]
        mu = c.loc[c.ood_acc.idxmax(), "regime"]
        print(f"  {tname:18s} most-novel={mn:22s} most-useful={mu}")

    # expert reference
    print("\nEXPERT known-hypotheses reference (topline):")
    print(EXPERT[["task", "test_acc", "ood_acc", "novelty_distance"]].round(3).to_string(index=False))

    make_figures(agg, ci_rows, cell, stat_rows)
    print("\nFigures written to figures/. Stats CSVs in results/.")


def make_figures(agg, ci_rows, cell, stat_rows):
    cidf = pd.DataFrame(ci_rows)
    labels = [RLAB[r] for r in ORDER]
    x = np.arange(len(ORDER))

    # Fig 1: IND & OOD accuracy bars with bootstrap CI
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for off, split, col in [(-0.2, "test", "#4C72B0"), (0.2, "ood", "#DD8452")]:
        sub = cidf[cidf.split == split].set_index("regime").reindex(ORDER)
        err = np.vstack([sub.acc - sub.lo, sub.hi - sub.acc])
        ax.bar(x + off, sub.acc, 0.4, yerr=err, capsize=3,
               label="IND test" if split == "test" else "OOD", color=col)
    # control line + chance
    c_ood = cidf[(cidf.split == "ood") & (cidf.regime == CONTROL)].acc.values[0]
    ax.axhline(0.5, ls=":", c="gray", label="chance (0.50)")
    ax.axhline(c_ood, ls="--", c="#DD8452", alpha=0.6, label="C0 OOD control")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (pooled items, 95% bootstrap CI)")
    ax.set_title("Downstream usefulness of hypothesis banks by ideation regime\n"
                 "(3 tasks x 3 seeds, gpt-4.1-mini generator & evaluator)")
    ax.set_ylim(0.45, max(0.85, cidf.hi.max() + 0.03)); ax.legend(ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig1_accuracy_by_regime.png", dpi=140); plt.close(fig)

    # Fig 2: grounding advantage (gain vs control) IND vs OOD
    sdf = pd.DataFrame(stat_rows)
    g = [r for r in ORDER if r != CONTROL]
    xx = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(10, 5))
    for off, split, col in [(-0.2, "test", "#4C72B0"), (0.2, "ood", "#DD8452")]:
        s = sdf[sdf.split == split].set_index("regime").reindex(g)
        ax.bar(xx + off, s.mean_gain, 0.4, label=("IND" if split == "test" else "OOD"), color=col)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels([RLAB[r] for r in g], rotation=20, ha="right")
    ax.set_ylabel("Accuracy gain vs ungrounded control")
    ax.set_title("Grounding advantage over the ungrounded control (H1/H2/H3)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "fig2_grounding_advantage.png", dpi=140); plt.close(fig)

    # Fig 3: novelty vs usefulness scatter (H4)
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"headline_binary": "o", "deceptive_reviews": "s", "dreaddit": "^"}
    cmap = {r: plt.cm.tab10(i) for i, r in enumerate(ORDER)}
    for _, row in cell.iterrows():
        ax.scatter(row.novelty, row.ood_acc, marker=markers[row.task],
                   color=cmap[row.regime], s=120, edgecolor="k", zorder=3)
    for r in ORDER:
        ax.scatter([], [], color=cmap[r], label=RLAB[r], s=80)
    for t, mk in markers.items():
        ax.scatter([], [], marker=mk, color="gray", label=t, s=80)
    ax.set_xlabel("Novelty: mean distance from expert hypotheses")
    ax.set_ylabel("Usefulness: OOD accuracy")
    ax.set_title("Novelty vs usefulness (H4): more novel is NOT more useful")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(FIG / "fig3_novelty_vs_usefulness.png", dpi=140); plt.close(fig)

    # Fig 4: cost/benefit — OOD acc vs generation tokens per regime
    gen_tok = generation_tokens_per_regime()
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in ORDER:
        a = agg.loc[r, "ood_acc"]; tk = gen_tok.get(r, np.nan)
        ax.scatter(tk / 1000, a, color=cmap[r], s=140, edgecolor="k", zorder=3)
        ax.annotate(RLAB[r], (tk / 1000, a), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Generation cost (1k tokens per bank, mean)")
    ax.set_ylabel("OOD accuracy")
    ax.set_title("Cost/benefit of grounding (gap #4): accuracy vs generation tokens")
    fig.tight_layout(); fig.savefig(FIG / "fig4_cost_benefit.png", dpi=140); plt.close(fig)

    # Fig 5: per-task OOD accuracy heatmap
    piv = DF.groupby(["regime", "task"]).ood_acc.mean().unstack().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto", vmin=0.45, vmax=0.8)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([RLAB[r] for r in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center",
                    color="w", fontsize=9)
    fig.colorbar(im, label="OOD accuracy"); ax.set_title("OOD accuracy per task x regime")
    fig.tight_layout(); fig.savefig(FIG / "fig5_per_task_heatmap.png", dpi=140); plt.close(fig)

    # save generation tokens
    (RES / "generation_tokens.json").write_text(json.dumps(gen_tok, indent=2))


def generation_tokens_per_regime() -> dict:
    """Attribute generation tokens per regime by re-running (cached) bank generation."""
    out = {}
    tasks = [data.Task(t) for t in RESULTS["config"]["tasks"]]
    for r in ORDER:
        llm.TOKENS.clear()
        n = 0
        for task in tasks:
            for seed in RESULTS["config"]["seeds"]:
                R.REGIMES[r](task, seed)  # cached -> instant, records gen tokens
                n += 1
        rep = llm.token_report().get(R.GEN_MODEL, {"input": 0, "output": 0})
        out[r] = (rep["input"] + rep["output"]) / n
    return out


if __name__ == "__main__":
    main()
