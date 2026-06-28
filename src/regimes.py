"""The six ideation regimes (1 ungrounded control + 5 grounding mechanisms).

Each regime returns a hypothesis bank: a list of K decision-rule strings for the task.
Generator model, decoding params and bank size K are held fixed across regimes; only the
*grounding scaffold* differs. Seed varies for variance estimation.
"""
from __future__ import annotations
import json
import random
import re

from llm import chat
from evaluate import predict_one

GEN_MODEL = "gpt-4.1-mini"
K = 15            # hypotheses per bank
K_EXAMPLES = 20   # labeled examples shown to data-grounded regimes
TEMP = 0.7

SYS = (
    "You are a scientist generating hypotheses for a binary classification task. "
    "A hypothesis is a clear, general, testable decision rule that helps predict the label "
    "from the features. Good hypotheses are specific enough to act on, general enough to "
    "transfer to new instances, and grounded in plausible mechanisms."
)


def _labels_line(task) -> str:
    return f'The two possible labels are: "{task.label_values[0]}" and "{task.label_values[1]}".'


def _parse_bank(text: str, k: int = K) -> list[str]:
    """Extract a list of hypothesis strings from JSON or numbered/bulleted text."""
    text = text.strip()
    # try fenced or raw JSON array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            hyps = [str(x).strip() for x in arr if str(x).strip()]
            if hyps:
                return hyps[:k]
        except Exception:  # noqa: BLE001
            pass
    # fallback: split numbered / bulleted lines
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        ln = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln)
        if len(ln) > 15:
            lines.append(ln)
    return lines[:k]


def _ask_bank(user: str, k: int, seed: int, sys: str = SYS) -> list[str]:
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    out = chat(msgs, model=GEN_MODEL, temperature=TEMP, max_tokens=1600, seed=seed)
    return _parse_bank(out, k)


def _format_examples(recs) -> str:
    return "\n\n".join(f"[Example {i+1}] {r['text']}\nLabel: {r['label']}"
                       for i, r in enumerate(recs))


def _sample_train(task, n, seed):
    rng = random.Random(seed)
    pool = list(task.splits["train"])
    rng.shuffle(pool)
    # balance across the two labels
    a, b = task.label_values
    pa = [r for r in pool if r["label"] == a]
    pb = [r for r in pool if r["label"] == b]
    half = n // 2
    sel = pa[:half] + pb[:n - half]
    rng.shuffle(sel)
    return sel


# ----------------------------------------------------------------------------- regimes
def gen_ungrounded(task, seed) -> list[str]:
    """C0 control: only the task description; no data, no reasoning scaffold."""
    user = (
        f"Task: {task.task_description}\n{_labels_line(task)}\n\n"
        f"Propose {K} novel, insightful hypotheses (decision rules) that would help predict "
        f"the label. Return them as a JSON array of {K} concise strings."
    )
    return _ask_bank(user, K, seed)


def gen_deduction(task, seed) -> list[str]:
    """C1: reason from first principles / decompose causal factors, then derive rules."""
    user = (
        f"Task: {task.task_description}\n{_labels_line(task)}\n\n"
        "Reason from first principles. Step 1: decompose the phenomenon into the underlying "
        "causal factors and mechanisms that plausibly drive the label. Step 2: from those "
        "mechanisms, DERIVE concrete, testable decision rules.\n\n"
        f"After your reasoning, output a JSON array of exactly {K} concise hypothesis strings "
        "(the derived rules)."
    )
    return _ask_bank(user, K, seed)


def gen_induction(task, seed) -> list[str]:
    """C2: induce hypotheses directly from labeled training examples."""
    recs = _sample_train(task, K_EXAMPLES, seed)
    user = (
        f"Task: {task.task_description}\n{_labels_line(task)}\n\n"
        f"Below are {len(recs)} labeled examples. Examine them carefully and INDUCE general "
        f"hypotheses (decision rules) that explain why each example has its label and that "
        f"would generalize to new instances.\n\n{_format_examples(recs)}\n\n"
        f"Output a JSON array of exactly {K} concise hypothesis strings."
    )
    return _ask_bank(user, K, seed)


def gen_empirical(task, seed, rounds: int = 2) -> list[str]:
    """C3: small-experiments + outcome-imagination.

    Start from an inductive bank, then iteratively (a) TEST it on a fresh training batch,
    (b) observe which examples it gets wrong (the empirical outcome), and (c) refine the
    bank to fix those errors. Grounding in empirical feedback.
    """
    bank = gen_induction(task, seed)
    for r in range(rounds):
        batch = _sample_train(task, 24, seed + 100 * (r + 1))
        # small experiment: predict labels for the batch with the CURRENT bank
        wrong = []
        for i, rec in enumerate(batch):
            pred = predict_one(task, bank, rec["text"], i, seed=0)
            if pred != rec["label"]:
                wrong.append((rec, pred))
        if not wrong:
            break
        wrong = wrong[:10]
        err_str = "\n\n".join(
            f"[Misclassified {j+1}] {rec['text']}\nTrue label: {rec['label']}\n"
            f"Your current rules predicted: {pred}"
            for j, (rec, pred) in enumerate(wrong)
        )
        user = (
            f"Task: {task.task_description}\n{_labels_line(task)}\n\n"
            f"Current hypotheses:\n" + "\n".join(f"{i+1}. {h}" for i, h in enumerate(bank)) +
            "\n\nWe TESTED these hypotheses on fresh examples. Here are the ones they got "
            f"WRONG:\n\n{err_str}\n\n"
            "Diagnose why the current rules failed on these cases, then REVISE the hypothesis "
            "set so it would classify them correctly while staying general. "
            f"Output a JSON array of exactly {K} improved hypothesis strings."
        )
        bank = _ask_bank(user, K, seed + 100 * (r + 1))
    return bank


def gen_literature_compare(task, seed, rounds: int = 3, per_round: int = 5) -> list[str]:
    """C4: SciMON-style novelty boosting by comparison to a growing idea pool.

    Generates in rounds; each round must produce ideas substantively DIFFERENT from all
    prior ones (grounding novelty in comparison). To avoid leakage we differentiate against
    the model's own accumulating pool, not the dataset's held-out expert hypotheses.
    """
    pool: list[str] = []
    for r in range(rounds):
        if pool:
            prior = "\n".join(f"- {h}" for h in pool)
            diff = (f"\n\nThe following hypotheses have ALREADY been proposed:\n{prior}\n\n"
                    f"Propose {per_round} NEW hypotheses that are substantively DIFFERENT from "
                    "all of the above (different mechanisms/angles, not rephrasings).")
        else:
            diff = f"\n\nPropose {per_round} hypotheses (decision rules)."
        user = (
            f"Task: {task.task_description}\n{_labels_line(task)}{diff}\n\n"
            f"Output a JSON array of exactly {per_round} concise hypothesis strings."
        )
        new = _ask_bank(user, per_round, seed + 17 * r)
        pool.extend(new)
    return pool[:K]


def gen_proposal(task, seed) -> list[str]:
    """C5: write a structured research proposal, then distill it into hypotheses."""
    p_user = (
        f"Task: {task.task_description}\n{_labels_line(task)}\n\n"
        "Write a concise research proposal (~250 words) for understanding what distinguishes "
        "the two labels. Include: (1) Background/motivation, (2) Hypothesized underlying "
        "mechanisms, (3) Concrete predictions about observable features, (4) How each "
        "prediction could be tested."
    )
    proposal = chat([{"role": "system", "content": SYS},
                     {"role": "user", "content": p_user}],
                    model=GEN_MODEL, temperature=TEMP, max_tokens=900, seed=seed)
    d_user = (
        f"Task: {task.task_description}\n{_labels_line(task)}\n\n"
        f"Here is a research proposal:\n\n{proposal}\n\n"
        f"Distill it into exactly {K} concrete, testable decision rules for predicting the "
        f"label. Output a JSON array of {K} concise hypothesis strings."
    )
    return _ask_bank(d_user, K, seed + 1)


REGIMES = {
    "C0_ungrounded": gen_ungrounded,
    "C1_deduction": gen_deduction,
    "C2_induction": gen_induction,
    "C3_empirical": gen_empirical,
    "C4_literature_compare": gen_literature_compare,
    "C5_proposal": gen_proposal,
}

REGIME_LABELS = {
    "C0_ungrounded": "Ungrounded (control)",
    "C1_deduction": "Deduction",
    "C2_induction": "Induction",
    "C3_empirical": "Small-exp + outcome",
    "C4_literature_compare": "Literature-compare",
    "C5_proposal": "Proposal-writing",
}
