"""Shared, regime-agnostic evaluation: apply a hypothesis bank to classify examples.

This procedure is IDENTICAL for every regime (same model, same prompt template) so the
only thing that differs across conditions is *how the hypothesis bank was grounded*. That
is what makes the ablation apples-to-apples.
"""
from __future__ import annotations
import re

from llm import chat, map_concurrent

EVAL_MODEL = "gpt-4.1-mini"


def _bank_str(bank: list[str]) -> str:
    return "\n".join(f"{i+1}. {h}" for i, h in enumerate(bank))


def _eval_prompt(task, bank: list[str], example_text: str, opt_a: str, opt_b: str) -> list[dict]:
    sys = (
        "You are a careful classifier. You are given a set of hypotheses (decision rules) "
        "discovered for a task, plus one new instance. Apply the hypotheses to predict the "
        "correct label. Rely on the hypotheses and the evidence in the instance. "
        "Respond with EXACTLY one character: 'A' or 'B'. No other text."
    )
    user = (
        f"Task: {task.task_description}\n\n"
        f"Hypotheses (decision rules):\n{_bank_str(bank)}\n\n"
        f"New instance:\n{example_text}\n\n"
        f"Question: {task.question}\n"
        f"Option A: {opt_a}\n"
        f"Option B: {opt_b}\n\n"
        "Apply the hypotheses and answer with exactly 'A' or 'B'."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _parse_ab(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\b([AB])\b", text.strip().upper())
    if m:
        return m.group(1)
    c = text.strip().upper()[:1]
    return c if c in ("A", "B") else None


def predict_one(task, bank, example_text, idx: int, seed: int = 0):
    """Predict canonical label for one example. Position of options randomized by idx."""
    # deterministic option-order swap to neutralize position bias
    swap = (idx % 2 == 1)
    a, b = task.label_values
    opt_a, opt_b = (b, a) if swap else (a, b)
    msgs = _eval_prompt(task, bank, example_text, opt_a, opt_b)
    out = chat(msgs, model=EVAL_MODEL, temperature=0.0, max_tokens=5, seed=seed)
    ab = _parse_ab(out)
    if ab is None:
        return None
    return opt_a if ab == "A" else opt_b


def evaluate_bank(task, bank: list[str], split: str, n: int, seed: int = 0,
                  max_workers: int = 24, desc: str = "") -> dict:
    """Classify the first n items of a split with the bank. Returns accuracy + details."""
    items = task.splits[split][:n]

    def _fn(arg):
        i, rec = arg
        pred = predict_one(task, bank, rec["text"], i, seed=seed)
        return {"idx": i, "true": rec["label"], "pred": pred,
                "correct": int(pred == rec["label"]) if pred is not None else 0,
                "abstain": int(pred is None)}

    rows = map_concurrent(_fn, list(enumerate(items)), max_workers=max_workers, desc=desc)
    rows = [r for r in rows if r is not None]
    n_eff = len(rows)
    acc = sum(r["correct"] for r in rows) / n_eff if n_eff else 0.0
    abst = sum(r["abstain"] for r in rows) / n_eff if n_eff else 0.0
    return {"accuracy": acc, "abstain_rate": abst, "n": n_eff, "rows": rows}
