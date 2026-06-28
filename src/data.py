"""Task loading for the HypoGeniC real tasks used in the ablation.

Each task is normalized into a common schema:
  - task_description (str)
  - label_values: [label_A, label_B]  (canonical strings)
  - known_hypotheses: list[str]        (expert hypotheses; held out, eval-only)
  - splits: dict tag -> list[{"text": rendered_example, "label": canonical_label}]
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "datasets" / "hypogenic_datasets" / "real"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _canon(label: str, values: list[str]) -> str:
    """Map a raw label to one of the canonical metadata values (handles trailing '.')."""
    s = label.strip().rstrip(".").strip().lower()
    for v in values:
        if v.strip().rstrip(".").strip().lower() == s:
            return v
    # fall back to substring match
    for v in values:
        if s in v.lower() or v.lower() in s:
            return v
    raise ValueError(f"unmappable label {label!r} not in {values}")


# task_name -> (subdir, files dict, label_values, render_fn, label_key)
def _render_headline(rec):
    return f'Headline 1: "{rec["headline_1"]}"\nHeadline 2: "{rec["headline_2"]}"'


def _render_review(rec):
    return f'Hotel review: "{rec["review_sentence"]}"'


def _render_post(rec):
    return f'Reddit post: "{rec["text"]}"'


TASKS = {
    "headline_binary": {
        "dir": "headline_binary",
        "files": {"train": "headline_binary_train.json",
                  "test": "headline_binary_test.json",
                  "ood": "headline_binary_ood.json"},
        "keys": ["headline_1", "headline_2"],
        "render": _render_headline,
        "question": "Which headline gets more clicks?",
    },
    "deceptive_reviews": {
        "dir": "deceptive_reviews",
        "files": {"train": "hotel_reviews_train.json",
                  "test": "hotel_reviews_test.json",
                  "ood": "ood_hotel_reviews_all.json"},
        "keys": ["review_sentence"],
        "render": _render_review,
        "question": "Is this hotel review deceptive or truthful?",
    },
    "dreaddit": {
        "dir": "dreaddit",
        "files": {"train": "dreaddit_ind_train.json",
                  "test": "dreaddit_ind_test.json",
                  "ood": "dreaddit_ood.json"},
        "keys": ["text"],
        "render": _render_post,
        "question": "Does this Reddit post show signs of mental stress?",
    },
}


class Task:
    def __init__(self, name: str):
        spec = TASKS[name]
        d = BASE / spec["dir"]
        meta = _load(d / "metadata.json")
        self.name = name
        self.task_description = meta["task_description"]
        self.label_values = list(meta["labels"]["label"]["values"])
        self.known_hypotheses = [h.replace("**", "") for h in meta.get("known_hypotheses", [])]
        self.question = spec["question"]
        self.render = spec["render"]
        self.splits = {}
        for tag, fname in spec["files"].items():
            raw = _load(d / fname)
            keys = spec["keys"]
            n = len(raw[keys[0]])
            recs = []
            for i in range(n):
                rec = {k: raw[k][i] for k in keys}
                lab = _canon(raw["label"][i], self.label_values)
                recs.append({"text": spec["render"](rec), "label": lab})
            self.splits[tag] = recs

    def __repr__(self):
        sizes = {k: len(v) for k, v in self.splits.items()}
        return f"Task({self.name}, labels={self.label_values}, sizes={sizes})"


if __name__ == "__main__":
    for name in TASKS:
        t = Task(name)
        print(t)
        print("  desc:", t.task_description[:90])
        print("  example:", t.splits["train"][0]["text"][:120].replace("\n", " | "),
              "->", t.splits["train"][0]["label"])
        print("  #known_hyp:", len(t.known_hypotheses))
