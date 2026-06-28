"""LLM client: disk-cached, concurrent, retrying OpenAI calls with token tracking.

All experiment calls go through here so that (a) reruns are idempotent/cheap via an
on-disk cache, (b) we can parallelize evaluation, and (c) token usage is logged for the
cost/benefit analysis.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = OpenAI()
_token_lock = threading.Lock()
# global token counters (input, output) keyed by model
TOKENS: dict[str, dict[str, int]] = {}


def _key(model: str, messages: list[dict], **params) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _record_tokens(model: str, pt: int, ct: int):
    with _token_lock:
        d = TOKENS.setdefault(model, {"input": 0, "output": 0, "calls": 0})
        d["input"] += pt
        d["output"] += ct
        d["calls"] += 1


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def _raw_call(model: str, messages: list[dict], temperature: float, max_tokens: int,
              seed: int | None):
    return _client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, seed=seed,
    )


def chat(messages: list[dict], model: str = "gpt-4.1-mini", temperature: float = 0.7,
         max_tokens: int = 1200, seed: int | None = None, use_cache: bool = True) -> str:
    """Single cached chat completion. Returns the assistant text."""
    k = _key(model, messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
    fp = CACHE_DIR / f"{k}.json"
    if use_cache and fp.exists():
        obj = json.loads(fp.read_text())
        _record_tokens(model, obj.get("pt", 0), obj.get("ct", 0))  # count for cost accounting
        return obj["text"]
    resp = _raw_call(model, messages, temperature, max_tokens, seed)
    text = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens
    ct = resp.usage.completion_tokens
    fp.write_text(json.dumps({"text": text, "pt": pt, "ct": ct}))
    _record_tokens(model, pt, ct)
    return text


def map_concurrent(fn: Callable, items: list, max_workers: int = 24, desc: str = "") -> list:
    """Run fn over items concurrently, preserving order. Exceptions -> None."""
    results: list = [None] * len(items)
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def _wrap(i, x):
        try:
            return i, fn(x)
        except Exception as e:  # noqa: BLE001
            return i, ("__ERROR__", str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_wrap, i, x) for i, x in enumerate(items)]
        for f in futs:
            i, r = f.result()
            if isinstance(r, tuple) and r and r[0] == "__ERROR__":
                results[i] = None
            else:
                results[i] = r
            with lock:
                done[0] += 1
                if desc and (done[0] % 50 == 0 or done[0] == len(items)):
                    rate = done[0] / max(time.time() - t0, 1e-6)
                    print(f"  [{desc}] {done[0]}/{len(items)} ({rate:.1f}/s)", flush=True)
    return results


def embed(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Cached batch embeddings."""
    out: list[list[float] | None] = [None] * len(texts)
    to_fetch, idxs = [], []
    for i, t in enumerate(texts):
        k = hashlib.sha256(f"{model}::{t}".encode()).hexdigest()
        fp = CACHE_DIR / f"emb_{k}.json"
        if fp.exists():
            out[i] = json.loads(fp.read_text())
        else:
            to_fetch.append(t)
            idxs.append((i, fp))
    if to_fetch:
        resp = _client.embeddings.create(model=model, input=to_fetch)
        for (i, fp), d in zip(idxs, resp.data):
            out[i] = d.embedding
            fp.write_text(json.dumps(d.embedding))
    return out  # type: ignore


def token_report() -> dict:
    with _token_lock:
        return json.loads(json.dumps(TOKENS))
