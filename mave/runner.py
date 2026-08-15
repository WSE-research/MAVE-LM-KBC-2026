"""Run the system over a split and write the predictions.

A run produces two files: ``predictions.jsonl`` in the submission format, and
``meta.json`` recording exactly what produced it — the resolved config, the
endpoint, cache hits and misses, token usage and cost. A run with zero cache
misses and zero cost is a pure replay of the shipped draws, which is what a
reproduction should look like.
"""

import json
import logging
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import DATA_DIR, ROOT, cache, client, pipeline
from .client import LLM
from .config import System

log = logging.getLogger("mave")

BUDGET_B = 32.0     # the shared task's inference-time parameter budget


def load_split(split: str) -> list[dict]:
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def describe(system: System, split: str) -> None:
    """Print the resolved plan without touching the endpoint."""
    print(f"{system.name} — {system.description}\n")
    print(f"model  : {system.model}")
    print(f"budget : {system.params_b}B / {BUDGET_B:.0f}B, open weights")
    print(f"cache  : {system.cache_slug}")
    for relation, plan in system.plans.items():
        params = plan.params
        shape = params["answer"] + (f" · {params['quorum']}" if "quorum" in params else "")
        print(f"\n{relation}  [{shape}]")
        print(f"  prompts: {plan.prompt_name}")
        for source in params["sources"]:
            draws = source.get("context", source).get("draws", 1)
            calls = len(source.get("sweep", [])) or draws
            print(f"  source : {source['name']:<12} {calls} call(s) per item")
    counts = Counter(item["Relation"] for item in load_split(split))
    print(f"\n{split}: {sum(counts.values())} items — {json.dumps(dict(sorted(counts.items())))}")


def run(system: System, split: str, out_dir: Path, offline: bool,
        workers: int = 8, limit: int = 0, relations: list[str] | None = None) -> Path:
    if system.params_b > BUDGET_B:
        raise SystemExit(f"{system.params_b}B exceeds the {BUDGET_B:.0f}B inference budget.")

    data = load_split(split)
    if relations:
        data = [item for item in data if item["Relation"] in relations]
    if limit:
        data = data[:limit]

    llm = LLM(system.model, cache_slug=system.cache_slug, offline=offline)
    contexts = {relation: pipeline.Context(relation=relation, prompt=plan.prompt,
                                           params=plan.params, llm=llm)
                for relation, plan in system.plans.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s | %s | %d items | %s", system.name, split, len(data),
             "offline replay of the shipped cache" if offline else client.base_url())

    progress, lock = Counter(), threading.Lock()

    def predict(item: dict) -> dict:
        subject, relation = item["SubjectEntity"], item["Relation"]
        try:
            objects = pipeline.answer(contexts[relation], subject)
            failed = False
        except cache.CacheMiss:
            raise
        except Exception as exc:      # one bad item must not kill the run
            log.warning("%s / %s — %s", subject, relation, exc)
            objects, failed = [], True
        with lock:
            progress["done"] += 1
            progress["errors"] += failed
            if progress["done"] % 100 == 0:
                log.info("  %d/%d", progress["done"], len(data))
        return {"SubjectEntity": subject, "Relation": relation, "ObjectEntities": objects}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(predict, data))     # map preserves input order

    predictions = out_dir / "predictions.jsonl"
    with open(predictions, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats, usage = cache.stats(), client.usage()
    (out_dir / "meta.json").write_text(json.dumps({
        "system": system.name,
        "split": split,
        "items": len(data),
        "model": system.model,
        "cache_slug": system.cache_slug,
        "base_url": None if offline else client.base_url(),
        "offline": offline,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "errors": progress["errors"],
        "degraded": client.degraded(),
        "llm_cache": stats,
        "usage": usage,
        "config": system.raw,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("done — %d errors, cache %d hits / %d misses, $%.4f",
             progress["errors"], stats["hits"], stats["misses"], usage["cost_usd"])
    if client.degraded():
        log.warning("DEGRADED: %s — re-run before trusting these numbers", client.degraded())
    log.info("wrote %s", predictions)
    return predictions


def evaluate(predictions: Path, split: str) -> None:
    """Score against the split's own gold with the official evaluator."""
    result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "evaluate.py"),
                             "-g", str(DATA_DIR / f"{split}.jsonl"), "-p", str(predictions)],
                            capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        log.error(result.stderr)
