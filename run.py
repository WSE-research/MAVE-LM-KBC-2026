#!/usr/bin/env python3
"""MAVE — closed-book knowledge-base construction for LM-KBC 2026.

    python run.py --plan                 # what the system does, without calling anything
    python run.py --split val            # reproduce the validation numbers (replay, free)
    python run.py --split train          # reproduce the training numbers   (replay, free)
    python run.py --split test           # reproduce the submitted test predictions
    python run.py --split val --live     # draw from an endpoint instead of the cache

By default every model response is replayed from the draw set committed in
``cache/`` — no API key, no endpoint, no cost. ``--live`` draws whatever is not
cached from an OpenAI-compatible endpoint (see ``.env.example``).
"""

import argparse
import logging
from pathlib import Path

from mave import RELATIONS, ROOT, cache
from mave.config import load_system
from mave.runner import describe, evaluate, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from a repo-root .env, without overriding the shell."""
    import os

    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--plan", action="store_true",
                    help="print the resolved system and exit")
    ap.add_argument("--live", action="store_true",
                    help="draw uncached responses from the endpoint instead of failing")
    ap.add_argument("--model", help="override the model name sent over the wire "
                                    "(the cache identity stays fixed)")
    ap.add_argument("--out", type=Path, help="output directory (default runs/<split>)")
    ap.add_argument("--config", type=Path, help="system config (default config/system.yaml)")
    ap.add_argument("--relation", action="append", choices=RELATIONS, dest="relations",
                    help="restrict to one or more relations (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items")
    ap.add_argument("--workers", type=int, default=8, help="items answered concurrently")
    args = ap.parse_args()

    _load_dotenv()
    system = load_system(args.config)
    if args.model:
        system.model = args.model

    if args.plan:
        describe(system, args.split)
        return

    out_dir = args.out or ROOT / "runs" / args.split
    try:
        predictions = run(system, args.split, out_dir, offline=not args.live,
                          workers=args.workers, limit=args.limit, relations=args.relations)
    except cache.CacheMiss as miss:
        raise SystemExit(f"\n{miss}\n\nThis configuration asks for responses the shipped "
                         f"cache does not hold. Re-run with --live and an endpoint "
                         f"configured (see .env.example).")

    whole_split = not args.limit and not args.relations
    if whole_split:
        reference = ROOT / "predictions" / f"{args.split}.jsonl"
        if reference.exists():
            same = reference.read_bytes() == predictions.read_bytes()
            print(f"\n{'identical to' if same else 'DIFFERS from'} the published "
                  f"{reference.relative_to(ROOT)}")

    # train and val ship their labels; test does not — its predictions are the deliverable.
    if args.split in ("train", "val") and whole_split:
        evaluate(predictions, args.split)


if __name__ == "__main__":
    main()
