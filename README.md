# MAVE — closed-book knowledge-base construction for LM-KBC 2026

Given a subject and a relation — `Israel` + `hasArea`, `Pfizer` +
`companyTradesAtStockExchange` — predict **every** correct object from what a
language model already knows. No retrieval, no fine-tuning, no lookups at
inference time, and everything that runs must be open-weight and sum to at most
32B parameters. Answering `[]` is a real answer: many subjects genuinely have no
object.

This system runs **one 24B open-weight model** and scores **0.6906 macro-F1** on
the shared task's hidden test labels.

| | train | validation |
|---|---:|---:|
| **macro-F1** | **0.6373** | **0.6699** |

---

## Reproduce it

Nothing to configure, no API key, no endpoint, no cost. Every model response the
system needs is committed in `cache/` (44,008 responses, 6.8 MB), so a run
replays the exact draws the published predictions were built from.

```bash
pip install -r requirements.txt

python run.py --split train    # → runs/train/predictions.jsonl + the official score
python run.py --split val      # → runs/val/predictions.jsonl   + the official score
python run.py --split test     # → runs/test/predictions.jsonl  (the submission file)
```

Each takes about eight seconds. Every run also checks its output against the
published `predictions/<split>.jsonl` and says so:

```
done — 0 errors, cache 15119 hits / 0 misses, $0.0000
identical to the published predictions/val.jsonl
```

`train` and `val` ship their labels, so those two runs print the official
evaluator's table. `test` does not ship labels — its predictions **are** the
deliverable, and `predictions/test.jsonl` is the file that was submitted.

To see what the system does before running it:

```bash
python run.py --plan
```

## Run it on your own model

The system talks to any **OpenAI-compatible** endpoint — a local vLLM, Ollama,
llama.cpp, LM Studio, or a hosted provider. It was developed against OpenRouter,
but nothing in it is specific to that.

```bash
cp .env.example .env         # set MAVE_BASE_URL and MAVE_API_KEY
python run.py --split val --live
```

`--live` draws whatever is not already cached and appends it to
`cache/<slug>.jsonl`, leaving the shipped set untouched. Without `--live` a cache
miss is an error rather than a silent API call, so a reproduction run either
replays the published draws or fails loudly.

`--model` changes the name sent over the wire without changing the cache
identity, which is what you want when the same weights are served under a
different name:

```bash
python run.py --split val --live \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

A full split is roughly **15,200 model calls**.
