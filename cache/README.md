# The shipped draw set

`mistralai_mistral-small-3.2-24b-instruct.jsonl.gz` — 44,008 model responses, 6.8 MB
compressed. One JSON object per line: `key` is a SHA-256 over the exact request
(both message turns, temperature, token limit, seed), `text` is what the model
replied.

These are precisely the responses the published predictions were built from, and
nothing else — every draw the system makes on `train`, `val` and `test`, with the
rest of the development cache left out.

## Why a cache is committed here

Because it is not a regenerable artifact, and finding that out was expensive.

One model slug is served by several endpoints at different quantizations, and an
unpinned run is a different mixture of them every time. The same configuration
drawn fresh scored **0.02 macro-F1 below** the file it shipped. So the draws are
the artifact: replaying them is what makes the published numbers reproduce on
another machine, in another month, with no account anywhere.

Redrawing all of them takes several hours and costs real money. Replaying them
takes eight seconds and costs nothing.

## How it is used

`mave/cache.py` reads this file, and additionally a plain
`mistralai_mistral-small-3.2-24b-instruct.jsonl` next to it if one exists — that
is where `--live` appends anything drawn since, so the shipped set stays
immutable. Local draws win on a key collision.

## Confidentiality

Each line is a request and a model reply. The prompts are versioned in
`config/prompts/` anyway. Checked for credentials before committing — none found.
