# autoresearch-astro

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) pointed at a different research problem.

Upstream gives an AI agent a small LLM training setup and lets it experiment overnight, minimising validation loss. This fork keeps the machinery — one editable file, one fixed metric, a keep-or-discard git loop driven by `program.md` — and swaps the problem for a data engineering one:

> **Stream crossmatched galaxy images and spectra off HuggingFace as fast as possible.**

The core idea is unchanged: you're not touching the Python files like you normally would as a researcher. Instead you are programming the `program.md` Markdown file that provides context to the AI agents and sets up your autonomous research org.

## The problem

[`UniverseTBD/mmu_ssl_legacysurvey_north`](https://huggingface.co/datasets/UniverseTBD/mmu_ssl_legacysurvey_north) — 14,174,203 Legacy Survey North galaxy cutouts, 152x152 pixels in three optical bands, 3.4 TiB across 5488 partitions.

[`UniverseTBD/mmu_desi_edr_sv3`](https://huggingface.co/datasets/UniverseTBD/mmu_desi_edr_sv3) — 1,126,441 DESI EDR SV3 spectra, 62 GiB across 306 partitions.

Both are [HATS](https://hats.readthedocs.io/) catalogs with 10 arcsecond margin caches, so [LSDB](https://lsdb.readthedocs.io/) can crossmatch them directly off the Hub. Matching them at 1 arcsecond gives paired (image, spectrum) records — the training input for multimodal astronomy models. The agent's job is to get those records out of the pipe as fast as possible.

This is a real bottleneck, not a toy. Existing multimodal work here consumes *pre-computed* crossmatches or pre-baked file lists precisely because streaming a live one is slow.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time ground-truth prep, and the metric. Not modified.
- **`stream.py`** — the single file the agent edits. Contains the streaming implementation. Everything is fair game: the baseline is the raw LSDB stack, but the agent can throw all of it out and read parquet byte ranges by hand if that's faster. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, streaming runs for a **fixed 90-second time budget** (wall clock, cold start included). The metric is **rows_per_sec** — verified crossmatch records delivered per second, higher is better.

A record only counts if it is a real match, has not been yielded before, and carries the full payload: a (3, 152, 152) float32 image and the spectrum arrays. Dropping the payload to go faster isn't an optimisation, it's a different problem. `prepare.py` builds a ground-truth match table once and checks every record against it, verifying payloads bit-exactly on two pinned partitions.

## Quick start

**Requirements:** CPU and a network connection. No GPU. Tested on 24 cores / 187 GB RAM. Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Build the ground-truth match table (one-time, ~10 min)
uv run prepare.py

# ...or go wider if nothing else needs the network
uv run prepare.py --workers 16

# 4. Manually run a single streaming experiment (~90 sec)
uv run stream.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

### Skipping the prep build

[`truth.parquet`](truth.parquet) and [`canonical_order.json`](canonical_order.json) are committed at the repo root — the ground-truth match table for all 200 aligned partitions (137,906 matches, 3.5 MB) and the canonical traversal order derived from it. Copy both into place:

```bash
mkdir -p ~/.cache/autoresearch-astro
cp truth.parquet canonical_order.json ~/.cache/autoresearch-astro/
```

The third artifact, `audit.npz`, is not committed. Build it with:

```bash
uv run prepare.py --audit-only
```

That still downloads the two audit partitions in full (~2 GB), but it takes a couple of minutes instead of the ~25 the full build needs.

Note that `truth.parquet` is the answer key: it lists every matched `(object_id_ls, object_id_desi)` pair and the partition holding it. A `stream.py` that reads it is no longer solving the streaming problem — see the "What you CANNOT do" rules in `program.md`.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, ground-truth prep + the metric (do not modify)
stream.py       — the streaming implementation (agent modifies this)
program.md      — agent instructions
analysis.ipynb  — plots of results.tsv
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `stream.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Streaming always runs for 90 seconds, and the cold start counts against it. Opening the two catalogs and planning the crossmatch costs the baseline ~30 seconds, a third of the budget — so eliminating startup is as legitimate a target as raising steady-state throughput. 90 seconds means roughly 35 experiments an hour, or ~280 while you sleep.
- **Cold cache every run.** The harness redirects every HuggingFace and fsspec cache at a fresh temp dir, so no run benefits from a previous one's downloads. The problem is streaming, not downloading once.
- **Dependencies are open.** Unlike upstream, the agent may `uv add` anything. Most of the headroom in a data pipeline lives in which I/O library you reach for.
- **Verified output.** Speed with the wrong answer is not speed. Every record is checked against a ground-truth match table, and payloads on two pinned partitions are checked bit-exactly.
- **Canonical partition order.** Because the score is per-second, streaming only the match-dense partitions would inflate it for free. The harness fixes an ascending traversal order and allows a 32-partition in-flight window, which parallel prefetching passes and cherry-picking doesn't.

## Baseline

The raw LSDB stack — `open_catalog` twice, `crossmatch`, walk the 200 aligned partitions with `CatalogStream` — gets about **32 rows/sec**, with 30 seconds to the first row and ~57 MB/s of throughput, covering 5-6 of the 191 partitions that contain matches. There are 137,906 matches in total, so streaming all of them inside the budget would take ~1500 rows/sec.

Throughput is noisy: five baseline runs measured 24.14, 27.72, 32.45, 33.69 and 33.79 rows/sec with no code change, so treat anything under a ~10% difference as noise. Much of that spread is granularity rather than the network — the baseline emits a whole partition at a time, so a run ends on either 5 or 6 completed partitions depending on where the clock falls. Some measured starting points for where the headroom is are listed at the bottom of `program.md`.

## License

MIT
