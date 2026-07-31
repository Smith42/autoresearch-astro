# autoresearch-astro

This is an experiment to have the LLM do its own research.

The research problem: **stream crossmatched galaxy images and spectra off HuggingFace as fast as possible.** `UniverseTBD/mmu_ssl_legacysurvey_north` (14.2M Legacy Survey North cutouts, 152x152 in 3 bands, 3.4 TiB) crossmatched against `UniverseTBD/mmu_desi_edr_sv3` (1.1M DESI EDR SV3 spectra, 62 GiB), at a 1 arcsecond radius. Both are HATS catalogs with 10 arcsecond margin caches. The output — a stream of paired (image, spectrum) records — is the training input for multimodal astronomy models, and right now nothing streams it quickly.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, ground-truth prep, and the metric. Do not modify.
   - `stream.py` — the file you modify. The streaming implementation.
4. **Verify prep exists**: Check that `~/.cache/autoresearch-astro/` contains `truth.parquet`, `canonical_order.json` and `audit.npz`. If not, tell the human to run `uv run prepare.py` (takes ~10 minutes, one time only). Do not run experiments while a prep build is in flight — it competes for the same network link and will make your numbers look bad.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment streams for a **fixed time budget of 90 seconds** (wall clock, cold start INCLUDED — opening the catalogs and planning the crossmatch counts against you). You launch it simply as: `uv run stream.py`.

**What you CAN do:**
- Modify `stream.py` — this is the only file you edit. Everything is fair game: throw out LSDB entirely and read parquet byte ranges yourself, do the matching with your own kd-tree, go async, go multiprocess, prune row groups, whatever gets records out faster.
- **Add dependencies with `uv add`** (e.g. `duckdb`, `obstore`, `aiohttp`, `polars`, `pyarrow.fs`). Commit `pyproject.toml` and `uv.lock` along with your change.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed metric, the record schema and the ground-truth match table.
- Modify the evaluation harness. The `run_benchmark` function in `prepare.py` is the ground truth metric.
- Cache the survey data between runs, or read it from anywhere but HuggingFace. The harness points every HuggingFace/fsspec cache at a fresh temp dir for exactly this reason. There are raw HDF5 copies of both surveys elsewhere on this filesystem — using them is not solving the problem.
- Hardcode partition paths, object ids, match counts, or anything else you learned from a previous run's output. The implementation has to work by streaming the catalogs, not by remembering them.
- Skip ahead to the match-dense partitions. See `partition_order_ok` below.

**The goal is simple: get the highest rows_per_sec.** Since the time budget is fixed, you are simply trying to get as many verified crossmatched records out of the pipe as you can in 90 seconds.

**Memory** is a soft constraint. This machine has 187 GB of RAM and 24 cores. Some increase is acceptable for meaningful throughput gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 3 rows/sec improvement that adds 20 lines of hacky code? Probably not worth it — that is inside the noise floor anyway. A 3 rows/sec improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the streaming script as is.

## What counts as a record

The harness scores a record only if it is a real match, has not been yielded already, and carries the full payload. Every record must be a dict with exactly these keys:

```
object_id_ls, object_id_desi, ra_ls, dec_ls, ra_desi, dec_desi,
_dist_arcsec, image, spectrum_flux, spectrum_lambda
```

`image` must be a (3, 152, 152) float32 array and the spectrum arrays must be 1-D. Dropping the payload to go faster is not an optimisation, it is a different problem — records that fail the schema fail the run. Records from the audit partitions have their payloads checked bit-exactly against the originals.

`prepare.to_record()` converts an LSDB crossmatch row into this dict. You do not have to use it; you can build the dict any way you like, as long as it verifies.

## Output format

Once the script finishes it prints a summary like this:

```
---
matched_rows:        3500
rows_per_sec:        32.45
verify:              OK
partition_order_ok:  true
partitions_touched:  6
duplicate_rows:      0
bytes_downloaded_mb: 6107.5
mb_per_sec:          56.6
time_to_first_row:   30.6
elapsed_seconds:     107.9
peak_rss_mb:         6046.0
```

`elapsed_seconds` overshoots the 90 second budget because the clock is only checked between records — if you are 20 seconds into fetching a partition when the budget expires, the run ends when that partition's next record appears. The score divides by actual elapsed time, so this costs you nothing directly, but an implementation that delivers records continuously rather than in big lumps controls its own clock better.

`rows_per_sec` is the metric — **higher is better**. You can extract the key numbers from the log file:

```
grep "^rows_per_sec:\|^verify:\|^partition_order_ok:" run.log
```

**A run with `verify: FAIL` or `partition_order_ok: false` is a failure, no matter how fast it was.** Do not chase a number that came with either of those.

`partition_order_ok` guards against a degenerate strategy: because the score is rows per second, streaming only the partitions that happen to contain lots of matches would inflate it without making anything faster. The harness fixes a canonical partition order (ascending, published in `canonical_order.json`) and checks that you never leave more than 32 partitions behind the furthest one you have reached. That is enough slack to run every core on its own partition and still have work in flight when the clock stops, so parallel prefetch, pipelining and async fetching all pass comfortably. Cherry-picking does not.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	rows_per_sec	mb_per_sec	peak_rss_gb	status	description
```

1. git commit hash (short, 7 chars)
2. rows_per_sec achieved (e.g. 12.56) — use 0.00 for crashes and failed verification
3. mb_per_sec achieved, round to .1f — use 0.0 for crashes
4. peak memory in GB, round to .1f (divide peak_rss_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	rows_per_sec	mb_per_sec	peak_rss_gb	status	description
a1b2c3d	32.45	56.6	5.9	keep	baseline (raw LSDB CatalogStream)
b2c3d4e	88.30	154.2	9.6	keep	16-worker dask client, 4 partitions per chunk
c3d4e5f	30.10	52.1	5.2	discard	column projection, margin still read
d4e5f6g	0.00	0.0	0.0	crash	async byte-range reader (event loop deadlock)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `stream.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run stream.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^rows_per_sec:\|^verify:\|^partition_order_ok:\|^peak_rss_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If rows_per_sec improved (higher) and the run verified, you "advance" the branch, keeping the git commit
9. If rows_per_sec is equal or worse, or the run failed verification, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Run-to-run spread is large**: five baseline runs with no code change measured 24.14, 27.72, 32.45, 33.69 and 33.79 rows/sec. Treat anything under a ~10% difference as noise, re-run before believing anything marginal, and be suspicious of a "win" measured right after a heavy job, when the network is still settling.

Most of that spread is not the network — it is granularity. The baseline delivers a whole partition at a time, so a run ends on either 2613 rows (5 partitions) or 3500 (6 partitions) depending on whether the last one lands before the clock stops. Coarse batching makes your own score noisy. The finer the granularity at which you emit records, the less your measurement depends on where the 90 second boundary happens to fall.

**Timeout**: Each experiment takes ~105 seconds in practice (90 second budget, plus overshoot on the final partition, plus startup and teardown). If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the LSDB and HATS source in `.venv`, read the parquet spec, re-read the in-scope files for new angles, try combining previous near-misses, try more radical rewrites. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~90 seconds then you can run approx 35/hour, for a total of about 280 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Where the headroom is

Measurements taken on this machine before the baseline was written, so you don't have to rediscover them:

- The crossmatch plan aligns to **200 partitions**, 191 of which contain at least one match. There are **137,906 matches in total** — so a perfect run would need ~1500 rows/sec to stream all of them inside the budget.
- Only 9.3% of the Legacy Survey catalog's 5488 partitions overlap the DESI footprint at all.
- The baseline spends **~30 seconds before its first row** — opening two catalogs (29s cold) and planning the crossmatch (8s). That is a third of the entire budget spent before a single record exists.
- The baseline reaches **~32 rows/sec** while pulling ~57 MB/s, and gets through 5-6 partitions. A single HTTP stream from HuggingFace gets 11-33 MB/s, so the pipe is already being used in parallel — but nowhere near as wide as it goes.
- **Column projection is a disappointment.** Dropping to `["ra","dec","object_id"]` — throwing away the entire image payload — makes a partition only ~2.5x cheaper (12-16s vs 23-40s). The cost is dominated by per-request latency and row group overhead, not by image bytes. Read that as: reading *less* is worth much less than reading *concurrently*.
- Matched rows are a tiny fraction of the bytes a partition contains. The baseline downloads whole partitions to keep a few hundred rows.
- `CatalogStream` prefetches exactly one partition ahead, on one thread, by default.

These are starting points, not a plan. The interesting question is what the pipeline would look like if you designed it for this problem instead of using the general-purpose one.
