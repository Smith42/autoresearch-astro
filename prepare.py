"""
One-time preparation and the fixed metric for autoresearch-astro experiments.

Builds the ground-truth crossmatch table used to verify streamed records, and
provides the benchmark harness that every experiment is scored by.

Usage:
    python prepare.py                    # full prep (build truth table + audit digests)
    python prepare.py --max-partitions 8 # partial build (for testing the harness)

Artifacts are stored in ~/.cache/autoresearch-astro/.

This file is READ-ONLY for the research agent. It defines the metric.
"""

import os
import sys
import json
import time
import shutil
import atexit
import hashlib
import argparse
import tempfile
import threading

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Cold cache
#
# Every run must stream from HuggingFace with a cold cache -- the research
# problem is "stream this fast", not "download this once". We point every
# cache HuggingFace/fsspec might use at a fresh temp dir, created at import
# time (so it is set before anything imports lsdb) and removed at exit.
# ---------------------------------------------------------------------------

# Worker subprocesses import this module too, and inherit the variable below, so
# the whole process tree shares one cache and only the process that created it
# cleans it up.
RUN_CACHE = os.environ.get("AUTORESEARCH_RUN_CACHE")
if RUN_CACHE is None or not os.path.isdir(RUN_CACHE):
    RUN_CACHE = tempfile.mkdtemp(prefix="autoresearch-astro-", dir=os.environ.get("AUTORESEARCH_TMP") or None)
    os.environ["AUTORESEARCH_RUN_CACHE"] = RUN_CACHE
    atexit.register(shutil.rmtree, RUN_CACHE, True)

os.environ["HF_HOME"] = os.path.join(RUN_CACHE, "hf")
os.environ["HF_HUB_CACHE"] = os.path.join(RUN_CACHE, "hf", "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(RUN_CACHE, "hf", "datasets")
os.environ["XDG_CACHE_HOME"] = os.path.join(RUN_CACHE, "xdg")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

# Images: 14,174,203 galaxy cutouts, 152x152 in 3 bands, 3.4 TiB, 5488 partitions.
CATALOG_A = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
# Spectra: 1,126,441 DESI EDR SV3 spectra, 61.8 GiB, 306 partitions.
CATALOG_B = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"

RADIUS_ARCSEC = 1.0      # crossmatch radius
N_NEIGHBORS = 1          # nearest match only
SUFFIXES = ("_ls", "_desi")

TIME_BUDGET = 90         # wall clock seconds per experiment, cold start INCLUDED
# How far ahead of the canonical order a run may reach. Sized to leave room for
# deep parallel prefetch (this machine has 24 cores, so a partition-per-core
# implementation can legitimately leave ~24 partitions unfinished when the clock
# stops) while still refusing a run that skips ahead to the match-dense ones.
IN_FLIGHT_WINDOW = 32
AUDIT_PARTITIONS = (0, 1)  # partitions whose payloads are verified bit-exactly

IMAGE_SHAPE = (3, 152, 152)  # (band, y, x) -- des-g, des-r, des-z

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-astro")
TRUTH_PATH = os.path.join(CACHE_DIR, "truth.parquet")
ORDER_PATH = os.path.join(CACHE_DIR, "canonical_order.json")
AUDIT_PATH = os.path.join(CACHE_DIR, "audit.npz")

# Every streamed record must carry exactly these keys.
RECORD_KEYS = (
    "object_id_ls",
    "object_id_desi",
    "ra_ls",
    "dec_ls",
    "ra_desi",
    "dec_desi",
    "_dist_arcsec",
    "image",
    "spectrum_flux",
    "spectrum_lambda",
)

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def to_record(row):
    """Convert one crossmatch row (as produced by lsdb) into a stream record.

    A convenience for the lsdb path -- the harness scores the dict, not the
    route taken to build it. An implementation that reads parquet directly is
    free to construct the same dict however it likes.
    """
    image = row["image_ls"]
    spectrum = row["spectrum_desi"]
    return {
        "object_id_ls": str(row["object_id_ls"]),
        "object_id_desi": str(row["object_id_desi"]),
        "ra_ls": float(row["ra_ls"]),
        "dec_ls": float(row["dec_ls"]),
        "ra_desi": float(row["ra_desi"]),
        "dec_desi": float(row["dec_desi"]),
        "_dist_arcsec": float(row["_dist_arcsec"]),
        # Each band's flux arrives as an object array of 152 rows of 152 floats.
        "image": np.stack([np.stack(list(band)) for band in image["flux"]]).astype(np.float32, copy=False),
        "spectrum_flux": np.asarray(spectrum["flux"], dtype=np.float32),
        "spectrum_lambda": np.asarray(spectrum["lambda"], dtype=np.float32),
    }


def digest_record(record):
    """Bit-exact digest of a record's payload, used for audit verification."""
    h = hashlib.blake2b(digest_size=16)
    h.update(record["object_id_ls"].encode())
    h.update(record["object_id_desi"].encode())
    h.update(np.ascontiguousarray(record["image"], dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(record["spectrum_flux"], dtype=np.float32).tobytes())
    return h.digest()


class RecordError(Exception):
    """A streamed record violated the required schema."""


def check_schema(record):
    """Raise RecordError unless the record matches the required schema."""
    if not isinstance(record, dict):
        raise RecordError(f"record is {type(record).__name__}, expected dict")
    missing = [k for k in RECORD_KEYS if k not in record]
    if missing:
        raise RecordError(f"record missing keys: {missing}")

    image = np.asarray(record["image"])
    if image.shape != IMAGE_SHAPE:
        raise RecordError(f"image shape {image.shape}, expected {IMAGE_SHAPE}")
    if not np.isfinite(image).all():
        raise RecordError("image contains non-finite values")

    for key in ("spectrum_flux", "spectrum_lambda"):
        arr = np.asarray(record[key])
        if arr.ndim != 1 or arr.size < 1000:
            raise RecordError(f"{key} has shape {arr.shape}, expected a 1-D spectrum")

    dist = float(record["_dist_arcsec"])
    if not 0.0 <= dist <= RADIUS_ARCSEC:
        raise RecordError(f"_dist_arcsec {dist} outside [0, {RADIUS_ARCSEC}]")


# ---------------------------------------------------------------------------
# Crossmatch construction (shared by the truth builder and the baseline)
# ---------------------------------------------------------------------------


def open_crossmatch(columns_a=None, columns_b=None):
    """Open both catalogs and plan the crossmatch. Returns the lazy lsdb Catalog."""
    import lsdb

    cat_a = lsdb.open_catalog(CATALOG_A, columns=columns_a)
    cat_b = lsdb.open_catalog(CATALOG_B, columns=columns_b)
    return cat_a.crossmatch(
        cat_b,
        radius_arcsec=RADIUS_ARCSEC,
        n_neighbors=N_NEIGHBORS,
        suffixes=SUFFIXES,
        suffix_method="all_columns",
    )


def patch_lsdb_empty_margin():
    """Work around an lsdb crash when a partition's margin cache is empty."""
    import pandas as pd
    import nested_pandas as npd
    from lsdb.dask import merge_catalog_functions

    def _safe_concat(partition, margin):
        if margin is None or len(margin) == 0:
            return partition
        if len(partition) == 0:
            return npd.NestedFrame(margin)
        return npd.NestedFrame(pd.concat([partition, margin]))

    merge_catalog_functions.concat_partition_and_margin = _safe_concat


# ---------------------------------------------------------------------------
# One-time preparation
# ---------------------------------------------------------------------------


def build_truth(max_partitions=None, n_workers=8):
    """Build the ground-truth match table, canonical order, and audit digests.

    The truth table only needs coordinates and ids, so it is built from a
    column-projected crossmatch -- about 2.5x cheaper per partition than
    dragging the image payload along.

    The worker count is deliberately modest. This is a one-time job, and a
    build that saturates the network link makes any measurement taken
    alongside it look artificially slow.
    """
    import warnings

    import dask
    import pandas as pd
    from dask.distributed import Client

    warnings.filterwarnings("ignore")
    os.makedirs(CACHE_DIR, exist_ok=True)
    patch_lsdb_empty_margin()

    print("planning column-projected crossmatch...", flush=True)
    t0 = time.perf_counter()
    projected = ["ra", "dec", "object_id"]
    xmatch = open_crossmatch(columns_a=projected, columns_b=projected)
    npartitions = xmatch._ddf.npartitions
    if max_partitions is not None:
        npartitions = min(npartitions, max_partitions)
    print(f"  {npartitions} aligned partitions ({time.perf_counter() - t0:.1f}s)", flush=True)

    dask.config.set({"distributed.admin.large-graph-warning-threshold": "100 MiB"})
    client = Client(n_workers=n_workers, threads_per_worker=1)
    client.run(patch_lsdb_empty_margin)
    print(f"  dask client with {n_workers} workers: {client.dashboard_link}", flush=True)

    frames = []
    t0 = time.perf_counter()
    try:
        futures = [client.compute(xmatch._ddf.partitions[i]) for i in range(npartitions)]
        for i, future in enumerate(futures):
            part = future.result()
            if len(part):
                frames.append(
                    pd.DataFrame(
                        {
                            "object_id_ls": part["object_id_ls"].astype(str),
                            "object_id_desi": part["object_id_desi"].astype(str),
                            "dist_arcsec": part["_dist_arcsec"].astype("float64"),
                            "partition": np.full(len(part), i, dtype="int32"),
                        }
                    )
                )
            done = i + 1
            elapsed = time.perf_counter() - t0
            rate = done / elapsed
            print(
                f"  partition {i:4d}/{npartitions}  {len(part):5d} matches  "
                f"{elapsed:6.1f}s elapsed  eta {(npartitions - done) / rate:6.1f}s",
                flush=True,
            )
    finally:
        client.close()

    truth = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["object_id_ls", "object_id_desi", "dist_arcsec", "partition"]
    )
    truth.to_parquet(TRUTH_PATH, index=False)
    print(f"\nwrote {TRUTH_PATH}: {len(truth)} matches", flush=True)

    # Canonical traversal order: partitions holding at least one match, ascending.
    # Empty partitions are excluded because they are invisible to the harness.
    order = sorted(int(p) for p in truth["partition"].unique())
    with open(ORDER_PATH, "w") as f:
        json.dump({"npartitions": int(npartitions), "order": order}, f)
    print(f"wrote {ORDER_PATH}: {len(order)} non-empty partitions", flush=True)

    build_audit(max_partitions=max_partitions)


def build_audit(max_partitions=None):
    """Store bit-exact payload digests for the pinned audit partitions."""
    import warnings

    import dask

    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")
    dask.config.set(scheduler="synchronous")
    patch_lsdb_empty_margin()

    partitions = [p for p in AUDIT_PARTITIONS if max_partitions is None or p < max_partitions]
    print(f"\nbuilding audit digests for partitions {partitions}...", flush=True)
    xmatch = open_crossmatch()

    keys, digests = [], []
    for p in partitions:
        t0 = time.perf_counter()
        part = xmatch._ddf.partitions[p].compute()
        for _, row in part.iterrows():
            record = to_record(row)
            keys.append(f"{record['object_id_ls']}|{record['object_id_desi']}")
            digests.append(digest_record(record))
        print(f"  partition {p}: {len(part)} records ({time.perf_counter() - t0:.1f}s)", flush=True)

    np.savez(
        AUDIT_PATH,
        keys=np.array(keys, dtype=object),
        digests=np.array([np.frombuffer(d, dtype=np.uint8) for d in digests], dtype=np.uint8)
        if digests
        else np.zeros((0, 16), dtype=np.uint8),
        partitions=np.array(partitions, dtype=np.int32),
    )
    print(f"wrote {AUDIT_PATH}: {len(keys)} audited records", flush=True)


def load_truth():
    """Load the prep artifacts. Raises a clear error if prep has not been run."""
    import pandas as pd

    for path in (TRUTH_PATH, ORDER_PATH, AUDIT_PATH):
        if not os.path.exists(path):
            sys.exit(f"missing {path} -- run `uv run prepare.py` first")

    truth = pd.read_parquet(TRUTH_PATH)
    pairs = {
        f"{a}|{b}": int(p)
        for a, b, p in zip(truth["object_id_ls"], truth["object_id_desi"], truth["partition"])
    }
    with open(ORDER_PATH) as f:
        order = json.load(f)["order"]
    audit_npz = np.load(AUDIT_PATH, allow_pickle=True)
    audit = {
        str(k): bytes(d)
        for k, d in zip(audit_npz["keys"], audit_npz["digests"])
    }
    return pairs, {p: rank for rank, p in enumerate(order)}, audit


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------


class PeakMemoryTracker:
    """Background thread sampling RSS of this process and all its descendants."""

    def __init__(self, interval=0.25):
        self._interval = interval
        self._process = psutil.Process()
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _total_rss(self):
        try:
            rss = self._process.memory_info().rss
            for child in self._process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def start(self):
        self._peak = self._total_rss()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._total_rss())
            self._stop.wait(self._interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        return max(self._peak, self._total_rss())


def run_benchmark(stream_fn):
    """Score a streaming implementation. This is the ground truth metric.

    Iterates `stream_fn()` for TIME_BUDGET seconds and counts how many distinct,
    verified crossmatch records it delivered per second.

    A record scores if and only if it matches the required schema, corresponds to
    a real match in the truth table, and has not been yielded already. Records
    from the audit partitions additionally have their payload verified
    bit-exactly, so a run cannot score by fabricating images.
    """
    pairs, ranks, audit = load_truth()

    tracker = PeakMemoryTracker()
    tracker.start()
    # Host-wide counter, so it is a diagnostic rather than part of the score --
    # anything else on this machine using the network shows up here too.
    net0 = psutil.net_io_counters().bytes_recv

    matched, duplicates = 0, 0
    seen = set()
    seen_partitions = set()
    max_rank = -1
    failure = None
    time_to_first_row = None

    t0 = time.perf_counter()
    try:
        for record in stream_fn():
            now = time.perf_counter()
            if time_to_first_row is None:
                time_to_first_row = now - t0

            check_schema(record)
            key = f"{record['object_id_ls']}|{record['object_id_desi']}"
            if key not in pairs:
                raise RecordError(f"{key} is not a real match")
            if key in audit and digest_record(record) != audit[key]:
                raise RecordError(f"{key} payload does not match the audited original")

            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
                matched += 1
                partition = pairs[key]
                seen_partitions.add(partition)
                max_rank = max(max_rank, ranks[partition])

            if now - t0 >= TIME_BUDGET:
                break
    except RecordError as exc:
        failure = str(exc)

    elapsed = time.perf_counter() - t0
    peak_rss = tracker.stop()
    net_bytes = psutil.net_io_counters().bytes_recv - net0

    # A run may reach ahead of the canonical order only as far as its in-flight
    # window: enough slack for parallel prefetch, not enough to harvest the
    # match-dense partitions and skip the rest.
    order_ok = max_rank - (len(seen_partitions) - 1) <= IN_FLIGHT_WINDOW if seen_partitions else True

    if failure is None and not order_ok:
        failure = (
            f"reached partition rank {max_rank} having finished only "
            f"{len(seen_partitions)} -- more than {IN_FLIGHT_WINDOW} left behind"
        )
    if failure is None and matched == 0:
        failure = "no records streamed"

    rows_per_sec = matched / elapsed if elapsed > 0 else 0.0
    print("---")
    print(f"matched_rows:        {matched}")
    print(f"rows_per_sec:        {rows_per_sec:.2f}")
    print(f"verify:              {'OK' if failure is None else 'FAIL'}")
    print(f"partition_order_ok:  {str(order_ok).lower()}")
    print(f"partitions_touched:  {len(seen_partitions)}")
    print(f"duplicate_rows:      {duplicates}")
    print(f"bytes_downloaded_mb: {net_bytes / 1e6:.1f}")
    print(f"mb_per_sec:          {net_bytes / 1e6 / elapsed if elapsed > 0 else 0.0:.1f}")
    print(f"time_to_first_row:   {time_to_first_row if time_to_first_row is not None else -1:.1f}")
    print(f"elapsed_seconds:     {elapsed:.1f}")
    print(f"peak_rss_mb:         {peak_rss / 1e6:.1f}")
    if failure is not None:
        print(f"failure:             {failure}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-partitions", type=int, default=None,
                        help="only build truth for the first N partitions (testing)")
    parser.add_argument("--workers", type=int, default=8,
                        help="dask workers used to build the truth table. Kept modest on "
                             "purpose: saturating the link makes concurrent measurements noisy")
    parser.add_argument("--audit-only", action="store_true",
                        help="rebuild only the audit digests")
    args = parser.parse_args()

    if args.audit_only:
        build_audit(max_partitions=args.max_partitions)
    else:
        build_truth(max_partitions=args.max_partitions, n_workers=args.workers)
