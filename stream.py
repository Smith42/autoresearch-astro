"""
Stream crossmatched image+spectrum pairs as fast as possible.

Legacy Survey North cutouts crossmatched against DESI EDR SV3 spectra, streamed
straight off HuggingFace. This is the baseline: the raw LSDB stack, no
cleverness. Open both catalogs, plan the crossmatch, walk the aligned partitions
one at a time, hand each matched row to the harness.

This is the only file you edit.

Run it with:
    uv run stream.py
"""

# prepare must be imported first: it redirects every HuggingFace/fsspec cache at
# a fresh temp dir, and that has to happen before lsdb pulls fsspec in.
import prepare

from lsdb.streams import CatalogStream


class OrderedStream(CatalogStream):
    """CatalogStream that walks partitions in ascending (canonical) order.

    The stock CatalogStream consumes partitions back-to-front. The harness owns
    the traversal order, so take from the front instead.
    """

    def get_next_partitions(self, partitions_left, rng):
        n = self.partitions_per_chunk
        return partitions_left[n:], partitions_left[:n]


def stream_pairs():
    """Yield one record per crossmatched (image, spectrum) pair."""
    prepare.patch_lsdb_empty_margin()
    xmatch = prepare.open_crossmatch()

    for chunk in OrderedStream(catalog=xmatch, shuffle=False):
        for _, row in chunk.iterrows():
            yield prepare.to_record(row)


if __name__ == "__main__":
    prepare.run_benchmark(stream_pairs)
