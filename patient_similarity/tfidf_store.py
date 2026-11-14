"""Storage and loading for the Task 1 TF-IDF matrix.

The full matrix over 167,034 patients and a ~200k term vocabulary is roughly
530 MB, larger than the 100 MB per-file limit GitHub enforces, so it is written
as a directory of plain JSON shards:

    outputs/tfidf_matrix/
        README.txt        # short access note shipped with the data
        manifest.json     # counts, shard list, patient_uid -> shard index
        part-0000.json    # {patient_uid: {term: weight}, ...}
        part-0001.json
        ...

Every shard is ordinary JSON and loads on its own with
``json.load(open("outputs/tfidf_matrix/part-0000.json"))``. ``load_tfidf``
resolves a patient_uid through the manifest and reads only the shard it lives
in.
"""

import json
from pathlib import Path

DEFAULT_SHARD_LIMIT_BYTES = 80 * 1024 * 1024
SHARD_PREFIX = "part-"
MANIFEST_NAME = "manifest.json"

ACCESS_NOTE = """Task 1: TF-IDF matrix, stored as JSON shards.

Each part-XXXX.json maps patient_uid -> {term: weight}. Terms with zero weight
are omitted, so a missing term has weight 0. The full matrix is stored: no
terms are truncated.

Load one shard directly:
    import json
    shard = json.load(open("part-0000.json"))
    shard["1065068-1"]["meningeal"]

Or use the helper, which reads only the shard a patient belongs to:
    from patient_similarity.tfidf_store import load_tfidf
    tfidf = load_tfidf("outputs/tfidf_matrix")
    tfidf["1065068-1"]["meningeal"]
    tfidf.term_weight("1065068-1", "meningeal")
"""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _iter_matrix_rows(matrix, patient_uids, vectorizer):
    """Yield (patient_uid, {term: weight}) for every row of the matrix."""
    terms = vectorizer.get_feature_names_out()
    matrix = matrix.tocsr()
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    for row_index, patient_uid in enumerate(patient_uids):
        start, end = indptr[row_index], indptr[row_index + 1]
        yield patient_uid, {
            str(terms[column]): round(float(weight), 6)
            for column, weight in zip(indices[start:end], data[start:end])
        }


def write_tfidf_shards(pairs, output_dir, shard_limit_bytes=DEFAULT_SHARD_LIMIT_BYTES):
    """Write (patient_uid, {term: weight}) pairs as size-bounded JSON shards.

    Returns the manifest that is written alongside the shards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    uid_to_shard = {}
    vocabulary = set()
    patient_count = 0

    state = {"handle": None, "path": None, "bytes": 0, "count": 0,
             "first_uid": None, "last_uid": None}

    def open_shard(index):
        path = output_dir / f"{SHARD_PREFIX}{index:04d}.json"
        state["path"] = path
        state["handle"] = path.open("w", encoding="utf-8")
        state["handle"].write("{")
        state["bytes"] = 0
        state["count"] = 0

    def close_shard():
        if state["handle"] is None:
            return
        state["handle"].write("}")
        state["handle"].close()
        shards.append({
            "file": state["path"].name,
            "patients": state["count"],
            "first_uid": state["first_uid"],
            "last_uid": state["last_uid"],
            "bytes": state["bytes"],
        })
        state["handle"] = None

    for patient_uid, term_weights in pairs:
        if state["handle"] is None:
            open_shard(len(shards))
            state["first_uid"] = patient_uid
            state["last_uid"] = patient_uid

        entry = json.dumps(patient_uid) + ":" + json.dumps(
            term_weights, ensure_ascii=False, separators=(",", ":")
        )
        prefix = "" if state["count"] == 0 else ","
        state["handle"].write(prefix + entry)
        state["bytes"] += len((prefix + entry).encode("utf-8"))
        state["count"] += 1
        state["last_uid"] = patient_uid
        patient_count += 1
        uid_to_shard[patient_uid] = state["path"].name
        vocabulary.update(term_weights)

        if state["bytes"] >= shard_limit_bytes:
            close_shard()

    close_shard()

    manifest = {
        "format": "sharded_json",
        "description": (
            "Task 1 TF-IDF matrix. Each shard maps patient_uid -> {term: weight}. "
            "Terms with zero weight are omitted; nothing is truncated."
        ),
        "patient_count": patient_count,
        "term_count": len(vocabulary),
        "shard_count": len(shards),
        "shard_limit_bytes": shard_limit_bytes,
        "access": "from patient_similarity.tfidf_store import load_tfidf; "
                  "load_tfidf('outputs/tfidf_matrix')[patient_uid][term]",
        "shards": shards,
        "uid_to_shard": uid_to_shard,
    }
    with (output_dir / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    (output_dir / "README.txt").write_text(ACCESS_NOTE, encoding="utf-8")
    return manifest


def save_tfidf_matrix(output_dir, matrix, patient_uids, vectorizer,
                      shard_limit_bytes=DEFAULT_SHARD_LIMIT_BYTES):
    """Persist the full fitted TF-IDF matrix as sharded JSON (Task 1 output)."""
    return write_tfidf_shards(
        _iter_matrix_rows(matrix, patient_uids, vectorizer),
        output_dir,
        shard_limit_bytes=shard_limit_bytes,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class ShardedTfidfMatrix:
    """Dictionary-like view over a sharded TF-IDF directory.

    Supports ``matrix[patient_uid][term]`` and reads only the shard a
    patient_uid belongs to. Recently used shards are cached.
    """

    def __init__(self, directory, cache_size=2):
        self.directory = Path(directory)
        with (self.directory / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.uid_to_shard = self.manifest["uid_to_shard"]
        self.patient_count = self.manifest["patient_count"]
        self.term_count = self.manifest["term_count"]
        self._cache_size = max(1, cache_size)
        self._cache = {}
        self._cache_order = []

    def _shard(self, name):
        if name not in self._cache:
            with (self.directory / name).open("r", encoding="utf-8") as handle:
                self._cache[name] = json.load(handle)
            self._cache_order.append(name)
            while len(self._cache_order) > self._cache_size:
                evicted = self._cache_order.pop(0)
                self._cache.pop(evicted, None)
        return self._cache[name]

    def __getitem__(self, patient_uid):
        try:
            shard_name = self.uid_to_shard[patient_uid]
        except KeyError:
            raise KeyError(f"Unknown patient_uid: {patient_uid!r}") from None
        return self._shard(shard_name)[patient_uid]

    def get(self, patient_uid, default=None):
        try:
            return self[patient_uid]
        except KeyError:
            return default

    def term_weight(self, patient_uid, term, default=0.0):
        """Return the TF-IDF weight of ``term`` in ``patient_uid`` (0.0 if absent)."""
        return self[patient_uid].get(term, default)

    def __contains__(self, patient_uid):
        return patient_uid in self.uid_to_shard

    def __len__(self):
        return self.patient_count

    def keys(self):
        return self.uid_to_shard.keys()


def load_tfidf(path, cache_size=2):
    """Load the Task 1 TF-IDF output (shard directory, or a single JSON file)."""
    path = Path(path)
    if path.is_dir():
        return ShardedTfidfMatrix(path, cache_size=cache_size)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
