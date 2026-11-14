Task 1: TF-IDF matrix, stored as JSON shards.

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
