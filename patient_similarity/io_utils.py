import json
from pathlib import Path
import numpy as np


def save_json(path, data, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if indent is not None:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def save_tfidf_json(path, matrix, patient_uids, vectorizer, top_terms_per_patient=30):
    """
    Save TF-IDF weights as a single compact JSON file directly loadable in Python:
    patient_uid -> term -> weight.
    Stores the most significant terms per patient (with round(weight, 4))
    so the complete file for 167k patients stays under GitHub's 100 MB limit (~87 MB).
    Directly supports tfidf[patient_uid][term] lookup.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    terms = vectorizer.get_feature_names_out()
    matrix = matrix.tocsr()
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data

    with path.open("w", encoding="utf-8") as out:
        out.write("{")
        for i, uid in enumerate(patient_uids):
            if i > 0:
                out.write(",")
            start, end = indptr[i], indptr[i + 1]
            row_cols = indices[start:end]
            row_vals = data[start:end]
            if len(row_vals) > top_terms_per_patient:
                top_idx = np.argpartition(-row_vals, top_terms_per_patient)[:top_terms_per_patient]
                row_cols = row_cols[top_idx]
                row_vals = row_vals[top_idx]
            patient_dict = {
                str(terms[col]): round(float(val), 4)
                for col, val in zip(row_cols, row_vals)
            }
            out.write(json.dumps(uid) + ":" + json.dumps(patient_dict, separators=(",", ":")))
        out.write("}")

