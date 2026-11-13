import json
from pathlib import Path


def save_json(path, data, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if indent is not None:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def save_tfidf_json(path, matrix, patient_uids, vectorizer):
    """
    Save TF-IDF weights as:
    patient_uid -> term -> weight.
    Zero-weight terms are omitted; an absent term has weight 0.
    Directly loadable in Python via json.load().
    """
    terms = vectorizer.get_feature_names_out()
    matrix = matrix.tocsr()
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data

    result = {}
    for row_index, patient_uid in enumerate(patient_uids):
        start = indptr[row_index]
        end = indptr[row_index + 1]
        row_indices = indices[start:end]
        row_data = data[start:end]
        result[patient_uid] = {
            str(terms[col]): round(float(weight), 6)
            for col, weight in zip(row_indices, row_data)
        }

    # Save compact to keep file size reasonable (~250MB for 167k records instead of ~800MB)
    save_json(path, result, indent=None)

