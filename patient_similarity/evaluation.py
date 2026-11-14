def precision_at_k(retrieved, relevant, k):
    """Fraction of the top-k results that are relevant (hits / k)."""
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / k if k else 0.0


def recall_at_k(retrieved, relevant, k):
    """Fraction of all relevant patients that appear in the top-k results."""
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def evaluate(patients, rankings, k=5):
    """Score every retrieval method per patient.

    Returns ``(per_patient, summary)``. ``per_patient`` matches the required
    Task 3 format: ``{patient_uid: {patient_id, method: {precision_5, recall_5}}}``.
    Patients without ground-truth similar patients get ``recall_5 = None`` and
    are excluded from the average recall.
    """
    per_patient = {}
    for patient in patients:
        uid = patient["patient_uid"]
        relevant = patient["similar_patients"]
        entry = {"patient_id": patient["patient_id"]}
        for method, retrieved_by_uid in rankings.items():
            retrieved = retrieved_by_uid.get(uid, [])
            entry[method] = {
                f"precision_{k}": round(precision_at_k(retrieved, relevant, k), 6),
                f"recall_{k}": recall_at_k(retrieved, relevant, k),
            }
        per_patient[uid] = entry

    summary = {"metrics": {}, "k": k}
    for method in rankings:
        precisions = [values[method][f"precision_{k}"] for values in per_patient.values()]
        recalls = [
            values[method][f"recall_{k}"]
            for values in per_patient.values()
            if values[method][f"recall_{k}"] is not None
        ]
        summary["metrics"][method] = {
            f"average_precision_{k}": round(sum(precisions) / len(precisions), 6) if precisions else 0.0,
            f"average_recall_{k}": round(sum(recalls) / len(recalls), 6) if recalls else None,
            "precision_patients_count": len(precisions),
            "recall_patients_count": len(recalls),
        }

    summary[f"patients_with_no_ground_truth_excluded_from_average_recall_{k}"] = sum(
        1 for patient in patients if not patient["similar_patients"]
    )
    summary["total_patients"] = len(patients)
    summary[f"average_top{k}_jaccard_overlap"] = average_top_k_overlap(rankings, k)
    return per_patient, summary


def average_top_k_overlap(rankings, k=5):
    """Mean Jaccard overlap of the top-k lists of every pair of methods."""
    methods = list(rankings)
    overlaps = {}
    for i, first in enumerate(methods):
        for second in methods[i + 1:]:
            scores = []
            for uid, first_results in rankings[first].items():
                a = set(first_results[:k])
                b = set(rankings[second].get(uid, [])[:k])
                union = a | b
                if union:
                    scores.append(len(a & b) / len(union))
            overlaps[f"{first}|{second}"] = round(sum(scores) / len(scores), 6) if scores else None
    return overlaps
