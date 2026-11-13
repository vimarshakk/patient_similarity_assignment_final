from pathlib import Path
import statistics

from .data_loader import load_patients
from .preprocessing import preprocess_text, lemmatization_mode
from .retrieval import build_tfidf, build_bm25, retrieve_tfidf, retrieve_bm25
from .evaluation import evaluate
from .io_utils import save_json, save_tfidf_json
from .report import create_report


def run_pipeline(input_folder, output_folder, report_folder, top_k=5, max_features=None):
    output_folder = Path(output_folder)
    report_folder = Path(report_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    report_folder.mkdir(parents=True, exist_ok=True)

    print("1/6 Loading patient JSON files...")
    patients, stats = load_patients(input_folder)
    patient_uids = [p["patient_uid"] for p in patients]
    raw_texts = [p["text"] for p in patients]
    print(f"Loaded {len(patients)} unique patient records.")

    print("2/6 Preprocessing clinical text...")
    texts = [preprocess_text(text) for text in raw_texts]
    token_counts = [len(text.split()) for text in texts]
    text_stats = {
        "average": statistics.mean(token_counts) if token_counts else 0,
        "median": statistics.median(token_counts) if token_counts else 0,
        "empty": sum(1 for text in texts if not text.strip()),
    }

    print("3/6 Building TF-IDF matrix...")
    tfidf_matrix, vectorizer = build_tfidf(texts, max_features=max_features)
    save_tfidf_json(
        output_folder / "tfidf_matrix.json",
        tfidf_matrix,
        patient_uids,
        vectorizer,
    )

    print("4/6 Retrieving top 5 patients using TF-IDF and BM25...")
    print("  - Ranking patients via TF-IDF cosine similarity...")
    tfidf_rankings = retrieve_tfidf(tfidf_matrix, patient_uids, top_k)
    print("  - Fitting BM25 Okapi model...")
    bm25_model = build_bm25(texts)
    print("  - Ranking patients via BM25 scores...")
    bm25_rankings = retrieve_bm25(bm25_model, texts, patient_uids, top_k)

    patient_by_uid = {p["patient_uid"]: p for p in patients}
    task2_output = {
        uid: {
            "patient_id": patient_by_uid[uid]["patient_id"],
            "tfidf": tfidf_rankings[uid],
            "bm25": bm25_rankings[uid],
        }
        for uid in patient_uids
    }
    save_json(output_folder / "top5_similar_patients.json", task2_output)

    print("5/6 Evaluating Precision@5 and Recall@5...")
    rankings = {"tfidf": tfidf_rankings, "bm25": bm25_rankings}
    per_patient_results, summary = evaluate(patients, rankings, top_k)
    save_json(output_folder / "evaluation_per_patient.json", per_patient_results)
    save_json(output_folder / "evaluation_summary.json", summary)

    print("6/6 Generating PDF report...")
    create_report(
        report_folder / "patient_similarity_report.pdf",
        stats,
        text_stats,
        lemmatization_mode(),
        summary,
    )

    print("\n========== EVALUATION SUMMARY ==========")
    for method, values in summary["metrics"].items():
        print(
            f"{method.upper()}: "
            f"Precision@5={values['average_precision_5']:.4f}, "
            f"Recall@5={values['average_recall_5']}"
        )
    print(
        "Patients excluded from average recall:",
        summary["patients_with_no_ground_truth_excluded_from_average_recall"],
    )
    print("\nRequired output files:")
    print("1. outputs/tfidf_matrix.json")
    print("2. outputs/top5_similar_patients.json")
    print("3. outputs/evaluation_per_patient.json")
    print("Report: reports/patient_similarity_report.pdf")
