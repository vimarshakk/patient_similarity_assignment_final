# HI 744: Patient Similarity Retrieval Assignment

This repository implements the patient similarity retrieval system and quantitative evaluation framework required for **HI 744 (Fall 2026)**. The system processes clinical narratives from the PMC-Patients corpus, builds TF-IDF and BM25 document representations, retrieves top-5 similar patients, evaluates Precision@5 and Recall@5 against expert annotations, and generates a formal 3-page evaluation report.

---

## 1. Execution Instructions

The project conforms to the assignment grading specification:

```bash
python main.py PATH_TO_INPUT_DATA_FOLDER
```

### Environment Setup

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py /path/to/patient/json/folder
```

#### Windows
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py path\to\patient\json\folder
```

If no directory argument is passed, `main.py` automatically checks standard local dataset locations (`data/PMC_Patients/PMC-Patients-full-v2`, `~/Downloads/PMC-Patients-full-v2`).

---

## 2. Project Architecture & Modularization

The codebase is modularized into single-responsibility components inside the `patient_similarity/` package:

- `main.py`: Entry point supporting CLI invocation (`python main.py PATH_TO_INPUT_DATA_FOLDER`).
- `patient_similarity/data_loader.py`: Recursively reads, parses, normalizes, and deduplicates patient JSON files using Pandas and standard JSON parsers.
- `patient_similarity/preprocessing.py`: NLP pipeline performing lowercasing, regex stripping of URLs and numeric tokens, punctuation removal, stop word filtering (`scikit-learn`), and morphological lemmatization (`NLTK WordNetLemmatizer` with LRU caching).
- `patient_similarity/retrieval.py`: 
  - **Task 1 (TF-IDF):** Builds scikit-learn `TfidfVectorizer` matrices using smoothed IDF (`smooth_idf=True`), sublinear term frequency scaling (`sublinear_tf=True`), and L2 normalization (`norm='l2'`).
  - **Task 2 (BM25 Okapi):** Vectorized sparse matrix formulation ($S = Q \cdot M^T$) with $k_1 = 1.5$ and $b = 0.75$, scoring and ranking top-5 candidates while strictly excluding self-retrieval.
- `patient_similarity/evaluation.py`: Computes per-patient Precision@5 and Recall@5 against the `similar_patients` ground truth (treating relevance 1 and 2 equally), handles undefined recall for patients without ground truth, and calculates TF-IDF vs. BM25 Jaccard overlap.
- `patient_similarity/io_utils.py`: Serializes outputs into Python-loadable JSON formats (including the direct-lookup TF-IDF dictionary used for Task 1).
- `patient_similarity/report.py`: Programmatically generates a 3-page academic and clinical evaluation report in PDF format with running headers and page numbering.
- `patient_similarity/pipeline.py`: Orchestrates the end-to-end execution flow.

---

## 3. Required Output Deliverables

Running the command produces three required output files in `outputs/` and a comprehensive PDF report in `reports/`:

1. **`outputs/tfidf_matrix.json` (Task 1):**  
   Dictionary mapping each `patient_uid` to its term weights (`{term: weight}`), for all 167,034 patients:
   ```python
   import json
   tfidf = json.load(open("outputs/tfidf_matrix.json"))
   tfidf["1065068-1"]["meningitis"]          # 0.1825
   tfidf["1065068-1"].get("meningitis")      # None if a term was not stored
   ```
   **Size note:** to keep the artifact under GitHub's 100 MB per-file limit, each patient
   stores only their 30 highest-weighted terms, rounded to 4 decimals. Terms outside that
   top 30 are not stored (an absent term should be read as weight 0). Task 2 and Task 3
   rankings were computed from the full in-memory matrix, not from this truncated copy,
   so cosine similarity recomputed from this file will not exactly reproduce the reported
   top-5 lists or metrics.
2. **`outputs/top5_similar_patients.json` (Task 2):**  
   Dictionary where each key is a `patient_uid`, mapping to its `patient_id` and ranked top-5 lists for `tfidf` and `bm25`:
   ```json
   {
     "7451892-4": {
       "patient_id": "7451892",
       "tfidf": ["7665777-1", "7645412-3", "7664817-4", "7901777-5", "7665777-6"],
       "bm25": ["7665777-2", "7665777-3", "7665777-4", "7665777-5", "7665777-6"]
     }
   }
   ```
3. **`outputs/evaluation_per_patient.json` (Task 3):**  
   Dictionary containing per-patient Precision@5 and Recall@5 metrics for both retrieval methods:
   ```json
   {
     "7451892-4": {
       "patient_id": "7451892",
       "tfidf": {"precision_5": 0.4, "recall_5": 0.5},
       "bm25": {"precision_5": 0.6, "recall_5": 0.75}
     }
   }
   ```
   *(Recall is `null` for patients without ground-truth annotations).*
4. **`outputs/evaluation_summary.json`:**  
   Institutional summary reporting macro-average Precision@5, macro-average Recall@5, count of evaluated patients, count of excluded patients, and average top-5 Jaccard overlap between TF-IDF and BM25.
5. **`reports/patient_similarity_report.pdf`:**  
   A 3-page PDF report covering descriptive dataset summaries, preprocessing methods, TF-IDF and BM25 parameter rationales and limitations, evaluation metrics, empirical results, overlap analysis, and plain-language interpretations for healthcare leadership.

---

## 4. Evaluation Criteria & Methodological Rules

- **Relevance Definition:** Relevance annotations of 1 (moderate similarity) and 2 (high similarity) in `similar_patients` are treated equally as relevant.
- **Self-Exclusion:** Patients are strictly prevented from retrieving themselves as similar records.
- **Recall Exclusion Rule:** Patients lacking ground-truth similar patients are excluded from the macro-average recall calculation and reported in the evaluation summary.
