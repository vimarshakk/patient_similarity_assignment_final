import sys
from pathlib import Path
from patient_similarity.pipeline import run_pipeline

if __name__ == "__main__":
    # Support command line invocation: python main.py PATH_TO_INPUT_DATA_FOLDER
    if len(sys.argv) > 1:
        input_folder = Path(sys.argv[1])
    else:
        candidate_paths = [
            Path("/Users/vimarshakprudhvi/Downloads/PMC-Patients-full-v2"),
            Path("data/PMC_Patients/PMC-Patients-full-v2"),
            Path("data/PMC_Patients"),
        ]
        input_folder = next((p for p in candidate_paths if p.exists()), candidate_paths[0])

    print(f"Running Patient Similarity Pipeline on: {input_folder}")

    run_pipeline(
        input_folder=input_folder,
        output_folder=Path("outputs"),
        report_folder=Path("reports"),
        top_k=5,
        max_features=None,
    )
