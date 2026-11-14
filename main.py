import sys
from pathlib import Path

from patient_similarity.pipeline import run_pipeline


def resolve_input_folder(argv):
    """Take the data folder from the command line, or fall back to local defaults."""
    if len(argv) > 1:
        return Path(argv[1])

    for candidate in (Path("data/PMC_Patients/PMC-Patients-full-v2"), Path("data/PMC_Patients")):
        if candidate.exists():
            return candidate

    sys.exit(
        "Usage: python main.py PATH_TO_INPUT_DATA_FOLDER\n"
        "PATH_TO_INPUT_DATA_FOLDER must contain the PMC-Patients JSON files."
    )


if __name__ == "__main__":
    input_folder = resolve_input_folder(sys.argv)
    print(f"Running Patient Similarity Pipeline on: {input_folder}")

    run_pipeline(
        input_folder=input_folder,
        output_folder=Path("outputs"),
        report_folder=Path("reports"),
        top_k=5,
        max_features=None,
    )
