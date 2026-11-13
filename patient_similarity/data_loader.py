import ast
import json
from pathlib import Path

import pandas as pd


def parse_mapping(value):
    """Parse similar_patients whether it is a dict or a string representation."""
    if isinstance(value, dict):
        return value
    if value is None or value == "" or value == "{}":
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _read_json_records(file_path):
    """Read one JSON file and normalize common JSON layouts (single patient, list, or DataFrame)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            if "patient_uid" in raw:
                return [raw]
            for key in ("data", "patients", "records", "items"):
                if isinstance(raw.get(key), list):
                    return raw[key]
            return [raw]
        if isinstance(raw, list):
            return raw
    except Exception:
        pass

    # Fallback to pandas read_json if standard json parsing fails
    try:
        frame = pd.read_json(file_path)
        if "patient_uid" in frame.columns:
            return frame.to_dict(orient="records")
    except Exception:
        pass

    return []



def load_patients(input_folder):
    """
    Load every patient JSON file recursively with pandas, combine records,
    then normalize fields required by the retrieval pipeline.
    """
    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    json_files = sorted(folder.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    all_records = []
    invalid = 0

    for file_path in json_files:
        try:
            records = _read_json_records(file_path)
            if not records:
                invalid += 1
                continue
            all_records.extend(records)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            invalid += 1

    if not all_records:
        raise ValueError("No valid patient records were found in the JSON directory.")

    # Pandas DataFrame makes the combined dataset easy to inspect and clean.
    df = pd.json_normalize(all_records, sep=".")
    if "patient_uid" not in df.columns:
        raise ValueError("The JSON records do not contain a patient_uid field.")

    df = df.dropna(subset=["patient_uid"]).copy()
    df["patient_uid"] = df["patient_uid"].astype(str).str.strip()
    df = df[df["patient_uid"] != ""]
    df = df.drop_duplicates(subset=["patient_uid"], keep="first")

    patients = []
    for _, row in df.iterrows():
        title = str(row.get("title") or "")
        description = str(
            row.get("patient")
            or row.get("case_description")
            or row.get("description")
            or row.get("text")
            or ""
        )

        similar = parse_mapping(row.get("similar_patients"))
        relevant = set()
        for patient_uid, relevance in similar.items():
            try:
                if int(relevance) in (1, 2):
                    relevant.add(str(patient_uid))
            except (TypeError, ValueError):
                continue

        patients.append({
            "patient_uid": row["patient_uid"],
            "patient_id": str(row.get("patient_id", "")),
            "text": (title + " " + description).strip(),
            "similar_patients": relevant,
        })

    if not patients:
        raise ValueError("No valid records with patient_uid were found.")

    stats = {
        "json_files_found": len(json_files),
        "invalid_records_or_files": invalid,
        "unique_patients_loaded": len(patients),
    }
    return patients, stats
