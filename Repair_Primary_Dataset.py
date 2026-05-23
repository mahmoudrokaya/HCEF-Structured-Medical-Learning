"""
Repair_Primary_Dataset.py

Purpose:
    Repair the original biomedical dataset into a clean canonical tabular dataset.

Inputs:
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Data/Baseline/Data 16-1-2024.csv
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Data/Baseline/Data 16-1-2024.xlsx
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Data/Baseline/Data1.xlsx
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Data/Baseline/Augmented_Data_16-1-2024.xlsx

Outputs:
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Data/Processed/primary_repaired.csv
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Data/Processed/primary_repaired_numeric.csv
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Experiments/Results/Dataset_Repair/repair_report.json
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Experiments/Results/Dataset_Repair/repair_summary.csv

Important:
    This script does NOT guess the final clinical target blindly.
    It repairs the dataset and produces target-candidate diagnostics.
"""

from pathlib import Path
import json
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")
BASELINE_DIR = ROOT_DIR / "Data" / "Baseline"
PROCESSED_DIR = ROOT_DIR / "Data" / "Processed"
REPORT_DIR = ROOT_DIR / "Experiments" / "Results" / "Dataset_Repair"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "raw_csv": BASELINE_DIR / "Data 16-1-2024.csv",
    "raw_xlsx": BASELINE_DIR / "Data 16-1-2024.xlsx",
    "data1": BASELINE_DIR / "Data1.xlsx",
    "augmented_6000": BASELINE_DIR / "Augmented_Data_16-1-2024.xlsx",
}


# ============================================================
# COLUMN GROUPS BASED ON BIOMEDICAL SECTIONS
# ============================================================

SECTION_NAMES = [
    "CBC",
    "KIDNEY FUNCTION TEST",
    "LIVER FUNCTION TEST",
    "LIPID PROFILE",
    "CARDIAC PROFILE",
    "Hematology",
]


# ============================================================
# BASIC UTILITIES
# ============================================================

def read_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=0)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, header=0)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_text(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x)

    if x.lower() in ["nan", "none", "null", ""]:
        return np.nan

    return x


def clean_column_name(name):
    name = normalize_text(name)

    if pd.isna(name):
        return "unknown"

    name = str(name)
    name = name.replace("/", "_")
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if not name:
        name = "unknown"

    return name


def make_unique_columns(columns):
    seen = {}
    new_cols = []

    for c in columns:
        base = clean_column_name(c)

        if base not in seen:
            seen[base] = 0
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")

    return new_cols


def is_section_header(value):
    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return any(value == s.lower() for s in SECTION_NAMES)


def row_non_missing_count(row):
    return row.notna().sum()


def numeric_convert_ratio(series):
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean()


# ============================================================
# HEADER REPAIR
# ============================================================

def detect_header_row(df: pd.DataFrame):
    """
    Detects likely real header row.
    The raw file has biomedical section labels and unnamed columns.
    We search rows with many non-empty text labels.
    """
    scores = []

    max_scan = min(20, len(df))

    for idx in range(max_scan):
        row = df.iloc[idx]
        non_missing = row_non_missing_count(row)
        text_count = sum(
            isinstance(v, str) and len(str(v).strip()) > 0
            for v in row.values
        )
        section_count = sum(is_section_header(v) for v in row.values)

        score = non_missing + text_count + (section_count * 2)
        scores.append((idx, score, non_missing, text_count, section_count))

    scores_df = pd.DataFrame(
        scores,
        columns=["row_index", "score", "non_missing", "text_count", "section_count"]
    )

    best_idx = int(scores_df.sort_values("score", ascending=False).iloc[0]["row_index"])

    return best_idx, scores_df


def build_repaired_headers(df: pd.DataFrame, header_row_idx: int):
    """
    Builds feature names by combining section labels and local column labels.
    """
    raw_columns = list(df.columns)
    header_values = list(df.iloc[header_row_idx].values)

    repaired_columns = []
    current_section = None

    for i, value in enumerate(header_values):
        value_clean = normalize_text(value)

        if is_section_header(value_clean):
            current_section = clean_column_name(value_clean)
            repaired_columns.append(current_section)
            continue

        if pd.isna(value_clean):
            original_col = str(raw_columns[i])
            if str(original_col).startswith("Unnamed"):
                base = f"{current_section}_feature_{i}" if current_section else f"feature_{i}"
            else:
                base = original_col
        else:
            base = value_clean
            if current_section and not is_section_header(base):
                base = f"{current_section}_{base}"

        repaired_columns.append(base)

    repaired_columns = make_unique_columns(repaired_columns)
    return repaired_columns


def remove_non_data_rows(df: pd.DataFrame):
    """
    Removes rows that are likely headers, section labels, or nearly empty.
    """
    cleaned_rows = []

    for _, row in df.iterrows():
        values = row.values

        non_missing = row_non_missing_count(row)
        section_count = sum(is_section_header(v) for v in values)

        if non_missing <= 2:
            continue

        if section_count >= 2:
            continue

        cleaned_rows.append(row)

    if not cleaned_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(cleaned_rows).reset_index(drop=True)


# ============================================================
# DATA REPAIR
# ============================================================

def repair_raw_dataset(path: Path, dataset_label: str):
    print(f"\nRepairing: {dataset_label}")
    print(f"Path: {path}")

    df_raw = read_file(path)
    original_shape = df_raw.shape

    df_raw = df_raw.applymap(normalize_text)

    header_row_idx, header_scores = detect_header_row(df_raw)
    repaired_columns = build_repaired_headers(df_raw, header_row_idx)

    df = df_raw.copy()
    df.columns = repaired_columns

    df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
    df = remove_non_data_rows(df)

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    df.columns = make_unique_columns(df.columns)

    repair_info = {
        "dataset_label": dataset_label,
        "input_path": str(path),
        "original_rows": original_shape[0],
        "original_columns": original_shape[1],
        "detected_header_row_index": header_row_idx,
        "after_repair_rows": df.shape[0],
        "after_repair_columns": df.shape[1],
        "header_detection_scores": header_scores.to_dict(orient="records"),
        "columns": list(df.columns),
    }

    return df, repair_info


def coerce_numeric_columns(df: pd.DataFrame, threshold=0.70):
    """
    Converts columns to numeric if at least threshold of values are numeric.
    """
    df = df.copy()

    conversion_report = []

    for col in df.columns:
        ratio = numeric_convert_ratio(df[col])

        if ratio >= threshold:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted = True
        else:
            converted = False

        conversion_report.append({
            "column": col,
            "numeric_ratio": round(float(ratio), 4),
            "converted_to_numeric": converted
        })

    return df, pd.DataFrame(conversion_report)


def drop_bad_columns(df: pd.DataFrame):
    """
    Drops columns that are unusable:
    - all missing
    - one unique value only
    - extremely sparse columns > 95% missing
    """
    df = df.copy()

    drop_cols = []
    reasons = []

    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        unique_count = df[col].nunique(dropna=True)

        reason = None

        if missing_ratio >= 1.0:
            reason = "all_missing"

        elif missing_ratio > 0.95:
            reason = "more_than_95_percent_missing"

        elif unique_count <= 1:
            reason = "constant_or_single_unique_value"

        if reason:
            drop_cols.append(col)
            reasons.append({
                "column": col,
                "reason": reason,
                "missing_ratio": round(float(missing_ratio), 4),
                "unique_count": int(unique_count)
            })

    df = df.drop(columns=drop_cols)

    return df, pd.DataFrame(reasons)


def identify_target_candidates(df: pd.DataFrame):
    """
    Produces candidate target columns but does not force one.
    """
    candidates = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        missing_ratio = df[col].isna().mean()

        if 2 <= unique_count <= 10 and missing_ratio < 0.30:
            value_counts = df[col].value_counts(dropna=True).to_dict()

            min_class_count = min(value_counts.values()) if value_counts else 0

            candidates.append({
                "column": col,
                "unique_count": int(unique_count),
                "missing_ratio": round(float(missing_ratio), 4),
                "min_class_count": int(min_class_count),
                "value_counts": {str(k): int(v) for k, v in value_counts.items()}
            })

    candidates = sorted(
        candidates,
        key=lambda x: (x["unique_count"], -x["min_class_count"])
    )

    return candidates


def compare_with_data1(repaired_df):
    """
    Uses Data1.xlsx as a sanity-check version because audit showed it had mostly numeric columns.
    """
    data1_path = INPUT_FILES["data1"]

    if not data1_path.exists():
        return None

    data1 = pd.read_excel(data1_path, header=0)
    data1_shape = data1.shape

    return {
        "data1_path": str(data1_path),
        "data1_shape": {
            "rows": data1_shape[0],
            "columns": data1_shape[1]
        },
        "repaired_shape": {
            "rows": repaired_df.shape[0],
            "columns": repaired_df.shape[1]
        },
        "note": "Data1.xlsx is used only as a structural sanity check, not as the canonical source unless manually confirmed."
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Repairing primary biomedical dataset")
    print("=" * 80)

    raw_csv = INPUT_FILES["raw_csv"]
    raw_xlsx = INPUT_FILES["raw_xlsx"]

    repaired_outputs = {}
    reports = {}

    for label, path in [("raw_csv", raw_csv), ("raw_xlsx", raw_xlsx)]:
        try:
            repaired_df, repair_info = repair_raw_dataset(path, label)

            numeric_df, conversion_report = coerce_numeric_columns(repaired_df)
            numeric_df, dropped_report = drop_bad_columns(numeric_df)

            target_candidates = identify_target_candidates(numeric_df)

            repaired_outputs[label] = numeric_df

            reports[label] = {
                "repair_info": repair_info,
                "numeric_conversion_report": conversion_report.to_dict(orient="records"),
                "dropped_columns": dropped_report.to_dict(orient="records"),
                "target_candidates": target_candidates
            }

            out_raw = PROCESSED_DIR / f"primary_repaired_from_{label}.csv"
            numeric_df.to_csv(out_raw, index=False, encoding="utf-8-sig")

            conversion_report.to_csv(
                REPORT_DIR / f"{label}_numeric_conversion_report.csv",
                index=False,
                encoding="utf-8-sig"
            )

            dropped_report.to_csv(
                REPORT_DIR / f"{label}_dropped_columns_report.csv",
                index=False,
                encoding="utf-8-sig"
            )

            pd.DataFrame(target_candidates).to_csv(
                REPORT_DIR / f"{label}_target_candidates.csv",
                index=False,
                encoding="utf-8-sig"
            )

            print(f"Saved repaired dataset: {out_raw}")
            print(f"Shape: {numeric_df.shape}")
            print(f"Target candidates found: {len(target_candidates)}")

        except Exception as e:
            reports[label] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"Failed to repair {label}: {e}")

    # Prefer CSV repair as canonical if available
    if "raw_csv" in repaired_outputs:
        canonical_df = repaired_outputs["raw_csv"]
        canonical_source = "raw_csv"
    elif "raw_xlsx" in repaired_outputs:
        canonical_df = repaired_outputs["raw_xlsx"]
        canonical_source = "raw_xlsx"
    else:
        raise RuntimeError("No repaired dataset could be generated.")

    canonical_path = PROCESSED_DIR / "primary_repaired.csv"
    canonical_numeric_path = PROCESSED_DIR / "primary_repaired_numeric.csv"

    canonical_df.to_csv(canonical_path, index=False, encoding="utf-8-sig")
    canonical_df.to_csv(canonical_numeric_path, index=False, encoding="utf-8-sig")

    sanity_check = compare_with_data1(canonical_df)

    final_report = {
        "canonical_source": canonical_source,
        "canonical_output": str(canonical_path),
        "canonical_numeric_output": str(canonical_numeric_path),
        "canonical_shape": {
            "rows": canonical_df.shape[0],
            "columns": canonical_df.shape[1]
        },
        "reports": reports,
        "data1_sanity_check": sanity_check,
        "important_note": (
            "The script repairs the biomedical table and generates target candidates. "
            "The final clinical target must be manually confirmed before modeling."
        )
    }

    with open(REPORT_DIR / "repair_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    summary_rows = []

    for label, report in reports.items():
        if "repair_info" in report:
            summary_rows.append({
                "source": label,
                "original_rows": report["repair_info"]["original_rows"],
                "original_columns": report["repair_info"]["original_columns"],
                "repaired_rows": report["repair_info"]["after_repair_rows"],
                "repaired_columns": report["repair_info"]["after_repair_columns"],
                "target_candidates": len(report["target_candidates"]),
                "status": "success"
            })
        else:
            summary_rows.append({
                "source": label,
                "original_rows": None,
                "original_columns": None,
                "repaired_rows": None,
                "repaired_columns": None,
                "target_candidates": None,
                "status": "failed"
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_DIR / "repair_summary.csv", index=False, encoding="utf-8-sig")

    with open(REPORT_DIR / "REPAIR_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Primary Dataset Repair Summary\n\n")
        f.write(f"Canonical source: `{canonical_source}`\n\n")
        f.write(f"Canonical repaired dataset: `{canonical_path}`\n\n")
        f.write(f"Rows: {canonical_df.shape[0]}\n\n")
        f.write(f"Columns: {canonical_df.shape[1]}\n\n")
        f.write("## Important Note\n\n")
        f.write(
            "The dataset has been structurally repaired, but the final target column "
            "must be manually confirmed before modeling.\n\n"
        )
        f.write("## Summary\n\n")
        f.write(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Primary dataset repair completed.")
    print("=" * 80)
    print(f"Canonical repaired dataset: {canonical_path}")
    print(f"Repair report: {REPORT_DIR / 'repair_report.json'}")
    print(f"Repair summary: {REPORT_DIR / 'repair_summary.csv'}")
    print("\nNext step:")
    print("Open the target candidate files and confirm the true clinical target column before modeling.")


if __name__ == "__main__":
    main()