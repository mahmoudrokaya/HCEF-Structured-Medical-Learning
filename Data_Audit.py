"""
Data_Audit.py

Purpose:
    Audit all datasets available for the PeerJ AI Application reproducibility pipeline.

Root:
    D://47//471//New Papers//Paper 3 IJOCTA//Sub

Outputs:
    Saved under:
    D://47//471//New Papers//Paper 3 IJOCTA//Sub//Experiments//Results//Data_Audit

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
import os
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ============================================================
# PATH SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")
DATA_DIR = ROOT_DIR / "Data"
OUTPUT_DIR = ROOT_DIR / "Experiments" / "Results" / "Data_Audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SUPPORTED DATA FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = [
    ".csv",
    ".xlsx",
    ".xls",
    ".data",
    ".txt",
    ".arff"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def file_size_mb(file_path):
    return round(file_path.stat().st_size / (1024 * 1024), 4)


def detect_separator(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.readline()

        separators = [",", ";", "\t", "|"]
        counts = {sep: sample.count(sep) for sep in separators}
        best_sep = max(counts, key=counts.get)

        return best_sep if counts[best_sep] > 0 else None

    except Exception:
        return None


def read_arff_as_dataframe(file_path):
    """
    Basic ARFF reader without scipy dependency.
    Reads data after @data line.
    """
    rows = []
    columns = []
    data_started = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("%"):
                continue

            lower_line = line.lower()

            if lower_line.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[1].replace("'", "").replace('"', "")
                    columns.append(col_name)

            elif lower_line.startswith("@data"):
                data_started = True

            elif data_started:
                rows.append([x.strip() for x in line.split(",")])

    if rows:
        df = pd.DataFrame(rows, columns=columns if len(columns) == len(rows[0]) else None)
        return df

    return pd.DataFrame()


def load_dataset(file_path):
    ext = file_path.suffix.lower()

    try:
        if ext == ".csv":
            return pd.read_csv(file_path)

        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        elif ext in [".data", ".txt"]:
            sep = detect_separator(file_path)
            if sep:
                return pd.read_csv(file_path, sep=sep, header=None)
            return pd.read_csv(file_path, header=None)

        elif ext == ".arff":
            return read_arff_as_dataframe(file_path)

        else:
            return None

    except Exception as e:
        return f"READ_ERROR: {str(e)}"


def infer_possible_target_columns(df):
    """
    Detect likely target columns using common target names and low-cardinality rule.
    """
    possible_names = [
        "target", "class", "label", "outcome", "diagnosis",
        "result", "status", "disease", "num", "y"
    ]

    targets = []

    for col in df.columns:
        col_lower = str(col).lower()
        if any(name in col_lower for name in possible_names):
            targets.append(str(col))

    for col in df.columns:
        try:
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 10 and str(col) not in targets:
                targets.append(str(col))
        except Exception:
            pass

    return targets[:10]


def audit_dataframe(df):
    rows, cols = df.shape

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_total = int(df.isna().sum().sum())
    missing_percent = round((missing_total / (rows * cols)) * 100, 4) if rows * cols > 0 else 0

    duplicate_rows = int(df.duplicated().sum()) if rows > 0 else 0

    possible_targets = infer_possible_target_columns(df)

    summary = {
        "rows": rows,
        "columns": cols,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "missing_values_total": missing_total,
        "missing_values_percent": missing_percent,
        "duplicate_rows": duplicate_rows,
        "possible_target_columns": possible_targets,
        "column_names": [str(c) for c in df.columns]
    }

    return summary


def save_column_profile(df, dataset_name):
    profile_rows = []

    for col in df.columns:
        series = df[col]

        row = {
            "dataset": dataset_name,
            "column": str(col),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(series.isna().mean() * 100, 4),
            "unique_values": int(series.nunique(dropna=True))
        }

        if pd.api.types.is_numeric_dtype(series):
            row.update({
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "median": series.median(),
                "max": series.max()
            })
        else:
            top_value = series.mode(dropna=True)
            row.update({
                "mean": None,
                "std": None,
                "min": None,
                "median": None,
                "max": None,
                "top_value": top_value.iloc[0] if not top_value.empty else None
            })

        profile_rows.append(row)

    return pd.DataFrame(profile_rows)


# ============================================================
# MAIN AUDIT
# ============================================================

def run_data_audit():
    print("=" * 80)
    print("Starting data audit")
    print("=" * 80)

    all_files = []
    audit_records = []
    column_profiles = []

    for file_path in DATA_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_files.append(file_path)

    print(f"Detected data files: {len(all_files)}")

    for file_path in all_files:
        relative_path = file_path.relative_to(ROOT_DIR)
        dataset_name = str(relative_path).replace("\\", "__").replace("/", "__")

        print(f"\nAuditing: {relative_path}")

        record = {
            "file_name": file_path.name,
            "relative_path": str(relative_path),
            "extension": file_path.suffix.lower(),
            "size_mb": file_size_mb(file_path),
            "status": "pending"
        }

        df = load_dataset(file_path)

        if isinstance(df, str) and df.startswith("READ_ERROR"):
            record["status"] = "read_error"
            record["error"] = df
            audit_records.append(record)
            print(f"  Failed: {df}")
            continue

        if df is None or df.empty:
            record["status"] = "empty_or_unsupported"
            audit_records.append(record)
            print("  Empty or unsupported.")
            continue

        try:
            summary = audit_dataframe(df)
            record.update(summary)
            record["status"] = "success"

            audit_records.append(record)

            profile_df = save_column_profile(df, dataset_name)
            column_profiles.append(profile_df)

            preview_path = OUTPUT_DIR / f"{dataset_name}_preview.csv"
            df.head(20).to_csv(preview_path, index=False, encoding="utf-8-sig")

            print(f"  Rows: {summary['rows']}")
            print(f"  Columns: {summary['columns']}")
            print(f"  Missing %: {summary['missing_values_percent']}")
            print(f"  Possible targets: {summary['possible_target_columns']}")

        except Exception as e:
            record["status"] = "audit_error"
            record["error"] = str(e)
            audit_records.append(record)
            print(f"  Audit error: {e}")

    # Save summary
    audit_df = pd.DataFrame(audit_records)
    audit_summary_path = OUTPUT_DIR / "data_audit_summary.csv"
    audit_df.to_csv(audit_summary_path, index=False, encoding="utf-8-sig")

    # Save Excel report
    excel_report_path = OUTPUT_DIR / "data_audit_report.xlsx"

    with pd.ExcelWriter(excel_report_path, engine="openpyxl") as writer:
        audit_df.to_excel(writer, sheet_name="File_Level_Audit", index=False)

        if column_profiles:
            all_profiles_df = pd.concat(column_profiles, ignore_index=True)
            all_profiles_df.to_excel(writer, sheet_name="Column_Profile", index=False)

    # Save JSON report
    json_report_path = OUTPUT_DIR / "data_audit_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=4, ensure_ascii=False)

    # Save markdown summary
    markdown_path = OUTPUT_DIR / "DATA_AUDIT_SUMMARY.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# Data Audit Summary\n\n")
        f.write(f"Root directory: `{ROOT_DIR}`\n\n")
        f.write(f"Data directory: `{DATA_DIR}`\n\n")
        f.write(f"Total detected files: **{len(all_files)}**\n\n")

        f.write("## Output Files\n\n")
        f.write(f"- `data_audit_summary.csv`\n")
        f.write(f"- `data_audit_report.xlsx`\n")
        f.write(f"- `data_audit_report.json`\n")
        f.write(f"- Preview CSV files for each readable dataset\n\n")

        f.write("## Successfully Read Files\n\n")
        success_df = audit_df[audit_df["status"] == "success"]

        for _, row in success_df.iterrows():
            f.write(f"### {row['file_name']}\n")
            f.write(f"- Path: `{row['relative_path']}`\n")
            f.write(f"- Rows: {row.get('rows', 'NA')}\n")
            f.write(f"- Columns: {row.get('columns', 'NA')}\n")
            f.write(f"- Missing values %: {row.get('missing_values_percent', 'NA')}\n")
            f.write(f"- Possible target columns: {row.get('possible_target_columns', 'NA')}\n\n")

    print("\n" + "=" * 80)
    print("Data audit completed successfully.")
    print("=" * 80)
    print(f"CSV summary saved to: {audit_summary_path}")
    print(f"Excel report saved to: {excel_report_path}")
    print(f"JSON report saved to: {json_report_path}")
    print(f"Markdown report saved to: {markdown_path}")


if __name__ == "__main__":
    run_data_audit()