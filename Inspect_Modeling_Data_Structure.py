from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

DATASETS = {
    "internal_augmented_supervised": ROOT_DIR / "Data" / "Processed" / "augmented_repaired_supervised.csv",
    "external_breast_cancer": ROOT_DIR / "Data" / "Breast Cancer Wisconsin Diagnostic Dataset" / "data.csv",
    "external_heart_disease": ROOT_DIR / "Data" / "UCI Heart Disease Data" / "heart_disease_uci.csv",
}

OUTPUT_DIR = ROOT_DIR / "Experiments" / "Results" / "Modeling_Data_Inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_file(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def inspect_dataset(name, path):
    print("=" * 80)
    print(f"Inspecting: {name}")
    print(path)

    df = read_file(path)

    report = {
        "dataset": name,
        "path": str(path),
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": [],
        "candidate_targets": [],
        "all_missing_columns": [],
        "duplicate_columns": [],
    }

    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    report["duplicate_columns"] = [str(c) for c in duplicate_cols]

    for col in df.columns:
        s = df[col]

        numeric_converted = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float(numeric_converted.notna().mean())

        unique_count = int(s.nunique(dropna=True))
        missing_count = int(s.isna().sum())
        missing_percent = float(s.isna().mean() * 100)

        value_counts = s.value_counts(dropna=True).head(20).to_dict()

        col_report = {
            "column": str(col),
            "dtype": str(s.dtype),
            "missing_count": missing_count,
            "missing_percent": round(missing_percent, 6),
            "unique_count": unique_count,
            "numeric_convert_ratio": round(numeric_ratio, 6),
            "sample_values": [str(x) for x in s.dropna().head(10).tolist()],
            "top_value_counts": {str(k): int(v) for k, v in value_counts.items()},
        }

        if missing_count == len(df):
            report["all_missing_columns"].append(str(col))

        if 2 <= unique_count <= 10 and missing_percent < 50:
            report["candidate_targets"].append(col_report)

        report["columns"].append(col_report)

    with open(OUTPUT_DIR / f"{name}_inspection.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    pd.DataFrame(report["columns"]).to_csv(
        OUTPUT_DIR / f"{name}_column_inspection.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(report["candidate_targets"]).to_csv(
        OUTPUT_DIR / f"{name}_candidate_targets.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicate columns: {len(report['duplicate_columns'])}")
    print(f"All-missing columns: {len(report['all_missing_columns'])}")
    print(f"Candidate targets: {[c['column'] for c in report['candidate_targets']]}")
    print()


def main():
    summary = []

    for name, path in DATASETS.items():
        try:
            inspect_dataset(name, path)
            summary.append({"dataset": name, "status": "success", "path": str(path)})
        except Exception as e:
            print(f"Failed: {name}: {e}")
            summary.append({"dataset": name, "status": "failed", "error": str(e), "path": str(path)})

    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "inspection_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 80)
    print("Inspection completed.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()