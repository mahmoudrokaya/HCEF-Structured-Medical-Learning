"""
Repair_Augmented_Dataset.py

Purpose:
    Repair Augmented_Data_16-1-2024.xlsx by mapping generic columns
    Column 1 ... Column 54 to meaningful biomedical feature names recovered
    from the repaired primary dataset.

Scientific role:
    - Column 1 is treated as the supervised target label.
    - Column 2 onward are aligned with the repaired biomedical variables.
    - The repaired augmented dataset becomes the canonical supervised dataset
      for model training, ablation, augmentation analysis, and reproducibility.

Inputs:
    Data/Baseline/Augmented_Data_16-1-2024.xlsx
    Data/Processed/primary_repaired.csv

Outputs:
    Data/Processed/augmented_repaired_supervised.csv
    Data/Processed/augmented_repaired_supervised_numeric.csv
    Experiments/Results/Augmented_Dataset_Repair/repair_augmented_report.json
    Experiments/Results/Augmented_Dataset_Repair/label_distribution.csv
    Experiments/Results/Augmented_Dataset_Repair/column_mapping.csv

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
import json
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
REPORT_DIR = ROOT_DIR / "Experiments" / "Results" / "Augmented_Dataset_Repair"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AUGMENTED_PATH = BASELINE_DIR / "Augmented_Data_16-1-2024.xlsx"
PRIMARY_REPAIRED_PATH = PROCESSED_DIR / "primary_repaired.csv"


# ============================================================
# SETTINGS
# ============================================================

TARGET_RAW_COLUMN = "Column 1"
TARGET_FINAL_COLUMN = "target"

LABEL_MAPPING = {
    "P": 0,
    "D": 1
}

NON_FEATURE_PRIMARY_COLUMNS = [
    "patient_No"
]


# ============================================================
# FUNCTIONS
# ============================================================

def read_excel_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_excel(path)


def read_csv_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def clean_text_value(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x.lower() in ["nan", "none", "null", ""]:
        return np.nan

    return x


def validate_augmented_structure(df: pd.DataFrame):
    expected_cols = [f"Column {i}" for i in range(1, 55)]

    missing_cols = [c for c in expected_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Augmented dataset does not contain the expected generic columns. "
            f"Missing columns: {missing_cols}"
        )

    if TARGET_RAW_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_RAW_COLUMN}")

    return expected_cols


def get_primary_feature_names(primary_df: pd.DataFrame):
    """
    Extract feature names from the repaired original dataset.
    The augmented file has Column 1 as target and Column 2..Column 54 as data.
    Therefore, we need 53 feature names.

    The repaired original contains:
        patient_No + 53 biomedical/demographic variables

    We remove patient_No because it is an identifier, not a predictive feature.
    """
    primary_cols = list(primary_df.columns)

    feature_names = [
        c for c in primary_cols
        if c not in NON_FEATURE_PRIMARY_COLUMNS
    ]

    if len(feature_names) != 53:
        raise ValueError(
            f"Expected 53 feature names after removing identifiers, "
            f"but found {len(feature_names)}.\n"
            f"Columns found: {feature_names}"
        )

    return feature_names


def build_column_mapping(feature_names):
    """
    Column 1 -> target
    Column 2 -> first repaired feature name
    ...
    Column 54 -> last repaired feature name
    """
    mapping = {TARGET_RAW_COLUMN: TARGET_FINAL_COLUMN}

    for i, feature_name in enumerate(feature_names, start=2):
        mapping[f"Column {i}"] = feature_name

    mapping_df = pd.DataFrame({
        "original_column": list(mapping.keys()),
        "repaired_column": list(mapping.values())
    })

    return mapping, mapping_df


def encode_target_labels(series: pd.Series):
    labels_clean = series.apply(clean_text_value)

    unique_labels = sorted(labels_clean.dropna().unique().tolist())

    unknown_labels = [x for x in unique_labels if x not in LABEL_MAPPING]

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found in target column: {unknown_labels}. "
            f"Expected labels: {list(LABEL_MAPPING.keys())}"
        )

    encoded = labels_clean.map(LABEL_MAPPING)

    if encoded.isna().any():
        raise ValueError("Missing or unmapped target labels detected after encoding.")

    label_report = pd.DataFrame({
        "original_label": list(LABEL_MAPPING.keys()),
        "encoded_label": list(LABEL_MAPPING.values()),
        "count": [
            int((labels_clean == label).sum())
            for label in LABEL_MAPPING.keys()
        ]
    })

    return encoded.astype(int), label_report


def coerce_features_to_numeric(df: pd.DataFrame, target_col: str):
    """
    Converts all feature columns to numeric when possible.
    For the augmented supervised dataset, all non-target columns should be numeric.
    """
    df = df.copy()

    conversion_rows = []

    for col in df.columns:
        if col == target_col:
            continue

        before_missing = int(df[col].isna().sum())

        converted = pd.to_numeric(df[col], errors="coerce")

        numeric_ratio = converted.notna().mean()

        df[col] = converted

        after_missing = int(df[col].isna().sum())

        conversion_rows.append({
            "column": col,
            "numeric_ratio_after_conversion": round(float(numeric_ratio), 6),
            "missing_before_conversion": before_missing,
            "missing_after_conversion": after_missing
        })

    conversion_report = pd.DataFrame(conversion_rows)

    return df, conversion_report


def dataset_quality_report(df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]

    missing_total = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])

    class_counts = df[target_col].value_counts().sort_index().to_dict()

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_columns": int(len(feature_cols)),
        "target_column": target_col,
        "missing_values_total": missing_total,
        "missing_values_percent": round((missing_total / total_cells) * 100, 6),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts_encoded": {str(k): int(v) for k, v in class_counts.items()},
        "class_balance_percent": {
            str(k): round((int(v) / len(df)) * 100, 4)
            for k, v in class_counts.items()
        }
    }

    return report


def save_feature_profile(df: pd.DataFrame, target_col: str):
    rows = []

    for col in df.columns:
        s = df[col]

        row = {
            "column": col,
            "role": "target" if col == target_col else "feature",
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_percent": round(float(s.isna().mean() * 100), 6),
            "unique_values": int(s.nunique(dropna=True))
        }

        if col != target_col:
            row.update({
                "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else None,
                "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) else None,
                "min": float(s.min()) if pd.api.types.is_numeric_dtype(s) else None,
                "median": float(s.median()) if pd.api.types.is_numeric_dtype(s) else None,
                "max": float(s.max()) if pd.api.types.is_numeric_dtype(s) else None,
            })
        else:
            row.update({
                "mean": None,
                "std": None,
                "min": None,
                "median": None,
                "max": None,
            })

        rows.append(row)

    profile_df = pd.DataFrame(rows)
    return profile_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Repairing augmented supervised dataset")
    print("=" * 80)

    print(f"Reading augmented dataset: {AUGMENTED_PATH}")
    augmented_df = read_excel_file(AUGMENTED_PATH)

    print(f"Reading repaired primary dataset: {PRIMARY_REPAIRED_PATH}")
    primary_df = read_csv_file(PRIMARY_REPAIRED_PATH)

    original_augmented_shape = augmented_df.shape
    original_primary_shape = primary_df.shape

    validate_augmented_structure(augmented_df)

    feature_names = get_primary_feature_names(primary_df)
    mapping, mapping_df = build_column_mapping(feature_names)

    repaired_df = augmented_df.rename(columns=mapping)

    repaired_df[TARGET_FINAL_COLUMN], label_report = encode_target_labels(
        repaired_df[TARGET_FINAL_COLUMN]
    )

    repaired_df, conversion_report = coerce_features_to_numeric(
        repaired_df,
        TARGET_FINAL_COLUMN
    )

    quality = dataset_quality_report(repaired_df, TARGET_FINAL_COLUMN)

    feature_profile = save_feature_profile(repaired_df, TARGET_FINAL_COLUMN)

    # Save repaired datasets
    repaired_path = PROCESSED_DIR / "augmented_repaired_supervised.csv"
    repaired_numeric_path = PROCESSED_DIR / "augmented_repaired_supervised_numeric.csv"

    repaired_df.to_csv(repaired_path, index=False, encoding="utf-8-sig")
    repaired_df.to_csv(repaired_numeric_path, index=False, encoding="utf-8-sig")

    # Save reports
    mapping_path = REPORT_DIR / "column_mapping.csv"
    label_distribution_path = REPORT_DIR / "label_distribution.csv"
    conversion_path = REPORT_DIR / "numeric_conversion_report.csv"
    profile_path = REPORT_DIR / "feature_profile.csv"

    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    label_report.to_csv(label_distribution_path, index=False, encoding="utf-8-sig")
    conversion_report.to_csv(conversion_path, index=False, encoding="utf-8-sig")
    feature_profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    final_report = {
        "input_augmented_dataset": str(AUGMENTED_PATH),
        "input_primary_repaired_dataset": str(PRIMARY_REPAIRED_PATH),
        "output_repaired_supervised_dataset": str(repaired_path),
        "output_repaired_supervised_numeric_dataset": str(repaired_numeric_path),
        "original_augmented_shape": {
            "rows": int(original_augmented_shape[0]),
            "columns": int(original_augmented_shape[1])
        },
        "original_primary_repaired_shape": {
            "rows": int(original_primary_shape[0]),
            "columns": int(original_primary_shape[1])
        },
        "repaired_supervised_shape": {
            "rows": int(repaired_df.shape[0]),
            "columns": int(repaired_df.shape[1])
        },
        "target_definition": {
            "raw_column": TARGET_RAW_COLUMN,
            "final_column": TARGET_FINAL_COLUMN,
            "label_mapping": LABEL_MAPPING,
            "interpretation_note": (
                "The labels are preserved from the augmented supervised dataset. "
                "The script maps P to 0 and D to 1 without redefining the clinical outcome."
            )
        },
        "quality_report": quality,
        "column_mapping_file": str(mapping_path),
        "label_distribution_file": str(label_distribution_path),
        "numeric_conversion_report_file": str(conversion_path),
        "feature_profile_file": str(profile_path),
        "important_note": (
            "This repaired augmented dataset should be used as the canonical supervised "
            "dataset for training and internal experiments. The original repaired dataset "
            "documents the source biomedical feature semantics."
        )
    }

    report_path = REPORT_DIR / "repair_augmented_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    # Markdown summary without requiring tabulate
    summary_path = REPORT_DIR / "REPAIR_AUGMENTED_SUMMARY.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Augmented Dataset Repair Summary\n\n")
        f.write(f"Input augmented dataset: `{AUGMENTED_PATH}`\n\n")
        f.write(f"Input repaired primary dataset: `{PRIMARY_REPAIRED_PATH}`\n\n")
        f.write(f"Output repaired supervised dataset: `{repaired_path}`\n\n")
        f.write("## Dataset Shape\n\n")
        f.write(f"- Original augmented shape: {original_augmented_shape}\n")
        f.write(f"- Repaired supervised shape: {repaired_df.shape}\n\n")
        f.write("## Target Definition\n\n")
        f.write(f"- Raw target column: `{TARGET_RAW_COLUMN}`\n")
        f.write(f"- Final target column: `{TARGET_FINAL_COLUMN}`\n")
        f.write(f"- Label mapping: `{LABEL_MAPPING}`\n\n")
        f.write("## Class Distribution\n\n")
        f.write(label_report.to_string(index=False))
        f.write("\n\n## Quality Report\n\n")
        for k, v in quality.items():
            f.write(f"- {k}: {v}\n")

    print("\n" + "=" * 80)
    print("Augmented dataset repair completed successfully.")
    print("=" * 80)
    print(f"Repaired supervised dataset: {repaired_path}")
    print(f"Column mapping: {mapping_path}")
    print(f"Label distribution: {label_distribution_path}")
    print(f"Repair report: {report_path}")
    print(f"Summary: {summary_path}")
    print("\nNext step:")
    print("Use augmented_repaired_supervised.csv in the corrected Prepare_Modeling_Datasets.py script.")


if __name__ == "__main__":
    main()