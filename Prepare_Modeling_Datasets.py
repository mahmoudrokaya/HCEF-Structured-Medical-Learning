"""
Prepare_Modeling_Datasets.py

Evidence-based dataset preparation script.

Confirmed by Inspect_Modeling_Data_Structure.py:
    internal_augmented_supervised:
        target = target
        all-missing columns = 2

    external_breast_cancer:
        target = diagnosis
        all-missing columns = 1

    external_heart_disease:
        target = num

Purpose:
    Prepare modeling-ready tabular datasets aligned with the revised paper methods.

Method:
    - Load verified datasets.
    - Clean column names.
    - Drop duplicate columns.
    - Drop all-missing columns before imputation.
    - Handle missing values.
    - Encode categorical variables.
    - Min-max normalize features.
    - Create deterministic stratified 70:15:15 splits.
    - Save full reports for reproducibility.

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

DATA_DIR = ROOT_DIR / "Data"
PROCESSED_DIR = DATA_DIR / "Processed"
SPLITS_DIR = DATA_DIR / "Splits"

MODELING_READY_DIR = PROCESSED_DIR / "Modeling_Ready"
MODELING_SPLITS_DIR = SPLITS_DIR / "Modeling_Ready"

REPORT_DIR = ROOT_DIR / "Experiments" / "Results" / "Modeling_Dataset_Preparation"

for folder in [MODELING_READY_DIR, MODELING_SPLITS_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "internal_augmented_supervised": {
        "path": PROCESSED_DIR / "augmented_repaired_supervised.csv",
        "target": "target",
        "role": "internal",
        "description": "Canonical repaired supervised augmented biomedical dataset.",
        "drop_columns": []
    },

    "external_breast_cancer": {
        "path": DATA_DIR / "Breast Cancer Wisconsin Diagnostic Dataset" / "data.csv",
        "target": "diagnosis",
        "role": "external",
        "description": "Breast Cancer Wisconsin Diagnostic Dataset.",
        "drop_columns": ["id", "Unnamed: 32", "Unnamed:_32"]
    },

    "external_heart_disease": {
        "path": DATA_DIR / "UCI Heart Disease Data" / "heart_disease_uci.csv",
        "target": "num",
        "role": "external",
        "description": "UCI Heart Disease Dataset.",
        "drop_columns": ["id"]
    }
}


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_VALIDATION_TOTAL_SIZE = 0.30
VALIDATION_FROM_TEMP_SIZE = 0.50


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported dataset format: {path}")


def clean_column_name(col):
    col = str(col).strip()
    col = col.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    col = col.replace("/", "_").replace("-", "_")
    col = "_".join(col.split())
    return col if col else "unknown"


def make_unique_columns(columns):
    seen = {}
    final_columns = []

    for col in columns:
        base = clean_column_name(col)

        if base not in seen:
            seen[base] = 0
            final_columns.append(base)
        else:
            seen[base] += 1
            final_columns.append(f"{base}_{seen[base]}")

    return final_columns


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns(df.columns)
    return df


def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    missing_tokens = [
        "?", "NA", "N/A", "na", "n/a",
        "null", "NULL", "None", "none",
        "", " "
    ]
    return df.replace(missing_tokens, np.nan)


def drop_configured_columns(df: pd.DataFrame, columns_to_drop):
    columns_to_drop_clean = [clean_column_name(c) for c in columns_to_drop]
    existing = [c for c in columns_to_drop_clean if c in df.columns]

    df = df.drop(columns=existing, errors="ignore")

    return df, existing


def drop_all_missing_columns(df: pd.DataFrame, target_col: str):
    all_missing_cols = [
        c for c in df.columns
        if c != target_col and df[c].isna().all()
    ]

    df = df.drop(columns=all_missing_cols, errors="ignore")

    return df, all_missing_cols


def drop_duplicate_columns(df: pd.DataFrame):
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df, duplicate_cols


# ============================================================
# TARGET PROCESSING
# ============================================================

def binarize_heart_target_if_needed(df: pd.DataFrame, target_col: str):
    """
    UCI Heart Disease:
        num = 0 means no disease
        num > 0 means disease
    """
    df = df.copy()

    if target_col == "num":
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df[df[target_col].notna()].reset_index(drop=True)
        df[target_col] = (df[target_col] > 0).astype(int)

    return df


def encode_target(y: pd.Series):
    if pd.api.types.is_numeric_dtype(y):
        values = sorted(pd.Series(y).dropna().unique().tolist())
        mapping = {str(v): int(v) for v in values}
        encoded = y.astype(int).values
        return encoded, mapping

    le = LabelEncoder()
    encoded = le.fit_transform(y.astype(str))
    mapping = {
        str(label): int(code)
        for code, label in enumerate(le.classes_)
    }

    return encoded, mapping


def summarize_target(y, mapping):
    counts = pd.Series(y).value_counts().sort_index()

    return {
        "class_mapping": mapping,
        "class_counts": {
            str(k): int(v)
            for k, v in counts.to_dict().items()
        },
        "class_percentages": {
            str(k): round((int(v) / len(y)) * 100, 4)
            for k, v in counts.to_dict().items()
        },
        "n_classes": int(len(counts))
    }


# ============================================================
# FEATURE PROCESSING
# ============================================================

def coerce_numeric_columns(df: pd.DataFrame, target_col: str, threshold=0.90):
    df = df.copy()
    rows = []

    for col in df.columns:
        if col == target_col:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        ratio = converted.notna().mean()

        if ratio >= threshold:
            df[col] = converted
            converted_flag = True
        else:
            converted_flag = False

        rows.append({
            "column": col,
            "numeric_conversion_ratio": round(float(ratio), 6),
            "converted_to_numeric": converted_flag
        })

    return df, pd.DataFrame(rows)


def preprocess_features(X: pd.DataFrame):
    """
    Method-aligned preprocessing:
        1. Separate numeric and categorical columns.
        2. Impute numeric values using mean.
        3. Impute categorical values using most frequent category.
        4. One-hot encode categorical variables.
        5. Min-max normalize final features.
    """
    X = X.copy()

    X, duplicate_cols = drop_duplicate_columns(X)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    report = {
        "duplicate_feature_columns_removed": duplicate_cols,
        "numeric_columns_before_encoding": numeric_cols,
        "categorical_columns_before_encoding": categorical_cols,
        "n_numeric_before_encoding": len(numeric_cols),
        "n_categorical_before_encoding": len(categorical_cols)
    }

    processed_parts = []

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="mean")
        numeric_imputed = numeric_imputer.fit_transform(X[numeric_cols])

        numeric_df = pd.DataFrame(
            numeric_imputed,
            columns=numeric_cols,
            index=X.index
        )

        processed_parts.append(numeric_df)

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        categorical_imputed = categorical_imputer.fit_transform(X[categorical_cols])

        categorical_df = pd.DataFrame(
            categorical_imputed,
            columns=categorical_cols,
            index=X.index
        )

        categorical_df = pd.get_dummies(
            categorical_df,
            columns=categorical_cols,
            drop_first=False
        )

        processed_parts.append(categorical_df)

    if not processed_parts:
        raise ValueError("No valid features remain after preprocessing.")

    X_processed = pd.concat(processed_parts, axis=1)
    X_processed, duplicate_after_encoding = drop_duplicate_columns(X_processed)

    scaler = MinMaxScaler()
    X_scaled_array = scaler.fit_transform(X_processed)

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X_processed.columns,
        index=X_processed.index
    )

    report["duplicate_columns_after_encoding_removed"] = duplicate_after_encoding
    report["final_feature_columns"] = list(X_scaled.columns)
    report["n_final_features"] = int(X_scaled.shape[1])

    return X_scaled, report


# ============================================================
# SPLITTING
# ============================================================

def can_stratify(y):
    counts = pd.Series(y).value_counts()
    return counts.min() >= 2


def create_splits(X: pd.DataFrame, y, dataset_name: str):
    stratify_y = y if can_stratify(y) else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_VALIDATION_TOTAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_y
    )

    stratify_temp = y_temp if can_stratify(y_temp) else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VALIDATION_FROM_TEMP_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_temp
    )

    split_dir = MODELING_SPLITS_DIR / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)

    split_data = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test)
    }

    for split_name, (X_part, y_part) in split_data.items():
        out_df = X_part.copy()
        out_df["target"] = y_part
        out_df.to_csv(
            split_dir / f"{split_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    return {
        "split_folder": str(split_dir),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_class_counts": {
            str(k): int(v)
            for k, v in pd.Series(y_train).value_counts().sort_index().items()
        },
        "validation_class_counts": {
            str(k): int(v)
            for k, v in pd.Series(y_val).value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(k): int(v)
            for k, v in pd.Series(y_test).value_counts().sort_index().items()
        },
        "stratified": bool(stratify_y is not None)
    }


# ============================================================
# REPORTING
# ============================================================

def create_feature_profile(X: pd.DataFrame, dataset_name: str):
    rows = []

    for col in X.columns:
        s = X[col]

        rows.append({
            "dataset": dataset_name,
            "feature": col,
            "dtype": str(s.dtype),
            "missing_after_preprocessing": int(s.isna().sum()),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std())
        })

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame):
    if df.empty:
        return "No rows available."

    columns = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = [str(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# ============================================================
# MAIN DATASET PREPARATION
# ============================================================

def prepare_dataset(dataset_name, config):
    print("\n" + "=" * 80)
    print(f"Preparing dataset: {dataset_name}")
    print("=" * 80)

    path = config["path"]
    target_col = config["target"]

    df = read_dataset(path)
    raw_shape = df.shape

    df = clean_column_names(df)
    df = normalize_missing_tokens(df)

    df, duplicate_initial = drop_duplicate_columns(df)
    df, configured_dropped = drop_configured_columns(
        df,
        config.get("drop_columns", [])
    )

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {dataset_name}. "
            f"Available columns: {list(df.columns)}"
        )

    df = binarize_heart_target_if_needed(df, target_col)
    df = df[df[target_col].notna()].reset_index(drop=True)

    df, all_missing_dropped = drop_all_missing_columns(df, target_col)

    df, numeric_report = coerce_numeric_columns(
        df,
        target_col=target_col,
        threshold=0.90
    )

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    y, target_mapping = encode_target(y_raw)

    X_processed, preprocessing_report = preprocess_features(X_raw)

    processed_df = X_processed.copy()
    processed_df["target"] = y

    processed_path = MODELING_READY_DIR / f"{dataset_name}_modeling_ready.csv"
    processed_df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    split_report = create_splits(
        X_processed,
        y,
        dataset_name
    )

    feature_profile_df = create_feature_profile(
        X_processed,
        dataset_name
    )

    feature_profile_path = REPORT_DIR / f"{dataset_name}_feature_profile.csv"
    numeric_report_path = REPORT_DIR / f"{dataset_name}_numeric_conversion_report.csv"
    preparation_report_path = REPORT_DIR / f"{dataset_name}_preparation_report.json"

    feature_profile_df.to_csv(
        feature_profile_path,
        index=False,
        encoding="utf-8-sig"
    )

    numeric_report.to_csv(
        numeric_report_path,
        index=False,
        encoding="utf-8-sig"
    )

    report = {
        "dataset_name": dataset_name,
        "description": config["description"],
        "role": config["role"],
        "input_path": str(path),
        "processed_path": str(processed_path),
        "raw_shape": {
            "rows": int(raw_shape[0]),
            "columns": int(raw_shape[1])
        },
        "processed_shape": {
            "rows": int(processed_df.shape[0]),
            "columns": int(processed_df.shape[1])
        },
        "target_column": target_col,
        "duplicate_initial_columns_removed": duplicate_initial,
        "configured_columns_dropped": configured_dropped,
        "all_missing_columns_dropped": all_missing_dropped,
        "target_summary": summarize_target(y, target_mapping),
        "preprocessing_report": preprocessing_report,
        "split_report": split_report,
        "feature_profile_path": str(feature_profile_path),
        "numeric_conversion_report_path": str(numeric_report_path)
    }

    with open(preparation_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"Input shape: {raw_shape}")
    print(f"Processed shape: {processed_df.shape}")
    print(f"Target: {target_col}")
    print(f"Dropped all-missing columns: {all_missing_dropped}")
    print(f"Saved modeling-ready dataset: {processed_path}")
    print(f"Saved splits: {split_report['split_folder']}")

    return report


# ============================================================
# RUN ALL
# ============================================================

def main():
    print("=" * 80)
    print("Preparing modeling-ready datasets")
    print("=" * 80)

    reports = []

    for dataset_name, config in DATASETS.items():
        try:
            reports.append(prepare_dataset(dataset_name, config))
        except Exception as e:
            print(f"Failed to prepare {dataset_name}: {e}")
            reports.append({
                "dataset_name": dataset_name,
                "status": "failed",
                "error": str(e)
            })

    summary_rows = []

    for r in reports:
        if "processed_shape" in r:
            summary_rows.append({
                "dataset": r["dataset_name"],
                "role": r["role"],
                "raw_rows": r["raw_shape"]["rows"],
                "raw_columns": r["raw_shape"]["columns"],
                "processed_rows": r["processed_shape"]["rows"],
                "processed_columns": r["processed_shape"]["columns"],
                "target": r["target_column"],
                "n_classes": r["target_summary"]["n_classes"],
                "train_rows": r["split_report"]["train_rows"],
                "validation_rows": r["split_report"]["validation_rows"],
                "test_rows": r["split_report"]["test_rows"],
                "status": "success"
            })
        else:
            summary_rows.append({
                "dataset": r["dataset_name"],
                "role": "unknown",
                "raw_rows": None,
                "raw_columns": None,
                "processed_rows": None,
                "processed_columns": None,
                "target": None,
                "n_classes": None,
                "train_rows": None,
                "validation_rows": None,
                "test_rows": None,
                "status": "failed"
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = REPORT_DIR / "modeling_dataset_preparation_summary.csv"
    summary_json = REPORT_DIR / "modeling_dataset_preparation_summary.json"
    summary_md = REPORT_DIR / "MODELING_DATASET_PREPARATION_SUMMARY.md"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4, ensure_ascii=False)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Modeling Dataset Preparation Summary\n\n")
        f.write(
            "This report summarizes the evidence-based preparation of "
            "the modeling-ready datasets.\n\n"
        )
        f.write(markdown_table(summary_df))
        f.write("\n\n## Methodological Notes\n\n")
        f.write("- Dataset targets were verified using inspection outputs.\n")
        f.write("- All-missing columns were removed before imputation.\n")
        f.write("- Numeric values were imputed using the mean.\n")
        f.write("- Categorical values were imputed using the most frequent category.\n")
        f.write("- Categorical variables were one-hot encoded.\n")
        f.write("- All final features were min-max normalized.\n")
        f.write("- Splits follow deterministic stratified 70:15:15 partitioning.\n")
        f.write("- No simulated image data or ResNet image workflow is used.\n")

    print("\n" + "=" * 80)
    print("Modeling dataset preparation completed.")
    print("=" * 80)
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()