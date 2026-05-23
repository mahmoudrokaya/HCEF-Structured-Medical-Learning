"""
Build_Refined_Modeling_Dataset.py

Purpose:
    Build refined modeling-ready datasets from the audited internal tabular dataset.

This corrected version safely handles empty audit CSV files, especially:
    high_correlation_pairs.csv
which can be empty when no highly correlated feature pairs are detected.

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
from datetime import datetime
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# PATH SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

TARGET_COLUMN = "target"
RANDOM_STATE = 42

SOURCE_FILE = (
    ROOT_DIR
    / "Data"
    / "Processed"
    / "Modeling_Ready"
    / "internal_augmented_supervised_modeling_ready.csv"
)

AUDIT_TABLE_DIR = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_0_Data_Audit"
    / "Tables"
)

OUTPUT_DATA_DIR = (
    ROOT_DIR
    / "Data"
    / "Processed"
    / "Modeling_Ready_Refined"
)

OUTPUT_SPLIT_DIR = (
    ROOT_DIR
    / "Data"
    / "Splits"
    / "Modeling_Ready_Refined"
)

RESULTS_DIR = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Refined_Dataset_Build"
)

TABLE_DIR = RESULTS_DIR / "Tables"
REPORT_DIR = RESULTS_DIR / "Reports"

for folder in [OUTPUT_DATA_DIR, OUTPUT_SPLIT_DIR, RESULTS_DIR, TABLE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# THRESHOLDS
# ============================================================

TOP_K_LIST = [10, 20, 30]
MI_ZERO_THRESHOLD = 1e-12
AUC_MARGIN_THRESHOLD = 0.003
STABILITY_RATIO_MAX = 0.020
CORRELATION_THRESHOLD = 0.90


# ============================================================
# HELPERS
# ============================================================

def load_required_csv(path: Path, required_columns=None):
    if not path.exists():
        raise FileNotFoundError(f"Required audit file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Required audit file is empty and cannot be used: {path}")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {missing}")

    return df


def load_optional_csv(path: Path, expected_columns=None):
    """
    Optional audit files may be empty.
    Example:
        high_correlation_pairs.csv is empty when no correlated pairs are detected.
    In that case, return an empty DataFrame with expected columns.
    """
    if expected_columns is None:
        expected_columns = []

    if not path.exists():
        return pd.DataFrame(columns=expected_columns)

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=expected_columns)


def normalize_series(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)

    if not higher_is_better:
        s = -s

    min_v = s.min()
    max_v = s.max()

    if max_v - min_v < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - min_v) / (max_v - min_v)


def safe_feature_list(features, available_features):
    available = set(available_features)
    seen = set()
    clean = []

    for feature in features:
        if feature in available and feature not in seen:
            clean.append(feature)
            seen.add(feature)

    return clean


def detect_clone_like_features(features):
    feature_set = set(features)
    clones = []

    for f in features:
        if f.endswith("_1"):
            base = f[:-2]
            if base in feature_set:
                clones.append(f)

    return clones


def remove_redundant_by_correlation(df_features, candidate_features, score_map, threshold=0.90):
    candidate_features = safe_feature_list(candidate_features, df_features.columns)

    if len(candidate_features) <= 1:
        return candidate_features

    ranked = sorted(candidate_features, key=lambda f: score_map.get(f, 0.0), reverse=True)
    selected = []

    corr = df_features[candidate_features].corr().abs()

    for feature in ranked:
        keep = True

        for selected_feature in selected:
            try:
                if corr.loc[feature, selected_feature] >= threshold:
                    keep = False
                    break
            except Exception:
                pass

        if keep:
            selected.append(feature)

    return selected


def make_split(df, dataset_name):
    split_dir = OUTPUT_SPLIT_DIR / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN],
    )

    train_df.to_csv(split_dir / "train.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(split_dir / "validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(split_dir / "test.csv", index=False, encoding="utf-8-sig")

    return {
        "split_folder": str(split_dir),
        "train_shape": list(train_df.shape),
        "validation_shape": list(validation_df.shape),
        "test_shape": list(test_df.shape),
        "train_target_distribution": train_df[TARGET_COLUMN].value_counts().to_dict(),
        "validation_target_distribution": validation_df[TARGET_COLUMN].value_counts().to_dict(),
        "test_target_distribution": test_df[TARGET_COLUMN].value_counts().to_dict(),
    }


def save_refined_dataset(source_df, dataset_name, features, reason):
    feature_columns = [c for c in source_df.columns if c != TARGET_COLUMN]
    features = safe_feature_list(features, feature_columns)

    if len(features) == 0:
        raise ValueError(f"No valid features selected for {dataset_name}")

    refined_df = source_df[features + [TARGET_COLUMN]].copy()

    output_file = OUTPUT_DATA_DIR / f"{dataset_name}.csv"
    refined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    split_info = make_split(refined_df, dataset_name)

    return {
        "dataset_name": dataset_name,
        "reason": reason,
        "output_file": str(output_file),
        "n_rows": int(refined_df.shape[0]),
        "n_columns": int(refined_df.shape[1]),
        "n_features": int(len(features)),
        "features": features,
        **split_info,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Building refined modeling datasets")
    print("=" * 80)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

    source_df = pd.read_csv(SOURCE_FILE)

    if TARGET_COLUMN not in source_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in source file.")

    feature_columns = [c for c in source_df.columns if c != TARGET_COLUMN]

    print(f"Source file: {SOURCE_FILE}")
    print(f"Source shape: {source_df.shape}")
    print(f"Number of input features: {len(feature_columns)}")

    # ------------------------------------------------------------
    # Load audit tables
    # ------------------------------------------------------------

    mi_df = load_required_csv(
        AUDIT_TABLE_DIR / "mutual_information_scores.csv",
        ["feature", "mutual_information"],
    )

    importance_df = load_required_csv(
        AUDIT_TABLE_DIR / "extra_trees_importance.csv",
        ["feature", "importance"],
    )

    stability_df = load_required_csv(
        AUDIT_TABLE_DIR / "feature_stability_analysis.csv",
        ["feature", "mean", "std", "stability_ratio"],
    )

    auc_df = load_required_csv(
        AUDIT_TABLE_DIR / "single_feature_auc_analysis.csv",
        ["feature", "single_feature_auc"],
    )

    corr_pairs_df = load_optional_csv(
        AUDIT_TABLE_DIR / "high_correlation_pairs.csv",
        ["feature_1", "feature_2", "correlation"],
    )

    low_var_df = load_optional_csv(
        AUDIT_TABLE_DIR / "low_variance_analysis.csv",
        ["feature", "variance", "selected"],
    )

    # ------------------------------------------------------------
    # Merge scores
    # ------------------------------------------------------------

    score_df = pd.DataFrame({"feature": feature_columns})

    score_df = score_df.merge(mi_df, on="feature", how="left")
    score_df = score_df.merge(importance_df, on="feature", how="left")
    score_df = score_df.merge(
        stability_df[["feature", "mean", "std", "stability_ratio"]],
        on="feature",
        how="left",
    )
    score_df = score_df.merge(auc_df, on="feature", how="left")

    score_df["mutual_information"] = score_df["mutual_information"].fillna(0.0)
    score_df["importance"] = score_df["importance"].fillna(0.0)
    score_df["mean"] = score_df["mean"].fillna(0.0)
    score_df["std"] = score_df["std"].fillna(0.0)

    if score_df["stability_ratio"].notna().any():
        score_df["stability_ratio"] = score_df["stability_ratio"].fillna(score_df["stability_ratio"].max())
    else:
        score_df["stability_ratio"] = 0.0

    score_df["single_feature_auc"] = score_df["single_feature_auc"].fillna(0.5)
    score_df["auc_distance_from_random"] = (score_df["single_feature_auc"] - 0.5).abs()

    score_df["mi_score_norm"] = normalize_series(score_df["mutual_information"], True)
    score_df["importance_score_norm"] = normalize_series(score_df["importance"], True)
    score_df["auc_score_norm"] = normalize_series(score_df["auc_distance_from_random"], True)
    score_df["stability_score_norm"] = normalize_series(score_df["stability_ratio"], False)

    score_df["hybrid_score"] = (
        0.35 * score_df["mi_score_norm"]
        + 0.30 * score_df["importance_score_norm"]
        + 0.20 * score_df["auc_score_norm"]
        + 0.15 * score_df["stability_score_norm"]
    )

    clone_like_features = detect_clone_like_features(feature_columns)

    if "selected" in low_var_df.columns and not low_var_df.empty:
        low_variance_features = low_var_df[low_var_df["selected"] == False]["feature"].tolist()
    else:
        low_variance_features = []

    zero_mi_features = score_df[score_df["mutual_information"] <= MI_ZERO_THRESHOLD]["feature"].tolist()
    weak_auc_features = score_df[score_df["auc_distance_from_random"] < AUC_MARGIN_THRESHOLD]["feature"].tolist()
    unstable_features = score_df[score_df["stability_ratio"] > STABILITY_RATIO_MAX]["feature"].tolist()

    score_df["flag_clone_like"] = score_df["feature"].isin(clone_like_features)
    score_df["flag_low_variance"] = score_df["feature"].isin(low_variance_features)
    score_df["flag_zero_mi"] = score_df["feature"].isin(zero_mi_features)
    score_df["flag_weak_auc"] = score_df["feature"].isin(weak_auc_features)
    score_df["flag_unstable"] = score_df["feature"].isin(unstable_features)

    score_df = score_df.sort_values(by="hybrid_score", ascending=False)

    feature_score_file = TABLE_DIR / "refined_feature_score_table.csv"
    score_df.to_csv(feature_score_file, index=False, encoding="utf-8-sig")

    score_map = dict(zip(score_df["feature"], score_df["hybrid_score"]))

    # ------------------------------------------------------------
    # Build candidate subsets
    # ------------------------------------------------------------

    ranked_features = score_df["feature"].tolist()

    ranked_clean = [
        f for f in ranked_features
        if f not in clone_like_features
        and f not in low_variance_features
    ]

    ranked_corr_filtered = remove_redundant_by_correlation(
        source_df[feature_columns],
        ranked_clean,
        score_map,
        threshold=CORRELATION_THRESHOLD,
    )

    mi_positive_features = score_df[
        (score_df["mutual_information"] > MI_ZERO_THRESHOLD)
        & (~score_df["feature"].isin(clone_like_features))
        & (~score_df["feature"].isin(low_variance_features))
    ]["feature"].tolist()

    mi_positive_features = remove_redundant_by_correlation(
        source_df[feature_columns],
        mi_positive_features,
        score_map,
        threshold=CORRELATION_THRESHOLD,
    )

    stable_informative_features = score_df[
        (score_df["mutual_information"] > MI_ZERO_THRESHOLD)
        & (score_df["auc_distance_from_random"] >= AUC_MARGIN_THRESHOLD)
        & (score_df["stability_ratio"] <= STABILITY_RATIO_MAX)
        & (~score_df["feature"].isin(clone_like_features))
        & (~score_df["feature"].isin(low_variance_features))
    ]["feature"].tolist()

    stable_informative_features = remove_redundant_by_correlation(
        source_df[feature_columns],
        stable_informative_features,
        score_map,
        threshold=CORRELATION_THRESHOLD,
    )

    medical_core_preferred = [
        "RBC",
        "D_DIMER",
        "MPV",
        "GLUCOSE",
        "HBA1C",
        "SGOT_AST",
        "ESR",
        "SODIUM_NA",
        "PLATELET_COUNT",
        "POTASSIUM_K",
        "CREATININE",
        "BLOOD_UREA",
        "RDW_SD",
        "CPK",
        "RDW_CV",
        "LDH",
        "HDL",
        "PTT",
        "TRIGLYCERIDES",
        "SGPT_ALT",
        "LYMPHOCYTE",
        "MAGNESIUM",
        "PCT",
        "MCHC",
        "MCV",
        "NEUTRPHILS",
    ]

    medical_core_features = safe_feature_list(
        [f for f in medical_core_preferred if f not in clone_like_features],
        feature_columns,
    )

    hybrid_selected_features = score_df[
        (~score_df["feature"].isin(clone_like_features))
        & (~score_df["feature"].isin(low_variance_features))
        & ~(
            (score_df["mutual_information"] <= MI_ZERO_THRESHOLD)
            & (score_df["auc_distance_from_random"] < AUC_MARGIN_THRESHOLD)
        )
    ]["feature"].tolist()

    hybrid_selected_features = remove_redundant_by_correlation(
        source_df[feature_columns],
        hybrid_selected_features,
        score_map,
        threshold=CORRELATION_THRESHOLD,
    )

    # Fallbacks
    if len(stable_informative_features) < 5:
        stable_informative_features = ranked_corr_filtered[:20]

    if len(hybrid_selected_features) < 5:
        hybrid_selected_features = ranked_corr_filtered[:20]

    if len(mi_positive_features) < 5:
        mi_positive_features = ranked_corr_filtered[:20]

    if len(medical_core_features) < 5:
        medical_core_features = ranked_corr_filtered[:20]

    # ------------------------------------------------------------
    # Save refined datasets
    # ------------------------------------------------------------

    summaries = []

    for k in TOP_K_LIST:
        summaries.append(
            save_refined_dataset(
                source_df,
                f"refined_top_{k}_features",
                ranked_corr_filtered[:k],
                f"Top {k} features by hybrid audit score after clone and redundancy filtering.",
            )
        )

    summaries.append(
        save_refined_dataset(
            source_df,
            "refined_mi_positive_features",
            mi_positive_features,
            "Features with positive mutual information after clone and redundancy filtering.",
        )
    )

    summaries.append(
        save_refined_dataset(
            source_df,
            "refined_stable_informative_features",
            stable_informative_features,
            "Features passing mutual information, single-feature AUC distance, and stability filters.",
        )
    )

    summaries.append(
        save_refined_dataset(
            source_df,
            "refined_medical_core_features",
            medical_core_features,
            "Conservative medical-core feature subset based on clinically interpretable biomarkers.",
        )
    )

    summaries.append(
        save_refined_dataset(
            source_df,
            "refined_hybrid_selected_features",
            hybrid_selected_features,
            "Hybrid selected subset removing clone-like, low-variance, and zero-MI plus weak-AUC features.",
        )
    )

    summary_df = pd.DataFrame([
        {
            "dataset_name": s["dataset_name"],
            "reason": s["reason"],
            "output_file": s["output_file"],
            "n_rows": s["n_rows"],
            "n_columns": s["n_columns"],
            "n_features": s["n_features"],
            "split_folder": s["split_folder"],
            "train_shape": s["train_shape"],
            "validation_shape": s["validation_shape"],
            "test_shape": s["test_shape"],
        }
        for s in summaries
    ])

    summary_df.to_csv(TABLE_DIR / "refined_dataset_summary.csv", index=False, encoding="utf-8-sig")

    with open(REPORT_DIR / "refined_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4, ensure_ascii=False, default=str)

    filtering_flags = {
        "source_file": str(SOURCE_FILE),
        "source_shape": list(source_df.shape),
        "target_column": TARGET_COLUMN,
        "low_variance_features": low_variance_features,
        "clone_like_features": clone_like_features,
        "zero_mi_features": zero_mi_features,
        "weak_auc_features": weak_auc_features,
        "unstable_features": unstable_features,
        "high_correlation_pair_count": int(len(corr_pairs_df)),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "mi_zero_threshold": MI_ZERO_THRESHOLD,
        "auc_margin_threshold": AUC_MARGIN_THRESHOLD,
        "stability_ratio_max": STABILITY_RATIO_MAX,
    }

    with open(REPORT_DIR / "feature_filtering_flags.json", "w", encoding="utf-8") as f:
        json.dump(filtering_flags, f, indent=4, ensure_ascii=False, default=str)

    readme = f"""# Refined Modeling Dataset Build

Generated: `{datetime.now()}`

## Purpose

This workflow builds refined versions of the internal modeling-ready dataset after the deep feature audit. The goal is to test whether Experiment 1 and Experiment 2 performance were limited by noisy, weak, redundant, or unstable features.

## Source Dataset

`{SOURCE_FILE}`

Source shape:
`{source_df.shape}`

Target column:
`{TARGET_COLUMN}`

## Audit Inputs

Audit tables were read from:

`{AUDIT_TABLE_DIR}`

Used audit files:
- `mutual_information_scores.csv`
- `extra_trees_importance.csv`
- `feature_stability_analysis.csv`
- `single_feature_auc_analysis.csv`
- `high_correlation_pairs.csv`
- `low_variance_analysis.csv`

## Important Note

An empty `high_correlation_pairs.csv` is valid and means that no feature pairs exceeded the high-correlation threshold during the audit.

## Feature Scoring

Hybrid audit score:

`0.35 * normalized mutual information + 0.30 * normalized ExtraTrees importance + 0.20 * normalized AUC distance from random + 0.15 * normalized stability score`

## Generated Refined Datasets

"""

    for s in summaries:
        readme += f"""### {s['dataset_name']}

Reason:
{s['reason']}

Features:
{s['n_features']}

File:
`{s['output_file']}`

Split folder:
`{s['split_folder']}`

"""

    readme += """## Next Step

Rerun Experiment 1 on each refined dataset and compare results against the original internal baseline.

Recommended next script:
`Experiment_1B_Refined_Baselines.py`
"""

    (REPORT_DIR / "README_Refined_Modeling_Dataset_Build.md").write_text(readme, encoding="utf-8")

    print("=" * 80)
    print("Refined modeling datasets generated successfully.")
    print("=" * 80)
    print(f"Feature score table: {feature_score_file}")
    print(f"Summary CSV: {TABLE_DIR / 'refined_dataset_summary.csv'}")
    print(f"Summary JSON: {REPORT_DIR / 'refined_dataset_summary.json'}")
    print(f"Filtering flags: {REPORT_DIR / 'feature_filtering_flags.json'}")
    print(f"README: {REPORT_DIR / 'README_Refined_Modeling_Dataset_Build.md'}")
    print("\nGenerated refined datasets:")

    for s in summaries:
        print(f"  - {s['dataset_name']}: {s['n_features']} features")


if __name__ == "__main__":
    main()
