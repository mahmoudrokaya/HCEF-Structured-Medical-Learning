# Experiment_0_Data_Audit_and_Feature_Analysis.py


"""
Experiment_0_Data_Audit_and_Feature_Analysis.py

Purpose:
    Perform a deep data audit before rebuilding Experiment 1.

Goals:
    1. Detect noisy features.
    2. Detect redundant/highly correlated columns.
    3. Detect low-variance features.
    4. Detect leakage-like predictors.
    5. Detect unstable distributions.
    6. Detect duplicated information.
    7. Measure feature importance stability.
    8. Analyze class separability.
    9. Generate publication-quality audit outputs.

Why this is necessary:
    The current HCE-F and baseline performance is unexpectedly low.
    Before modifying architectures, the dataset itself must be audited.

Expected outcome:
    A refined feature subset that removes:
        - noisy variables,
        - redundant variables,
        - unstable variables,
        - leakage-risk variables,
        - weak predictors.

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
from datetime import datetime
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

INPUT_FILE = (
    ROOT_DIR
    / "Data"
    / "Processed"
    / "Modeling_Ready"
    / "internal_augmented_supervised_modeling_ready.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_0_Data_Audit"
)

TABLE_DIR = OUTPUT_DIR / "Tables"
FIGURE_DIR = OUTPUT_DIR / "Figures"
REPORT_DIR = OUTPUT_DIR / "Reports"

for folder in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "target"
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("Experiment 0: Deep Dataset Audit and Feature Analysis")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")


df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Target column: {TARGET_COLUMN}")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")


X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(int)


# ============================================================
# BASIC AUDIT
# ============================================================

basic_audit = {
    "n_rows": int(df.shape[0]),
    "n_columns": int(df.shape[1]),
    "n_features": int(X.shape[1]),
    "target_distribution": y.value_counts().to_dict(),
    "missing_values_total": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
}

with open(REPORT_DIR / "basic_audit.json", "w", encoding="utf-8") as f:
    json.dump(basic_audit, f, indent=4)


# ============================================================
# LOW VARIANCE FEATURES
# ============================================================

selector = VarianceThreshold(threshold=0.001)
selector.fit(X)

variance_df = pd.DataFrame({
    "feature": X.columns,
    "variance": X.var().values,
    "selected": selector.get_support()
})

variance_df = variance_df.sort_values(by="variance")
variance_df.to_csv(TABLE_DIR / "low_variance_analysis.csv", index=False)

low_variance_features = variance_df[
    variance_df["selected"] == False
]["feature"].tolist()


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

corr_matrix = X.corr().abs()

corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]

        if value >= 0.90:
            corr_pairs.append({
                "feature_1": corr_matrix.columns[i],
                "feature_2": corr_matrix.columns[j],
                "correlation": value,
            })

corr_pairs_df = pd.DataFrame(corr_pairs)

corr_pairs_df.to_csv(
    TABLE_DIR / "high_correlation_pairs.csv",
    index=False
)


plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar()
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "correlation_matrix.png", dpi=300)
plt.close()


# ============================================================
# MUTUAL INFORMATION
# ============================================================

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=RANDOM_STATE
)

mi_df = pd.DataFrame({
    "feature": X.columns,
    "mutual_information": mi_scores
}).sort_values(by="mutual_information", ascending=False)

mi_df.to_csv(TABLE_DIR / "mutual_information_scores.csv", index=False)


# ============================================================
# EXTRA TREES IMPORTANCE
# ============================================================

forest = ExtraTreesClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

forest.fit(X, y)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": forest.feature_importances_
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(TABLE_DIR / "extra_trees_importance.csv", index=False)


# ============================================================
# FEATURE STABILITY ANALYSIS
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

stability_records = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    for feature, importance in zip(X.columns, model.feature_importances_):
        stability_records.append({
            "fold": fold,
            "feature": feature,
            "importance": importance,
        })

stability_df = pd.DataFrame(stability_records)

stability_summary = stability_df.groupby("feature")["importance"].agg([
    "mean",
    "std"
]).reset_index()

stability_summary["stability_ratio"] = (
    stability_summary["std"] /
    (stability_summary["mean"] + 1e-8)
)

stability_summary = stability_summary.sort_values(
    by="stability_ratio"
)

stability_summary.to_csv(
    TABLE_DIR / "feature_stability_analysis.csv",
    index=False
)


# ============================================================
# LEAKAGE-RISK ANALYSIS
# ============================================================

leakage_records = []

for col in X.columns:
    values = X[col].values.reshape(-1, 1)

    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    model = LogisticRegression(max_iter=2000)
    model.fit(values_scaled, y)

    prob = model.predict_proba(values_scaled)[:, 1]

    auc_score = roc_auc_score(y, prob)

    leakage_records.append({
        "feature": col,
        "single_feature_auc": auc_score,
    })

leakage_df = pd.DataFrame(leakage_records)
leakage_df = leakage_df.sort_values(
    by="single_feature_auc",
    ascending=False
)

leakage_df.to_csv(
    TABLE_DIR / "single_feature_auc_analysis.csv",
    index=False
)


# ============================================================
# PCA ANALYSIS
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": pca_coords[:, 0],
    "PC2": pca_coords[:, 1],
    "target": y.values,
})

pca_df.to_csv(TABLE_DIR / "pca_projection.csv", index=False)

plt.figure(figsize=(8, 6))

for cls in sorted(y.unique()):
    idx = y.values == cls
    plt.scatter(
        pca_coords[idx, 0],
        pca_coords[idx, 1],
        label=f"Class {cls}",
        alpha=0.7,
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Class Separability")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "pca_class_separability.png", dpi=300)
plt.close()


# ============================================================
# t-SNE ANALYSIS
# ============================================================

sample_size = min(2000, len(X_scaled))

sample_idx = np.random.choice(
    len(X_scaled),
    sample_size,
    replace=False
)

X_sample = X_scaled[sample_idx]
y_sample = y.values[sample_idx]


tsne = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    perplexity=30,
)

coords = tsne.fit_transform(X_sample)

plt.figure(figsize=(8, 6))

for cls in sorted(np.unique(y_sample)):
    idx = y_sample == cls
    plt.scatter(
        coords[idx, 0],
        coords[idx, 1],
        label=f"Class {cls}",
        alpha=0.7,
    )

plt.title("t-SNE Class Separability")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "tsne_class_separability.png", dpi=300)
plt.close()


# ============================================================
# FINAL RECOMMENDATIONS
# ============================================================

recommendations = {
    "low_variance_features": low_variance_features,
    "high_correlation_pair_count": int(len(corr_pairs_df)),
    "top_mutual_information_features": mi_df.head(15).to_dict(orient="records"),
    "lowest_mutual_information_features": mi_df.tail(15).to_dict(orient="records"),
    "top_single_feature_auc": leakage_df.head(15).to_dict(orient="records"),
    "unstable_features": stability_summary.sort_values(
        by="stability_ratio",
        ascending=False
    ).head(15).to_dict(orient="records"),
}

with open(
    REPORT_DIR / "feature_engineering_recommendations.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(recommendations, f, indent=4)


# ============================================================
# MARKDOWN REPORT
# ============================================================

report = f"""
# Deep Dataset Audit Report

Generated:
{datetime.now()}

## Dataset Summary

- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Features: {X.shape[1]}
- Target: {TARGET_COLUMN}

## Key Findings

### Low Variance Features

Detected: {len(low_variance_features)}

### High Correlation Pairs

Detected: {len(corr_pairs_df)}

### Potential Leakage Indicators

Features with extremely high single-feature AUC should be manually reviewed.

### Feature Stability

Features with unstable importance across folds may introduce noise.

### Class Separability

PCA and t-SNE projections help determine whether the classes are naturally separable.

## Recommended Next Step

Construct a refined dataset version:

1. Remove low variance features.
2. Remove highly correlated redundant columns.
3. Remove unstable noisy features.
4. Remove suspicious leakage-like predictors.
5. Rebuild Experiment 1 using the refined dataset.
"""

(REPORT_DIR / "DATA_AUDIT_REPORT.md").write_text(
    report,
    encoding="utf-8"
)


print("=" * 80)
print("Experiment 0 completed successfully.")
print("=" * 80)
print(f"Results folder: {OUTPUT_DIR}")
print(f"Audit report: {REPORT_DIR / 'DATA_AUDIT_REPORT.md'}")
print(f"Feature importance: {TABLE_DIR / 'extra_trees_importance.csv'}")
print(f"Leakage analysis: {TABLE_DIR / 'single_feature_auc_analysis.csv'}")
print(f"Correlation analysis: {TABLE_DIR / 'high_correlation_pairs.csv'}")
print(f"Feature stability: {TABLE_DIR / 'feature_stability_analysis.csv'}")
print(f"Recommendations: {REPORT_DIR / 'feature_engineering_recommendations.json'}")
