"""
Experiment_1B_Refined_Baselines.py

Experiment 1B:
    Refined Feature-Subset Baseline Evaluation

Purpose:
    Re-run baseline models on all refined datasets generated after deep feature auditing.
    This determines whether weak performance in Experiment 1 was caused by noisy,
    redundant, unstable, or weakly informative features.

Methodological alignment:
    - Tabular data only.
    - Same deterministic split strategy already created for refined datasets.
    - Same baseline families used in Experiment 1.
    - Same validation/test metrics.
    - No image pipeline.
    - No ResNet workflow.
    - Outputs reproducibility documentation required by journal.

Journal/reviewer alignment:
    - Algorithms and code provided.
    - README generated.
    - Evaluation methodology generated for Methods.
    - Baseline training and learning dynamics included.
    - Transparent metrics, figures, tables, and limitations note generated.
    - Clean reproducibility outputs generated.

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
from datetime import datetime
import json
import time
import warnings
import platform
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATH SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

TARGET_COLUMN = "target"
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS_FOR_LEARNING_CURVES = 5

REFINED_DATA_DIR = (
    ROOT_DIR
    / "Data"
    / "Processed"
    / "Modeling_Ready_Refined"
)

REFINED_SPLIT_DIR = (
    ROOT_DIR
    / "Data"
    / "Splits"
    / "Modeling_Ready_Refined"
)

REFINED_BUILD_SUMMARY = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Refined_Dataset_Build"
    / "Tables"
    / "refined_dataset_summary.csv"
)

EXPERIMENT_1_BASELINE_METRICS = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_1_Internal_Baselines"
    / "Tables"
    / "baseline_metrics.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_1B_Refined_Baselines"
)

TABLE_DIR = OUTPUT_DIR / "Tables"
FIGURE_DIR = OUTPUT_DIR / "Figures"
METRIC_DIR = OUTPUT_DIR / "Metrics"
LOG_DIR = OUTPUT_DIR / "Logs"
CM_DIR = OUTPUT_DIR / "Confusion_Matrices"
ROC_DIR = OUTPUT_DIR / "ROC_Curves"
PR_DIR = OUTPUT_DIR / "Precision_Recall_Curves"
LEARNING_DIR = OUTPUT_DIR / "Learning_Curves"
README_DIR = OUTPUT_DIR / "Reproducibility"

for folder in [
    OUTPUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    LOG_DIR,
    CM_DIR,
    ROC_DIR,
    PR_DIR,
    LEARNING_DIR,
    README_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASETS
# ============================================================

REFINED_DATASETS = [
    "refined_top_10_features",
    "refined_top_20_features",
    "refined_top_30_features",
    "refined_mi_positive_features",
    "refined_stable_informative_features",
    "refined_medical_core_features",
    "refined_hybrid_selected_features",
]


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)
    with open(LOG_DIR / "experiment_1b_console_log.txt", "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def save_error(error):
    with open(LOG_DIR / "experiment_1b_error_log.txt", "w", encoding="utf-8") as f:
        f.write(str(error) + "\n\n")
        f.write(traceback.format_exc())


# ============================================================
# DATA LOADING
# ============================================================

def load_refined_dataset_splits(dataset_name):
    split_dir = REFINED_SPLIT_DIR / dataset_name

    train_path = split_dir / "train.csv"
    val_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing split files for {dataset_name}. Expected train.csv, validation.csv, test.csv in {split_dir}"
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def split_xy(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


# ============================================================
# MODELS
# ============================================================

def get_models():
    return {
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        "Support_Vector_Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=N_JOBS,
        ),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Extra_Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=N_JOBS,
        ),
        "K_Nearest_Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Multi_Layer_Perceptron": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True,
            ))
        ]),
    }


def get_prediction_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T
        return scores

    return None


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, y_score, dataset_name, model_name, split_name):
    labels = sorted(np.unique(y_true))
    n_classes = len(labels)

    record = {
        "dataset": dataset_name,
        "model": model_name,
        "split": split_name,
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if y_score is not None:
        try:
            if n_classes == 2:
                pos = y_score[:, 1] if y_score.ndim > 1 else y_score
                record["roc_auc"] = float(roc_auc_score(y_true, pos))
                record["pr_auc"] = float(average_precision_score(y_true, pos))
            else:
                y_bin = label_binarize(y_true, classes=labels)
                record["roc_auc"] = float(roc_auc_score(y_bin, y_score, average="weighted", multi_class="ovr"))
                record["pr_auc"] = float(average_precision_score(y_bin, y_score, average="weighted"))
        except Exception:
            pass

    return record


# ============================================================
# PLOTS
# ============================================================

def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_").replace("+", "plus")


def save_confusion_matrix_plot(cm, labels, dataset_name, model_name, split_name):
    name = f"{safe_name(dataset_name)}_{safe_name(model_name)}_{split_name}"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(f"{dataset_name} - {model_name} - {split_name} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{name}_confusion_matrix.png", dpi=300)
    plt.close("all")


def save_roc_curve(y_true, y_score, dataset_name, model_name, split_name):
    if y_score is None:
        return

    labels = sorted(np.unique(y_true))
    n_classes = len(labels)
    name = f"{safe_name(dataset_name)}_{safe_name(model_name)}_{split_name}"

    try:
        plt.figure(figsize=(7, 6))

        if n_classes == 2:
            pos = y_score[:, 1] if y_score.ndim > 1 else y_score
            fpr, tpr, _ = roc_curve(y_true, pos)
            plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
        else:
            y_bin = label_binarize(y_true, classes=labels)
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                plt.plot(fpr, tpr, label=f"Class {label}, AUC = {auc(fpr, tpr):.4f}")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset_name} - {model_name} - {split_name} ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"{name}_roc_curve.png", dpi=300)
        plt.close("all")
    except Exception as e:
        plt.close("all")
        log(f"ROC skipped for {dataset_name} {model_name} {split_name}: {e}")


def save_pr_curve(y_true, y_score, dataset_name, model_name, split_name):
    if y_score is None:
        return

    labels = sorted(np.unique(y_true))
    n_classes = len(labels)
    name = f"{safe_name(dataset_name)}_{safe_name(model_name)}_{split_name}"

    try:
        plt.figure(figsize=(7, 6))

        if n_classes == 2:
            pos = y_score[:, 1] if y_score.ndim > 1 else y_score
            precision, recall, _ = precision_recall_curve(y_true, pos)
            plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, pos):.4f}")
        else:
            y_bin = label_binarize(y_true, classes=labels)
            for i, label in enumerate(labels):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                plt.plot(recall, precision, label=f"Class {label}, PR-AUC = {average_precision_score(y_bin[:, i], y_score[:, i]):.4f}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{dataset_name} - {model_name} - {split_name} Precision-Recall")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(PR_DIR / f"{name}_precision_recall_curve.png", dpi=300)
        plt.close("all")
    except Exception as e:
        plt.close("all")
        log(f"PR skipped for {dataset_name} {model_name} {split_name}: {e}")


def save_learning_curve_plot(model, X_train, y_train, dataset_name, model_name):
    name = f"{safe_name(dataset_name)}_{safe_name(model_name)}"

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS_FOR_LEARNING_CURVES,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=N_JOBS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        lc_df = pd.DataFrame({
            "dataset": dataset_name,
            "model": model_name,
            "train_size": train_sizes,
            "train_f1_weighted_mean": train_mean,
            "train_f1_weighted_std": train_std,
            "validation_f1_weighted_mean": val_mean,
            "validation_f1_weighted_std": val_std,
        })

        lc_df.to_csv(TABLE_DIR / f"{name}_learning_curve_values.csv", index=False, encoding="utf-8-sig")

        plt.figure(figsize=(7, 6))
        plt.plot(train_sizes, train_mean, marker="o", label="Training F1-weighted")
        plt.plot(train_sizes, val_mean, marker="o", label="Validation F1-weighted")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
        plt.xlabel("Training examples")
        plt.ylabel("F1-weighted")
        plt.title(f"{dataset_name} - {model_name} Learning Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(LEARNING_DIR / f"{name}_learning_curve.png", dpi=300)
        plt.close("all")

    except Exception as e:
        log(f"Learning curve skipped for {dataset_name} {model_name}: {e}")


def save_summary_plots(best_per_dataset_df):
    if best_per_dataset_df.empty:
        return

    plot_df = best_per_dataset_df.sort_values(by="f1_weighted", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["dataset"], plot_df["f1_weighted"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best Test F1-weighted")
    plt.title("Best Baseline Performance Across Refined Feature Subsets")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_test_f1_by_refined_dataset.png", dpi=300)
    plt.close("all")

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["dataset"], plot_df["roc_auc"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best Test ROC-AUC")
    plt.title("Best ROC-AUC Across Refined Feature Subsets")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_test_roc_auc_by_refined_dataset.png", dpi=300)
    plt.close("all")

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["dataset"], plot_df["pr_auc"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best Test PR-AUC")
    plt.title("Best PR-AUC Across Refined Feature Subsets")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_test_pr_auc_by_refined_dataset.png", dpi=300)
    plt.close("all")


# ============================================================
# COMPARISON WITH EXPERIMENT 1
# ============================================================

def compare_with_original_experiment_1(best_per_dataset_df):
    rows = []

    if EXPERIMENT_1_BASELINE_METRICS.exists():
        original = pd.read_csv(EXPERIMENT_1_BASELINE_METRICS)
        original_test = original[original["split"] == "test"].copy()

        if not original_test.empty:
            best_original = original_test.sort_values(by=["f1_weighted", "roc_auc"], ascending=False).iloc[0]

            rows.append({
                "source": "Experiment_1_Original_All_Features",
                "dataset": "internal_augmented_supervised",
                "model": best_original["model"],
                "accuracy": best_original["accuracy"],
                "f1_weighted": best_original["f1_weighted"],
                "roc_auc": best_original["roc_auc"],
                "pr_auc": best_original["pr_auc"],
                "delta_f1_vs_original": 0.0,
                "delta_auc_vs_original": 0.0,
                "delta_pr_auc_vs_original": 0.0,
            })

            original_f1 = best_original["f1_weighted"]
            original_auc = best_original["roc_auc"]
            original_pr = best_original["pr_auc"]
        else:
            original_f1 = np.nan
            original_auc = np.nan
            original_pr = np.nan
    else:
        original_f1 = np.nan
        original_auc = np.nan
        original_pr = np.nan

    for _, row in best_per_dataset_df.iterrows():
        rows.append({
            "source": "Experiment_1B_Refined_Feature_Subset",
            "dataset": row["dataset"],
            "model": row["model"],
            "accuracy": row["accuracy"],
            "f1_weighted": row["f1_weighted"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "delta_f1_vs_original": row["f1_weighted"] - original_f1 if not pd.isna(original_f1) else np.nan,
            "delta_auc_vs_original": row["roc_auc"] - original_auc if not pd.isna(original_auc) else np.nan,
            "delta_pr_auc_vs_original": row["pr_auc"] - original_pr if not pd.isna(original_pr) else np.nan,
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(TABLE_DIR / "refined_vs_original_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    return comparison_df


# ============================================================
# DOCUMENTATION
# ============================================================

def write_reproducibility_files(dataset_summaries, best_per_dataset_df, comparison_df):
    readme = f"""# Experiment 1B: Refined Feature-Subset Baseline Evaluation

## Title
Paper 3 IJOCTA Revision - Refined Baseline Evaluation After Feature Audit

## Description
This experiment reruns baseline classifiers on refined feature subsets created after the deep dataset audit. The goal is to determine whether the low performance in Experiment 1 was caused by noisy, redundant, unstable, or weakly informative features.

The experiment uses tabular data only. No image simulation, image preprocessing, ResNet feature extraction, or computer-vision pipeline is used.

## Dataset Information
Refined datasets are stored in:

`{REFINED_DATA_DIR}`

Refined splits are stored in:

`{REFINED_SPLIT_DIR}`

Target column:

`{TARGET_COLUMN}`

Evaluated refined datasets:
"""

    for item in dataset_summaries:
        readme += f"""
- `{item['dataset']}`: {item['n_features']} features, train {item['train_shape']}, validation {item['validation_shape']}, test {item['test_shape']}
"""

    readme += f"""

## Code Information
Script:

`Experiment_1B_Refined_Baselines.py`

Algorithms:
- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- Extra Trees
- k-Nearest Neighbors
- Multi-Layer Perceptron

## Usage Instructions
Run:

```powershell
cd "D:\\47\\471\\New Papers\\Paper 3 IJOCTA\\Sub\\Experiments\\Code"
python Experiment_1B_Refined_Baselines.py
```

## Requirements
Python 3.10 or later.

Required libraries:
- numpy
- pandas
- matplotlib
- scikit-learn

Install:

```powershell
pip install numpy pandas matplotlib scikit-learn
```

## Methodology
1. Load each refined feature-subset dataset using its prepared stratified train, validation, and test splits.
2. Train the same baseline classifiers used in Experiment 1.
3. Evaluate each model using accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, ROC-AUC, PR-AUC, and confusion matrix analysis.
4. Generate ROC curves, precision-recall curves, confusion matrices, and learning curves.
5. Rank feature subsets by best test F1-weighted, ROC-AUC, and PR-AUC.
6. Compare all refined-feature results with the original all-feature Experiment 1 baseline.

## Citations
The internal dataset repository citation should be added if publicly archived. Third-party external dataset URLs/DOIs are handled in the external validation experiment.

## License and Contribution Guidelines
Add repository license and contribution rules before public release.
"""

    methods_text = """# Methods Text: Refined Feature-Subset Baseline Evaluation

Following the initial baseline and full-model experiments, a refined feature-subset evaluation was conducted to determine whether weak classification performance was caused by noisy, unstable, redundant, or weakly informative variables. The refined datasets were created from the audited internal tabular dataset using mutual information, ExtraTrees importance, feature-stability analysis, single-feature ROC-AUC, clone-like feature detection, and redundancy filtering.

Each refined dataset was evaluated using the same baseline algorithms and the same deterministic split structure used in the internal baseline stage. The evaluated algorithms included Logistic Regression, Support Vector Machine, Random Forest, Gradient Boosting, Extra Trees, k-Nearest Neighbors, and Multi-Layer Perceptron classifiers. For each refined feature subset, models were trained on the training partition and evaluated on validation and test partitions.

Evaluation metrics included accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, ROC-AUC, PR-AUC, and confusion matrix analysis. ROC curves, precision-recall curves, and learning curves were generated to document model behavior and learning dynamics. Results were compared against the original all-feature baseline to quantify whether feature refinement improved generalization. No image preprocessing, simulated image data, convolutional feature extraction, or ResNet-based workflow was used.
"""

    limitations = """# Limitations Note for Conclusion

The refined feature-subset baseline experiment evaluates whether performance improves after removing weak, unstable, or redundant variables. However, feature selection is based on statistical audit criteria from the same internal dataset, and therefore improvement should be interpreted as internal feature-space refinement rather than external validation. If refined subsets do not substantially improve performance, this may indicate weak target-feature association or possible limitations in the augmented dataset construction. Findings should be interpreted together with full HCE-F evaluation, ablation analysis, robustness testing, and external dataset validation.
"""

    (README_DIR / "README_Experiment_1B_Refined_Baselines.md").write_text(readme, encoding="utf-8")
    (OUTPUT_DIR / "METHODS_TEXT_Experiment_1B_Refined_Baselines.md").write_text(methods_text, encoding="utf-8")
    (OUTPUT_DIR / "LIMITATIONS_NOTE_FOR_CONCLUSION.md").write_text(limitations, encoding="utf-8")

    env = {
        "generated_at": str(datetime.now()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "root_dir": str(ROOT_DIR),
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "refined_data_dir": str(REFINED_DATA_DIR),
        "refined_split_dir": str(REFINED_SPLIT_DIR),
        "evaluated_datasets": REFINED_DATASETS,
    }

    with open(LOG_DIR / "environment_report.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=4, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================

def main():
    (LOG_DIR / "experiment_1b_console_log.txt").write_text("", encoding="utf-8")

    log("=" * 80)
    log("Experiment 1B: Refined Feature-Subset Baseline Evaluation")
    log("=" * 80)

    start_time = time.time()

    if not REFINED_SPLIT_DIR.exists():
        raise FileNotFoundError(f"Refined split directory not found: {REFINED_SPLIT_DIR}")

    all_metrics = []
    all_reports = {}
    dataset_summaries = []

    for dataset_name in REFINED_DATASETS:
        log("\n" + "=" * 80)
        log(f"Evaluating refined dataset: {dataset_name}")
        log("=" * 80)

        train_df, val_df, test_df = load_refined_dataset_splits(dataset_name)

        X_train, y_train = split_xy(train_df)
        X_val, y_val = split_xy(val_df)
        X_test, y_test = split_xy(test_df)

        labels = sorted(np.unique(pd.concat([y_train, y_val, y_test], axis=0)))

        dataset_summary = {
            "dataset": dataset_name,
            "n_features": int(X_train.shape[1]),
            "train_shape": list(train_df.shape),
            "validation_shape": list(val_df.shape),
            "test_shape": list(test_df.shape),
            "target_distribution_train": y_train.value_counts().to_dict(),
            "target_distribution_validation": y_val.value_counts().to_dict(),
            "target_distribution_test": y_test.value_counts().to_dict(),
        }

        dataset_summaries.append(dataset_summary)

        log(f"Train shape: {train_df.shape}")
        log(f"Validation shape: {val_df.shape}")
        log(f"Test shape: {test_df.shape}")
        log(f"Features: {X_train.shape[1]}")
        log(f"Classes: {labels}")

        models = get_models()

        for model_name, model in models.items():
            log("-" * 80)
            log(f"Training {model_name} on {dataset_name}")

            model_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - model_start

            for split_name, X_split, y_split in [
                ("validation", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                y_pred = model.predict(X_split)
                y_score = get_prediction_scores(model, X_split)

                metrics = compute_metrics(
                    y_true=y_split,
                    y_pred=y_pred,
                    y_score=y_score,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    split_name=split_name,
                )

                metrics["training_time_seconds"] = float(train_time)
                metrics["n_features"] = int(X_train.shape[1])
                all_metrics.append(metrics)

                cm = confusion_matrix(y_split, y_pred, labels=labels)

                cm_df = pd.DataFrame(
                    cm,
                    index=[f"true_{x}" for x in labels],
                    columns=[f"pred_{x}" for x in labels],
                )

                cm_df.to_csv(
                    TABLE_DIR / f"{safe_name(dataset_name)}_{safe_name(model_name)}_{split_name}_confusion_matrix.csv",
                    encoding="utf-8-sig",
                )

                save_confusion_matrix_plot(cm, labels, dataset_name, model_name, split_name)
                save_roc_curve(y_split, y_score, dataset_name, model_name, split_name)
                save_pr_curve(y_split, y_score, dataset_name, model_name, split_name)

                report = classification_report(
                    y_split,
                    y_pred,
                    zero_division=0,
                    output_dict=True,
                )

                all_reports[f"{dataset_name}_{model_name}_{split_name}"] = report

                with open(
                    METRIC_DIR / f"{safe_name(dataset_name)}_{safe_name(model_name)}_{split_name}_classification_report.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(report, f, indent=4, ensure_ascii=False)

            save_learning_curve_plot(model, X_train, y_train, dataset_name, model_name)

            log(f"Completed {model_name} on {dataset_name} in {train_time:.4f} seconds.")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLE_DIR / "refined_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    with open(METRIC_DIR / "refined_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False, default=str)

    with open(METRIC_DIR / "classification_reports_all_refined_models.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=4, ensure_ascii=False, default=str)

    dataset_summary_df = pd.DataFrame(dataset_summaries)
    dataset_summary_df.to_csv(TABLE_DIR / "evaluated_refined_dataset_summary.csv", index=False, encoding="utf-8-sig")

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    validation_metrics = metrics_df[metrics_df["split"] == "validation"].copy()

    best_per_dataset = (
        test_metrics
        .sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False)
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_per_dataset.to_csv(TABLE_DIR / "best_test_model_per_refined_dataset.csv", index=False, encoding="utf-8-sig")

    best_overall_test = (
        test_metrics
        .sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False)
        .reset_index(drop=True)
    )

    best_overall_test.to_csv(TABLE_DIR / "refined_baseline_test_model_ranking.csv", index=False, encoding="utf-8-sig")

    best_validation = (
        validation_metrics
        .sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False)
        .reset_index(drop=True)
    )

    best_validation.to_csv(TABLE_DIR / "refined_baseline_validation_model_ranking.csv", index=False, encoding="utf-8-sig")

    comparison_df = compare_with_original_experiment_1(best_per_dataset)
    save_summary_plots(best_per_dataset)

    total_runtime = time.time() - start_time

    summary = {
        "experiment": "Experiment 1B - Refined Feature-Subset Baseline Evaluation",
        "generated_at": str(datetime.now()),
        "target": TARGET_COLUMN,
        "evaluated_datasets": dataset_summaries,
        "best_per_refined_dataset": best_per_dataset.to_dict(orient="records"),
        "best_overall_test_model": best_overall_test.iloc[0].to_dict() if not best_overall_test.empty else None,
        "comparison_with_original_experiment_1": comparison_df.to_dict(orient="records"),
        "total_runtime_seconds": float(total_runtime),
    }

    with open(OUTPUT_DIR / "refined_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)

    with open(OUTPUT_DIR / "refined_baseline_summary.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 1B Refined Baseline Summary\n\n")
        f.write(f"Generated at: `{datetime.now()}`\n\n")
        f.write(f"Target: `{TARGET_COLUMN}`\n\n")
        f.write("## Evaluated Refined Datasets\n\n")

        for item in dataset_summaries:
            f.write(
                f"- {item['dataset']}: {item['n_features']} features, "
                f"train {item['train_shape']}, validation {item['validation_shape']}, test {item['test_shape']}\n"
            )

        f.write("\n## Best Test Model Per Refined Dataset\n\n")
        for _, row in best_per_dataset.iterrows():
            f.write(
                f"- {row['dataset']} | {row['model']} | "
                f"F1-weighted={row['f1_weighted']:.6f} | "
                f"ROC-AUC={row['roc_auc']:.6f} | "
                f"PR-AUC={row['pr_auc']:.6f}\n"
            )

        if not best_overall_test.empty:
            row = best_overall_test.iloc[0]
            f.write("\n## Best Overall Refined Baseline\n\n")
            f.write(
                f"- Dataset: {row['dataset']}\n"
                f"- Model: {row['model']}\n"
                f"- Accuracy: {row['accuracy']}\n"
                f"- F1-weighted: {row['f1_weighted']}\n"
                f"- ROC-AUC: {row['roc_auc']}\n"
                f"- PR-AUC: {row['pr_auc']}\n"
            )

        f.write("\n## Comparison With Original Experiment 1\n\n")
        for _, row in comparison_df.iterrows():
            f.write(
                f"- {row['source']} | {row['dataset']} | {row['model']} | "
                f"F1-weighted={row['f1_weighted']:.6f} | "
                f"ROC-AUC={row['roc_auc']:.6f} | "
                f"PR-AUC={row['pr_auc']:.6f}\n"
            )

        f.write(f"\nTotal runtime seconds: `{total_runtime:.4f}`\n")

    write_reproducibility_files(dataset_summaries, best_per_dataset, comparison_df)

    log("=" * 80)
    log("Experiment 1B completed successfully.")
    log("=" * 80)
    log(f"Metrics: {TABLE_DIR / 'refined_baseline_metrics.csv'}")
    log(f"Best per dataset: {TABLE_DIR / 'best_test_model_per_refined_dataset.csv'}")
    log(f"Comparison with original: {TABLE_DIR / 'refined_vs_original_baseline_comparison.csv'}")
    log(f"Summary: {OUTPUT_DIR / 'refined_baseline_summary.md'}")
    log(f"README: {README_DIR / 'README_Experiment_1B_Refined_Baselines.md'}")
    log(f"Total runtime: {total_runtime:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Experiment 1B failed. See error log:", flush=True)
        print(LOG_DIR / "experiment_1b_error_log.txt", flush=True)
        save_error(e)
        raise
