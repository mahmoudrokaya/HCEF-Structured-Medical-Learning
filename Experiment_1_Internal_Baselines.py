"""
Experiment_1_Internal_Baselines.py

Experiment 1:
    Internal Baseline Training and Learning Dynamics

Purpose:
    Run reproducible baseline models on the canonical internal tabular dataset.

Outputs:
    D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Experiments/Results/Experiment_1_Internal_Baselines

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
DATASET_NAME = "internal_augmented_supervised"
TARGET_COLUMN = "target"

MODEL_READY_FILE = ROOT_DIR / "Data" / "Processed" / "Modeling_Ready" / "internal_augmented_supervised_modeling_ready.csv"
SPLIT_DIR = ROOT_DIR / "Data" / "Splits" / "Modeling_Ready" / DATASET_NAME

OUTPUT_DIR = ROOT_DIR / "Experiments" / "Results" / "Experiment_1_Internal_Baselines"
TABLE_DIR = OUTPUT_DIR / "Tables"
METRIC_DIR = OUTPUT_DIR / "Metrics"
LOG_DIR = OUTPUT_DIR / "Logs"
CM_DIR = OUTPUT_DIR / "Confusion_Matrices"
ROC_DIR = OUTPUT_DIR / "ROC_Curves"
PR_DIR = OUTPUT_DIR / "Precision_Recall_Curves"
LEARNING_DIR = OUTPUT_DIR / "Learning_Curves"
README_DIR = OUTPUT_DIR / "Reproducibility"

for folder in [OUTPUT_DIR, TABLE_DIR, METRIC_DIR, LOG_DIR, CM_DIR, ROC_DIR, PR_DIR, LEARNING_DIR, README_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS_FOR_LEARNING_CURVES = 5


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)
    with open(LOG_DIR / "experiment_1_console_log.txt", "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def save_error(error):
    with open(LOG_DIR / "experiment_1_error_log.txt", "w", encoding="utf-8") as f:
        f.write(str(error) + "\n\n")
        f.write(traceback.format_exc())


# ============================================================
# DATA LOADING
# ============================================================

def find_split_file(split_name):
    candidates = [
        SPLIT_DIR / f"{split_name}.csv",
        SPLIT_DIR / f"{DATASET_NAME}_{split_name}.csv",
        SPLIT_DIR / f"{split_name}_data.csv",
        SPLIT_DIR / f"{split_name}_split.csv",
    ]

    if split_name == "validation":
        candidates.extend([
            SPLIT_DIR / "val.csv",
            SPLIT_DIR / f"{DATASET_NAME}_val.csv",
            SPLIT_DIR / "validation_data.csv",
            SPLIT_DIR / "val_data.csv",
        ])

    for path in candidates:
        if path.exists():
            return path

    # broader fallback
    if SPLIT_DIR.exists():
        for path in SPLIT_DIR.glob("*.csv"):
            low = path.name.lower()
            if split_name == "train" and "train" in low:
                return path
            if split_name == "validation" and ("val" in low or "validation" in low):
                return path
            if split_name == "test" and "test" in low:
                return path

    return None


def load_data():
    log("Checking prepared split files...")
    log(f"Split folder: {SPLIT_DIR}")

    train_path = find_split_file("train")
    val_path = find_split_file("validation")
    test_path = find_split_file("test")

    if train_path and val_path and test_path:
        log(f"Using prepared train file: {train_path}")
        log(f"Using prepared validation file: {val_path}")
        log(f"Using prepared test file: {test_path}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        return train_df, val_df, test_df, "prepared_split_files"

    log("Prepared split files were not detected with expected names.")
    log("Using fallback stratified 70:15:15 split from modeling-ready dataset.")

    if not MODEL_READY_FILE.exists():
        raise FileNotFoundError(f"Modeling-ready dataset not found: {MODEL_READY_FILE}")

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(MODEL_READY_FILE)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {MODEL_READY_FILE}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET_COLUMN],
        random_state=RANDOM_STATE
    )

    return train_df, val_df, test_df, "fallback_split_from_modeling_ready_file"


def split_xy(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from dataframe.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
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
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=N_JOBS
        ),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Extra_Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=N_JOBS
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
                early_stopping=True
            ))
        ]),
    }


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T
        return scores
    return None


# ============================================================
# METRICS AND PLOTS
# ============================================================

def compute_metrics(y_true, y_pred, y_score, model_name, split_name):
    labels = sorted(np.unique(y_true))
    n_classes = len(labels)

    out = {
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
                out["roc_auc"] = float(roc_auc_score(y_true, pos))
                out["pr_auc"] = float(average_precision_score(y_true, pos))
            else:
                y_bin = label_binarize(y_true, classes=labels)
                out["roc_auc"] = float(roc_auc_score(y_bin, y_score, average="weighted", multi_class="ovr"))
                out["pr_auc"] = float(average_precision_score(y_bin, y_score, average="weighted"))
        except Exception:
            pass

    return out


def plot_confusion_matrix(cm, labels, model_name, split_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title(f"{model_name} - {split_name} Confusion Matrix")
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
    plt.savefig(CM_DIR / f"{model_name}_{split_name}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc(y_true, y_score, model_name, split_name):
    if y_score is None:
        return

    labels = sorted(np.unique(y_true))
    n_classes = len(labels)

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
        plt.title(f"{model_name} - {split_name} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"{model_name}_{split_name}_roc_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        plt.close()
        log(f"ROC plot skipped for {model_name} {split_name}: {e}")


def plot_pr(y_true, y_score, model_name, split_name):
    if y_score is None:
        return

    labels = sorted(np.unique(y_true))
    n_classes = len(labels)

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
        plt.title(f"{model_name} - {split_name} Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(PR_DIR / f"{model_name}_{split_name}_precision_recall_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        plt.close()
        log(f"PR plot skipped for {model_name} {split_name}: {e}")


def plot_learning_curve(model, X_train, y_train, model_name):
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
            random_state=RANDOM_STATE
        )

        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        pd.DataFrame({
            "train_size": train_sizes,
            "train_f1_weighted_mean": train_mean,
            "train_f1_weighted_std": train_std,
            "validation_f1_weighted_mean": val_mean,
            "validation_f1_weighted_std": val_std,
        }).to_csv(TABLE_DIR / f"{model_name}_learning_curve_values.csv", index=False, encoding="utf-8-sig")

        plt.figure(figsize=(7, 6))
        plt.plot(train_sizes, train_mean, marker="o", label="Training F1-weighted")
        plt.plot(train_sizes, val_mean, marker="o", label="Validation F1-weighted")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
        plt.xlabel("Training examples")
        plt.ylabel("F1-weighted")
        plt.title(f"{model_name} Learning Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(LEARNING_DIR / f"{model_name}_learning_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        log(f"Learning curve skipped for {model_name}: {e}")


# ============================================================
# DOCUMENTATION OUTPUTS
# ============================================================

def write_reproducibility_files(train_df, val_df, test_df, source_mode):
    readme = f"""# Experiment 1: Internal Baseline Training and Learning Dynamics

## Title
Paper 3 IJOCTA Revision - Internal Baseline Training

## Description
This experiment establishes baseline performance on the canonical internal tabular biomedical dataset before applying the full Hybrid Contrastive-Ensemble Framework.

No image simulation, image preprocessing, ResNet feature extraction, or computer-vision workflow is used.

## Dataset Information
Dataset name: `{DATASET_NAME}`

Modeling-ready file:
`{MODEL_READY_FILE}`

Split folder:
`{SPLIT_DIR}`

Target column:
`{TARGET_COLUMN}`

Data source mode:
`{source_mode}`

Training samples: {len(train_df)}
Validation samples: {len(val_df)}
Test samples: {len(test_df)}

## Code Information
Script:
`Experiment_1_Internal_Baselines.py`

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
python Experiment_1_Internal_Baselines.py
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
1. Load the prepared internal tabular dataset splits.
2. Separate features and target.
3. Train seven baseline classifiers.
4. Evaluate validation and test performance.
5. Report accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrices, ROC curves, precision-recall curves, and learning curves.
6. Save all tables, figures, logs, and summaries.

## Citations
The internal dataset repository citation should be added if the dataset is publicly archived. Third-party dataset URLs/DOIs are handled in the external validation experiment.

## License and Contribution Guidelines
Add the selected repository license and contribution rules before public release.
"""

    methods_text = """# Methods Text: Baseline Training and Learning Dynamics

Baseline evaluation was conducted on the canonical internal tabular dataset prepared through deterministic preprocessing. The dataset was evaluated using stratified training, validation, and test partitions. The baseline evaluation included Logistic Regression, Support Vector Machine, Random Forest, Gradient Boosting, Extra Trees, k-Nearest Neighbors, and Multi-Layer Perceptron classifiers.

Each model was trained on the training partition and evaluated on the validation and test partitions using accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, ROC-AUC, PR-AUC, and confusion matrix analysis. ROC and precision-recall curves were generated when probabilistic predictions were available. Learning dynamics were assessed using cross-validated learning curves based on weighted F1-score across increasing training-set sizes.

No image preprocessing, simulated image data, convolutional feature extraction, or ResNet-based workflow was used.
"""

    limitations = """# Limitations Note for Conclusion

This baseline experiment provides an internal reference for comparison but does not by itself establish the effectiveness of the proposed HCE-F model. The internal dataset is augmented and may not fully represent variability across independent clinical cohorts. Conclusions about the full framework should rely on the combined results of the full HCE-F evaluation, ablation study, robustness analysis, and external validation.
"""

    (README_DIR / "README_Experiment_1_Internal_Baselines.md").write_text(readme, encoding="utf-8")
    (OUTPUT_DIR / "METHODS_TEXT_Experiment_1_Baseline_Evaluation.md").write_text(methods_text, encoding="utf-8")
    (OUTPUT_DIR / "LIMITATIONS_NOTE_FOR_CONCLUSION.md").write_text(limitations, encoding="utf-8")

    env = {
        "generated_at": str(datetime.now()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "root_dir": str(ROOT_DIR),
        "dataset": DATASET_NAME,
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
    }

    with open(LOG_DIR / "environment_report.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=4, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================

def main():
    # reset log
    (LOG_DIR / "experiment_1_console_log.txt").write_text("", encoding="utf-8")

    log("=" * 80)
    log("Experiment 1: Internal Baseline Training and Learning Dynamics")
    log("=" * 80)

    start = time.time()

    train_df, val_df, test_df, source_mode = load_data()

    log(f"Data source mode: {source_mode}")
    log(f"Train shape: {train_df.shape}")
    log(f"Validation shape: {val_df.shape}")
    log(f"Test shape: {test_df.shape}")

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    labels = sorted(np.unique(pd.concat([y_train, y_val, y_test], axis=0)))
    log(f"Target classes: {labels}")

    models = get_models()
    all_metrics = []
    all_reports = {}

    for model_name, model in models.items():
        log("-" * 80)
        log(f"Training model: {model_name}")

        model_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - model_start
        log(f"Training completed in {train_time:.4f} seconds.")

        for split_name, X_split, y_split in [
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            y_pred = model.predict(X_split)
            y_score = get_scores(model, X_split)

            metric_record = compute_metrics(y_split, y_pred, y_score, model_name, split_name)
            metric_record["training_time_seconds"] = float(train_time)
            all_metrics.append(metric_record)

            cm = confusion_matrix(y_split, y_pred, labels=labels)
            pd.DataFrame(
                cm,
                index=[f"true_{x}" for x in labels],
                columns=[f"pred_{x}" for x in labels],
            ).to_csv(TABLE_DIR / f"{model_name}_{split_name}_confusion_matrix.csv", encoding="utf-8-sig")

            plot_confusion_matrix(cm, labels, model_name, split_name)
            plot_roc(y_split, y_score, model_name, split_name)
            plot_pr(y_split, y_score, model_name, split_name)

            report = classification_report(y_split, y_pred, zero_division=0, output_dict=True)
            all_reports[f"{model_name}_{split_name}"] = report

            with open(METRIC_DIR / f"{model_name}_{split_name}_classification_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)

        plot_learning_curve(model, X_train, y_train, model_name)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df.to_csv(TABLE_DIR / "baseline_metrics.csv", index=False, encoding="utf-8-sig")

    with open(METRIC_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    with open(METRIC_DIR / "classification_reports_all_models.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=4, ensure_ascii=False)

    validation_rank = metrics_df[metrics_df["split"] == "validation"].sort_values(
        by=["f1_weighted", "roc_auc"], ascending=False
    ).reset_index(drop=True)

    test_rank = metrics_df[metrics_df["split"] == "test"].sort_values(
        by=["f1_weighted", "roc_auc"], ascending=False
    ).reset_index(drop=True)

    validation_rank.to_csv(TABLE_DIR / "baseline_validation_model_ranking.csv", index=False, encoding="utf-8-sig")
    test_rank.to_csv(TABLE_DIR / "baseline_test_model_ranking.csv", index=False, encoding="utf-8-sig")

    best_validation = validation_rank.iloc[0].to_dict()
    best_test = test_rank.iloc[0].to_dict()

    runtime = time.time() - start

    summary = {
        "experiment": "Experiment 1 - Internal Baseline Training and Learning Dynamics",
        "generated_at": str(datetime.now()),
        "dataset": DATASET_NAME,
        "target": TARGET_COLUMN,
        "source_mode": source_mode,
        "train_shape": list(train_df.shape),
        "validation_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "best_validation_model": best_validation,
        "best_test_model": best_test,
        "total_runtime_seconds": float(runtime),
    }

    with open(OUTPUT_DIR / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)

    with open(OUTPUT_DIR / "baseline_summary.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 1 Baseline Summary\n\n")
        f.write(f"Generated at: `{datetime.now()}`\n\n")
        f.write(f"Dataset: `{DATASET_NAME}`\n\n")
        f.write(f"Target: `{TARGET_COLUMN}`\n\n")
        f.write(f"Source mode: `{source_mode}`\n\n")
        f.write(f"Training shape: `{train_df.shape}`\n\n")
        f.write(f"Validation shape: `{val_df.shape}`\n\n")
        f.write(f"Test shape: `{test_df.shape}`\n\n")
        f.write("## Best Validation Model\n\n")
        for key, value in best_validation.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n## Best Test Model\n\n")
        for key, value in best_test.items():
            f.write(f"- {key}: {value}\n")
        f.write(f"\nTotal runtime seconds: `{runtime:.4f}`\n")

    write_reproducibility_files(train_df, val_df, test_df, source_mode)

    log("=" * 80)
    log("Experiment 1 completed successfully.")
    log("=" * 80)
    log(f"Metrics: {TABLE_DIR / 'baseline_metrics.csv'}")
    log(f"Summary: {OUTPUT_DIR / 'baseline_summary.md'}")
    log(f"README: {README_DIR / 'README_Experiment_1_Internal_Baselines.md'}")
    log(f"Total runtime: {runtime:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Experiment 1 failed. See error log:", flush=True)
        print(LOG_DIR / "experiment_1_error_log.txt", flush=True)
        save_error(e)
        raise
