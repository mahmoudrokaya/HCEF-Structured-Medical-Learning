"""
Experiment_Design_Map.py

Purpose:
    Create a formal experiment design map for the revised Paper 3 IJOCTA pipeline.

Why this script is needed:
    - Ensures all experiments are aligned with the manuscript methods.
    - Prevents mismatch between code, datasets, and reviewer requirements.
    - Documents the exact purpose, dataset, target, method component, and outputs for each experiment.

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
from datetime import datetime
import json
import pandas as pd


# ============================================================
# PATH SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

CODE_DIR = ROOT_DIR / "Experiments" / "Code"
RESULTS_DIR = ROOT_DIR / "Experiments" / "Results"

MODEL_READY_DIR = ROOT_DIR / "Data" / "Processed" / "Modeling_Ready"
SPLITS_DIR = ROOT_DIR / "Data" / "Splits" / "Modeling_Ready"

OUTPUT_DIR = RESULTS_DIR / "Experiment_Design_Map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CANONICAL MODELING DATASETS
# ============================================================

DATASETS = {
    "internal_augmented_supervised": {
        "modeling_ready_file": MODEL_READY_DIR / "internal_augmented_supervised_modeling_ready.csv",
        "split_folder": SPLITS_DIR / "internal_augmented_supervised",
        "target": "target",
        "role": "main internal dataset",
        "shape_from_last_run": "6000 rows x 52 columns",
        "notes": "Nationality and Gender were removed because they became fully missing."
    },
    "external_breast_cancer": {
        "modeling_ready_file": MODEL_READY_DIR / "external_breast_cancer_modeling_ready.csv",
        "split_folder": SPLITS_DIR / "external_breast_cancer",
        "target": "diagnosis",
        "role": "external validation dataset",
        "shape_from_last_run": "569 rows x 31 columns",
        "notes": "Used only for external generalization and reproducibility validation."
    },
    "external_heart_disease": {
        "modeling_ready_file": MODEL_READY_DIR / "external_heart_disease_modeling_ready.csv",
        "split_folder": SPLITS_DIR / "external_heart_disease",
        "target": "num",
        "role": "external validation dataset",
        "shape_from_last_run": "920 rows x 28 columns",
        "notes": "Used only for external generalization and reproducibility validation."
    }
}


# ============================================================
# EXPERIMENT DESIGN MAP
# ============================================================

EXPERIMENTS = [
    {
        "experiment_id": "Experiment_1",
        "experiment_name": "Internal Baseline Training and Learning Dynamics",
        "primary_goal": (
            "Establish transparent baseline performance on the canonical internal "
            "tabular dataset before introducing the proposed HCE-F model."
        ),
        "dataset": "internal_augmented_supervised",
        "input_file": str(DATASETS["internal_augmented_supervised"]["modeling_ready_file"]),
        "split_folder": str(DATASETS["internal_augmented_supervised"]["split_folder"]),
        "target_column": DATASETS["internal_augmented_supervised"]["target"],
        "method_alignment": (
            "Corresponds to the evaluation methodology required in Methods. "
            "Uses tabular inputs only and evaluates standard machine-learning classifiers "
            "before applying tensorized residual encoding, contrastive embedding, and ensemble fusion."
        ),
        "models_or_components": [
            "Logistic Regression",
            "Support Vector Machine",
            "Random Forest",
            "Gradient Boosting",
            "Extra Trees",
            "k-Nearest Neighbors",
            "Multi-Layer Perceptron"
        ],
        "metrics": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "PR-AUC",
            "Confusion matrix",
            "Learning curve or validation behavior where applicable"
        ],
        "required_outputs": [
            "baseline_metrics.csv",
            "baseline_metrics.json",
            "baseline_confusion_matrices",
            "baseline_roc_curves",
            "baseline_precision_recall_curves",
            "baseline_summary.md"
        ],
        "output_folder": str(RESULTS_DIR / "Experiment_1_Internal_Baselines"),
        "reviewer_requirement_addressed": (
            "Evaluation method in Methods, baseline training, learning dynamics, "
            "reproducibility, and transparent benchmark comparison."
        )
    },

    {
        "experiment_id": "Experiment_2",
        "experiment_name": "Full HCE-F Model Evaluation",
        "primary_goal": (
            "Evaluate the complete proposed framework using tensorized residual encoding, "
            "supervised contrastive projection, and probabilistic ensemble fusion."
        ),
        "dataset": "internal_augmented_supervised",
        "input_file": str(DATASETS["internal_augmented_supervised"]["modeling_ready_file"]),
        "split_folder": str(DATASETS["internal_augmented_supervised"]["split_folder"]),
        "target_column": DATASETS["internal_augmented_supervised"]["target"],
        "method_alignment": (
            "Directly implements the paper method: tabular input -> tensorized residual encoder "
            "-> supervised contrastive projection head -> latent embeddings -> probabilistic ensemble fusion."
        ),
        "models_or_components": [
            "Tensorized residual tabular encoder",
            "Supervised contrastive projection head",
            "Latent embedding extraction",
            "Probabilistic ensemble fusion",
            "Weighted decision aggregation"
        ],
        "metrics": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "PR-AUC",
            "Training loss",
            "Validation loss",
            "Embedding separability",
            "Runtime",
            "Model size"
        ],
        "required_outputs": [
            "hcef_metrics.csv",
            "hcef_metrics.json",
            "training_loss_curve.png",
            "validation_loss_curve.png",
            "embedding_pca.png",
            "embedding_tsne.png",
            "hcef_confusion_matrix.png",
            "hcef_roc_curve.png",
            "hcef_precision_recall_curve.png",
            "ensemble_weights.csv",
            "hcef_summary.md"
        ],
        "output_folder": str(RESULTS_DIR / "Experiment_2_HCEF_Full_Model"),
        "reviewer_requirement_addressed": (
            "Algorithms and code implementation, complete reproducible AI Application workflow, "
            "and direct empirical evaluation of the proposed method."
        )
    },

    {
        "experiment_id": "Experiment_3",
        "experiment_name": "Ablation Study and Component Contribution",
        "primary_goal": (
            "Quantify the contribution of each HCE-F component by removing or replacing "
            "individual modules."
        ),
        "dataset": "internal_augmented_supervised",
        "input_file": str(DATASETS["internal_augmented_supervised"]["modeling_ready_file"]),
        "split_folder": str(DATASETS["internal_augmented_supervised"]["split_folder"]),
        "target_column": DATASETS["internal_augmented_supervised"]["target"],
        "method_alignment": (
            "Matches the Methods requirement that ablation and component contribution "
            "must be described and evaluated as part of the evaluation methodology."
        ),
        "models_or_components": [
            "Full HCE-F",
            "Without supervised contrastive loss",
            "Without tensorized residual encoder",
            "Without probabilistic ensemble fusion",
            "Single best base learner only",
            "Simple averaging ensemble instead of optimized weighted fusion"
        ],
        "metrics": [
            "Accuracy",
            "F1-score",
            "ROC-AUC",
            "PR-AUC",
            "Delta compared with full HCE-F",
            "Runtime",
            "Embedding separability"
        ],
        "required_outputs": [
            "ablation_metrics.csv",
            "ablation_delta_table.csv",
            "component_contribution_plot.png",
            "ablation_roc_comparison.png",
            "ablation_summary.md"
        ],
        "output_folder": str(RESULTS_DIR / "Experiment_3_Ablation_Study"),
        "reviewer_requirement_addressed": (
            "Ablation study and component contribution required by the technical revision request."
        )
    },

    {
        "experiment_id": "Experiment_4",
        "experiment_name": "Robustness Under Controlled Noise and Perturbation",
        "primary_goal": (
            "Assess robustness of the full HCE-F model and selected baselines under "
            "controlled tabular feature perturbations."
        ),
        "dataset": "internal_augmented_supervised",
        "input_file": str(DATASETS["internal_augmented_supervised"]["modeling_ready_file"]),
        "split_folder": str(DATASETS["internal_augmented_supervised"]["split_folder"]),
        "target_column": DATASETS["internal_augmented_supervised"]["target"],
        "method_alignment": (
            "Implements controlled Gaussian perturbation and noise robustness evaluation "
            "as described in the methods-oriented implementation framework."
        ),
        "models_or_components": [
            "Full HCE-F under clean data",
            "Full HCE-F under low noise",
            "Full HCE-F under moderate noise",
            "Full HCE-F under high noise",
            "Best baseline under matching perturbation levels"
        ],
        "metrics": [
            "Accuracy",
            "F1-score",
            "ROC-AUC",
            "PR-AUC",
            "Performance degradation percentage",
            "Robustness index"
        ],
        "required_outputs": [
            "noise_robustness_metrics.csv",
            "noise_degradation_table.csv",
            "noise_robustness_curve.png",
            "baseline_vs_hcef_noise_comparison.png",
            "robustness_summary.md"
        ],
        "output_folder": str(RESULTS_DIR / "Experiment_4_Noise_Robustness"),
        "reviewer_requirement_addressed": (
            "Supports claims of stability and robustness under noisy tabular conditions."
        )
    },

    {
        "experiment_id": "Experiment_5",
        "experiment_name": "External Dataset Generalization",
        "primary_goal": (
            "Evaluate whether the proposed tabular framework remains reproducible and "
            "competitive across independent public biomedical tabular datasets."
        ),
        "dataset": "external_breast_cancer + external_heart_disease",
        "input_file": (
            str(DATASETS["external_breast_cancer"]["modeling_ready_file"])
            + " | "
            + str(DATASETS["external_heart_disease"]["modeling_ready_file"])
        ),
        "split_folder": (
            str(DATASETS["external_breast_cancer"]["split_folder"])
            + " | "
            + str(DATASETS["external_heart_disease"]["split_folder"])
        ),
        "target_column": "diagnosis | num",
        "method_alignment": (
            "Uses the same tabular preprocessing and evaluation pipeline on external datasets "
            "without introducing any image-based or unrelated workflow."
        ),
        "models_or_components": [
            "Best baseline model from Experiment 1",
            "Full HCE-F trained and evaluated separately on each external dataset",
            "Comparable metric reporting across datasets"
        ],
        "metrics": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC where applicable",
            "PR-AUC where applicable",
            "Runtime",
            "Model size"
        ],
        "required_outputs": [
            "external_generalization_metrics.csv",
            "external_dataset_comparison_table.csv",
            "external_roc_curves.png",
            "external_precision_recall_curves.png",
            "external_generalization_summary.md"
        ],
        "output_folder": str(RESULTS_DIR / "Experiment_5_External_Generalization"),
        "reviewer_requirement_addressed": (
            "Third-party dataset reproducibility, public dataset validation, and improved "
            "generalization evidence."
        )
    }
]


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_paths():
    validation_records = []

    for dataset_name, info in DATASETS.items():
        file_path = Path(info["modeling_ready_file"])
        split_folder = Path(info["split_folder"])

        validation_records.append({
            "dataset": dataset_name,
            "modeling_ready_file": str(file_path),
            "modeling_ready_file_exists": file_path.exists(),
            "split_folder": str(split_folder),
            "split_folder_exists": split_folder.exists(),
            "target": info["target"],
            "role": info["role"],
            "notes": info["notes"]
        })

    return validation_records


def create_output_folders():
    for exp in EXPERIMENTS:
        out = Path(exp["output_folder"])
        out.mkdir(parents=True, exist_ok=True)

        for sub in [
            "Tables",
            "Figures",
            "Metrics",
            "Models",
            "Logs",
            "Embeddings",
            "Confusion_Matrices",
            "ROC_Curves",
            "Precision_Recall_Curves"
        ]:
            (out / sub).mkdir(parents=True, exist_ok=True)


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports():
    create_output_folders()

    validation_records = validate_paths()

    experiments_df = pd.DataFrame(EXPERIMENTS)
    validation_df = pd.DataFrame(validation_records)

    csv_path = OUTPUT_DIR / "experiment_design_map.csv"
    json_path = OUTPUT_DIR / "experiment_design_map.json"
    validation_csv_path = OUTPUT_DIR / "dataset_path_validation.csv"
    md_path = OUTPUT_DIR / "EXPERIMENT_DESIGN_MAP.md"

    experiments_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    validation_df.to_csv(validation_csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": str(datetime.now()),
                "root_dir": str(ROOT_DIR),
                "methodological_constraints": {
                    "data_type": "tabular only",
                    "no_image_pipeline": True,
                    "no_resnet_workflow": True,
                    "preprocessing": "deterministic",
                    "split_strategy": "stratified 70:15:15",
                    "main_internal_dataset": "internal_augmented_supervised_modeling_ready.csv"
                },
                "datasets": {
                    k: {
                        kk: str(vv) if isinstance(vv, Path) else vv
                        for kk, vv in val.items()
                    }
                    for k, val in DATASETS.items()
                },
                "experiments": EXPERIMENTS,
                "path_validation": validation_records
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Design Map\n\n")
        f.write(f"Generated at: `{datetime.now()}`\n\n")
        f.write(f"Root directory: `{ROOT_DIR}`\n\n")

        f.write("## Methodological Constraints\n\n")
        f.write("- Tabular data only.\n")
        f.write("- No image pipeline.\n")
        f.write("- No ResNet workflow.\n")
        f.write("- Deterministic preprocessing.\n")
        f.write("- Stratified 70:15:15 train/validation/test splits.\n")
        f.write("- Experiments must align directly with the paper methods.\n\n")

        f.write("## Dataset Path Validation\n\n")
        for item in validation_records:
            f.write(f"### {item['dataset']}\n")
            f.write(f"- Role: {item['role']}\n")
            f.write(f"- Target: `{item['target']}`\n")
            f.write(f"- Modeling-ready file exists: `{item['modeling_ready_file_exists']}`\n")
            f.write(f"- Split folder exists: `{item['split_folder_exists']}`\n")
            f.write(f"- File: `{item['modeling_ready_file']}`\n")
            f.write(f"- Split folder: `{item['split_folder']}`\n")
            f.write(f"- Notes: {item['notes']}\n\n")

        f.write("## Experiments\n\n")

        for exp in EXPERIMENTS:
            f.write(f"## {exp['experiment_id']}: {exp['experiment_name']}\n\n")
            f.write(f"**Primary goal:** {exp['primary_goal']}\n\n")
            f.write(f"**Dataset:** `{exp['dataset']}`\n\n")
            f.write(f"**Target column:** `{exp['target_column']}`\n\n")
            f.write(f"**Input file:** `{exp['input_file']}`\n\n")
            f.write(f"**Split folder:** `{exp['split_folder']}`\n\n")
            f.write(f"**Method alignment:** {exp['method_alignment']}\n\n")

            f.write("**Models/components:**\n\n")
            for item in exp["models_or_components"]:
                f.write(f"- {item}\n")

            f.write("\n**Metrics:**\n\n")
            for item in exp["metrics"]:
                f.write(f"- {item}\n")

            f.write("\n**Required outputs:**\n\n")
            for item in exp["required_outputs"]:
                f.write(f"- {item}\n")

            f.write(f"\n**Output folder:** `{exp['output_folder']}`\n\n")

            f.write(
                f"**Reviewer requirement addressed:** "
                f"{exp['reviewer_requirement_addressed']}\n\n"
            )

            f.write("---\n\n")

    print("=" * 80)
    print("Experiment design map generated successfully.")
    print("=" * 80)
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    print(f"Dataset validation saved to: {validation_csv_path}")
    print(f"Markdown saved to: {md_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    save_reports()