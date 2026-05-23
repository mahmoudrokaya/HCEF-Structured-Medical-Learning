"""
Project Folder Structure Initializer
Paper: Paper 3 IJOCTA
Purpose:
    - Create and organize the complete project structure
    - Keep all experiment codes inside:
        D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Experiments/Code
    - Save all experiment outputs inside:
        D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Experiments
    - Store datasets inside:
        D:/47/471/New Papers/Paper 3 IJOCTA/Sub/Data

Author: Mahmoud Rokaya
"""

import os
from pathlib import Path

# ============================================================
# ROOT DIRECTORY
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

# ============================================================
# MAIN FOLDERS
# ============================================================

folders = [

    # Main folders
    ROOT_DIR / "Data",
    ROOT_DIR / "Experiments",

    # Experiment code folder
    ROOT_DIR / "Experiments" / "Code",

    # Reproducibility folders
    ROOT_DIR / "Experiments" / "Logs",
    ROOT_DIR / "Experiments" / "Configs",
    ROOT_DIR / "Experiments" / "Checkpoints",
    ROOT_DIR / "Experiments" / "Saved_Models",

    # Results folders
    ROOT_DIR / "Experiments" / "Results",
    ROOT_DIR / "Experiments" / "Results" / "Tables",
    ROOT_DIR / "Experiments" / "Results" / "Figures",
    ROOT_DIR / "Experiments" / "Results" / "Metrics",
    ROOT_DIR / "Experiments" / "Results" / "Embeddings",
    ROOT_DIR / "Experiments" / "Results" / "ROC_Curves",
    ROOT_DIR / "Experiments" / "Results" / "Confusion_Matrices",
    ROOT_DIR / "Experiments" / "Results" / "PCA",
    ROOT_DIR / "Experiments" / "Results" / "tSNE",
    ROOT_DIR / "Experiments" / "Results" / "Ablation_Study",
    ROOT_DIR / "Experiments" / "Results" / "Generalization",
    ROOT_DIR / "Experiments" / "Results" / "Augmentation",

    # Dataset folders
    ROOT_DIR / "Data" / "Original",
    ROOT_DIR / "Data" / "Augmented",
    ROOT_DIR / "Data" / "External_Datasets",
    ROOT_DIR / "Data" / "Processed",
    ROOT_DIR / "Data" / "Splits",

    # External datasets
    ROOT_DIR / "Data" / "External_Datasets" / "Heart_Disease",
    ROOT_DIR / "Data" / "External_Datasets" / "Pima_Diabetes",
    ROOT_DIR / "Data" / "External_Datasets" / "Breast_Cancer",

    # Documentation folders
    ROOT_DIR / "Documentation",
    ROOT_DIR / "Documentation" / "README",
    ROOT_DIR / "Documentation" / "Code_Manual",
    ROOT_DIR / "Documentation" / "PeerJ_Submission",

    # Supplementary materials
    ROOT_DIR / "Supplementary",
    ROOT_DIR / "Supplementary" / "Code",
    ROOT_DIR / "Supplementary" / "Figures",
    ROOT_DIR / "Supplementary" / "Tables",
    ROOT_DIR / "Supplementary" / "Datasets",
]

# ============================================================
# CREATE FOLDERS
# ============================================================

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Project folder structure created successfully.")
print("=" * 70)

# ============================================================
# DISPLAY TREE STRUCTURE
# ============================================================

def print_tree(start_path, prefix=""):
    items = sorted(start_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(item, prefix + extension)

print("\nGenerated Folder Structure:\n")
print(ROOT_DIR.name)
print_tree(ROOT_DIR)

# ============================================================
# OPTIONAL: CREATE README FILES
# ============================================================

readme_files = {

    ROOT_DIR / "Data" / "README.txt":
        "This folder contains original, processed, augmented, and external datasets.",

    ROOT_DIR / "Experiments" / "README.txt":
        "This folder contains experiment outputs, logs, models, and evaluation results.",

    ROOT_DIR / "Experiments" / "Code" / "README.txt":
        "Place all experiment and training scripts here.",

    ROOT_DIR / "Documentation" / "README" / "README.md":
        "# Project Documentation\n\nThis folder contains README and reproducibility documents.",

    ROOT_DIR / "Supplementary" / "README.txt":
        "Supplementary materials prepared for PeerJ submission.",
}

for file_path, content in readme_files.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("\nREADME files generated successfully.")

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nProject initialization completed successfully.")