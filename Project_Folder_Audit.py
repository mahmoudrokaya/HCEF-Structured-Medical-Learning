"""
Project_Folder_Audit.py

Purpose:
    Generate a complete folder and file tree for the Paper 3 IJOCTA project.

This helps:
    - verify preprocessing outputs
    - verify experiment structure
    - verify reproducibility organization
    - verify dataset preparation pipeline
    - document the current project state

Author:
    Mahmoud Rokaya
"""

from pathlib import Path
from datetime import datetime

# ============================================================
# ROOT DIRECTORY
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = ROOT_DIR / "Experiments" / "Results" / "Project_Audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TREE_TEXT_FILE = OUTPUT_DIR / "project_folder_tree.txt"
TREE_MARKDOWN_FILE = OUTPUT_DIR / "PROJECT_FOLDER_TREE.md"

# ============================================================
# SETTINGS
# ============================================================

IGNORE_FOLDERS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode"
}

# ============================================================
# TREE GENERATION
# ============================================================

tree_lines = []

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"

    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"

    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"

    return f"{size_bytes / (1024 ** 3):.2f} GB"


def generate_tree(path, prefix=""):

    try:
        items = sorted(
            [p for p in path.iterdir()
             if p.name not in IGNORE_FOLDERS],
            key=lambda x: (x.is_file(), x.name.lower())
        )

    except PermissionError:
        return

    for i, item in enumerate(items):

        connector = "└── " if i == len(items) - 1 else "├── "

        if item.is_dir():

            line = f"{prefix}{connector}[DIR] {item.name}"
            tree_lines.append(line)

            extension = "    " if i == len(items) - 1 else "│   "

            generate_tree(item, prefix + extension)

        else:

            try:
                size = format_size(item.stat().st_size)
            except:
                size = "Unknown"

            line = f"{prefix}{connector}{item.name} ({size})"
            tree_lines.append(line)


# ============================================================
# GENERATE TREE
# ============================================================

header = [
    "=" * 80,
    "PROJECT FOLDER STRUCTURE AUDIT",
    "=" * 80,
    f"Generated: {datetime.now()}",
    f"Root: {ROOT_DIR}",
    "=" * 80,
    ""
]

tree_lines.extend(header)

tree_lines.append(ROOT_DIR.name)

generate_tree(ROOT_DIR)

# ============================================================
# SAVE TXT
# ============================================================

with open(TREE_TEXT_FILE, "w", encoding="utf-8") as f:
    for line in tree_lines:
        f.write(line + "\n")

# ============================================================
# SAVE MARKDOWN
# ============================================================

with open(TREE_MARKDOWN_FILE, "w", encoding="utf-8") as f:

    f.write("# Project Folder Structure Audit\n\n")

    f.write(f"Generated: `{datetime.now()}`\n\n")
    f.write(f"Root Directory:\n\n")
    f.write(f"`{ROOT_DIR}`\n\n")

    f.write("```text\n")

    for line in tree_lines:
        f.write(line + "\n")

    f.write("```\n")

# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("=" * 80)
print("Project folder audit completed successfully.")
print("=" * 80)

print(f"\nTXT tree saved to:\n{TREE_TEXT_FILE}")
print(f"\nMarkdown tree saved to:\n{TREE_MARKDOWN_FILE}")

print("\nPreview:\n")

for line in tree_lines[:80]:
    print(line)