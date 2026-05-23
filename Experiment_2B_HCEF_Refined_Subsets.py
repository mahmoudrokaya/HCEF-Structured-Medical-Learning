"""
Experiment_2B_HCEF_Refined_Subsets.py

Experiment 2B:
    HCE-F Evaluation on Refined Feature Subsets

Purpose:
    Re-run the full Hybrid Contrastive-Ensemble Framework (HCE-F) on all refined
    feature subsets produced after the deep dataset audit.

Why this experiment is needed:
    Experiment 1 and Experiment 2 showed near-random performance on the original
    full feature set. Experiment 1B tested conventional baselines on refined
    feature subsets. Experiment 2B tests whether the proposed HCE-F framework
    benefits from feature-space refinement.

Methodological alignment:
    - Tabular data only.
    - Same HCE-F method as Experiment 2:
        tabular input -> tensorized residual encoder -> supervised contrastive projection
        -> latent embeddings -> probabilistic ensemble fusion.
    - Same refined datasets and deterministic splits used in Experiment 1B.
    - No image pipeline.
    - No ResNet workflow.
    - Generates README, Methods text, limitations note, metrics, figures, logs,
      and reproducibility outputs.

Journal/reviewer alignment:
    - Algorithms and code provided.
    - Evaluation method documented.
    - Component learning dynamics reported.
    - Dataset refinement and reproducibility documented.
    - Results are saved in machine-readable tables and manuscript-ready summaries.

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
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
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
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# PYTORCH IMPORT
# ============================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise ImportError(
        "PyTorch is required for Experiment 2B. Install it first:\n"
        "pip install torch\n\n"
        f"Original import error: {e}"
    )


# ============================================================
# PATH SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

TARGET_COLUMN = "target"
RANDOM_STATE = 42

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

EXPERIMENT_1B_METRICS = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_1B_Refined_Baselines"
    / "Tables"
    / "refined_baseline_metrics.csv"
)

EXPERIMENT_2_METRICS = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_2_HCEF_Full_Model"
    / "Tables"
    / "hcef_metrics.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_2B_HCEF_Refined_Subsets"
)

TABLE_DIR = OUTPUT_DIR / "Tables"
FIGURE_DIR = OUTPUT_DIR / "Figures"
METRIC_DIR = OUTPUT_DIR / "Metrics"
MODEL_DIR = OUTPUT_DIR / "Models"
LOG_DIR = OUTPUT_DIR / "Logs"
EMBED_DIR = OUTPUT_DIR / "Embeddings"
CM_DIR = OUTPUT_DIR / "Confusion_Matrices"
ROC_DIR = OUTPUT_DIR / "ROC_Curves"
PR_DIR = OUTPUT_DIR / "Precision_Recall_Curves"
README_DIR = OUTPUT_DIR / "Reproducibility"

for folder in [
    OUTPUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    MODEL_DIR,
    LOG_DIR,
    EMBED_DIR,
    CM_DIR,
    ROC_DIR,
    PR_DIR,
    README_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# REFINED DATASETS
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
# MODEL SETTINGS
# ============================================================

BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
EMBED_DIM = 32
TENSOR_RANK = 16
CONTRASTIVE_TEMPERATURE = 0.20
CONTRASTIVE_WEIGHT = 0.25
DROPOUT = 0.15
N_JOBS = 1   # keep single-process for Windows stability

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)
    with open(LOG_DIR / "experiment_2b_console_log.txt", "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def save_error(error):
    with open(LOG_DIR / "experiment_2b_error_log.txt", "w", encoding="utf-8") as f:
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
            f"Missing refined split files for {dataset_name}. Expected train.csv, validation.csv, test.csv in {split_dir}"
        )

    return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)


def split_xy(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


# ============================================================
# DATASET CLASS
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ============================================================
# HCE-F NETWORK
# ============================================================

class TensorizedResidualBlock(nn.Module):
    """
    Compact tensorized residual block for tabular data.

    Implements low-rank multiplicative feature interaction with residual identity
    retention, consistent with the HCE-F method.
    """

    def __init__(self, dim, rank=16, dropout=0.15, alpha=0.60):
        super().__init__()
        self.alpha = alpha

        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.factor_u = nn.Linear(dim, rank)
        self.factor_v = nn.Linear(dim, rank)
        self.tensor_project = nn.Linear(rank, dim)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        linear_term = self.linear(x)
        interaction = torch.tanh(self.factor_u(x) * self.factor_v(x))
        tensor_term = self.tensor_project(interaction)
        out = self.alpha * x + (1.0 - self.alpha) * (linear_term + tensor_term)
        out = self.norm(out)
        return F.relu(out)


class HCEFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embed_dim=32, n_classes=2, rank=16, dropout=0.15):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.block1 = TensorizedResidualBlock(hidden_dim, rank=rank, dropout=dropout, alpha=0.60)
        self.block2 = TensorizedResidualBlock(hidden_dim, rank=rank, dropout=dropout, alpha=0.60)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.classifier_head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_embedding=False):
        h = self.input_layer(x)
        h = self.block1(h)
        h = self.block2(h)

        z = self.projection_head(h)
        z_norm = F.normalize(z, p=2, dim=1)

        logits = self.classifier_head(z_norm)

        if return_embedding:
            return logits, z_norm

        return logits


# ============================================================
# LOSS
# ============================================================

def supervised_contrastive_loss(embeddings, labels, temperature=0.20):
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)

    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
    logits = similarity - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=embeddings.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    positive_count = mask.sum(1)
    valid = positive_count > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()


# ============================================================
# METRICS AND PLOTS
# ============================================================

def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_").replace("+", "plus")


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


def plot_confusion_matrix(cm, labels, dataset_name, model_name, split_name):
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


def plot_roc(y_true, y_score, dataset_name, model_name, split_name):
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


def plot_pr(y_true, y_score, dataset_name, model_name, split_name):
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


def plot_training_history(history, dataset_name):
    df = pd.DataFrame(history)
    name = safe_name(dataset_name)

    df.to_csv(TABLE_DIR / f"{name}_hcef_training_history.csv", index=False, encoding="utf-8-sig")

    for key in ["train_total_loss", "train_ce_loss", "train_contrastive_loss", "validation_total_loss"]:
        if key not in df.columns:
            continue

        plt.figure(figsize=(7, 6))
        plt.plot(df["epoch"], df[key], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f"{dataset_name}: {key.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{name}_{key}.png", dpi=300)
        plt.close("all")

    plt.figure(figsize=(7, 6))
    plotted = False

    if "validation_f1_weighted" in df.columns:
        plt.plot(df["epoch"], df["validation_f1_weighted"], marker="o", label="Validation F1-weighted")
        plotted = True

    if "validation_roc_auc" in df.columns:
        plt.plot(df["epoch"], df["validation_roc_auc"], marker="o", label="Validation ROC-AUC")
        plotted = True

    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dataset_name}: HCE-F Validation Learning Dynamics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{name}_validation_learning_dynamics.png", dpi=300)

    plt.close("all")


def plot_embedding_pca(embeddings, labels, dataset_name, split_name):
    name = f"{safe_name(dataset_name)}_{split_name}"

    emb_df = pd.DataFrame(embeddings)
    emb_df["label"] = labels
    emb_df.to_csv(EMBED_DIR / f"{name}_embeddings.csv", index=False, encoding="utf-8-sig")

    try:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(embeddings)

        plt.figure(figsize=(7, 6))
        for cls in sorted(np.unique(labels)):
            idx = labels == cls
            plt.scatter(coords[idx, 0], coords[idx, 1], label=f"Class {cls}", alpha=0.7)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{dataset_name} {split_name} HCE-F Embeddings - PCA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EMBED_DIR / f"{name}_embedding_pca.png", dpi=300)
        plt.close("all")
    except Exception as e:
        plt.close("all")
        log(f"PCA embedding plot skipped for {dataset_name} {split_name}: {e}")


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def evaluate_network(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_y = []
    all_prob = []
    all_pred = []
    all_emb = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits, emb = model(xb, return_embedding=True)
            loss = criterion(logits, yb)

            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred = np.argmax(prob, axis=1)

            total_loss += loss.item() * xb.size(0)
            all_y.extend(yb.cpu().numpy())
            all_prob.append(prob)
            all_pred.extend(pred)
            all_emb.append(emb.cpu().numpy())

    y_true = np.array(all_y)
    y_pred = np.array(all_pred)
    y_prob = np.vstack(all_prob)
    embeddings = np.vstack(all_emb)

    return total_loss / len(y_true), y_true, y_pred, y_prob, embeddings


def train_hcef_network(dataset_name, X_train, y_train, X_val, y_val, input_dim, n_classes):
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = HCEFNetwork(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        n_classes=n_classes,
        rank=TENSOR_RANK,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_f1 = -np.inf
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        running_total = 0.0
        running_ce = 0.0
        running_contrastive = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits, emb = model(xb, return_embedding=True)
            ce_loss = criterion(logits, yb)
            con_loss = supervised_contrastive_loss(emb, yb, temperature=CONTRASTIVE_TEMPERATURE)
            total_loss = ce_loss + CONTRASTIVE_WEIGHT * con_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_n = xb.size(0)
            running_total += total_loss.item() * batch_n
            running_ce += ce_loss.item() * batch_n
            running_contrastive += con_loss.item() * batch_n
            n_seen += batch_n

        val_loss, y_val_true, y_val_pred, y_val_prob, _ = evaluate_network(model, val_loader, criterion)
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_prob, dataset_name, "HCEF_Neural_Model", "validation")

        record = {
            "dataset": dataset_name,
            "epoch": epoch,
            "train_total_loss": running_total / n_seen,
            "train_ce_loss": running_ce / n_seen,
            "train_contrastive_loss": running_contrastive / n_seen,
            "validation_total_loss": val_loss,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_f1_weighted": val_metrics["f1_weighted"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_pr_auc": val_metrics["pr_auc"],
        }

        history.append(record)

        log(
            f"{dataset_name} | Epoch {epoch:03d} | "
            f"loss={record['train_total_loss']:.4f} | "
            f"ce={record['train_ce_loss']:.4f} | "
            f"contrastive={record['train_contrastive_loss']:.4f} | "
            f"val_f1={record['validation_f1_weighted']:.4f} | "
            f"val_auc={record['validation_roc_auc']:.4f}"
        )

        if record["validation_f1_weighted"] > best_val_f1:
            best_val_f1 = record["validation_f1_weighted"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            log(f"{dataset_name}: Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_DIR / f"{safe_name(dataset_name)}_hcef_model.pt")

    return model, history, best_epoch


def extract_outputs(model, X, y):
    loader = DataLoader(TabularDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    return evaluate_network(model, loader, criterion)


# ============================================================
# EMBEDDING ENSEMBLE
# ============================================================

def get_embedding_ensemble_models():
    return {
        "Embedding_Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        "Embedding_SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        "Embedding_Random_Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=N_JOBS,
        ),
        "Embedding_Extra_Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=N_JOBS,
        ),
        "Embedding_Gradient_Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Embedding_KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Embedding_MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ))
        ]),
    }


def compute_validation_weights(model_names, val_probs, y_val):
    losses = []
    labels = sorted(np.unique(y_val))

    for name in model_names:
        try:
            losses.append(log_loss(y_val, val_probs[name], labels=labels))
        except Exception:
            losses.append(10.0)

    losses = np.array(losses, dtype=float)
    scores = -losses
    scores = scores - scores.max()
    weights = np.exp(scores) / np.exp(scores).sum()

    return dict(zip(model_names, weights)), dict(zip(model_names, losses))


def run_embedding_ensemble(dataset_name, train_emb, y_train, val_emb, y_val, test_emb, y_test):
    models = get_embedding_ensemble_models()

    val_probs = {}
    test_probs = {}
    records = []

    for model_name, model in models.items():
        log(f"{dataset_name}: training embedding learner {model_name}")

        start = time.time()
        model.fit(train_emb, y_train)
        train_time = time.time() - start

        val_prob = model.predict_proba(val_emb)
        test_prob = model.predict_proba(test_emb)

        val_pred = np.argmax(val_prob, axis=1)
        test_pred = np.argmax(test_prob, axis=1)

        val_record = compute_metrics(y_val, val_pred, val_prob, dataset_name, model_name, "validation")
        test_record = compute_metrics(y_test, test_pred, test_prob, dataset_name, model_name, "test")

        val_record["training_time_seconds"] = float(train_time)
        test_record["training_time_seconds"] = float(train_time)

        records.append(val_record)
        records.append(test_record)

        val_probs[model_name] = val_prob
        test_probs[model_name] = test_prob

        with open(MODEL_DIR / f"{safe_name(dataset_name)}_{safe_name(model_name)}.pkl", "wb") as f:
            pickle.dump(model, f)

    model_names = list(val_probs.keys())
    weights, validation_losses = compute_validation_weights(model_names, val_probs, y_val)

    weighted_val_prob = np.zeros_like(next(iter(val_probs.values())))
    weighted_test_prob = np.zeros_like(next(iter(test_probs.values())))
    average_val_prob = np.mean(list(val_probs.values()), axis=0)
    average_test_prob = np.mean(list(test_probs.values()), axis=0)

    for name in model_names:
        weighted_val_prob += weights[name] * val_probs[name]
        weighted_test_prob += weights[name] * test_probs[name]

    weights_df = pd.DataFrame({
        "dataset": dataset_name,
        "base_learner": model_names,
        "validation_log_loss": [validation_losses[name] for name in model_names],
        "fusion_weight": [weights[name] for name in model_names],
    })

    weights_df.to_csv(TABLE_DIR / f"{safe_name(dataset_name)}_ensemble_weights.csv", index=False, encoding="utf-8-sig")

    fusion_outputs = [
        ("HCEF_Weighted_Ensemble_Fusion", weighted_val_prob, weighted_test_prob),
        ("HCEF_Average_Ensemble_Fusion", average_val_prob, average_test_prob),
    ]

    for fusion_name, val_prob, test_prob in fusion_outputs:
        val_pred = np.argmax(val_prob, axis=1)
        test_pred = np.argmax(test_prob, axis=1)

        val_record = compute_metrics(y_val, val_pred, val_prob, dataset_name, fusion_name, "validation")
        test_record = compute_metrics(y_test, test_pred, test_prob, dataset_name, fusion_name, "test")

        records.append(val_record)
        records.append(test_record)

        cm = confusion_matrix(y_test, test_pred, labels=sorted(np.unique(y_test)))
        pd.DataFrame(
            cm,
            index=[f"true_{x}" for x in sorted(np.unique(y_test))],
            columns=[f"pred_{x}" for x in sorted(np.unique(y_test))],
        ).to_csv(TABLE_DIR / f"{safe_name(dataset_name)}_{fusion_name}_test_confusion_matrix.csv", encoding="utf-8-sig")

        plot_confusion_matrix(cm, sorted(np.unique(y_test)), dataset_name, fusion_name, "test")
        plot_roc(y_test, test_prob, dataset_name, fusion_name, "test")
        plot_pr(y_test, test_prob, dataset_name, fusion_name, "test")

    return records, weights_df


# ============================================================
# COMPARISON TABLES
# ============================================================

def compare_with_1b_and_2(best_per_dataset):
    rows = []

    if EXPERIMENT_1B_METRICS.exists():
        exp1b = pd.read_csv(EXPERIMENT_1B_METRICS)
        exp1b_test = exp1b[exp1b["split"] == "test"].copy()

        for dataset in REFINED_DATASETS:
            part = exp1b_test[exp1b_test["dataset"] == dataset]
            if not part.empty:
                best = part.sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False).iloc[0]
                rows.append({
                    "source": "Experiment_1B_Refined_Baseline",
                    "dataset": dataset,
                    "model": best["model"],
                    "accuracy": best["accuracy"],
                    "f1_weighted": best["f1_weighted"],
                    "roc_auc": best["roc_auc"],
                    "pr_auc": best["pr_auc"],
                })

    for _, row in best_per_dataset.iterrows():
        rows.append({
            "source": "Experiment_2B_HCEF_Refined",
            "dataset": row["dataset"],
            "model": row["model"],
            "accuracy": row["accuracy"],
            "f1_weighted": row["f1_weighted"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
        })

    comparison = pd.DataFrame(rows)

    if not comparison.empty:
        comparison.to_csv(TABLE_DIR / "experiment_1b_vs_2b_comparison.csv", index=False, encoding="utf-8-sig")

    # pairwise delta HCE-F vs best baseline per dataset
    delta_rows = []

    for dataset in REFINED_DATASETS:
        base = comparison[
            (comparison["source"] == "Experiment_1B_Refined_Baseline")
            & (comparison["dataset"] == dataset)
        ]

        hcef = comparison[
            (comparison["source"] == "Experiment_2B_HCEF_Refined")
            & (comparison["dataset"] == dataset)
        ]

        if not base.empty and not hcef.empty:
            b = base.iloc[0]
            h = hcef.iloc[0]

            delta_rows.append({
                "dataset": dataset,
                "best_1b_model": b["model"],
                "best_2b_model": h["model"],
                "baseline_f1": b["f1_weighted"],
                "hcef_f1": h["f1_weighted"],
                "delta_f1_hcef_minus_baseline": h["f1_weighted"] - b["f1_weighted"],
                "baseline_auc": b["roc_auc"],
                "hcef_auc": h["roc_auc"],
                "delta_auc_hcef_minus_baseline": h["roc_auc"] - b["roc_auc"],
                "baseline_pr_auc": b["pr_auc"],
                "hcef_pr_auc": h["pr_auc"],
                "delta_pr_auc_hcef_minus_baseline": h["pr_auc"] - b["pr_auc"],
            })

    delta_df = pd.DataFrame(delta_rows)

    if not delta_df.empty:
        delta_df.to_csv(TABLE_DIR / "hcef_refined_vs_baseline_delta_table.csv", index=False, encoding="utf-8-sig")

    return comparison, delta_df


def plot_best_hcef_by_dataset(best_per_dataset):
    if best_per_dataset.empty:
        return

    plot_df = best_per_dataset.sort_values(by="f1_weighted", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["dataset"], plot_df["f1_weighted"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best HCE-F Test F1-weighted")
    plt.title("Best HCE-F Performance Across Refined Feature Subsets")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_hcef_f1_by_refined_dataset.png", dpi=300)
    plt.close("all")

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["dataset"], plot_df["roc_auc"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best HCE-F Test ROC-AUC")
    plt.title("Best HCE-F ROC-AUC Across Refined Feature Subsets")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_hcef_roc_auc_by_refined_dataset.png", dpi=300)
    plt.close("all")


# ============================================================
# DOCUMENTATION
# ============================================================

def write_reproducibility_files(dataset_summaries, best_per_dataset, comparison_df, delta_df):
    readme = f"""# Experiment 2B: HCE-F Evaluation on Refined Feature Subsets

## Title
Paper 3 IJOCTA Revision - HCE-F Refined Feature-Subset Evaluation

## Description
This experiment evaluates the full Hybrid Contrastive-Ensemble Framework on the refined feature subsets generated after deep dataset auditing.

The HCE-F workflow is:

Tabular input -> tensorized residual encoder -> supervised contrastive projection head -> latent embeddings -> probabilistic ensemble fusion.

No image simulation, image preprocessing, ResNet feature extraction, or computer-vision workflow is used.

## Dataset Information
Refined split directory:

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

`Experiment_2B_HCEF_Refined_Subsets.py`

Implemented components:
- Tensorized residual encoder
- Supervised contrastive projection head
- Cross-entropy classification head
- Latent embedding extraction
- Embedding-based probabilistic learners
- Weighted and average ensemble fusion

## Usage Instructions
Run:

```powershell
cd "D:\\47\\471\\New Papers\\Paper 3 IJOCTA\\Sub\\Experiments\\Code"
python Experiment_2B_HCEF_Refined_Subsets.py
```

## Requirements
Python 3.10 or later.

Required libraries:
- numpy
- pandas
- matplotlib
- scikit-learn
- torch

Install:

```powershell
pip install numpy pandas matplotlib scikit-learn torch
```

## Methodology
1. Load each refined dataset split.
2. Standardize training features and apply the same scaler to validation and test features.
3. Train a tensorized residual neural encoder using a combined cross-entropy and supervised contrastive objective.
4. Extract normalized latent embeddings.
5. Train probabilistic base learners on embeddings.
6. Apply validation-weighted and average probabilistic ensemble fusion.
7. Evaluate validation and test results using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrices, ROC curves, precision-recall curves, embedding PCA, runtime, and model size.
8. Compare HCE-F refined-subset performance with Experiment 1B refined baseline results.

## Citations
Internal dataset citation or repository information should be added if publicly archived. Third-party external dataset URLs/DOIs are handled in the external validation experiment.

## License and Contribution Guidelines
Add repository license and contribution rules before public release.
"""

    methods_text = """# Methods Text: HCE-F Evaluation on Refined Feature Subsets

After auditing and refining the original feature space, the full HCE-F framework was reevaluated on each refined tabular feature subset. Each refined dataset used the same stratified training, validation, and test partitions as the corresponding refined baseline experiment, ensuring fair comparison between conventional models and the proposed representation-learning framework.

For each refined subset, the model followed the HCE-F pipeline: tabular input standardization, tensorized residual feature encoding, supervised contrastive projection, latent embedding extraction, and probabilistic ensemble fusion. The neural encoder was trained using a combined objective containing cross-entropy loss and supervised contrastive loss. The learned embeddings were then used to train probabilistic base learners, and final predictions were obtained through validation-weighted and average ensemble fusion.

Evaluation included accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, ROC-AUC, PR-AUC, confusion matrices, ROC curves, precision-recall curves, embedding PCA visualizations, training-loss curves, validation learning dynamics, runtime, and model-size reporting. HCE-F performance on each refined subset was compared against the strongest baseline obtained from Experiment 1B. No image preprocessing, simulated image data, convolutional feature extraction, or ResNet-based workflow was used.
"""

    limitations = """# Limitations Note for Conclusion

The refined-subset HCE-F experiment tests whether the proposed representation-learning method benefits from feature-space auditing and refinement. However, the refined subsets are still derived from the same internal augmented dataset. Therefore, any observed improvement should be interpreted as internal feature-space optimization rather than definitive evidence of external generalization. If performance remains limited across refined subsets, this may indicate weak target-feature association or limitations in the augmented dataset construction. External validation remains necessary.
"""

    (README_DIR / "README_Experiment_2B_HCEF_Refined_Subsets.md").write_text(readme, encoding="utf-8")
    (OUTPUT_DIR / "METHODS_TEXT_Experiment_2B_HCEF_Refined_Subsets.md").write_text(methods_text, encoding="utf-8")
    (OUTPUT_DIR / "LIMITATIONS_NOTE_FOR_CONCLUSION.md").write_text(limitations, encoding="utf-8")

    env = {
        "generated_at": str(datetime.now()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "root_dir": str(ROOT_DIR),
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "hidden_dim": HIDDEN_DIM,
        "embedding_dim": EMBED_DIM,
        "tensor_rank": TENSOR_RANK,
        "refined_datasets": REFINED_DATASETS,
    }

    with open(LOG_DIR / "environment_report.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=4, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================

def main():
    (LOG_DIR / "experiment_2b_console_log.txt").write_text("", encoding="utf-8")

    log("=" * 80)
    log("Experiment 2B: HCE-F Evaluation on Refined Feature Subsets")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    start_time = time.time()

    all_records = []
    all_histories = []
    dataset_summaries = []
    parameter_counts = {}
    best_epochs = {}
    model_sizes = {}

    for dataset_name in REFINED_DATASETS:
        log("\n" + "=" * 80)
        log(f"Evaluating HCE-F on refined dataset: {dataset_name}")
        log("=" * 80)

        train_df, val_df, test_df = load_refined_dataset_splits(dataset_name)

        X_train_df, y_train_s = split_xy(train_df)
        X_val_df, y_val_s = split_xy(val_df)
        X_test_df, y_test_s = split_xy(test_df)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df).astype(np.float32)
        X_val = scaler.transform(X_val_df).astype(np.float32)
        X_test = scaler.transform(X_test_df).astype(np.float32)

        y_train = y_train_s.values.astype(np.int64)
        y_val = y_val_s.values.astype(np.int64)
        y_test = y_test_s.values.astype(np.int64)

        labels = sorted(np.unique(np.concatenate([y_train, y_val, y_test])))
        n_classes = len(labels)
        input_dim = X_train.shape[1]

        dataset_summary = {
            "dataset": dataset_name,
            "n_features": int(input_dim),
            "train_shape": list(train_df.shape),
            "validation_shape": list(val_df.shape),
            "test_shape": list(test_df.shape),
            "target_distribution_train": y_train_s.value_counts().to_dict(),
            "target_distribution_validation": y_val_s.value_counts().to_dict(),
            "target_distribution_test": y_test_s.value_counts().to_dict(),
        }

        dataset_summaries.append(dataset_summary)

        log(f"Train shape: {train_df.shape}")
        log(f"Validation shape: {val_df.shape}")
        log(f"Test shape: {test_df.shape}")
        log(f"Features: {input_dim}")
        log(f"Classes: {labels}")

        with open(MODEL_DIR / f"{safe_name(dataset_name)}_feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        train_start = time.time()
        hcef_model, history, best_epoch = train_hcef_network(
            dataset_name,
            X_train,
            y_train,
            X_val,
            y_val,
            input_dim=input_dim,
            n_classes=n_classes,
        )
        neural_training_time = time.time() - train_start

        best_epochs[dataset_name] = int(best_epoch)
        parameter_counts[dataset_name] = int(sum(p.numel() for p in hcef_model.parameters()))

        model_path = MODEL_DIR / f"{safe_name(dataset_name)}_hcef_model.pt"
        model_sizes[dataset_name] = float(model_path.stat().st_size / (1024 * 1024)) if model_path.exists() else np.nan

        plot_training_history(history, dataset_name)
        all_histories.extend(history)

        train_loss, y_train_true, y_train_pred, y_train_prob, train_emb = extract_outputs(hcef_model, X_train, y_train)
        val_loss, y_val_true, y_val_pred, y_val_prob, val_emb = extract_outputs(hcef_model, X_val, y_val)
        test_loss, y_test_true, y_test_pred, y_test_prob, test_emb = extract_outputs(hcef_model, X_test, y_test)

        plot_embedding_pca(test_emb, y_test_true, dataset_name, "test")

        for split_name, y_true, y_pred, y_prob in [
            ("train", y_train_true, y_train_pred, y_train_prob),
            ("validation", y_val_true, y_val_pred, y_val_prob),
            ("test", y_test_true, y_test_pred, y_test_prob),
        ]:
            rec = compute_metrics(y_true, y_pred, y_prob, dataset_name, "HCEF_Neural_Model", split_name)
            rec["best_epoch"] = int(best_epoch)
            rec["neural_training_time_seconds"] = float(neural_training_time)
            rec["parameter_count"] = int(parameter_counts[dataset_name])
            rec["model_size_mb"] = float(model_sizes[dataset_name])
            all_records.append(rec)

            if split_name == "test":
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                pd.DataFrame(
                    cm,
                    index=[f"true_{x}" for x in labels],
                    columns=[f"pred_{x}" for x in labels],
                ).to_csv(TABLE_DIR / f"{safe_name(dataset_name)}_HCEF_Neural_Model_test_confusion_matrix.csv", encoding="utf-8-sig")

                plot_confusion_matrix(cm, labels, dataset_name, "HCEF_Neural_Model", split_name)
                plot_roc(y_true, y_prob, dataset_name, "HCEF_Neural_Model", split_name)
                plot_pr(y_true, y_prob, dataset_name, "HCEF_Neural_Model", split_name)

                with open(METRIC_DIR / f"{safe_name(dataset_name)}_HCEF_Neural_Model_test_classification_report.json", "w", encoding="utf-8") as f:
                    json.dump(classification_report(y_true, y_pred, output_dict=True, zero_division=0), f, indent=4, ensure_ascii=False)

        ensemble_start = time.time()
        ensemble_records, weights_df = run_embedding_ensemble(
            dataset_name,
            train_emb,
            y_train_true,
            val_emb,
            y_val_true,
            test_emb,
            y_test_true,
        )
        ensemble_time = time.time() - ensemble_start

        for rec in ensemble_records:
            rec["best_epoch"] = int(best_epoch)
            rec["neural_training_time_seconds"] = float(neural_training_time)
            rec["ensemble_training_time_seconds"] = float(ensemble_time)
            rec["parameter_count"] = int(parameter_counts[dataset_name])
            rec["model_size_mb"] = float(model_sizes[dataset_name])
            all_records.append(rec)

    metrics_df = pd.DataFrame(all_records)
    metrics_df.to_csv(TABLE_DIR / "hcef_refined_metrics.csv", index=False, encoding="utf-8-sig")

    with open(METRIC_DIR / "hcef_refined_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False, default=str)

    pd.DataFrame(all_histories).to_csv(TABLE_DIR / "hcef_refined_training_histories.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dataset_summaries).to_csv(TABLE_DIR / "evaluated_refined_dataset_summary.csv", index=False, encoding="utf-8-sig")

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()

    best_per_dataset = (
        test_metrics
        .sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False)
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_per_dataset.to_csv(TABLE_DIR / "best_hcef_test_model_per_refined_dataset.csv", index=False, encoding="utf-8-sig")

    best_overall = (
        test_metrics
        .sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False)
        .reset_index(drop=True)
    )

    best_overall.to_csv(TABLE_DIR / "hcef_refined_test_model_ranking.csv", index=False, encoding="utf-8-sig")

    comparison_df, delta_df = compare_with_1b_and_2(best_per_dataset)
    plot_best_hcef_by_dataset(best_per_dataset)

    total_runtime = time.time() - start_time

    summary = {
        "experiment": "Experiment 2B - HCE-F Evaluation on Refined Feature Subsets",
        "generated_at": str(datetime.now()),
        "target": TARGET_COLUMN,
        "device": str(DEVICE),
        "evaluated_datasets": dataset_summaries,
        "best_epochs": best_epochs,
        "parameter_counts": parameter_counts,
        "model_sizes_mb": model_sizes,
        "best_hcef_per_refined_dataset": best_per_dataset.to_dict(orient="records"),
        "best_overall_hcef_test_model": best_overall.iloc[0].to_dict() if not best_overall.empty else None,
        "comparison_with_experiment_1b": comparison_df.to_dict(orient="records") if not comparison_df.empty else [],
        "delta_against_experiment_1b": delta_df.to_dict(orient="records") if not delta_df.empty else [],
        "total_runtime_seconds": float(total_runtime),
    }

    with open(OUTPUT_DIR / "hcef_refined_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)

    with open(OUTPUT_DIR / "hcef_refined_summary.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 2B HCE-F Refined Feature-Subset Summary\n\n")
        f.write(f"Generated at: `{datetime.now()}`\n\n")
        f.write(f"Target: `{TARGET_COLUMN}`\n\n")
        f.write(f"Device: `{DEVICE}`\n\n")

        f.write("## Evaluated Refined Datasets\n\n")
        for item in dataset_summaries:
            f.write(
                f"- {item['dataset']}: {item['n_features']} features, "
                f"train {item['train_shape']}, validation {item['validation_shape']}, test {item['test_shape']}\n"
            )

        f.write("\n## Best HCE-F Test Model Per Refined Dataset\n\n")
        for _, row in best_per_dataset.iterrows():
            f.write(
                f"- {row['dataset']} | {row['model']} | "
                f"F1-weighted={row['f1_weighted']:.6f} | "
                f"ROC-AUC={row['roc_auc']:.6f} | "
                f"PR-AUC={row['pr_auc']:.6f}\n"
            )

        if not best_overall.empty:
            row = best_overall.iloc[0]
            f.write("\n## Best Overall HCE-F Refined Result\n\n")
            f.write(
                f"- Dataset: {row['dataset']}\n"
                f"- Model: {row['model']}\n"
                f"- Accuracy: {row['accuracy']}\n"
                f"- F1-weighted: {row['f1_weighted']}\n"
                f"- ROC-AUC: {row['roc_auc']}\n"
                f"- PR-AUC: {row['pr_auc']}\n"
            )

        if not delta_df.empty:
            f.write("\n## HCE-F Refined vs Experiment 1B Baseline Delta\n\n")
            for _, row in delta_df.iterrows():
                f.write(
                    f"- {row['dataset']} | "
                    f"Baseline={row['best_1b_model']} | "
                    f"HCE-F={row['best_2b_model']} | "
                    f"Delta F1={row['delta_f1_hcef_minus_baseline']:.6f} | "
                    f"Delta ROC-AUC={row['delta_auc_hcef_minus_baseline']:.6f} | "
                    f"Delta PR-AUC={row['delta_pr_auc_hcef_minus_baseline']:.6f}\n"
                )

        f.write(f"\nTotal runtime seconds: `{total_runtime:.4f}`\n")

    write_reproducibility_files(dataset_summaries, best_per_dataset, comparison_df, delta_df)

    log("=" * 80)
    log("Experiment 2B completed successfully.")
    log("=" * 80)
    log(f"Metrics: {TABLE_DIR / 'hcef_refined_metrics.csv'}")
    log(f"Best per dataset: {TABLE_DIR / 'best_hcef_test_model_per_refined_dataset.csv'}")
    log(f"Comparison with Experiment 1B: {TABLE_DIR / 'experiment_1b_vs_2b_comparison.csv'}")
    log(f"Delta table: {TABLE_DIR / 'hcef_refined_vs_baseline_delta_table.csv'}")
    log(f"Summary: {OUTPUT_DIR / 'hcef_refined_summary.md'}")
    log(f"README: {README_DIR / 'README_Experiment_2B_HCEF_Refined_Subsets.md'}")
    log(f"Total runtime: {total_runtime:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Experiment 2B failed. See error log:", flush=True)
        print(LOG_DIR / "experiment_2b_error_log.txt", flush=True)
        save_error(e)
        raise
