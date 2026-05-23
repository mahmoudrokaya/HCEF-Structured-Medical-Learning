"""
Experiment_2_HCEF_Full_Model.py

Experiment 2:
    Full Hybrid Contrastive-Ensemble Framework (HCE-F) Evaluation

Purpose:
    Implement the full method described in the revised paper:
        Tabular input
        -> tensorized residual encoder
        -> supervised contrastive projection head
        -> latent embeddings
        -> probabilistic ensemble fusion

Journal/reviewer alignment:
    - Algorithms and implementation code are provided.
    - Evaluation method is explicit and reproducible.
    - Training/validation/test results are saved.
    - Learning dynamics, embeddings, ensemble weights, ROC/PR curves,
      confusion matrices, runtime, model size, README, and Methods text are generated.
    - Tabular data only.
    - No image pipeline.
    - No ResNet workflow.

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL TORCH IMPORT WITH CLEAR ERROR MESSAGE
# ============================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise ImportError(
        "PyTorch is required for Experiment 2. Install it first, for example:\n"
        "pip install torch\n\n"
        f"Original import error: {e}"
    )


# ============================================================
# PATH SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")
DATASET_NAME = "internal_augmented_supervised"
TARGET_COLUMN = "target"

MODEL_READY_FILE = ROOT_DIR / "Data" / "Processed" / "Modeling_Ready" / "internal_augmented_supervised_modeling_ready.csv"
SPLIT_DIR = ROOT_DIR / "Data" / "Splits" / "Modeling_Ready" / DATASET_NAME

BASELINE_METRICS_FILE = ROOT_DIR / "Experiments" / "Results" / "Experiment_1_Internal_Baselines" / "Tables" / "baseline_metrics.csv"

OUTPUT_DIR = ROOT_DIR / "Experiments" / "Results" / "Experiment_2_HCEF_Full_Model"
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
    OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, METRIC_DIR, MODEL_DIR, LOG_DIR,
    EMBED_DIR, CM_DIR, ROC_DIR, PR_DIR, README_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

RANDOM_STATE = 42
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
EMBED_DIM = 32
TENSOR_RANK = 16
CONTRASTIVE_TEMPERATURE = 0.20
CONTRASTIVE_WEIGHT = 0.25
DROPOUT = 0.15
N_JOBS = -1

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)
    with open(LOG_DIR / "experiment_2_console_log.txt", "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def save_error(error):
    with open(LOG_DIR / "experiment_2_error_log.txt", "w", encoding="utf-8") as f:
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
    train_path = find_split_file("train")
    val_path = find_split_file("validation")
    test_path = find_split_file("test")

    if train_path and val_path and test_path:
        log(f"Using train file: {train_path}")
        log(f"Using validation file: {val_path}")
        log(f"Using test file: {test_path}")
        return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path), "prepared_split_files"

    log("Prepared splits not found. Creating fallback stratified split from modeling-ready dataset.")

    if not MODEL_READY_FILE.exists():
        raise FileNotFoundError(f"Modeling-ready file not found: {MODEL_READY_FILE}")

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(MODEL_READY_FILE)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN]
    )

    return train_df, val_df, test_df, "fallback_split_from_modeling_ready_file"


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
# HCE-F MODEL
# ============================================================

class TensorizedResidualBlock(nn.Module):
    """
    Compact tensorized residual block for tabular data.

    The block combines:
        - linear transformation
        - low-rank multiplicative interaction
        - residual identity retention

    This operationalizes tensorized residual encoding without using any
    image-based or convolutional workflow.
    """

    def __init__(self, dim, rank=16, dropout=0.15, alpha=0.50):
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
# LOSS FUNCTIONS
# ============================================================

def supervised_contrastive_loss(embeddings, labels, temperature=0.20):
    """
    Supervised contrastive loss using labels within each mini-batch.
    """

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
    loss = -mean_log_prob_pos.mean()

    return loss


# ============================================================
# METRICS AND PLOTS
# ============================================================

def get_probabilities_from_logits(logits):
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


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


def plot_confusion_matrix(cm, labels, name):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title(f"{name} Confusion Matrix")
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
    plt.close()


def plot_roc(y_true, y_score, name):
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
        plt.title(f"{name} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"{name}_roc_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        plt.close()
        log(f"ROC plot skipped for {name}: {e}")


def plot_pr(y_true, y_score, name):
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
        plt.title(f"{name} Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(PR_DIR / f"{name}_precision_recall_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        plt.close()
        log(f"PR plot skipped for {name}: {e}")


def plot_training_history(history):
    """
    Save and plot HCE-F training history.

    Important:
        The training loop stores history as a list of dictionaries.
        Therefore, it must be converted to a DataFrame before column access.
    """

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        TABLE_DIR / "hcef_training_history.csv",
        index=False,
        encoding="utf-8-sig"
    )

    loss_keys = [
        "train_total_loss",
        "train_ce_loss",
        "train_contrastive_loss",
        "validation_total_loss"
    ]

    for key in loss_keys:
        if key not in history_df.columns:
            continue

        plt.figure(figsize=(7, 6))
        plt.plot(history_df["epoch"], history_df[key], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(key.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{key}.png", dpi=300)
        plt.close()

    plt.figure(figsize=(7, 6))

    plotted = False

    if "validation_f1_weighted" in history_df.columns:
        plt.plot(
            history_df["epoch"],
            history_df["validation_f1_weighted"],
            marker="o",
            label="Validation F1-weighted"
        )
        plotted = True

    if "validation_roc_auc" in history_df.columns:
        plt.plot(
            history_df["epoch"],
            history_df["validation_roc_auc"],
            marker="o",
            label="Validation ROC-AUC"
        )
        plotted = True

    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Learning Dynamics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "validation_learning_dynamics.png", dpi=300)

    plt.close()


def plot_embeddings(embeddings, labels, split_name):
    emb_df = pd.DataFrame(embeddings)
    emb_df["label"] = labels
    emb_df.to_csv(EMBED_DIR / f"{split_name}_embeddings.csv", index=False, encoding="utf-8-sig")

    try:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(embeddings)

        plt.figure(figsize=(7, 6))
        for cls in sorted(np.unique(labels)):
            idx = labels == cls
            plt.scatter(coords[idx, 0], coords[idx, 1], label=f"Class {cls}", alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{split_name} HCE-F Embeddings - PCA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EMBED_DIR / f"{split_name}_embedding_pca.png", dpi=300)
        plt.close()
    except Exception as e:
        log(f"PCA embedding plot skipped for {split_name}: {e}")

    try:
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
        coords = tsne.fit_transform(embeddings)

        plt.figure(figsize=(7, 6))
        for cls in sorted(np.unique(labels)):
            idx = labels == cls
            plt.scatter(coords[idx, 0], coords[idx, 1], label=f"Class {cls}", alpha=0.7)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"{split_name} HCE-F Embeddings - t-SNE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EMBED_DIR / f"{split_name}_embedding_tsne.png", dpi=300)
        plt.close()
    except Exception as e:
        log(f"t-SNE embedding plot skipped for {split_name}: {e}")


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

            prob = get_probabilities_from_logits(logits)
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


def train_hcef_network(X_train, y_train, X_val, y_val, input_dim, n_classes):
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = HCEFNetwork(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        n_classes=n_classes,
        rank=TENSOR_RANK,
        dropout=DROPOUT
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
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_prob, "HCEF_Neural_Model", "validation")

        record = {
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
            f"Epoch {epoch:03d} | "
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
            log(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_DIR / "hcef_tensorized_residual_contrastive_model.pt")

    return model, history, best_epoch


def extract_embeddings(model, X, y):
    ds = TabularDataset(X, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    loss, y_true, y_pred, y_prob, embeddings = evaluate_network(model, loader, criterion)

    return loss, y_true, y_pred, y_prob, embeddings


# ============================================================
# ENSEMBLE FUSION ON HCE-F EMBEDDINGS
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
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=N_JOBS
        ),
        "Embedding_Extra_Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=N_JOBS
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
                random_state=RANDOM_STATE
            ))
        ]),
    }


def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def compute_validation_weights(model_names, val_probs, y_val):
    losses = []
    for name in model_names:
        prob = val_probs[name]
        try:
            losses.append(log_loss(y_val, prob, labels=sorted(np.unique(y_val))))
        except Exception:
            losses.append(10.0)

    losses = np.array(losses, dtype=float)

    # Softmax over negative loss. Lower validation loss gets higher weight.
    scores = -losses
    scores = scores - scores.max()
    weights = np.exp(scores) / np.exp(scores).sum()

    return dict(zip(model_names, weights)), dict(zip(model_names, losses))


def fit_embedding_ensemble(train_emb, y_train, val_emb, y_val, test_emb, y_test):
    models = get_embedding_ensemble_models()

    val_probs = {}
    test_probs = {}
    base_records = []

    for name, model in models.items():
        log(f"Training embedding base learner: {name}")
        start = time.time()
        model.fit(train_emb, y_train)
        train_time = time.time() - start

        val_prob = safe_predict_proba(model, val_emb)
        test_prob = safe_predict_proba(model, test_emb)

        if val_prob is None or test_prob is None:
            continue

        val_pred = np.argmax(val_prob, axis=1)
        test_pred = np.argmax(test_prob, axis=1)

        val_record = compute_metrics(y_val, val_pred, val_prob, name, "validation")
        test_record = compute_metrics(y_test, test_pred, test_prob, name, "test")
        val_record["training_time_seconds"] = float(train_time)
        test_record["training_time_seconds"] = float(train_time)

        base_records.append(val_record)
        base_records.append(test_record)

        val_probs[name] = val_prob
        test_probs[name] = test_prob

        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    model_names = list(val_probs.keys())
    weights, validation_losses = compute_validation_weights(model_names, val_probs, y_val)

    weighted_val_prob = np.zeros_like(next(iter(val_probs.values())))
    weighted_test_prob = np.zeros_like(next(iter(test_probs.values())))

    for name in model_names:
        weighted_val_prob += weights[name] * val_probs[name]
        weighted_test_prob += weights[name] * test_probs[name]

    val_pred = np.argmax(weighted_val_prob, axis=1)
    test_pred = np.argmax(weighted_test_prob, axis=1)

    fusion_val_record = compute_metrics(y_val, val_pred, weighted_val_prob, "HCEF_Weighted_Ensemble_Fusion", "validation")
    fusion_test_record = compute_metrics(y_test, test_pred, weighted_test_prob, "HCEF_Weighted_Ensemble_Fusion", "test")

    weights_df = pd.DataFrame({
        "base_learner": model_names,
        "validation_log_loss": [validation_losses[name] for name in model_names],
        "fusion_weight": [weights[name] for name in model_names],
    })

    weights_df.to_csv(TABLE_DIR / "ensemble_weights.csv", index=False, encoding="utf-8-sig")

    with open(METRIC_DIR / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=4, ensure_ascii=False)

    return base_records, fusion_val_record, fusion_test_record, weighted_val_prob, weighted_test_prob, weights_df


# ============================================================
# COMPARISON WITH EXPERIMENT 1
# ============================================================

def compare_with_baseline(hcef_test_record):
    if not BASELINE_METRICS_FILE.exists():
        return None

    baseline_df = pd.read_csv(BASELINE_METRICS_FILE)
    baseline_test = baseline_df[baseline_df["split"] == "test"].copy()

    if baseline_test.empty:
        return None

    best_baseline = baseline_test.sort_values(by=["f1_weighted", "roc_auc"], ascending=False).iloc[0].to_dict()

    comparison = {
        "best_experiment_1_baseline_model": best_baseline.get("model"),
        "baseline_test_accuracy": float(best_baseline.get("accuracy")),
        "baseline_test_f1_weighted": float(best_baseline.get("f1_weighted")),
        "baseline_test_roc_auc": float(best_baseline.get("roc_auc")),
        "baseline_test_pr_auc": float(best_baseline.get("pr_auc")),
        "hcef_test_accuracy": float(hcef_test_record.get("accuracy")),
        "hcef_test_f1_weighted": float(hcef_test_record.get("f1_weighted")),
        "hcef_test_roc_auc": float(hcef_test_record.get("roc_auc")),
        "hcef_test_pr_auc": float(hcef_test_record.get("pr_auc")),
        "delta_accuracy": float(hcef_test_record.get("accuracy") - best_baseline.get("accuracy")),
        "delta_f1_weighted": float(hcef_test_record.get("f1_weighted") - best_baseline.get("f1_weighted")),
        "delta_roc_auc": float(hcef_test_record.get("roc_auc") - best_baseline.get("roc_auc")),
        "delta_pr_auc": float(hcef_test_record.get("pr_auc") - best_baseline.get("pr_auc")),
    }

    pd.DataFrame([comparison]).to_csv(TABLE_DIR / "hcef_vs_best_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    return comparison


# ============================================================
# DOCUMENTATION OUTPUTS
# ============================================================

def write_reproducibility_files(train_df, val_df, test_df, source_mode):
    readme = f"""# Experiment 2: Full HCE-F Model Evaluation

## Title
Paper 3 IJOCTA Revision - Full Hybrid Contrastive-Ensemble Framework Evaluation

## Description
This experiment implements the full HCE-F method on tabular biomedical data. The workflow is:

Tabular input -> tensorized residual encoder -> supervised contrastive projection head -> latent embeddings -> probabilistic ensemble fusion.

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
`Experiment_2_HCEF_Full_Model.py`

Implemented components:
- Tensorized residual tabular encoder
- Supervised contrastive projection head
- Cross-entropy classification head
- Latent embedding extraction
- Probabilistic ensemble base learners
- Validation-optimized weighted ensemble fusion

## Usage Instructions
Run:

```powershell
cd "D:\\47\\471\\New Papers\\Paper 3 IJOCTA\\Sub\\Experiments\\Code"
python Experiment_2_HCEF_Full_Model.py
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
1. Load prepared tabular train, validation, and test splits.
2. Train a compact tensorized residual neural encoder.
3. Optimize a combined loss containing cross-entropy and supervised contrastive loss.
4. Extract normalized latent embeddings.
5. Train probabilistic ensemble learners on the learned embeddings.
6. Compute validation-based fusion weights using negative log-loss.
7. Evaluate validation and test performance using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix.
8. Save training dynamics, embeddings, ensemble weights, figures, tables, logs, and summaries.

## Citations
Third-party external dataset URLs/DOIs are handled in the external validation experiment. This experiment uses the internal prepared dataset.

## License and Contribution Guidelines
Add the selected repository license and contribution rules before public release.
"""

    methods_text = """# Methods Text: Full HCE-F Model Evaluation

The full HCE-F model was evaluated using the canonical internal tabular dataset and the same stratified training, validation, and test partitions used in the baseline experiment. The implementation followed the proposed methodological pipeline: structured tabular input, tensorized residual feature encoding, supervised contrastive projection, latent embedding extraction, and probabilistic ensemble fusion.

The tensorized residual encoder used compact low-rank multiplicative interactions combined with residual identity retention to model nonlinear inter-feature dependencies while preserving stable information flow. The projection head mapped encoded features into a normalized latent space. Training minimized a combined objective containing cross-entropy loss and supervised contrastive loss, thereby jointly optimizing predictive discrimination and latent-space compactness.

After training, normalized embeddings were extracted for the training, validation, and test partitions. Multiple probabilistic base learners were trained on the learned embeddings. Their validation probabilities were used to compute fusion weights based on validation negative log-loss, assigning higher weights to more reliable learners. Final predictions were obtained through weighted probabilistic aggregation.

Evaluation included accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, ROC-AUC, PR-AUC, confusion matrices, ROC curves, precision-recall curves, embedding PCA/t-SNE visualizations, training-loss curves, validation learning dynamics, runtime, and model-size reporting. No image preprocessing, simulated image data, convolutional feature extraction, or ResNet-based workflow was used.
"""

    limitations = """# Limitations Note for Conclusion

The full HCE-F experiment evaluates the proposed method on the internal prepared tabular dataset. Although the experiment directly tests the core methodological components, internal evaluation alone cannot fully establish cross-dataset generalization. The internal dataset is augmented and may not capture the full variability of independent clinical cohorts. Therefore, the findings should be interpreted together with ablation, robustness, and external-validation experiments.
"""

    (README_DIR / "README_Experiment_2_HCEF_Full_Model.md").write_text(readme, encoding="utf-8")
    (OUTPUT_DIR / "METHODS_TEXT_Experiment_2_HCEF_Evaluation.md").write_text(methods_text, encoding="utf-8")
    (OUTPUT_DIR / "LIMITATIONS_NOTE_FOR_CONCLUSION.md").write_text(limitations, encoding="utf-8")

    env = {
        "generated_at": str(datetime.now()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "root_dir": str(ROOT_DIR),
        "dataset": DATASET_NAME,
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
    }

    with open(LOG_DIR / "environment_report.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=4, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================

def main():
    (LOG_DIR / "experiment_2_console_log.txt").write_text("", encoding="utf-8")

    log("=" * 80)
    log("Experiment 2: Full HCE-F Model Evaluation")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    start_time = time.time()

    train_df, val_df, test_df, source_mode = load_data()

    log(f"Data source mode: {source_mode}")
    log(f"Train shape: {train_df.shape}")
    log(f"Validation shape: {val_df.shape}")
    log(f"Test shape: {test_df.shape}")

    X_train_df, y_train_s = split_xy(train_df)
    X_val_df, y_val_s = split_xy(val_df)
    X_test_df, y_test_s = split_xy(test_df)

    feature_names = list(X_train_df.columns)

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

    log(f"Input features: {input_dim}")
    log(f"Target classes: {labels}")

    with open(MODEL_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    pd.DataFrame({"feature_name": feature_names}).to_csv(TABLE_DIR / "feature_names.csv", index=False, encoding="utf-8-sig")

    training_start = time.time()
    hcef_model, history, best_epoch = train_hcef_network(
        X_train, y_train,
        X_val, y_val,
        input_dim=input_dim,
        n_classes=n_classes
    )
    neural_training_time = time.time() - training_start

    plot_training_history(history)

    train_loss, y_train_true, y_train_pred, y_train_prob, train_emb = extract_embeddings(hcef_model, X_train, y_train)
    val_loss, y_val_true, y_val_pred, y_val_prob, val_emb = extract_embeddings(hcef_model, X_val, y_val)
    test_loss, y_test_true, y_test_pred, y_test_prob, test_emb = extract_embeddings(hcef_model, X_test, y_test)

    plot_embeddings(train_emb, y_train_true, "train")
    plot_embeddings(val_emb, y_val_true, "validation")
    plot_embeddings(test_emb, y_test_true, "test")

    neural_records = []
    for split_name, y_true, y_pred, y_prob in [
        ("train", y_train_true, y_train_pred, y_train_prob),
        ("validation", y_val_true, y_val_pred, y_val_prob),
        ("test", y_test_true, y_test_pred, y_test_prob),
    ]:
        rec = compute_metrics(y_true, y_pred, y_prob, "HCEF_Neural_Model", split_name)
        rec["neural_training_time_seconds"] = float(neural_training_time)
        rec["best_epoch"] = int(best_epoch)
        neural_records.append(rec)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(
            cm,
            index=[f"true_{x}" for x in labels],
            columns=[f"pred_{x}" for x in labels],
        ).to_csv(TABLE_DIR / f"HCEF_Neural_Model_{split_name}_confusion_matrix.csv", encoding="utf-8-sig")

        plot_confusion_matrix(cm, labels, f"HCEF_Neural_Model_{split_name}")
        plot_roc(y_true, y_prob, f"HCEF_Neural_Model_{split_name}")
        plot_pr(y_true, y_prob, f"HCEF_Neural_Model_{split_name}")

        with open(METRIC_DIR / f"HCEF_Neural_Model_{split_name}_classification_report.json", "w", encoding="utf-8") as f:
            json.dump(classification_report(y_true, y_pred, output_dict=True, zero_division=0), f, indent=4, ensure_ascii=False)

    ensemble_start = time.time()
    base_records, fusion_val_record, fusion_test_record, fusion_val_prob, fusion_test_prob, weights_df = fit_embedding_ensemble(
        train_emb, y_train_true,
        val_emb, y_val_true,
        test_emb, y_test_true
    )
    ensemble_time = time.time() - ensemble_start

    fusion_val_record["ensemble_training_time_seconds"] = float(ensemble_time)
    fusion_test_record["ensemble_training_time_seconds"] = float(ensemble_time)
    fusion_val_record["best_epoch"] = int(best_epoch)
    fusion_test_record["best_epoch"] = int(best_epoch)

    for split_name, y_true, prob in [
        ("validation", y_val_true, fusion_val_prob),
        ("test", y_test_true, fusion_test_prob),
    ]:
        pred = np.argmax(prob, axis=1)

        cm = confusion_matrix(y_true, pred, labels=labels)
        pd.DataFrame(
            cm,
            index=[f"true_{x}" for x in labels],
            columns=[f"pred_{x}" for x in labels],
        ).to_csv(TABLE_DIR / f"HCEF_Weighted_Ensemble_Fusion_{split_name}_confusion_matrix.csv", encoding="utf-8-sig")

        plot_confusion_matrix(cm, labels, f"HCEF_Weighted_Ensemble_Fusion_{split_name}")
        plot_roc(y_true, prob, f"HCEF_Weighted_Ensemble_Fusion_{split_name}")
        plot_pr(y_true, prob, f"HCEF_Weighted_Ensemble_Fusion_{split_name}")

        with open(METRIC_DIR / f"HCEF_Weighted_Ensemble_Fusion_{split_name}_classification_report.json", "w", encoding="utf-8") as f:
            json.dump(classification_report(y_true, pred, output_dict=True, zero_division=0), f, indent=4, ensure_ascii=False)

    all_records = neural_records + base_records + [fusion_val_record, fusion_test_record]
    metrics_df = pd.DataFrame(all_records)
    metrics_df.to_csv(TABLE_DIR / "hcef_metrics.csv", index=False, encoding="utf-8-sig")

    with open(METRIC_DIR / "hcef_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)

    test_rank = metrics_df[metrics_df["split"] == "test"].sort_values(
        by=["f1_weighted", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    test_rank.to_csv(TABLE_DIR / "hcef_test_model_ranking.csv", index=False, encoding="utf-8-sig")

    validation_rank = metrics_df[metrics_df["split"] == "validation"].sort_values(
        by=["f1_weighted", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    validation_rank.to_csv(TABLE_DIR / "hcef_validation_model_ranking.csv", index=False, encoding="utf-8-sig")

    comparison = compare_with_baseline(fusion_test_record)

    parameter_count = sum(p.numel() for p in hcef_model.parameters())
    model_size_mb = (MODEL_DIR / "hcef_tensorized_residual_contrastive_model.pt").stat().st_size / (1024 * 1024)

    total_runtime = time.time() - start_time

    summary = {
        "experiment": "Experiment 2 - Full HCE-F Model Evaluation",
        "generated_at": str(datetime.now()),
        "dataset": DATASET_NAME,
        "target": TARGET_COLUMN,
        "source_mode": source_mode,
        "train_shape": list(train_df.shape),
        "validation_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "device": str(DEVICE),
        "input_dim": int(input_dim),
        "n_classes": int(n_classes),
        "best_epoch": int(best_epoch),
        "parameter_count": int(parameter_count),
        "model_size_mb": float(model_size_mb),
        "neural_training_time_seconds": float(neural_training_time),
        "ensemble_training_time_seconds": float(ensemble_time),
        "total_runtime_seconds": float(total_runtime),
        "best_validation_model": validation_rank.iloc[0].to_dict(),
        "best_test_model": test_rank.iloc[0].to_dict(),
        "hcef_weighted_fusion_validation": fusion_val_record,
        "hcef_weighted_fusion_test": fusion_test_record,
        "comparison_with_best_baseline": comparison,
    }

    with open(OUTPUT_DIR / "hcef_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)

    with open(OUTPUT_DIR / "hcef_summary.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 2 HCE-F Full Model Summary\n\n")
        f.write(f"Generated at: `{datetime.now()}`\n\n")
        f.write(f"Dataset: `{DATASET_NAME}`\n\n")
        f.write(f"Target: `{TARGET_COLUMN}`\n\n")
        f.write(f"Source mode: `{source_mode}`\n\n")
        f.write(f"Training shape: `{train_df.shape}`\n\n")
        f.write(f"Validation shape: `{val_df.shape}`\n\n")
        f.write(f"Test shape: `{test_df.shape}`\n\n")
        f.write(f"Device: `{DEVICE}`\n\n")
        f.write(f"Best epoch: `{best_epoch}`\n\n")
        f.write(f"Parameter count: `{parameter_count}`\n\n")
        f.write(f"Model size MB: `{model_size_mb:.4f}`\n\n")
        f.write("## HCE-F Weighted Ensemble Test Results\n\n")
        for key, value in fusion_test_record.items():
            f.write(f"- {key}: {value}\n")
        if comparison is not None:
            f.write("\n## Comparison With Best Experiment 1 Baseline\n\n")
            for key, value in comparison.items():
                f.write(f"- {key}: {value}\n")
        f.write(f"\nTotal runtime seconds: `{total_runtime:.4f}`\n")

    write_reproducibility_files(train_df, val_df, test_df, source_mode)

    log("=" * 80)
    log("Experiment 2 completed successfully.")
    log("=" * 80)
    log(f"Metrics: {TABLE_DIR / 'hcef_metrics.csv'}")
    log(f"Summary: {OUTPUT_DIR / 'hcef_summary.md'}")
    log(f"README: {README_DIR / 'README_Experiment_2_HCEF_Full_Model.md'}")
    log(f"Total runtime: {total_runtime:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Experiment 2 failed. See error log:", flush=True)
        print(LOG_DIR / "experiment_2_error_log.txt", flush=True)
        save_error(e)
        raise
