"""
Experiment_3_Formal_Ablation_Study.py

Experiment 3:
    Formal Ablation Study on the Best Refined Feature Subset

Purpose:
    Quantify the contribution of each HCE-F component after feature-space refinement.

Primary refined subset:
    refined_top_20_features

Why this subset:
    Experiment 2B showed that refined_top_20_features produced the strongest
    HCE-F validation behavior among refined subsets. Therefore, it is the most
    appropriate controlled setting for component-level ablation.

Ablation variants:
    1. Full_HCEF:
        tensorized residual encoder + supervised contrastive loss + embedding ensemble fusion
    2. No_Contrastive_Loss:
        tensorized residual encoder + cross-entropy only + embedding ensemble fusion
    3. No_Tensorized_Residual:
        plain residual MLP encoder + supervised contrastive loss + embedding ensemble fusion
    4. Neural_Only:
        tensorized residual encoder + supervised contrastive loss without ensemble fusion
    5. Tensorized_Residual_Only:
        tensorized residual encoder + cross-entropy only without ensemble fusion
    6. Contrastive_Only:
        plain residual MLP encoder + supervised contrastive loss without ensemble fusion

Fusion variants:
    - Weighted fusion
    - Average fusion
    - Max-confidence fusion
    - Stacking fusion

Journal/reviewer alignment:
    - Directly addresses ablation study and component contribution.
    - Uses tabular data only.
    - Uses refined feature subset after data audit.
    - No image pipeline.
    - No ResNet workflow.
    - Generates README, Methods text, limitations note, tables, figures, logs,
      and machine-readable metrics.

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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
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
        "PyTorch is required for Experiment 3. Install it first:\n"
        "pip install torch\n\n"
        f"Original import error: {e}"
    )


# ============================================================
# PATH SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

TARGET_COLUMN = "target"
DATASET_NAME = "refined_top_20_features"
RANDOM_STATE = 42

REFINED_SPLIT_DIR = (
    ROOT_DIR
    / "Data"
    / "Splits"
    / "Modeling_Ready_Refined"
    / DATASET_NAME
)

EXPERIMENT_1B_METRICS = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_1B_Refined_Baselines"
    / "Tables"
    / "refined_baseline_metrics.csv"
)

EXPERIMENT_2B_METRICS = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_2B_HCEF_Refined_Subsets"
    / "Tables"
    / "hcef_refined_metrics.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "Experiments"
    / "Results"
    / "Experiment_3_Formal_Ablation_Study"
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
# SETTINGS
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
N_JOBS = 1

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)
    with open(LOG_DIR / "experiment_3_ablation_console_log.txt", "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def save_error(error):
    with open(LOG_DIR / "experiment_3_ablation_error_log.txt", "w", encoding="utf-8") as f:
        f.write(str(error) + "\n\n")
        f.write(traceback.format_exc())


# ============================================================
# DATA LOADING
# ============================================================

def load_splits():
    train_path = REFINED_SPLIT_DIR / "train.csv"
    val_path = REFINED_SPLIT_DIR / "validation.csv"
    test_path = REFINED_SPLIT_DIR / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

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
# NETWORK COMPONENTS
# ============================================================

class TensorizedResidualBlock(nn.Module):
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


class PlainResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15, alpha=0.60):
        super().__init__()
        self.alpha = alpha

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return F.relu(self.alpha * x + (1.0 - self.alpha) * self.net(x))


class AblationNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        embed_dim=32,
        n_classes=2,
        rank=16,
        dropout=0.15,
        use_tensorized_residual=True,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_tensorized_residual:
            self.block1 = TensorizedResidualBlock(hidden_dim, rank=rank, dropout=dropout)
            self.block2 = TensorizedResidualBlock(hidden_dim, rank=rank, dropout=dropout)
        else:
            self.block1 = PlainResidualBlock(hidden_dim, dropout=dropout)
            self.block2 = PlainResidualBlock(hidden_dim, dropout=dropout)

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


def compute_metrics(y_true, y_pred, y_score, variant, model_name, split_name):
    labels = sorted(np.unique(y_true))
    n_classes = len(labels)

    record = {
        "dataset": DATASET_NAME,
        "variant": variant,
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


def plot_confusion_matrix(cm, labels, variant, model_name):
    name = f"{safe_name(variant)}_{safe_name(model_name)}"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(f"{variant} - {model_name} Test Confusion Matrix")
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
    plt.savefig(CM_DIR / f"{name}_test_confusion_matrix.png", dpi=300)
    plt.close("all")


def plot_roc(y_true, y_score, variant, model_name):
    if y_score is None:
        return

    labels = sorted(np.unique(y_true))
    n_classes = len(labels)
    name = f"{safe_name(variant)}_{safe_name(model_name)}"

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
        plt.title(f"{variant} - {model_name} Test ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"{name}_test_roc_curve.png", dpi=300)
        plt.close("all")

    except Exception as e:
        plt.close("all")
        log(f"ROC skipped for {variant} {model_name}: {e}")


def plot_pr(y_true, y_score, variant, model_name):
    if y_score is None:
        return

    labels = sorted(np.unique(y_true))
    n_classes = len(labels)
    name = f"{safe_name(variant)}_{safe_name(model_name)}"

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
        plt.title(f"{variant} - {model_name} Test Precision-Recall")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(PR_DIR / f"{name}_test_precision_recall_curve.png", dpi=300)
        plt.close("all")

    except Exception as e:
        plt.close("all")
        log(f"PR skipped for {variant} {model_name}: {e}")


def plot_training_history(history, variant):
    df = pd.DataFrame(history)
    name = safe_name(variant)

    df.to_csv(TABLE_DIR / f"{name}_training_history.csv", index=False, encoding="utf-8-sig")

    if df.empty:
        return

    for key in ["train_total_loss", "train_ce_loss", "train_contrastive_loss", "validation_total_loss"]:
        if key not in df.columns:
            continue

        plt.figure(figsize=(7, 6))
        plt.plot(df["epoch"], df[key], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f"{variant}: {key.replace('_', ' ').title()}")
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
        plt.title(f"{variant}: Validation Learning Dynamics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{name}_validation_learning_dynamics.png", dpi=300)

    plt.close("all")


def plot_embedding_pca(embeddings, labels, variant):
    name = safe_name(variant)

    emb_df = pd.DataFrame(embeddings)
    emb_df["label"] = labels
    emb_df.to_csv(EMBED_DIR / f"{name}_test_embeddings.csv", index=False, encoding="utf-8-sig")

    try:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(embeddings)

        plt.figure(figsize=(7, 6))
        for cls in sorted(np.unique(labels)):
            idx = labels == cls
            plt.scatter(coords[idx, 0], coords[idx, 1], label=f"Class {cls}", alpha=0.7)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{variant}: Test Embeddings PCA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EMBED_DIR / f"{name}_test_embedding_pca.png", dpi=300)
        plt.close("all")
    except Exception as e:
        plt.close("all")
        log(f"PCA skipped for {variant}: {e}")


# ============================================================
# TRAINING
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


def train_variant(config, X_train, y_train, X_val, y_val, input_dim, n_classes):
    variant = config["variant"]

    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = AblationNetwork(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        n_classes=n_classes,
        rank=TENSOR_RANK,
        dropout=DROPOUT,
        use_tensorized_residual=config["use_tensorized_residual"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_f1 = -np.inf
    best_epoch = 0
    patience_counter = 0
    history = []

    log("=" * 80)
    log(f"Training ablation variant: {variant}")
    log("=" * 80)

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

            if config["use_contrastive_loss"]:
                con_loss = supervised_contrastive_loss(emb, yb, temperature=CONTRASTIVE_TEMPERATURE)
                total_loss = ce_loss + CONTRASTIVE_WEIGHT * con_loss
            else:
                con_loss = torch.tensor(0.0, device=DEVICE)
                total_loss = ce_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_n = xb.size(0)
            running_total += total_loss.item() * batch_n
            running_ce += ce_loss.item() * batch_n
            running_contrastive += con_loss.item() * batch_n
            n_seen += batch_n

        val_loss, y_val_true, y_val_pred, y_val_prob, _ = evaluate_network(model, val_loader, criterion)
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_prob, variant, "Neural_Model", "validation")

        record = {
            "variant": variant,
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
            f"{variant} | Epoch {epoch:03d} | "
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
            log(f"{variant}: Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = MODEL_DIR / f"{safe_name(variant)}_model.pt"
    torch.save(model.state_dict(), model_path)

    return model, history, best_epoch, model_path


def extract_outputs(model, X, y):
    loader = DataLoader(TabularDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    return evaluate_network(model, loader, criterion)


# ============================================================
# ENSEMBLE FUSION
# ============================================================

def get_embedding_models():
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


def validation_weights(model_names, val_probs, y_val):
    labels = sorted(np.unique(y_val))
    losses = []

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


def max_confidence_fusion(prob_dict):
    names = list(prob_dict.keys())
    stacked = np.stack([prob_dict[name] for name in names], axis=0)
    confidence = np.max(stacked, axis=2)
    best_idx = np.argmax(confidence, axis=0)

    fused = np.zeros_like(stacked[0])

    for i, idx in enumerate(best_idx):
        fused[i] = stacked[idx, i]

    return fused


def stacking_fusion(val_probs, test_probs, y_val):
    names = list(val_probs.keys())

    X_meta_val = np.hstack([val_probs[name] for name in names])
    X_meta_test = np.hstack([test_probs[name] for name in names])

    meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
    meta.fit(X_meta_val, y_val)

    return meta.predict_proba(X_meta_test), meta


def run_embedding_ensemble(variant, train_emb, y_train, val_emb, y_val, test_emb, y_test):
    models = get_embedding_models()

    val_probs = {}
    test_probs = {}
    records = []

    for model_name, model in models.items():
        log(f"{variant}: training embedding learner {model_name}")

        start = time.time()
        model.fit(train_emb, y_train)
        train_time = time.time() - start

        val_prob = model.predict_proba(val_emb)
        test_prob = model.predict_proba(test_emb)

        val_pred = np.argmax(val_prob, axis=1)
        test_pred = np.argmax(test_prob, axis=1)

        val_record = compute_metrics(y_val, val_pred, val_prob, variant, model_name, "validation")
        test_record = compute_metrics(y_test, test_pred, test_prob, variant, model_name, "test")

        val_record["training_time_seconds"] = float(train_time)
        test_record["training_time_seconds"] = float(train_time)

        records.append(val_record)
        records.append(test_record)

        val_probs[model_name] = val_prob
        test_probs[model_name] = test_prob

        with open(MODEL_DIR / f"{safe_name(variant)}_{safe_name(model_name)}.pkl", "wb") as f:
            pickle.dump(model, f)

    model_names = list(val_probs.keys())
    weights, losses = validation_weights(model_names, val_probs, y_val)

    weights_df = pd.DataFrame({
        "variant": variant,
        "base_learner": model_names,
        "validation_log_loss": [losses[name] for name in model_names],
        "fusion_weight": [weights[name] for name in model_names],
    })

    weights_df.to_csv(TABLE_DIR / f"{safe_name(variant)}_fusion_weights.csv", index=False, encoding="utf-8-sig")

    weighted_test_prob = np.zeros_like(next(iter(test_probs.values())))

    for name in model_names:
        weighted_test_prob += weights[name] * test_probs[name]

    average_test_prob = np.mean(list(test_probs.values()), axis=0)
    max_conf_test_prob = max_confidence_fusion(test_probs)
    stacking_test_prob, stacking_model = stacking_fusion(val_probs, test_probs, y_val)

    with open(MODEL_DIR / f"{safe_name(variant)}_stacking_meta_model.pkl", "wb") as f:
        pickle.dump(stacking_model, f)

    fusion_outputs = {
        "Weighted_Fusion": weighted_test_prob,
        "Average_Fusion": average_test_prob,
        "Max_Confidence_Fusion": max_conf_test_prob,
        "Stacking_Fusion": stacking_test_prob,
    }

    for fusion_name, prob in fusion_outputs.items():
        pred = np.argmax(prob, axis=1)

        rec = compute_metrics(y_test, pred, prob, variant, fusion_name, "test")
        records.append(rec)

        cm = confusion_matrix(y_test, pred, labels=sorted(np.unique(y_test)))

        pd.DataFrame(
            cm,
            index=[f"true_{x}" for x in sorted(np.unique(y_test))],
            columns=[f"pred_{x}" for x in sorted(np.unique(y_test))],
        ).to_csv(TABLE_DIR / f"{safe_name(variant)}_{fusion_name}_test_confusion_matrix.csv", encoding="utf-8-sig")

        plot_confusion_matrix(cm, sorted(np.unique(y_test)), variant, fusion_name)
        plot_roc(y_test, prob, variant, fusion_name)
        plot_pr(y_test, prob, variant, fusion_name)

    return records


# ============================================================
# SUMMARY TABLES
# ============================================================

def compare_with_prior_results(best_per_variant):
    rows = []

    if EXPERIMENT_1B_METRICS.exists():
        exp1b = pd.read_csv(EXPERIMENT_1B_METRICS)
        part = exp1b[
            (exp1b["dataset"] == DATASET_NAME)
            & (exp1b["split"] == "test")
        ]

        if not part.empty:
            best = part.sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False).iloc[0]
            rows.append({
                "source": "Experiment_1B_Best_Refined_Baseline",
                "variant_or_model": best["model"],
                "f1_weighted": best["f1_weighted"],
                "roc_auc": best["roc_auc"],
                "pr_auc": best["pr_auc"],
                "accuracy": best["accuracy"],
            })

    if EXPERIMENT_2B_METRICS.exists():
        exp2b = pd.read_csv(EXPERIMENT_2B_METRICS)
        part = exp2b[
            (exp2b["dataset"] == DATASET_NAME)
            & (exp2b["split"] == "test")
        ]

        if not part.empty:
            best = part.sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False).iloc[0]
            rows.append({
                "source": "Experiment_2B_Best_HCEF_Refined",
                "variant_or_model": best["model"],
                "f1_weighted": best["f1_weighted"],
                "roc_auc": best["roc_auc"],
                "pr_auc": best["pr_auc"],
                "accuracy": best["accuracy"],
            })

    if not best_per_variant.empty:
        best = best_per_variant.sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False).iloc[0]
        rows.append({
            "source": "Experiment_3_Best_Ablation",
            "variant_or_model": f"{best['variant']} | {best['model']}",
            "f1_weighted": best["f1_weighted"],
            "roc_auc": best["roc_auc"],
            "pr_auc": best["pr_auc"],
            "accuracy": best["accuracy"],
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(TABLE_DIR / "comparison_with_experiments_1b_2b.csv", index=False, encoding="utf-8-sig")

    return comparison_df


def build_delta_table(best_per_variant):
    if best_per_variant.empty:
        return pd.DataFrame()

    full = best_per_variant[best_per_variant["variant"] == "Full_HCEF"]

    if full.empty:
        ref = best_per_variant.sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False).iloc[0]
    else:
        ref = full.iloc[0]

    rows = []

    for _, row in best_per_variant.iterrows():
        rows.append({
            "variant": row["variant"],
            "best_model": row["model"],
            "f1_weighted": row["f1_weighted"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "accuracy": row["accuracy"],
            "reference_variant": ref["variant"],
            "reference_model": ref["model"],
            "delta_f1_vs_reference": row["f1_weighted"] - ref["f1_weighted"],
            "delta_auc_vs_reference": row["roc_auc"] - ref["roc_auc"],
            "delta_pr_auc_vs_reference": row["pr_auc"] - ref["pr_auc"],
        })

    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(TABLE_DIR / "formal_ablation_delta_table.csv", index=False, encoding="utf-8-sig")

    return delta_df


def plot_component_contribution(best_per_variant):
    if best_per_variant.empty:
        return

    plot_df = best_per_variant.sort_values(by="f1_weighted", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["variant"], plot_df["f1_weighted"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best Test F1-weighted")
    plt.title("Formal Ablation: Best Test F1 by Variant")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "formal_ablation_best_f1_by_variant.png", dpi=300)
    plt.close("all")

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["variant"], plot_df["roc_auc"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best Test ROC-AUC")
    plt.title("Formal Ablation: Best ROC-AUC by Variant")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "formal_ablation_best_auc_by_variant.png", dpi=300)
    plt.close("all")


# ============================================================
# DOCUMENTATION
# ============================================================

def write_reproducibility_files(train_df, val_df, test_df, best_per_variant, comparison_df, delta_df):
    readme = f"""# Experiment 3: Formal Ablation Study

## Title
Paper 3 IJOCTA Revision - Formal HCE-F Ablation Study on Refined Top-20 Features

## Description
This experiment quantifies the contribution of each HCE-F component after feature-space refinement. The experiment uses the strongest refined subset identified in Experiment 2B: `{DATASET_NAME}`.

The workflow uses tabular data only. No image simulation, image preprocessing, ResNet feature extraction, or computer-vision workflow is used.

## Dataset Information
Dataset:

`{DATASET_NAME}`

Split folder:

`{REFINED_SPLIT_DIR}`

Target column:

`{TARGET_COLUMN}`

Training samples: {len(train_df)}
Validation samples: {len(val_df)}
Test samples: {len(test_df)}

## Code Information
Script:

`Experiment_3_Formal_Ablation_Study.py`

Ablation variants:
- Full_HCEF
- No_Contrastive_Loss
- No_Tensorized_Residual
- Neural_Only
- Tensorized_Residual_Only
- Contrastive_Only

Fusion strategies:
- Weighted fusion
- Average fusion
- Max-confidence fusion
- Stacking fusion

## Usage Instructions
Run:

```powershell
cd "D:\\47\\471\\New Papers\\Paper 3 IJOCTA\\Sub\\Experiments\\Code"
python Experiment_3_Formal_Ablation_Study.py
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
1. Load the refined top-20 feature subset using its prepared train, validation, and test splits.
2. Standardize features based on the training partition only.
3. Train each ablation variant under the same split and random state.
4. Evaluate neural-only outputs.
5. Extract latent embeddings.
6. Train embedding-level base learners when fusion is enabled.
7. Compare fusion strategies.
8. Save all metrics, figures, confusion matrices, ROC curves, precision-recall curves, embedding PCA plots, training curves, and summary reports.

## License and Contribution Guidelines
Add repository license and contribution rules before public release.
"""

    methods_text = """# Methods Text: Formal Ablation Study and Component Contribution

A formal ablation study was conducted using the refined top-20 feature subset, which showed the strongest HCE-F learning behavior in the refined-subset evaluation. All ablation variants used the same stratified training, validation, and test partitions to ensure that performance differences reflected component-level effects rather than data-split variation.

The evaluated variants included the full HCE-F model, a version without supervised contrastive loss, a version without tensorized residual encoding, a neural-only version without ensemble fusion, a tensorized-residual-only version, and a contrastive-only version. For variants using embedding-level fusion, multiple probabilistic learners were trained on the learned latent embeddings. Fusion strategies included validation-weighted fusion, average fusion, max-confidence fusion, and stacking fusion.

Performance was assessed using accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, ROC-AUC, PR-AUC, confusion matrices, ROC curves, precision-recall curves, embedding PCA visualizations, training-loss curves, validation learning dynamics, runtime, and parameter count. This design directly evaluates the contribution of tensorized residual encoding, supervised contrastive representation learning, and probabilistic ensemble fusion. No image preprocessing, simulated image data, convolutional feature extraction, or ResNet-based workflow was used.
"""

    limitations = """# Limitations Note for Conclusion

The formal ablation study identifies the relative contribution of HCE-F components under the refined top-20 feature setting. However, the ablation is still based on the internal augmented dataset and should not be interpreted as external validation. If component effects are modest, this may reflect intrinsic limitations in the target-feature relationship rather than failure of a specific module. External validation and robustness testing remain necessary.
"""

    (README_DIR / "README_Experiment_3_Formal_Ablation_Study.md").write_text(readme, encoding="utf-8")
    (OUTPUT_DIR / "METHODS_TEXT_Experiment_3_Formal_Ablation_Study.md").write_text(methods_text, encoding="utf-8")
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
    (LOG_DIR / "experiment_3_ablation_console_log.txt").write_text("", encoding="utf-8")

    log("=" * 80)
    log("Experiment 3: Formal Ablation Study")
    log("=" * 80)
    log(f"Dataset: {DATASET_NAME}")
    log(f"Device: {DEVICE}")

    start_time = time.time()

    train_df, val_df, test_df = load_splits()

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

    log(f"Train shape: {train_df.shape}")
    log(f"Validation shape: {val_df.shape}")
    log(f"Test shape: {test_df.shape}")
    log(f"Input features: {input_dim}")
    log(f"Classes: {labels}")

    with open(MODEL_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    variants = [
        {
            "variant": "Full_HCEF",
            "use_tensorized_residual": True,
            "use_contrastive_loss": True,
            "use_embedding_ensemble": True,
        },
        {
            "variant": "No_Contrastive_Loss",
            "use_tensorized_residual": True,
            "use_contrastive_loss": False,
            "use_embedding_ensemble": True,
        },
        {
            "variant": "No_Tensorized_Residual",
            "use_tensorized_residual": False,
            "use_contrastive_loss": True,
            "use_embedding_ensemble": True,
        },
        {
            "variant": "Neural_Only",
            "use_tensorized_residual": True,
            "use_contrastive_loss": True,
            "use_embedding_ensemble": False,
        },
        {
            "variant": "Tensorized_Residual_Only",
            "use_tensorized_residual": True,
            "use_contrastive_loss": False,
            "use_embedding_ensemble": False,
        },
        {
            "variant": "Contrastive_Only",
            "use_tensorized_residual": False,
            "use_contrastive_loss": True,
            "use_embedding_ensemble": False,
        },
    ]

    all_records = []
    all_histories = []
    best_epochs = {}
    parameter_counts = {}
    model_sizes = {}

    for config in variants:
        variant = config["variant"]

        variant_start = time.time()

        model, history, best_epoch, model_path = train_variant(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
            input_dim=input_dim,
            n_classes=n_classes,
        )

        variant_training_time = time.time() - variant_start

        best_epochs[variant] = int(best_epoch)
        parameter_counts[variant] = int(sum(p.numel() for p in model.parameters()))
        model_sizes[variant] = float(model_path.stat().st_size / (1024 * 1024)) if model_path.exists() else np.nan

        plot_training_history(history, variant)
        all_histories.extend(history)

        train_loss, y_train_true, y_train_pred, y_train_prob, train_emb = extract_outputs(model, X_train, y_train)
        val_loss, y_val_true, y_val_pred, y_val_prob, val_emb = extract_outputs(model, X_val, y_val)
        test_loss, y_test_true, y_test_pred, y_test_prob, test_emb = extract_outputs(model, X_test, y_test)

        plot_embedding_pca(test_emb, y_test_true, variant)

        for split_name, y_true, y_pred, y_prob in [
            ("train", y_train_true, y_train_pred, y_train_prob),
            ("validation", y_val_true, y_val_pred, y_val_prob),
            ("test", y_test_true, y_test_pred, y_test_prob),
        ]:
            rec = compute_metrics(y_true, y_pred, y_prob, variant, "Neural_Model", split_name)
            rec["best_epoch"] = int(best_epoch)
            rec["variant_training_time_seconds"] = float(variant_training_time)
            rec["parameter_count"] = int(parameter_counts[variant])
            rec["model_size_mb"] = float(model_sizes[variant])
            all_records.append(rec)

            if split_name == "test":
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                pd.DataFrame(
                    cm,
                    index=[f"true_{x}" for x in labels],
                    columns=[f"pred_{x}" for x in labels],
                ).to_csv(TABLE_DIR / f"{safe_name(variant)}_Neural_Model_test_confusion_matrix.csv", encoding="utf-8-sig")

                plot_confusion_matrix(cm, labels, variant, "Neural_Model")
                plot_roc(y_true, y_prob, variant, "Neural_Model")
                plot_pr(y_true, y_prob, variant, "Neural_Model")

                with open(METRIC_DIR / f"{safe_name(variant)}_Neural_Model_test_classification_report.json", "w", encoding="utf-8") as f:
                    json.dump(classification_report(y_true, y_pred, output_dict=True, zero_division=0), f, indent=4, ensure_ascii=False)

        if config["use_embedding_ensemble"]:
            ensemble_start = time.time()
            ensemble_records = run_embedding_ensemble(
                variant,
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
                rec["variant_training_time_seconds"] = float(variant_training_time)
                rec["ensemble_training_time_seconds"] = float(ensemble_time)
                rec["parameter_count"] = int(parameter_counts[variant])
                rec["model_size_mb"] = float(model_sizes[variant])
                all_records.append(rec)

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(TABLE_DIR / "formal_ablation_metrics.csv", index=False, encoding="utf-8-sig")

    with open(METRIC_DIR / "formal_ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False, default=str)

    pd.DataFrame(all_histories).to_csv(TABLE_DIR / "formal_ablation_training_histories.csv", index=False, encoding="utf-8-sig")

    test_df_metrics = results_df[results_df["split"] == "test"].copy()

    best_per_variant = (
        test_df_metrics
        .sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False)
        .groupby("variant", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_per_variant.to_csv(TABLE_DIR / "best_test_model_per_ablation_variant.csv", index=False, encoding="utf-8-sig")

    full_ranking = (
        test_df_metrics
        .sort_values(by=["f1_weighted", "roc_auc", "pr_auc"], ascending=False)
        .reset_index(drop=True)
    )

    full_ranking.to_csv(TABLE_DIR / "formal_ablation_test_model_ranking.csv", index=False, encoding="utf-8-sig")

    delta_df = build_delta_table(best_per_variant)
    comparison_df = compare_with_prior_results(best_per_variant)
    plot_component_contribution(best_per_variant)

    total_runtime = time.time() - start_time

    summary = {
        "experiment": "Experiment 3 - Formal Ablation Study",
        "generated_at": str(datetime.now()),
        "dataset": DATASET_NAME,
        "target": TARGET_COLUMN,
        "device": str(DEVICE),
        "train_shape": list(train_df.shape),
        "validation_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "input_dim": int(input_dim),
        "n_classes": int(n_classes),
        "best_epochs": best_epochs,
        "parameter_counts": parameter_counts,
        "model_sizes_mb": model_sizes,
        "best_per_variant": best_per_variant.to_dict(orient="records"),
        "delta_table": delta_df.to_dict(orient="records") if not delta_df.empty else [],
        "comparison_with_experiments_1b_2b": comparison_df.to_dict(orient="records") if not comparison_df.empty else [],
        "best_overall_test_result": full_ranking.iloc[0].to_dict() if not full_ranking.empty else None,
        "total_runtime_seconds": float(total_runtime),
    }

    with open(OUTPUT_DIR / "formal_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)

    with open(OUTPUT_DIR / "formal_ablation_summary.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 3 Formal Ablation Study Summary\n\n")
        f.write(f"Generated at: `{datetime.now()}`\n\n")
        f.write(f"Dataset: `{DATASET_NAME}`\n\n")
        f.write(f"Target: `{TARGET_COLUMN}`\n\n")
        f.write(f"Device: `{DEVICE}`\n\n")
        f.write(f"Input features: `{input_dim}`\n\n")

        f.write("## Best Test Model Per Ablation Variant\n\n")
        for _, row in best_per_variant.iterrows():
            f.write(
                f"- {row['variant']} | {row['model']} | "
                f"F1-weighted={row['f1_weighted']:.6f} | "
                f"ROC-AUC={row['roc_auc']:.6f} | "
                f"PR-AUC={row['pr_auc']:.6f}\n"
            )

        f.write("\n## Delta Table\n\n")
        for _, row in delta_df.iterrows():
            f.write(
                f"- {row['variant']} | {row['best_model']} | "
                f"Delta F1={row['delta_f1_vs_reference']:.6f} | "
                f"Delta ROC-AUC={row['delta_auc_vs_reference']:.6f} | "
                f"Delta PR-AUC={row['delta_pr_auc_vs_reference']:.6f}\n"
            )

        f.write("\n## Comparison With Previous Experiments\n\n")
        for _, row in comparison_df.iterrows():
            f.write(
                f"- {row['source']} | {row['variant_or_model']} | "
                f"F1-weighted={row['f1_weighted']:.6f} | "
                f"ROC-AUC={row['roc_auc']:.6f} | "
                f"PR-AUC={row['pr_auc']:.6f}\n"
            )

        if not full_ranking.empty:
            row = full_ranking.iloc[0]
            f.write("\n## Best Overall Ablation Result\n\n")
            f.write(
                f"- Variant: {row['variant']}\n"
                f"- Model: {row['model']}\n"
                f"- Accuracy: {row['accuracy']}\n"
                f"- F1-weighted: {row['f1_weighted']}\n"
                f"- ROC-AUC: {row['roc_auc']}\n"
                f"- PR-AUC: {row['pr_auc']}\n"
            )

        f.write(f"\nTotal runtime seconds: `{total_runtime:.4f}`\n")

    write_reproducibility_files(train_df, val_df, test_df, best_per_variant, comparison_df, delta_df)

    log("=" * 80)
    log("Experiment 3 formal ablation completed successfully.")
    log("=" * 80)
    log(f"Metrics: {TABLE_DIR / 'formal_ablation_metrics.csv'}")
    log(f"Best per variant: {TABLE_DIR / 'best_test_model_per_ablation_variant.csv'}")
    log(f"Delta table: {TABLE_DIR / 'formal_ablation_delta_table.csv'}")
    log(f"Summary: {OUTPUT_DIR / 'formal_ablation_summary.md'}")
    log(f"README: {README_DIR / 'README_Experiment_3_Formal_Ablation_Study.md'}")
    log(f"Total runtime: {total_runtime:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Experiment 3 formal ablation failed. See error log:", flush=True)
        print(LOG_DIR / "experiment_3_ablation_error_log.txt", flush=True)
        save_error(e)
        raise
