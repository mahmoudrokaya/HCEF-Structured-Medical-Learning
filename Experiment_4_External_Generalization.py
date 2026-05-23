"""
Experiment_4_External_Generalization.py

Experiment 4:
    External Cross-Dataset Generalization and Robustness Evaluation

Purpose:
    Evaluate the reproducibility and robustness of the finalized tabular HCE-F
    protocol on independent external medical tabular datasets.

External datasets:
    - external_breast_cancer
    - external_heart_disease

Important methodological note:
    The external datasets have different feature spaces and different clinical
    targets from the internal dataset. Therefore, this script evaluates
    protocol-level generalization, not direct feature-transfer from the internal
    model.

Protocol:
    - Load prepared external train/validation/test splits.
    - Train classical baselines.
    - Train Full HCE-F.
    - Train Contrastive-Only HCE-F, motivated by Experiment 3.
    - Compare best HCE-F result with best baseline per external dataset.
    - Save metrics, figures, summaries, README, methods text, and limitations.

Tabular data only.
No image pipeline.
No ResNet workflow.
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

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, auc
)

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT_DIR = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub")

SPLIT_ROOT = ROOT_DIR / "Data" / "Splits" / "Modeling_Ready"
PROCESSED_DIR = ROOT_DIR / "Data" / "Processed" / "Modeling_Ready"

OUTPUT_DIR = ROOT_DIR / "Experiments" / "Results" / "Experiment_4_External_Generalization"
TABLE_DIR = OUTPUT_DIR / "Tables"
FIGURE_DIR = OUTPUT_DIR / "Figures"
METRIC_DIR = OUTPUT_DIR / "Metrics"
MODEL_DIR = OUTPUT_DIR / "Models"
LOG_DIR = OUTPUT_DIR / "Logs"
CM_DIR = OUTPUT_DIR / "Confusion_Matrices"
ROC_DIR = OUTPUT_DIR / "ROC_Curves"
PR_DIR = OUTPUT_DIR / "Precision_Recall_Curves"
EMBED_DIR = OUTPUT_DIR / "Embeddings"
README_DIR = OUTPUT_DIR / "Reproducibility"

for d in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, METRIC_DIR, MODEL_DIR, LOG_DIR, CM_DIR, ROC_DIR, PR_DIR, EMBED_DIR, README_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
EMBED_DIM = 32
TENSOR_RANK = 16
DROPOUT = 0.15
CONTRASTIVE_TEMPERATURE = 0.20
CONTRASTIVE_WEIGHT = 0.25
N_JOBS = 1

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXTERNAL_DATASETS = [
    {
        "dataset": "external_breast_cancer",
        "target": "diagnosis",
        "target_candidates": ["diagnosis", "target", "Target", "class", "Class", "label", "Label", "y"],
        "source_note": "Breast Cancer Wisconsin Diagnostic Dataset. Add original DOI/URL in manuscript Materials & Methods.",
    },
    {
        "dataset": "external_heart_disease",
        "target": "num",
        "target_candidates": ["num", "target", "Target", "class", "Class", "label", "Label", "y"],
        "source_note": "Heart Disease dataset. Add original DOI/URL in manuscript Materials & Methods.",
    },
]


# ============================================================
# UTILITIES
# ============================================================

def log(msg):
    print(msg, flush=True)
    with open(LOG_DIR / "experiment_4_console_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def save_error(e):
    with open(LOG_DIR / "experiment_4_error_log.txt", "w", encoding="utf-8") as f:
        f.write(str(e) + "\n\n")
        f.write(traceback.format_exc())


def safe_name(x):
    return str(x).replace(" ", "_").replace("/", "_").replace("\\", "_").replace("+", "plus")


def resolve_target_column(train_df, preferred_target, target_candidates=None, dataset_name="dataset"):
    """
    Resolve target column robustly.

    Prepared modeling datasets may rename original targets such as diagnosis/num
    into a generic column such as target. This function prevents false failure
    while still avoiding unsafe guessing where possible.
    """
    if target_candidates is None:
        target_candidates = []

    candidates = [preferred_target] + list(target_candidates) + [
        "target", "Target", "TARGET",
        "label", "Label", "LABEL",
        "class", "Class", "CLASS",
        "y", "Y"
    ]

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for c in candidates:
        if c in train_df.columns:
            return c

    # Safe fallback: if exactly one column is binary/small-cardinality and appears last,
    # use the last column only after recording it in the logs.
    last_col = train_df.columns[-1]
    unique_count = train_df[last_col].nunique(dropna=True)

    if unique_count <= max(10, int(0.05 * len(train_df))):
        log(
            f"WARNING: Target column for {dataset_name} was not found by name. "
            f"Using last column '{last_col}' as target because it has {unique_count} unique values."
        )
        return last_col

    raise ValueError(
        f"Could not resolve target column for {dataset_name}. "
        f"Preferred target='{preferred_target}'. "
        f"Available columns={list(train_df.columns)}"
    )


def load_splits(dataset_name, target_col, target_candidates=None):
    split_dir = SPLIT_ROOT / dataset_name
    train_path = split_dir / "train.csv"
    val_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        resolved_target = resolve_target_column(train_df, target_col, target_candidates, dataset_name)
        return train_df, val_df, test_df, resolved_target, "prepared_split_files"

    full_file = PROCESSED_DIR / f"{dataset_name}_modeling_ready.csv"
    if not full_file.exists():
        raise FileNotFoundError(f"Missing splits and full file for {dataset_name}")

    from sklearn.model_selection import train_test_split
    df = pd.read_csv(full_file)

    resolved_target = resolve_target_column(df, target_col, target_candidates, dataset_name)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df[resolved_target]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df[resolved_target]
    )
    return train_df, val_df, test_df, resolved_target, "fallback_split_from_modeling_ready_file"


def prepare_xy(train_df, val_df, test_df, target_col):
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {target_col} not found.")

    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    for c in feature_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    target_encoder = LabelEncoder()
    target_encoder.fit(train_df[target_col].astype(str))

    y_train = target_encoder.transform(train_df[target_col].astype(str))
    y_val = target_encoder.transform(val_df[target_col].astype(str))
    y_test = target_encoder.transform(test_df[target_col].astype(str))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, target_encoder, scaler


# ============================================================
# METRICS AND PLOTS
# ============================================================

def compute_metrics(y_true, y_pred, y_score, dataset, model, split):
    labels = sorted(np.unique(y_true))
    n_classes = len(labels)

    out = {
        "dataset": dataset,
        "model": model,
        "split": split,
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


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T
        return scores
    return None


def plot_cm(y_true, y_pred, dataset, model, split):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    name = f"{safe_name(dataset)}_{safe_name(model)}_{split}"

    pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(
        TABLE_DIR / f"{name}_confusion_matrix.csv", encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title(f"{dataset} - {model} - {split} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
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


def plot_roc_pr(y_true, y_score, dataset, model, split):
    if y_score is None:
        return

    labels = sorted(np.unique(y_true))
    n_classes = len(labels)
    name = f"{safe_name(dataset)}_{safe_name(model)}_{split}"

    try:
        plt.figure(figsize=(7, 6))
        if n_classes == 2:
            pos = y_score[:, 1]
            fpr, tpr, _ = roc_curve(y_true, pos)
            plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
        else:
            y_bin = label_binarize(y_true, classes=labels)
            for i, lab in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                plt.plot(fpr, tpr, label=f"Class {lab}, AUC={auc(fpr,tpr):.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset} - {model} - {split} ROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"{name}_roc_curve.png", dpi=300)
        plt.close("all")
    except Exception:
        plt.close("all")

    try:
        plt.figure(figsize=(7, 6))
        if n_classes == 2:
            pos = y_score[:, 1]
            p, r, _ = precision_recall_curve(y_true, pos)
            plt.plot(r, p, label=f"PR-AUC={average_precision_score(y_true,pos):.4f}")
        else:
            y_bin = label_binarize(y_true, classes=labels)
            for i, lab in enumerate(labels):
                p, r, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                plt.plot(r, p, label=f"Class {lab}, PR-AUC={average_precision_score(y_bin[:, i], y_score[:, i]):.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{dataset} - {model} - {split} PR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PR_DIR / f"{name}_precision_recall_curve.png", dpi=300)
        plt.close("all")
    except Exception:
        plt.close("all")


def plot_history(history, dataset, model):
    df = pd.DataFrame(history)
    name = f"{safe_name(dataset)}_{safe_name(model)}"
    df.to_csv(TABLE_DIR / f"{name}_training_history.csv", index=False, encoding="utf-8-sig")

    if df.empty:
        return

    for key in ["train_total_loss", "train_ce_loss", "train_contrastive_loss", "validation_total_loss"]:
        if key in df.columns:
            plt.figure(figsize=(7, 6))
            plt.plot(df["epoch"], df[key], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(key)
            plt.title(f"{dataset} - {model}: {key}")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"{name}_{key}.png", dpi=300)
            plt.close("all")

    plt.figure(figsize=(7, 6))
    if "validation_f1_weighted" in df.columns:
        plt.plot(df["epoch"], df["validation_f1_weighted"], marker="o", label="F1-weighted")
    if "validation_roc_auc" in df.columns:
        plt.plot(df["epoch"], df["validation_roc_auc"], marker="o", label="ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{dataset} - {model}: Validation Learning Dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}_validation_learning_dynamics.png", dpi=300)
    plt.close("all")


def plot_embedding_pca(emb, y, dataset, model):
    name = f"{safe_name(dataset)}_{safe_name(model)}"
    pd.DataFrame(emb).assign(label=y).to_csv(EMBED_DIR / f"{name}_test_embeddings.csv", index=False, encoding="utf-8-sig")
    try:
        coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(emb)
        plt.figure(figsize=(7, 6))
        for cls in sorted(np.unique(y)):
            idx = y == cls
            plt.scatter(coords[idx, 0], coords[idx, 1], label=f"Class {cls}", alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{dataset} - {model}: Test Embeddings PCA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EMBED_DIR / f"{name}_test_embedding_pca.png", dpi=300)
        plt.close("all")
    except Exception:
        plt.close("all")


# ============================================================
# MODELS
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class TensorizedResidualBlock(nn.Module):
    def __init__(self, dim, rank=16, dropout=0.15, alpha=0.60):
        super().__init__()
        self.alpha = alpha
        self.linear = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout))
        self.u = nn.Linear(dim, rank)
        self.v = nn.Linear(dim, rank)
        self.project = nn.Linear(rank, dim)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        lin = self.linear(x)
        inter = torch.tanh(self.u(x) * self.v(x))
        ten = self.project(inter)
        return F.relu(self.norm(self.alpha * x + (1 - self.alpha) * (lin + ten)))


class PlainResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15, alpha=0.60):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return F.relu(self.alpha * x + (1 - self.alpha) * self.net(x))


class ExternalHCEF(nn.Module):
    def __init__(self, input_dim, n_classes, use_tensorized=True):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT))
        block = TensorizedResidualBlock if use_tensorized else PlainResidualBlock
        self.b1 = block(HIDDEN_DIM, TENSOR_RANK, DROPOUT) if use_tensorized else block(HIDDEN_DIM, DROPOUT)
        self.b2 = block(HIDDEN_DIM, TENSOR_RANK, DROPOUT) if use_tensorized else block(HIDDEN_DIM, DROPOUT)
        self.proj = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, EMBED_DIM))
        self.cls = nn.Linear(EMBED_DIM, n_classes)

    def forward(self, x, return_embedding=False):
        h = self.input(x)
        h = self.b1(h)
        h = self.b2(h)
        z = F.normalize(self.proj(h), p=2, dim=1)
        logits = self.cls(z)
        if return_embedding:
            return logits, z
        return logits


def supervised_contrastive_loss(emb, labels, temperature=0.20):
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(emb.device)
    sim = torch.matmul(emb, emb.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=emb.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    pos = mask.sum(1)
    valid = pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=emb.device)
    return -((mask * log_prob).sum(1)[valid] / pos[valid]).mean()


def baseline_models():
    return {
        "Logistic_Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Support_Vector_Machine": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random_Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "K_Nearest_Neighbors": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]),
        "Multi_Layer_Perceptron": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, early_stopping=True, random_state=RANDOM_STATE))]),
    }


def run_baselines(dataset, X_train, y_train, X_val, y_val, X_test, y_test):
    records = []
    for model_name, model in baseline_models().items():
        log(f"{dataset}: training baseline {model_name}")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        for split, Xs, ys in [("validation", X_val, y_val), ("test", X_test, y_test)]:
            pred = model.predict(Xs)
            score = get_scores(model, Xs)
            rec = compute_metrics(ys, pred, score, dataset, model_name, split)
            rec["training_time_seconds"] = float(train_time)
            rec["model_family"] = "Classical_Baseline"
            records.append(rec)
            if split == "test":
                plot_cm(ys, pred, dataset, model_name, split)
                plot_roc_pr(ys, score, dataset, model_name, split)
                with open(METRIC_DIR / f"{safe_name(dataset)}_{safe_name(model_name)}_test_classification_report.json", "w", encoding="utf-8") as f:
                    json.dump(classification_report(ys, pred, output_dict=True, zero_division=0), f, indent=4)
    return records


def evaluate_net(model, X, y):
    loader = DataLoader(TabularDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0.0
    yy, pp, pr, ee = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, emb = model(xb, return_embedding=True)
            loss = criterion(logits, yb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            total += loss.item() * xb.size(0)
            yy.extend(yb.cpu().numpy())
            pp.append(prob)
            pr.extend(pred)
            ee.append(emb.cpu().numpy())
    return total / len(yy), np.array(yy), np.array(pr), np.vstack(pp), np.vstack(ee)


def train_hcef(dataset, model_name, X_train, y_train, X_val, y_val, n_classes, use_tensorized=True, use_contrastive=True):
    model = ExternalHCEF(X_train.shape[1], n_classes, use_tensorized=use_tensorized).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    best_state, best_f1, best_epoch, patience = None, -np.inf, 0, 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = total_ce = total_con = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, emb = model(xb, return_embedding=True)
            ce_loss = ce(logits, yb)
            con_loss = supervised_contrastive_loss(emb, yb, CONTRASTIVE_TEMPERATURE) if use_contrastive else torch.tensor(0.0, device=DEVICE)
            loss = ce_loss + CONTRASTIVE_WEIGHT * con_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_ce += ce_loss.item() * bs
            total_con += con_loss.item() * bs
            n += bs

        val_loss, yv, predv, probv, _ = evaluate_net(model, X_val, y_val)
        met = compute_metrics(yv, predv, probv, dataset, model_name, "validation")
        row = {
            "dataset": dataset,
            "model": model_name,
            "epoch": epoch,
            "train_total_loss": total_loss / n,
            "train_ce_loss": total_ce / n,
            "train_contrastive_loss": total_con / n,
            "validation_total_loss": val_loss,
            "validation_f1_weighted": met["f1_weighted"],
            "validation_roc_auc": met["roc_auc"],
            "validation_pr_auc": met["pr_auc"],
        }
        history.append(row)
        log(f"{dataset} | {model_name} | epoch {epoch:03d} | val_f1={row['validation_f1_weighted']:.4f} | val_auc={row['validation_roc_auc']:.4f}")

        if row["validation_f1_weighted"] > best_f1:
            best_f1, best_epoch, patience = row["validation_f1_weighted"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            log(f"{dataset} | {model_name}: early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    model_path = MODEL_DIR / f"{safe_name(dataset)}_{safe_name(model_name)}.pt"
    torch.save(model.state_dict(), model_path)
    plot_history(history, dataset, model_name)
    return model, history, best_epoch, model_path


def run_hcef(dataset, X_train, y_train, X_val, y_val, X_test, y_test, n_classes):
    records = []
    variants = [
        ("External_Full_HCEF", True, True),
        ("External_Contrastive_Only", False, True),
    ]

    for name, use_tensorized, use_contrastive in variants:
        log("=" * 80)
        log(f"{dataset}: training {name}")
        log("=" * 80)
        start = time.time()
        model, hist, best_epoch, model_path = train_hcef(dataset, name, X_train, y_train, X_val, y_val, n_classes, use_tensorized, use_contrastive)
        train_time = time.time() - start
        param_count = sum(p.numel() for p in model.parameters())
        size_mb = model_path.stat().st_size / (1024 * 1024)

        for split, Xs, ys in [("validation", X_val, y_val), ("test", X_test, y_test)]:
            loss, yt, pred, prob, emb = evaluate_net(model, Xs, ys)
            rec = compute_metrics(yt, pred, prob, dataset, name, split)
            rec["training_time_seconds"] = float(train_time)
            rec["best_epoch"] = int(best_epoch)
            rec["parameter_count"] = int(param_count)
            rec["model_size_mb"] = float(size_mb)
            rec["model_family"] = "HCEF_Protocol"
            records.append(rec)
            if split == "test":
                plot_cm(yt, pred, dataset, name, split)
                plot_roc_pr(yt, prob, dataset, name, split)
                plot_embedding_pca(emb, yt, dataset, name)
                with open(METRIC_DIR / f"{safe_name(dataset)}_{safe_name(name)}_test_classification_report.json", "w", encoding="utf-8") as f:
                    json.dump(classification_report(yt, pred, output_dict=True, zero_division=0), f, indent=4)
    return records


def build_tables(metrics_df):
    test = metrics_df[metrics_df["split"] == "test"].copy()
    best_all = test.sort_values(["f1_weighted", "roc_auc", "pr_auc"], ascending=False).groupby("dataset", as_index=False).head(1).reset_index(drop=True)
    best_hcef = test[test["model_family"] == "HCEF_Protocol"].sort_values(["f1_weighted", "roc_auc", "pr_auc"], ascending=False).groupby("dataset", as_index=False).head(1).reset_index(drop=True)
    best_base = test[test["model_family"] == "Classical_Baseline"].sort_values(["f1_weighted", "roc_auc", "pr_auc"], ascending=False).groupby("dataset", as_index=False).head(1).reset_index(drop=True)

    best_all.to_csv(TABLE_DIR / "best_external_model_per_dataset.csv", index=False, encoding="utf-8-sig")
    best_hcef.to_csv(TABLE_DIR / "best_hcef_external_model_per_dataset.csv", index=False, encoding="utf-8-sig")
    best_base.to_csv(TABLE_DIR / "best_external_baseline_per_dataset.csv", index=False, encoding="utf-8-sig")

    rows = []
    for ds in sorted(test["dataset"].unique()):
        h = best_hcef[best_hcef["dataset"] == ds]
        b = best_base[best_base["dataset"] == ds]
        if not h.empty and not b.empty:
            h = h.iloc[0]
            b = b.iloc[0]
            rows.append({
                "dataset": ds,
                "best_external_baseline_model": b["model"],
                "best_hcef_model": h["model"],
                "baseline_f1_weighted": b["f1_weighted"],
                "hcef_f1_weighted": h["f1_weighted"],
                "delta_f1_hcef_minus_baseline": h["f1_weighted"] - b["f1_weighted"],
                "baseline_roc_auc": b["roc_auc"],
                "hcef_roc_auc": h["roc_auc"],
                "delta_roc_auc_hcef_minus_baseline": h["roc_auc"] - b["roc_auc"],
                "baseline_pr_auc": b["pr_auc"],
                "hcef_pr_auc": h["pr_auc"],
                "delta_pr_auc_hcef_minus_baseline": h["pr_auc"] - b["pr_auc"],
            })
    delta = pd.DataFrame(rows)
    delta.to_csv(TABLE_DIR / "external_hcef_vs_baseline_delta_table.csv", index=False, encoding="utf-8-sig")
    return best_all, best_hcef, best_base, delta


def write_docs(dataset_summaries):
    readme = f"""# Experiment 4: External Cross-Dataset Generalization

Generated: `{datetime.now()}`

## Purpose
Evaluate protocol-level generalization of the tabular HCE-F workflow on independent external medical tabular datasets.

## Important Methodological Note
The external datasets have different feature spaces and targets. Therefore, this experiment does not claim direct model transfer. It evaluates whether the same tabular HCE-F protocol remains reproducible and competitive across heterogeneous external datasets.

## External Datasets
"""
    for d in dataset_summaries:
        readme += f"""
### {d['dataset']}
- Target: `{d['target']}`
- Features: `{d['n_features']}`
- Classes: `{d['n_classes']}`
- Train: `{d['train_shape']}`
- Validation: `{d['validation_shape']}`
- Test: `{d['test_shape']}`
- Source note: {d['source_note']}
"""

    readme += """
## Models
- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- Extra Trees
- KNN
- MLP
- External Full HCE-F
- External Contrastive-Only HCE-F

## Usage
```powershell
cd "D:\\47\\471\\New Papers\\Paper 3 IJOCTA\\Sub\\Experiments\\Code"
python Experiment_4_External_Generalization.py
```

## Requirements
```powershell
pip install numpy pandas matplotlib scikit-learn torch
```

## Third-Party Data Requirement
The manuscript Materials & Methods section must include the original DOI/URL for each external dataset.
"""

    methods = """# Methods Text: External Cross-Dataset Generalization

External validation was conducted using independent third-party medical tabular datasets. Because the external datasets have heterogeneous feature spaces and clinical targets, the experiment evaluated protocol-level generalization rather than direct feature-transfer from the internal model. The same preprocessing logic, baseline models, HCE-F training objective, and evaluation metrics were applied independently to each dataset.

For each external dataset, classical baseline models and two HCE-F variants were trained using the prepared training split and evaluated on validation and test splits. The HCE-F variants included the full tensorized residual model with supervised contrastive learning and a contrastive-only variant motivated by the formal ablation study. Evaluation included accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, ROC-AUC, PR-AUC, confusion matrices, ROC curves, precision-recall curves, embedding PCA visualizations, runtime, parameter count, and model size. No image preprocessing, simulated image data, convolutional feature extraction, or ResNet-based workflow was used.
"""

    limitations = """# Limitations Note for Conclusion

This experiment evaluates protocol-level external generalization across heterogeneous medical tabular datasets. Since the feature spaces and targets differ between datasets, the results should not be interpreted as direct model transfer. Final conclusions should distinguish protocol robustness from direct cross-feature transfer.
"""

    (README_DIR / "README_Experiment_4_External_Generalization.md").write_text(readme, encoding="utf-8")
    (OUTPUT_DIR / "METHODS_TEXT_Experiment_4_External_Generalization.md").write_text(methods, encoding="utf-8")
    (OUTPUT_DIR / "LIMITATIONS_NOTE_FOR_CONCLUSION.md").write_text(limitations, encoding="utf-8")

    env = {
        "generated_at": str(datetime.now()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "random_state": RANDOM_STATE,
        "external_datasets": EXTERNAL_DATASETS,
    }
    with open(LOG_DIR / "environment_report.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=4, ensure_ascii=False)


def main():
    (LOG_DIR / "experiment_4_console_log.txt").write_text("", encoding="utf-8")
    log("=" * 80)
    log("Experiment 4: External Cross-Dataset Generalization")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    start_all = time.time()
    all_records = []
    summaries = []

    for cfg in EXTERNAL_DATASETS:
        dataset = cfg["dataset"]
        target = cfg["target"]
        target_candidates = cfg.get("target_candidates", [])

        log("\n" + "=" * 80)
        log(f"Evaluating external dataset: {dataset}")
        log("=" * 80)

        train_df, val_df, test_df, resolved_target, source_mode = load_splits(dataset, target, target_candidates)
        log(f"Resolved target column for {dataset}: {resolved_target}")
        Xtr_raw, Xv_raw, Xte_raw, Xtr, Xv, Xte, ytr, yv, yte, enc, scaler = prepare_xy(train_df, val_df, test_df, resolved_target)

        labels = sorted(np.unique(np.concatenate([ytr, yv, yte])))
        n_classes = len(labels)

        summaries.append({
            "dataset": dataset,
            "target": resolved_target,
            "original_configured_target": target,
            "source_mode": source_mode,
            "train_shape": list(train_df.shape),
            "validation_shape": list(val_df.shape),
            "test_shape": list(test_df.shape),
            "n_features": int(Xtr.shape[1]),
            "n_classes": int(n_classes),
            "target_classes": list(map(str, enc.classes_)),
            "source_note": cfg["source_note"],
        })

        log(f"Source mode: {source_mode}")
        log(f"Train shape: {train_df.shape}")
        log(f"Validation shape: {val_df.shape}")
        log(f"Test shape: {test_df.shape}")
        log(f"Features: {Xtr.shape[1]}")
        log(f"Classes: {labels}")

        with open(MODEL_DIR / f"{safe_name(dataset)}_feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(MODEL_DIR / f"{safe_name(dataset)}_target_encoder.pkl", "wb") as f:
            pickle.dump(enc, f)

        all_records.extend(run_baselines(dataset, Xtr_raw, ytr, Xv_raw, yv, Xte_raw, yte))
        all_records.extend(run_hcef(dataset, Xtr, ytr, Xv, yv, Xte, yte, n_classes))

    metrics_df = pd.DataFrame(all_records)
    metrics_df.to_csv(TABLE_DIR / "external_generalization_metrics.csv", index=False, encoding="utf-8-sig")
    with open(METRIC_DIR / "external_generalization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False, default=str)

    pd.DataFrame(summaries).to_csv(TABLE_DIR / "external_dataset_summary.csv", index=False, encoding="utf-8-sig")

    best_all, best_hcef, best_base, delta = build_tables(metrics_df)

    total = time.time() - start_all

    summary = {
        "experiment": "Experiment 4 - External Cross-Dataset Generalization",
        "generated_at": str(datetime.now()),
        "device": str(DEVICE),
        "external_datasets": summaries,
        "best_external_model_per_dataset": best_all.to_dict(orient="records"),
        "best_hcef_external_model_per_dataset": best_hcef.to_dict(orient="records"),
        "best_external_baseline_per_dataset": best_base.to_dict(orient="records"),
        "hcef_vs_baseline_delta_table": delta.to_dict(orient="records"),
        "total_runtime_seconds": float(total),
    }

    with open(OUTPUT_DIR / "external_generalization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)

    with open(OUTPUT_DIR / "external_generalization_summary.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 4 External Cross-Dataset Generalization Summary\n\n")
        f.write(f"Generated at: `{datetime.now()}`\n\n")
        f.write(f"Device: `{DEVICE}`\n\n")
        f.write("## External Datasets\n\n")
        for s in summaries:
            f.write(f"- {s['dataset']} | Target={s['target']} | Features={s['n_features']} | Classes={s['n_classes']} | Train={s['train_shape']} | Validation={s['validation_shape']} | Test={s['test_shape']}\n")
        f.write("\n## Best External Model Per Dataset\n\n")
        for _, row in best_all.iterrows():
            f.write(f"- {row['dataset']} | {row['model']} | F1={row['f1_weighted']:.6f} | ROC-AUC={row['roc_auc']:.6f} | PR-AUC={row['pr_auc']:.6f}\n")
        f.write("\n## HCE-F vs Baseline Delta\n\n")
        for _, row in delta.iterrows():
            f.write(f"- {row['dataset']} | Baseline={row['best_external_baseline_model']} | HCE-F={row['best_hcef_model']} | Delta F1={row['delta_f1_hcef_minus_baseline']:.6f} | Delta AUC={row['delta_roc_auc_hcef_minus_baseline']:.6f}\n")
        f.write(f"\nTotal runtime seconds: `{total:.4f}`\n")

    write_docs(summaries)

    log("=" * 80)
    log("Experiment 4 completed successfully.")
    log("=" * 80)
    log(f"Metrics: {TABLE_DIR / 'external_generalization_metrics.csv'}")
    log(f"Summary: {OUTPUT_DIR / 'external_generalization_summary.md'}")
    log(f"Delta table: {TABLE_DIR / 'external_hcef_vs_baseline_delta_table.csv'}")
    log(f"README: {README_DIR / 'README_Experiment_4_External_Generalization.md'}")
    log(f"Total runtime: {total:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Experiment 4 failed. See error log:", flush=True)
        print(LOG_DIR / "experiment_4_error_log.txt", flush=True)
        save_error(e)
        raise
