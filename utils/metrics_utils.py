from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import numpy as np
import matplotlib
import csv
matplotlib.use("Agg")


def _to_numpy(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _fpr_at_tpr(binary_labels, scores, tpr_target=0.95):
    fpr_arr, tpr_arr, _ = roc_curve(binary_labels, scores)
    # find the threshold index where TPR first reaches tpr_target
    idx = np.searchsorted(tpr_arr, tpr_target)
    if idx >= len(fpr_arr):
        return float("nan")
    return float(fpr_arr[idx])


class OODMetrics:
    def __init__(self, setup, abundance_filter, dataset_name, model_path="", hierarchy_level="", datetime_tag=None):
        self.setup = setup
        self.abundance_filter = abundance_filter
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.hierarchy_level = hierarchy_level
        self.datetime_tag = datetime_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.ood_metrics= {}
        self.id_metrics= {}
        self.per_class_df = None
        self._binary_labels = None
        self._ood_scores = None
        self._confusion_mat = None
        self._class_names= []
        self._computed = False


    def compute(self, ood_scores, test_labels, id_preds, class_names=None, ood_higher=False):
        scores = _to_numpy(ood_scores).astype(float)
        labels = _to_numpy(test_labels).astype(int)
        preds_raw = _to_numpy(id_preds)
        preds = preds_raw.astype(int)

        if ood_higher:
            scores = -scores   # normalise to high = ID

        binary = (labels >= 0).astype(int)
        self._binary_labels = binary
        self._ood_scores = scores

        # metrics
        auroc = roc_auc_score(binary, scores)
        fpr95 = _fpr_at_tpr(binary, scores, tpr_target=0.95)
        #aupr not rply in benchmark but in case i want to add it later
        aupr_in = average_precision_score(binary,     scores)
        aupr_out = average_precision_score(1 - binary, -scores)


        # threshold tau at 95 % TPR
        id_scores_only = scores[binary == 1]
        threshold = np.percentile(id_scores_only, 5)
        bin_preds = (scores >= threshold).astype(int)
        bal_acc_ood = balanced_accuracy_score(binary, bin_preds)

        self.ood_metrics = dict(
            auroc=auroc,
            fpr95=fpr95,
            aupr_in=aupr_in,
            aupr_out=aupr_out,
            bal_acc_ood=bal_acc_ood,
            threshold=threshold,
        )

        # ID prediction metrics
        id_mask    = labels >= 0
        id_true = labels[id_mask]
        id_pred = preds[id_mask]
    
        n_classes = int(id_true.max()) + 1

        if class_names is not None:
            self._class_names = list(class_names)
        else:
            self._class_names = [str(i) for i in range(n_classes)]

        #ignore warning here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prec, rec, f1, support = precision_recall_fscore_support(
                id_true, id_pred, labels=list(range(n_classes)), zero_division=0
            )

        bal_acc_id = balanced_accuracy_score(id_true, id_pred)
        self.id_metrics = dict(
            balanced_accuracy=bal_acc_id,
            precision=dict(
                min=float(prec.min()),
                max=float(prec.max()),
                median=float(np.median(prec)),
                mean=float(prec.mean()),
            ),
            recall=dict(
                min=float(rec.min()),
                max=float(rec.max()),
                median=float(np.median(rec)),
                mean=float(rec.mean()),
            ),
            f1=dict(
                min=float(f1.min()),
                max=float(f1.max()),
                median=float(np.median(f1)),
                mean=float(f1.mean()),
            ),
            per_class=dict(
                precision=prec.tolist(),
                recall=rec.tolist(),
                f1=f1.tolist(),
                support=support.tolist(),
            ),
        )

        self._confusion_mat = confusion_matrix(
            id_true, id_pred, labels=list(range(n_classes)))

        self._computed = True
        return self

    # PLOTS FOR THE REPORT

    def plot_score_histogram(self, output_dir, method_name="OOD"):
        os.makedirs(output_dir, exist_ok=True)

        id_scores = self._ood_scores[self._binary_labels == 1]
        ood_scores = self._ood_scores[self._binary_labels == 0]
        threshold = self.ood_metrics["threshold"]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(id_scores,  bins=50, alpha=0.6, label="ID",
                density=True, color="steelblue")
        ax.hist(ood_scores, bins=50, alpha=0.6,
                label="OOD", density=True, color="tomato")
        ax.axvline(threshold, color="k", linestyle="--",
                   label=f"Threshold ({threshold:.3f})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{method_name} score distributions – ID vs OOD")
        ax.legend()
        fig.tight_layout()

        path = os.path.join(
            output_dir, f"score_histogram_{self.datetime_tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_roc_curve(self, output_dir, method_name="OOD"):
        os.makedirs(output_dir, exist_ok=True)

        fpr_arr, tpr_arr, _ = roc_curve(self._binary_labels, self._ood_scores)
        auroc = self.ood_metrics["auroc"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_arr, tpr_arr, lw=2, label=f"AUROC = {auroc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.axvline(self.ood_metrics["fpr95"], color="tomato",
                   linestyle=":", label=f"FPR95 = {self.ood_metrics['fpr95']:.4f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{method_name} – ROC Curve")
        ax.legend()
        fig.tight_layout()

        path = os.path.join(output_dir, f"roc_curve_{self.datetime_tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_pr_curve(self, output_dir, method_name="OOD"):
        os.makedirs(output_dir, exist_ok=True)

        prec_arr, rec_arr, _ = precision_recall_curve(
            self._binary_labels, self._ood_scores)
        aupr = self.ood_metrics["aupr_in"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec_arr, prec_arr, lw=2, label=f"AUPR = {aupr:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{method_name} – Precision-Recall Curve")
        ax.legend()
        fig.tight_layout()

        path = os.path.join(output_dir, f"pr_curve_{self.datetime_tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_confusion_matrix(self, output_dir):
        """Confusion matrix heatmap for the report"""
        os.makedirs(output_dir, exist_ok=True)

        cm = self._confusion_mat
        n = cm.shape[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        px_per_cell = 4 if n > 100 else 8
        fig_size = max(12, n * px_per_cell / 72)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        im = ax.imshow(cm_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label="Recall (row-normalised)")

        if n <= 40:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(self._class_names, rotation=90, fontsize=6)
            ax.set_yticklabels(self._class_names, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"Predicted class (0–{n-1})")
            ax.set_ylabel(f"True class (0–{n-1})")

        ax.set_title(
            f"Confusion Matrix (normalised) – {self.dataset_name}\n"
            f"Setup: {self.setup}  |  {self.datetime_tag}"
        )
        fig.tight_layout()

        img_path = os.path.join(
            output_dir, f"confusion_matrix_{self.datetime_tag}.png")
        fig.savefig(img_path, dpi=150)
        plt.close(fig)

        # save raw counts as CSV
        csv_path = os.path.join(
            output_dir, f"confusion_matrix_{self.datetime_tag}.csv")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([""] + self._class_names)
            for i, row in enumerate(cm):
                writer.writerow([self._class_names[i]] + row.tolist())

        return img_path

    def plot_per_class_bars(self, output_dir, top_n= 40):
        os.makedirs(output_dir, exist_ok=True)

        f1_arr = np.array(self.id_metrics["per_class"]["f1"])
        n = len(f1_arr)

        sorted_idx = np.argsort(f1_arr)
        worst_idx = sorted_idx[:top_n]
        best_idx = sorted_idx[-top_n:][::-1]

        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.22)))

        for ax, idx, title, color in [
            (axes[0], worst_idx, f"Worst {top_n} classes (F1)", "tomato"),
            (axes[1], best_idx,  f"Best {top_n} classes (F1)",  "steelblue"),
        ]:
            names = [self._class_names[i] for i in idx]
            values = f1_arr[idx]
            y_pos = range(len(idx))
            ax.barh(list(y_pos), values, color=color, alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_xlabel("F1 score")
            ax.set_title(title)
            ax.invert_yaxis()

        fig.suptitle(
            f"Per-class F1 – {self.dataset_name} | {self.setup} | {self.datetime_tag}",
            fontsize=10,
        )
        fig.tight_layout()

        path = os.path.join(
            output_dir, f"per_class_f1_{self.datetime_tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # print after every OOD detection run
    def print_summary(self):
        sep = "-*-" * 11
        print(sep)
        print(f"OOD METRICS  |  {self.dataset_name}  |  setup: {self.setup}")
        print(sep)
        m = self.ood_metrics
        print(f"  AUROC       : {m['auroc']:.4f}")
        print(f"  FPR@95TPR   : {m['fpr95']:.4f}")
        print(f"  AUPR-in     : {m['aupr_in']:.4f}")
        print(f"  AUPR-out    : {m['aupr_out']:.4f}")
        print(
            f"  Bal Acc OOD : {m['bal_acc_ood']:.4f}  (threshold={m['threshold']:.4f})")
        print(sep)
        print("ID CLASSIFICATION METRICS")
        print(sep)
        c = self.id_metrics
        print(f"  Balanced Accuracy : {c['balanced_accuracy']:.4f}")
        print(f"  Precision  – min {c['precision']['min']:.4f}  "
              f"median {c['precision']['median']:.4f}  "
              f"max {c['precision']['max']:.4f}  "
              f"mean {c['precision']['mean']:.4f}")
        print(f"  Recall     – min {c['recall']['min']:.4f}  "
              f"median {c['recall']['median']:.4f}  "
              f"max {c['recall']['max']:.4f}  "
              f"mean {c['recall']['mean']:.4f}")
        print(f"  F1         – min {c['f1']['min']:.4f}  "
              f"median {c['f1']['median']:.4f}  "
              f"max {c['f1']['max']:.4f}  "
              f"mean {c['f1']['mean']:.4f}")
        print(sep)
