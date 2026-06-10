import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, roc_curve,
)

from ood.msp import get_msp_scores
from utils.save_run import save_run
from ood.mahalonobis import extract_raw_features, fit_lda, MahalanobisOODDetector
import seaborn as sns
from ood.energy_maha import EmpiricalCDFCombiner


def calibrate_threshold(cal_combined, tpr_target = 0.95):
    threshold = np.percentile(cal_combined, 100 * tpr_target)
    print(f"[Threshold] {threshold:.4f}  (TPR target {tpr_target:.0%})")
    return threshold

# PLOTS

def plot_score_histogram(scores, labels, threshold, output_dir, tag):
    id_scores  = scores[labels >= 0]
    ood_scores = scores[labels == -1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(id_scores,  bins=50, alpha=0.6, density=True,
            color="steelblue", label="ID")
    ax.hist(ood_scores, bins=50, alpha=0.6, density=True,
            color="tomato",    label="OOD")
    ax.axvline(threshold, color="k", linestyle="--",
               label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel("eCDF combined score (higher = more OOD)")
    ax.set_ylabel("Density")
    ax.set_title("MSP + Mahalanobis eCDF score: ID vs OOD")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"msp_maha_histogram_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc(scores, labels, output_dir, tag):
    binary = (labels == -1).astype(int)
    fpr_arr, tpr_arr, _ = roc_curve(binary, scores)
    auroc = roc_auc_score(binary, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, color="steelblue",
            label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("MSP + Mahalanobis eCDF — ROC Curve")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"msp_maha_roc_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_components(msp_raw, maha_raw, labels, output_dir, tag):
    id_mask  = labels >= 0
    ood_mask = labels == -1

    palette = sns.color_palette("Set2")
    id_color  = palette[0]  # green
    ood_color = palette[1]  # orange
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(msp_raw[id_mask],   -maha_raw[id_mask],
               s=6, alpha=0.4, color=id_color, label="ID",  linewidths=0)
    ax.scatter(msp_raw[ood_mask],  -maha_raw[ood_mask],
               s=6, alpha=0.4, color=ood_color,    label="OOD", linewidths=0)
    ax.set_xlabel("MSP score (higher = more ID)")
    ax.set_ylabel("Mahalanobis score (negated, higher = more OOD)")
    ax.set_title("Score components: MSP vs Mahalanobis")
    ax.legend(markerscale=3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"msp_maha_components_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ecdf_surface(combiner, output_dir, tag):
    cal = combiner.cal_scores_
    m_min, m_max = cal[:, 0].min(), cal[:, 0].max()
    h_min, h_max = cal[:, 1].min(), cal[:, 1].max()

    mg = np.linspace(m_min, m_max, 80)
    hg = np.linspace(h_min, h_max, 80)
    MM, HH = np.meshgrid(mg, hg)
    grid_m = MM.ravel()
    grid_h = HH.ravel()

    grid_scores = combiner.transform(grid_m, grid_h).reshape(MM.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(MM, HH, grid_scores, levels=20, cmap="YlOrRd")
    fig.colorbar(cf, ax=ax, label="eCDF combined score")
    ax.scatter(cal[:, 0], cal[:, 1], s=3, alpha=0.3,
               color="steelblue", label="ID cal points", linewidths=0)
    ax.set_xlabel("MSP score (negated, higher = more OOD)")
    ax.set_ylabel("Mahalanobis score (higher = more OOD)")
    ax.set_title("eCDF decision surface")
    ax.legend(markerscale=4)
    fig.tight_layout()

    path = os.path.join(output_dir, f"msp_maha_ecdf_surface_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path



def run_msp_maha_ood(model, train_loader, val_loader, test_loader, device, output_dir,
 setup_name, cov_type="empirical", lda_components=None, fpr_target=0.05, ecdf_chunk=512, datetime_tag=None):
    """ Full pipeline, run in ood_utils.py."""
    tag = datetime_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    print("Computing MSP scores")
    val_msp,  val_labels_m,  _           = get_msp_scores(model, val_loader,  device)
    test_msp, test_labels,   class_preds = get_msp_scores(model, test_loader, device)

    val_msp_np     = -val_msp.numpy()
    test_msp_np    = -test_msp.numpy()
    val_labels_np  = val_labels_m.numpy()
    test_labels_np = test_labels.numpy()
    class_preds_np = class_preds.numpy()

    print("Computing Mahalanobis scores")
    X_train, y_train = extract_raw_features(train_loader, "Train")
    X_val,   y_val   = extract_raw_features(val_loader,   "Val")
    X_test,  y_test  = extract_raw_features(test_loader,  "Test")

    if (y_train == -1).any():
        mask    = y_train >= 0
        X_train = X_train[mask]
        y_train = y_train[mask]

    print("Fitting LDA")
    lda         = fit_lda(X_train, y_train, n_components=lda_components)
    X_train_lda = lda.transform(X_train)
    X_val_lda   = lda.transform(X_val)
    X_test_lda  = lda.transform(X_test)

    print("Fitting Mahalanobis detector")
    maha_det = MahalanobisOODDetector()
    maha_det.fit(X_train_lda, y_train, cov_type=cov_type)

    val_maha_np  = maha_det.score(X_val_lda)
    test_maha_np = maha_det.score(X_test_lda)

    val_id_mask  = val_labels_np >= 0
    val_msp_cal  = val_msp_np[val_id_mask]
    val_maha_cal = val_maha_np[val_id_mask]

    print(f"Fitting eCDF combiner on {val_id_mask.sum()} ID val samples")
    combiner = EmpiricalCDFCombiner(chunk_size=ecdf_chunk)
    combiner.fit(val_msp_cal, val_maha_cal)

    print("Computing eCDF combined scores")
    val_combined_id = combiner.transform(val_msp_cal, val_maha_cal)
    test_combined   = combiner.transform(test_msp_np, test_maha_np)     

    # debug prints
    print(f"Val ID eCDF   — mean: {val_combined_id.mean():.3f}  std: {val_combined_id.std():.3f}")
    print(f"Test ID eCDF  — mean: {test_combined[test_labels_np>=0].mean():.3f}  std: {test_combined[test_labels_np>=0].std():.3f}")
    print(f"Test OOD eCDF — mean: {test_combined[test_labels_np==-1].mean():.3f}  std: {test_combined[test_labels_np==-1].std():.3f}")

    threshold = calibrate_threshold(val_combined_id, tpr_target=1 - fpr_target)

    save_run(
        base_dir    = output_dir,
        setup_name  = setup_name,
        method      = "msp_maha",
        test_scores = test_combined,
        test_labels = test_labels_np,
        class_preds = class_preds_np,
        ood_higher  = True,
        meta        = {
            "combination_method": "empirical_cdf (Novello et al. 2024)",
            "cov_type":           cov_type,
            "threshold":          float(threshold),
            "n_cal_samples":      int(val_id_mask.sum()),
            "msp_direction":      "negated (higher MSP = more ID)",
        },
    )

    figure_paths = []

    p = plot_score_histogram(test_combined, test_labels_np, threshold,
                             output_dir, tag)
    figure_paths.append(p)

    p = plot_roc(test_combined, test_labels_np, output_dir, tag)
    figure_paths.append(p)

    p = plot_score_components(-test_msp_np, test_maha_np,
                              test_labels_np, output_dir, tag)
    figure_paths.append(p)

    p = plot_ecdf_surface(combiner, output_dir, tag)
    figure_paths.append(p)

    return test_combined, test_labels_np, class_preds_np, figure_paths
