from ood.mahalonobis import extract_raw_features, fit_lda, MahalanobisOODDetector
from ood.energy import get_energy_scores
from utils.save_run import save_run
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save img to files


class EmpiricalCDFCombiner:
    def __init__(self, chunk_size: int = 512):
        self.cal_scores_: Optional[np.ndarray] = None
        self.chunk_size = chunk_size

    def fit(self, *score_arrays):
        """Fit on val ID samples."""
        self.cal_scores_ = np.stack(score_arrays, axis=1)
        return self

    def transform(self, *score_arrays):
        """Combined score for test points."""
        test_scores = np.stack(score_arrays, axis=1)  # (N_test, d)
        N_cal = self.cal_scores_.shape[0]
        N_test = test_scores.shape[0]
        combined = np.empty(N_test, dtype=np.float64)

        # in chuncks
        for start in range(0, N_test, self.chunk_size):
            end = min(start + self.chunk_size, N_test)
            chunk = test_scores[start:end]          # (chunk, d)

            # dominated[i, j] = True if cal point j is <= test point i on ALL dims
            # cal: (1, N_cal, d)   chunk: (chunk, 1, d)
            dominated = np.all(
                self.cal_scores_[np.newaxis, :, :] <= chunk[:, np.newaxis, :],
                axis=2,
            )                                        # (chunk, N_cal)
            combined[start:end] = dominated.mean(axis=1)

        return combined


def calibrate_threshold(cal_combined, tpr_target=0.95):
    """tau"""
    threshold = np.percentile(cal_combined, 100 * tpr_target)
    print(f"[Threshold] {threshold:.4f}  (TPR target {tpr_target:.0%})")
    return threshold

############### PLOTS ############

def plot_score_histogram(scores, labels, threshold, output_dir, tag):
    id_scores = scores[labels >= 0]
    ood_scores = scores[labels == -1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        id_scores,
        bins=50,
        alpha=0.6,
        density=True,
        color="steelblue",
        label="ID",
    )
    ax.hist(
        ood_scores,
        bins=50,
        alpha=0.6,
        density=True,
        color="tomato",
        label="OOD",
    )
    ax.axvline(
        threshold,
        color="k",
        linestyle="--",
        label=f"Threshold ({threshold:.3f})",
    )
    ax.set_xlabel("eCDF combined score (higher = more OOD)")
    ax.set_ylabel("Density")
    ax.set_title("Energy + Mahalanobis eCDF score: ID vs OOD")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"energy_maha_histogram_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc(scores, labels, output_dir, tag):
    binary = (labels == -1).astype(int)
    fpr_arr, tpr_arr, _ = roc_curve(binary, scores)
    auroc = roc_auc_score(binary, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr_arr,
        tpr_arr,
        lw=2,
        color="steelblue",
        label=f"AUROC = {auroc:.4f}",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Energy + Mahalanobis eCDF — ROC Curve")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"energy_maha_roc_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_components(energy_raw, maha_raw, labels, output_dir, tag):
    """shows how complementary the signals are"""
    id_mask = labels >= 0
    ood_mask = labels == -1

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        energy_raw[id_mask],
        maha_raw[id_mask],
        s=6,
        alpha=0.4,
        color="steelblue",
        label="ID",
        linewidths=0,
    )
    ax.scatter(
        energy_raw[ood_mask],
        maha_raw[ood_mask],
        s=6,
        alpha=0.4,
        color="tomato",
        label="OOD",
        linewidths=0,
    )
    ax.set_xlabel("Energy score (negated, higher = more OOD)")
    ax.set_ylabel("Mahalanobis score (higher = more OOD)")
    ax.set_title("Score components: energy vs Mahalanobis")
    ax.legend(markerscale=3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"energy_maha_components_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_ecdf_surface(combiner, output_dir, tag):
    cal = combiner.cal_scores_
    e_min, e_max = cal[:, 0].min(), cal[:, 0].max()
    m_min, m_max = cal[:, 1].min(), cal[:, 1].max()

    eg = np.linspace(e_min, e_max, 80)
    mg = np.linspace(m_min, m_max, 80)
    EE, MM = np.meshgrid(eg, mg)

    grid_e = EE.ravel()
    grid_m = MM.ravel()

    grid_scores = combiner.transform(grid_e, grid_m).reshape(EE.shape)
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(EE, MM, grid_scores, levels=20, cmap="YlOrRd")
    fig.colorbar(cf, ax=ax, label="eCDF combined score")
    ax.scatter(
        cal[:, 0],
        cal[:, 1],
        s=3,
        alpha=0.3,
        color="steelblue",
        label="ID cal points",
        linewidths=0,
    )
    ax.set_xlabel("Energy score (negated, higher = more OOD)")
    ax.set_ylabel("Mahalanobis score (higher = more OOD)")
    ax.set_title("eCDF decision surface")
    ax.legend(markerscale=4)

    fig.tight_layout()
    path = os.path.join(output_dir, f"energy_maha_ecdf_surface_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path

def run_energy_maha_ood(model, train_loader, val_loader, test_loader, device, output_dir,
                        setup_name, cov_type="empirical", lda_components=None,
                        fpr_target=0.05, ecdf_chunk=512, datetime_tag=None):
    """ Full pipeline, run in ood_utils.py."""
    # seeds cuz idk if it gets here from the main
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tag = datetime_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    torch.set_grad_enabled(False)

    print("Computing energy scores")
    val_energy,  val_labels_e,  _ = get_energy_scores(
        model, val_loader,  device)
    test_energy, test_labels,   class_preds = get_energy_scores(
        model, test_loader, device)

    # negat for this data ID energy > OOD energy
    val_energy_np = -val_energy.numpy()
    test_energy_np = -test_energy.numpy()
    val_labels_np = val_labels_e.numpy()
    test_labels_np = test_labels.numpy()
    class_preds_np = class_preds.numpy()

    print("Computing Mahalanobis scores")
    X_train, y_train = extract_raw_features(train_loader, "Train")
    X_val,   y_val = extract_raw_features(val_loader,   "Val")
    X_test,  y_test = extract_raw_features(test_loader,  "Test")

    if (y_train == -1).any():
        mask = y_train >= 0
        X_train = X_train[mask]
        y_train = y_train[mask]

    print("Fitting LDA")
    lda = fit_lda(X_train, y_train, n_components=lda_components)
    X_train_lda = lda.transform(X_train)
    X_val_lda = lda.transform(X_val)
    X_test_lda = lda.transform(X_test)

    print("Fitting Mahalanobis detector")
    maha_det = MahalanobisOODDetector()
    maha_det.fit(X_train_lda, y_train, cov_type=cov_type)

    val_maha_np = maha_det.score(X_val_lda)
    test_maha_np = maha_det.score(X_test_lda)

    # all val are id but in case
    val_id_mask = val_labels_np >= 0
    val_energy_cal = val_energy_np[val_id_mask]
    val_maha_cal = val_maha_np[val_id_mask]

    print(f"Fitting eCDF on {val_id_mask.sum()} val samples")
    combiner = EmpiricalCDFCombiner(chunk_size=ecdf_chunk)
    combiner.fit(val_energy_cal, val_maha_cal)
    print("Computing eCDF combined scores")
    val_combined_id = combiner.transform(val_energy_cal, val_maha_cal)
    test_combined = combiner.transform(test_energy_np, test_maha_np)

    #tau
    threshold = calibrate_threshold(val_combined_id, tpr_target=1 - fpr_target)


    save_run(
        base_dir=output_dir,
        setup_name=setup_name,
        method="energy_maha",
        test_scores=test_combined,
        test_labels=test_labels_np,
        class_preds=class_preds_np,
        ood_higher=True,
        meta={
            "combination_method": "empirical_cdf (Novello et al. 2024)",
            "cov_type":           cov_type,
            "threshold":          float(threshold),
            "n_cal_samples":      int(val_id_mask.sum()),
            "energy_direction":   "negated (ID energy > OOD energy for this data)",
        },
    )

    #plot
    figure_paths = []
    p = plot_score_histogram(test_combined, test_labels_np, threshold,
                             output_dir, tag)
    figure_paths.append(p)

    p = plot_roc(test_combined, test_labels_np, output_dir, tag)
    figure_paths.append(p)

    p = plot_score_components(test_energy_np, test_maha_np,
                              test_labels_np, output_dir, tag)
    figure_paths.append(p)

    p = plot_ecdf_surface(combiner, output_dir, tag)
    figure_paths.append(p)

    return test_combined, test_labels_np, class_preds_np, figure_paths
