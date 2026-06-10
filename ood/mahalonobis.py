from sklearn.covariance import EmpiricalCovariance
from utils.save_run import save_run
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")


def extract_raw_features(loader, text="Extracting"):
    X, y = [], []
    for inputs, labels in tqdm(loader, desc=text):
        X.append(inputs.view(inputs.size(0), -1).numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)


def fit_lda(X_train, y_train, n_components=None):
    n_classes = len(np.unique(y_train))
    max_comps = n_classes - 1
    n_comps = min(n_components, max_comps) if n_components else max_comps

    lda = LinearDiscriminantAnalysis(n_components=n_comps, solver="svd")
    lda.fit(X_train, y_train)
    print(f"LDA fitted {n_comps} components  (max = {max_comps})")
    return lda


class MahalanobisOODDetector:
    def __init__(self):
        self.class_means = {}
        self.shared_prec = None
        self.classes_ = None
        self.threshold = None

    def fit(self, X_train, y_train, cov_type="empirical"):
        self.classes_ = np.unique(y_train)

        # per-class means
        for c in self.classes_:
            self.class_means[c] = X_train[y_train == c].mean(axis=0)

        residuals = X_train - np.stack([self.class_means[c] for c in y_train])

        if cov_type == "empirical":
            cov_est = EmpiricalCovariance().fit(residuals)
            self.shared_prec = cov_est.precision_
        else:
            raise ValueError("Only empirical covariance implemented.")

        print("Condition number:", np.linalg.cond(cov_est.covariance_))
        print("Precision matrix norm:", np.linalg.norm(self.shared_prec))

        print("Condition number:", np.linalg.cond(
            np.linalg.inv(self.shared_prec)))
        print("Precision matrix norm:", np.linalg.norm(self.shared_prec))
        print(f"used: {cov_type}")
        return self

    def _dist_to_class(self, X, c):
        """Mahalanobis distance from every row of X to class c mean."""
        diff = X - self.class_means[c]
        left = diff @ self.shared_prec
        dist_sq = (left * diff).sum(axis=1)
        return np.sqrt(np.clip(dist_sq, 0, None))

    def score(self, X):
        """min Mahalanobis distance to any class, Higher = more OOD."""
        all_dists = np.stack(
            [self._dist_to_class(X, c) for c in self.classes_], axis=1
        )
        return all_dists.min(axis=1)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Nearest-class-mean """
        all_dists = np.stack(
            [self._dist_to_class(X, c) for c in self.classes_], axis=1
        )
        return self.classes_[all_dists.argmin(axis=1)]

    # tau
    def set_threshold(self, X_val_id, fpr_target=0.05):
        val_scores = self.score(X_val_id)
        self.threshold = np.percentile(val_scores, 100 * (1 - fpr_target))
        print(
            f"[Threshold] {self.threshold:.4f}  (FPR target {fpr_target:.0%})")
        return self.threshold


#### PLOTS ###

def plot_score_histogram(test_scores, test_labels, threshold, output_dir, tag):
    id_scores = test_scores[test_labels >= 0]
    ood_scores = test_scores[test_labels == -1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(id_scores,  bins=60, alpha=0.6,
            density=True, color="steelblue", label="ID")
    ax.hist(ood_scores, bins=60, alpha=0.6,
            density=True, color="tomato",    label="OOD")
    ax.axvline(threshold, color="k", linestyle="--",
               label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Mahalanobis distance (min over classes)")
    ax.set_ylabel("Density")
    ax.set_title("Mahalanobis distance: ID vs OOD")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"mahal_histogram_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc(test_scores, test_labels, output_dir, tag):
    binary = (test_labels == -1).astype(int)   # 1 = OOD

    fpr_arr, tpr_arr, _ = roc_curve(binary, test_scores)
    auroc = roc_auc_score(binary, test_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mahalanobis – ROC Curve")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"mahal_roc_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lda_variance(lda, output_dir, tag):
    ratios = lda.explained_variance_ratio_
    cum = np.cumsum(ratios)
    n_comps = len(ratios)

    fig, ax1 = plt.subplots(figsize=(min(16, max(8, n_comps // 4)), 4))
    ax1.bar(range(n_comps), ratios, color="steelblue",
            alpha=0.7, label="Per-component")
    ax2 = ax1.twinx()
    ax2.plot(range(n_comps), cum, color="tomato", lw=2, label="Cumulative")
    ax2.axhline(0.95, color="tomato", linestyle="--", alpha=0.5, label="95 %")
    ax1.set_xlabel("LDA component")
    ax1.set_ylabel("Explained variance ratio")
    ax2.set_ylabel("Cumulative variance")
    ax1.set_title("LDA explained variance")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()

    path = os.path.join(output_dir, f"lda_variance_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lda_2d(X_lda_test, test_labels, output_dir, tag, max_id_classes=30):
    fig, ax = plt.subplots(figsize=(8, 6))

    id_mask = test_labels >= 0
    ood_mask = test_labels == -1

    id_classes = np.unique(test_labels[id_mask])[:max_id_classes]
    cmap = plt.cm.get_cmap("tab20", len(id_classes))

    for i, c in enumerate(id_classes):
        m = test_labels == c
        ax.scatter(X_lda_test[m, 0], X_lda_test[m, 1],
                   color=cmap(i), s=6, alpha=0.5, linewidths=0)

    ax.scatter(X_lda_test[ood_mask, 0], X_lda_test[ood_mask, 1],
               color="black", s=6, alpha=0.3, linewidths=0, label="OOD")

    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title(f"LDA projection – first 2 components\n"
                 f"(showing {len(id_classes)} ID classes + OOD)")
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()

    path = os.path.join(output_dir, f"lda_2d_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_distances(det, X_lda_test, test_labels, output_dir, tag):
    id_mask = test_labels >= 0
    id_labels = test_labels[id_mask]
    id_scores = det.score(X_lda_test[id_mask])

    classes = np.unique(id_labels)
    if len(classes) > 60:
        classes = classes[:60]

    data = [id_scores[id_labels == c] for c in classes]
    labels = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.25), 5))
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.axhline(det.threshold, color="tomato", linestyle="--",
               label=f"Threshold ({det.threshold:.2f})")
    ax.set_xlabel("True class")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("Per-class Mahalanobis distance (ID test samples)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"mahal_class_distances_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_mahalanobis_lda(train_loader, val_loader, test_loader,
                        output_dir, cov_type, setup_name, lda_components=None, fpr_target=0.05):
    """full Mahalanobis pipeline, run in ood_utils.py"""
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting features")
    X_train, y_train = extract_raw_features(train_loader, text="Train")
    X_val, y_val = extract_raw_features(val_loader, text="Val")
    X_test, y_test = extract_raw_features(test_loader, text="Test")

    print(cov_type)

    print("Fitting LDA.")
    lda = fit_lda(X_train, y_train, n_components=lda_components)

    X_train_lda = lda.transform(X_train)
    X_val_lda = lda.transform(X_val)
    X_val_id_lda = lda.transform(X_val)
    X_test_lda = lda.transform(X_test)

    print("Fitting Mahalanobis detector")
    det = MahalanobisOODDetector()
    det.fit(X_train_lda, y_train, cov_type=cov_type)
    det.set_threshold(X_val_id_lda, fpr_target=fpr_target)

    # print("Precision matrix norm:", np.linalg.norm(self.shared_prec))
    # print("Condition number:", np.linalg.cond(np.linalg.inv(self.shared_prec)))

    print("Scoring test set")
    test_scores = det.score(X_test_lda)           # higher = more OOD
    class_preds = det.predict_class(X_test_lda)   # nearest-class-mean

    save_run(
        base_dir=output_dir,
        setup_name=setup_name,
        # distinguishes knn_raw / knn_pca / knn_embedding
        method=f"maha_{cov_type}",
        test_scores=test_scores,
        test_labels=y_test,
        class_preds=class_preds,
        ood_higher=True,
        meta={
            "covariance estimate": cov_type,
        },
    )

    figure_paths = []

    p = plot_score_histogram(
        test_scores, y_test, det.threshold, output_dir, tag)
    figure_paths.append(p)

    p = plot_roc(test_scores, y_test, output_dir, tag)
    figure_paths.append(p)

    p = plot_lda_variance(lda, output_dir, tag)
    if p:
        figure_paths.append(p)

    p = plot_lda_2d(X_test_lda, y_test, output_dir, tag)
    if p:
        figure_paths.append(p)

    p = plot_class_distances(det, X_test_lda, y_test, output_dir, tag)
    figure_paths.append(p)

    return test_scores, y_test, class_preds, figure_paths
