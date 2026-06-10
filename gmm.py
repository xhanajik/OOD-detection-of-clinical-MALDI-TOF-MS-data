from utils.save_run import save_run
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")

# same as mahalanobis
def extract_raw_features(loader, text="Extracting"):
    X, y = [], []
    for inputs, labels in tqdm(loader, desc=text):
        X.append(inputs.view(inputs.size(0), -1).numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

# same as mahalanobis
def fit_lda(X_train, y_train, n_components=None):
    """
    Fit LDA on training data, n_components=None uses the maximum (n_classes - 1).
    """
    n_classes = len(np.unique(y_train))
    max_comps = n_classes - 1
    n_comps = min(n_components, max_comps) if n_components else max_comps

    lda = LinearDiscriminantAnalysis(n_components=n_comps, solver="svd")
    lda.fit(X_train, y_train)
    print(f"[LDA] fitted {n_comps} components  (max = {max_comps})")
    return lda


def select_n_components_bic(X, candidate_g, covariance_type="full", random_state=42):
    """Fit FMM for each candidate num of components and minimise BIC"""
    bic_scores = []
    for g in candidate_g:
        gmm = GaussianMixture(
            n_components=g,
            covariance_type=covariance_type,
            max_iter=200,
            random_state=random_state,
            reg_covar=1e-6,
        )
        try:
            gmm.fit(X.astype(np.float64))
            bic_scores.append(gmm.bic(X.astype(np.float64)))

        # sometimes G fails, throw error
        except ValueError as e:
            print(f"[BIC] G={g} failed: {e}")
            bic_scores.append(np.inf)

    bic_scores = np.array(bic_scores)
    best_g = candidate_g[int(np.argmin(bic_scores))]
    print(f"[BIC] selected G={best_g} (lowest BIC={bic_scores.min():.2f})")
    return best_g, bic_scores


class GMMOODDetector:
    """GMM density estimator in LDA space, OOD score = negative log-likelihood (higher → more OOD)"""

    def __init__(self, n_components=1, covariance_type="full", random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None
        self.threshold = None
        self.class_means = {}
        self.classes_ = None

    def fit(self, X_train, y_train):
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=200,
            random_state=self.random_state,
            reg_covar=1e-6,
        )
        self.gmm.fit(X_train)
        print(f"GMM fitted G={self.n_components} components "
              f"({self.covariance_type} covariance)")

        # same as Mahalanobis
        self.classes_ = np.unique(y_train)
        for c in self.classes_:
            self.class_means[c] = X_train[y_train == c].mean(axis=0)

        return self

    def score(self, X):
        return -self.gmm.score_samples(X)

    def predict_class(self, X):
        """ nearest class mean prediction in LDA space"""
        dists = np.stack(
            [np.linalg.norm(X - self.class_means[c], axis=1)
             for c in self.classes_],
            axis=1,
        )   # (N, n_classes)
        return self.classes_[dists.argmin(axis=1)]

    # tau
    def set_threshold(self, X_val_id, fpr_target=0.05):
        val_scores = self.score(X_val_id)
        self.threshold = np.percentile(val_scores, 100 * (1 - fpr_target))
        print(
            f"[Threshold] {self.threshold:.4f}  (FPR target {fpr_target:.0%})")
        return self.threshold


# PLOTS

def plot_bic_curve(candidate_g, bic_scores, best_g, output_dir, tag):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(candidate_g, bic_scores, marker="o", color="steelblue", lw=2)
    ax.axvline(best_g, color="tomato", linestyle="--",
               label=f"Selected G={best_g}")
    ax.set_xlabel("Number of components G")
    ax.set_ylabel("BIC")
    ax.set_title("GMM component selection via BIC")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"gmm_bic_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_histogram(test_scores, test_labels, threshold, output_dir, tag):
    id_scores = test_scores[test_labels >= 0]
    ood_scores = test_scores[test_labels == -1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(id_scores,  bins=60, alpha=0.6, density=True,
            color="steelblue", label="ID")
    ax.hist(ood_scores, bins=60, alpha=0.6, density=True,
            color="tomato",    label="OOD")
    ax.axvline(threshold, color="k", linestyle="--",
               label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Negative log-likelihood (GMM)")
    ax.set_ylabel("Density")
    ax.set_title("GMM NLL score: ID vs OOD")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"gmm_histogram_{tag}.png")
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
    ax.set_title("GMM – ROC Curve")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"gmm_roc_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lda_2d(X_lda_test, test_labels, output_dir, tag, max_id_classes=30):
    if X_lda_test.shape[1] < 2:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    id_classes = np.unique(test_labels[test_labels >= 0])[:max_id_classes]
    cmap = plt.cm.get_cmap("tab20", len(id_classes))

    for i, c in enumerate(id_classes):
        m = test_labels == c
        ax.scatter(X_lda_test[m, 0], X_lda_test[m, 1],
                   color=cmap(i), s=6, alpha=0.5, linewidths=0)

    ood_mask = test_labels == -1
    ax.scatter(X_lda_test[ood_mask, 0], X_lda_test[ood_mask, 1],
               color="black", s=6, alpha=0.3, linewidths=0, label="OOD")

    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title(f"LDA projection – first 2 components\n"
                 f"(showing {len(id_classes)} ID classes + OOD)")
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()

    path = os.path.join(output_dir, f"gmm_lda_2d_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_component_weights(det, output_dir, tag):
    weights = det.gmm.weights_
    g_range = np.arange(1, len(weights) + 1)

    fig, ax = plt.subplots(figsize=(max(6, len(weights) * 0.4), 4))
    ax.bar(g_range, weights, color="steelblue", alpha=0.7)
    ax.set_xlabel("Component")
    ax.set_ylabel("Mixing weight $\\pi_g$")
    ax.set_title(f"GMM component weights  (G={len(weights)})")
    ax.axhline(1 / len(weights), color="tomato", linestyle="--",
               label="Uniform weight")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"gmm_weights_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_gmm_lda(train_loader, val_loader, test_loader, output_dir, setup_name,
                candidate_g=None, covariance_type="full", lda_components=None, fpr_target=0.05, random_state=42):
    """GMM pipeline, run in ood_utils.py"""
    if candidate_g is None:
        candidate_g = [1, 2, 4, 8, 16, 32]

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting train features")
    X_train, y_train = extract_raw_features(train_loader, text="Train")

    print("Extracting val features")
    X_val, y_val = extract_raw_features(val_loader, text="Val")

    print("Extracting test features")
    X_test, y_test = extract_raw_features(test_loader, text="Test")

    X_val_id = X_val[y_val >= 0]
    y_val_id = y_val[y_val >= 0]

    print("Fitting LDA")
    lda = fit_lda(X_train, y_train, n_components=lda_components)
    X_train_lda = lda.transform(X_train)
    X_val_id_lda = lda.transform(X_val_id)
    X_test_lda = lda.transform(X_test)
    X_train_lda = X_train_lda.astype(np.float64)

    print("Selecting number of GMM components with BIC")
    best_g, bic_scores = select_n_components_bic(
        X_train_lda,
        candidate_g,
        covariance_type=covariance_type,
        random_state=random_state,
    )

    print(f"Fitting GMM with G={best_g} components")
    det = GMMOODDetector(
        n_components=best_g,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    det.fit(X_train_lda, y_train)
    det.set_threshold(X_val_id_lda, fpr_target=fpr_target)

    print("Scoring test set")
    test_scores = det.score(X_test_lda)          # higher = more OOD
    class_preds = det.predict_class(X_test_lda)  # nearest-class-mean

    save_run(
        base_dir=output_dir,
        setup_name=setup_name,
        method=f"gmm_g{best_g}_{covariance_type}",
        test_scores=test_scores,
        test_labels=y_test,
        class_preds=class_preds,
        ood_higher=True,
        meta={
            "n_components_selected": best_g,
            "covariance_type":       covariance_type,
            "candidate_g":           candidate_g,
            "bic_scores":            bic_scores.tolist(),
        },
    )

    figure_paths = []
    p = plot_bic_curve(candidate_g, bic_scores, best_g, output_dir, tag)
    figure_paths.append(p)

    p = plot_score_histogram(
        test_scores, y_test, det.threshold, output_dir, tag)
    figure_paths.append(p)

    p = plot_roc(test_scores, y_test, output_dir, tag)
    figure_paths.append(p)

    p = plot_lda_2d(X_test_lda, y_test, output_dir, tag)
    if p:
        figure_paths.append(p)

    p = plot_component_weights(det, output_dir, tag)
    figure_paths.append(p)

    return test_scores, y_test, class_preds, figure_paths
