from utils.save_run import save_run
from utils.model_utils import extract_embeddings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")


# same as maha
def extract_raw_features(loader, text="Extracting raw"):
    X, y = [], []
    for inputs, labels in tqdm(loader, desc=text):
        X.append(inputs.view(inputs.size(0), -1).numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)


def fit_pca(X_train, X_val, X_test, n_components=0.95, random_state=42):
    """n_components=0.95 means 95% explained variance, int for a fixed num of components"""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_tr = pca.fit_transform(X_train)  # fit on train
    X_v = pca.transform(X_val)  # transform val and test
    X_te = pca.transform(X_test)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(
        f"[PCA] {pca.n_components_} components → {explained:.1%} variance explained")
    return X_tr, X_v, X_te, pca


class KNNOODDetector:
    def __init__(self, k=5, metric="euclidean", n_jobs=-1, k_range=[1, 3, 5, 10, 15, 18]):
        self.k = k
        self.metric = metric
        self.n_jobs = n_jobs
        self.index = None
        self.threshold = None
        # store training 1-NN class prediction
        self._y_train = None
        self.k_range = k_range

    def fit(self, X_train_id, y_train_id):
        """Reference set for KNN index is the train set"""
        self.index = NearestNeighbors(
            n_neighbors=max(self.k_range),
            metric=self.metric,
            algorithm="auto",
            n_jobs=self.n_jobs,
        )
        self.index.fit(X_train_id)
        self._y_train = y_train_id
        return self

    def score(self, X):
        all_distances, _ = self.index.kneighbors(
            X, n_neighbors=max(self.k_range))
        scores = [all_distances[:, k-1] for k in self.k_range]
        return np.mean(scores, axis=0)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """ 1-NN is the prediction, so closest neighbor"""
        _, indices = self.index.kneighbors(X, n_neighbors=1)
        return self._y_train[indices[:, 0]]

    # tau
    def set_threshold(self, X_val, fpr_target=0.05):
        val_scores = self.score(X_val)
        self.threshold = np.percentile(val_scores, 100 * (1 - fpr_target))
        print(
            f"[Threshold] {self.threshold:.4f}  (FPR target {fpr_target:.0%})")
        return self.threshold

# PLOTS #


def plot_score_histogram(test_scores, test_labels, threshold, output_dir, tag, metric="", k=0):
    id_scores = test_scores[test_labels >= 0]
    ood_scores = test_scores[test_labels == -1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(id_scores,  bins=60, alpha=0.6,
            density=True, color="steelblue", label="ID")
    ax.hist(ood_scores, bins=60, alpha=0.6,
            density=True, color="tomato",    label="OOD")
    ax.axvline(threshold, color="k", linestyle="--",
               label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel(f"KNN distance (K={k}, metric={metric})")
    ax.set_ylabel("Density")
    ax.set_title("KNN distance distribution: ID vs OOD")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"knn_histogram_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc(test_scores, test_labels, output_dir, tag, metric="", k=0):
    binary = (test_labels == -1).astype(int)
    fpr_arr, tpr_arr, _ = roc_curve(binary, test_scores)
    auroc = roc_auc_score(binary, test_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, color="steelblue",
            label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"KNN OOD — ROC Curve  (K={k}, {metric})")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"knn_roc_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_distance_by_class(test_scores, test_labels, threshold, output_dir, tag, max_classes=60):
    id_mask = test_labels >= 0
    id_labels = test_labels[id_mask]
    id_scores = test_scores[id_mask]

    classes = np.unique(id_labels)
    if len(classes) > max_classes:
        classes = classes[:max_classes]

    data = [id_scores[id_labels == c] for c in classes]
    labels = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.25), 5))
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.axhline(threshold, color="tomato", linestyle="--",
               label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel("True class")
    ax.set_ylabel("KNN distance")
    ax.set_title("Per-class KNN distance distribution (ID test samples)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"knn_class_distances_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_knn_ood(train_loader, val_loader, test_loader, output_dir, setup_name, model=None, device="gpu",
                variant="raw", metric="euclidean", k_range=[1, 3, 5, 10, 15, 18],
                pca_components=0.8, fpr_target=0.05, datetime_tag=None):
    """Full KNN pipeline, runs only 1 variant (embedding, raw, pca) at a time, run in ood_utils.py"""
    tag = datetime_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, variant)
    os.makedirs(output_dir, exist_ok=True)

    if variant == "embedding":
        print("Extracting embeddings")
        X_train, y_train = extract_embeddings(
            train_loader, model, device, "Train")
        X_val,   y_val = extract_embeddings(val_loader,   model, device, "Val")
        X_test,  y_test = extract_embeddings(
            test_loader,  model, device, "Test")

    else:
        print("Extracting features")
        X_train, y_train = extract_raw_features(train_loader, "Train")
        X_val,   y_val = extract_raw_features(val_loader,   "Val")
        X_test,  y_test = extract_raw_features(test_loader,  "Test")

        if variant == "pca":
            print("Applying PCA")
            X_train, X_val, X_test, _ = fit_pca(
                X_train, X_val, X_test, n_components=pca_components
            )

    # not really best_k anymore, just fit on the max, so that only 1 detector can be used for all k-th NN distances
    best_k = max(k_range)
    print(f"Fitting KNN detector  (K={best_k}, metric={metric})")
    det = KNNOODDetector(k=best_k, metric=metric, k_range=k_range)
    det.fit(X_train, y_train)
    det.set_threshold(X_val, fpr_target=fpr_target)

    print("Scoring test set")
    test_scores = det.score(X_test)
    class_preds = det.predict_class(X_test)

    save_run(
        base_dir=output_dir,
        setup_name=setup_name,
        # change method name per variant: knn_raw / knn_pca / knn_embedding
        method=f"knn_{variant}",
        test_scores=test_scores,
        test_labels=y_test,
        class_preds=class_preds,
        ood_higher=True,
        meta={
            "K":      best_k,
            "metric": metric,
            "variant": variant,
        },
    )

    figure_paths = []
    p = plot_score_histogram(
        test_scores, y_test, det.threshold, output_dir, tag, metric, best_k
    )
    figure_paths.append(p)

    p = plot_roc(test_scores, y_test, output_dir, tag, metric, best_k)
    figure_paths.append(p)

    p = plot_distance_by_class(
        test_scores, y_test, det.threshold, output_dir, tag)
    figure_paths.append(p)

    return test_scores, y_test, class_preds, figure_paths
