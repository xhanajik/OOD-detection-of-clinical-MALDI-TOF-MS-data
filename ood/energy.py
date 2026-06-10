from utils.save_run import save_run
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Optional, Tuple
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")


def get_energy_scores(model, dataloader, device: str):
    model.eval()
    all_scores, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Energy scores"):
            x = x.to(device)
            logits = model(x)                            
            probs = F.softmax(logits, dim=1)
            energy = torch.logsumexp(logits, dim=1)     
            all_scores.append(energy.cpu())
            all_labels.append(y)
            # all_preds.append(logits.argmax(dim=1).cpu())

            # issues with the logits argmax on joltik cluster idk why 
            # get predictions from softmax probs but this is basically the same
            all_preds.append(probs.argmax(dim=1).cpu()) 

    return (torch.cat(all_scores), torch.cat(all_labels), torch.cat(all_preds),)


#tau
def calibrate_threshold(val_scores, val_labels, tpr_target: float = 0.95):
    id_scores = val_scores[val_labels >= 0].numpy()
    threshold = np.percentile(id_scores, 100 * (1 - tpr_target))
    print(f"[Threshold] {threshold:.4f}  (TPR target {tpr_target:.0%})")
    return threshold


####### PLOTS ###############

def plot_score_histogram(scores, labels, threshold, output_dir, tag):
    id_scores = scores[labels >= 0].numpy()
    ood_scores = scores[labels == -1].numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(id_scores,  bins=50, alpha=0.6,
            density=True, color="steelblue", label="ID")
    ax.hist(ood_scores, bins=50, alpha=0.6,
            density=True, color="tomato",    label="OOD")
    ax.axvline(threshold, color="k", linestyle="--",
               label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Energy score  (log Σ exp logit)")
    ax.set_ylabel("Density")
    ax.set_title("Energy score distributions: ID vs OOD")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"energy_histogram_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc(scores, labels, output_dir, tag):
    binary = (labels == -1).int().numpy()
    scores_np = scores.numpy()

    # negate so that higher = more OOD for roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(binary, -scores_np)
    auroc = roc_auc_score(binary, -scores_np)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, color="steelblue",
            label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Energy Score OOD — ROC Curve")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"energy_roc_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def run_energy_ood(model, val_loader, test_loader, device, output_dir, setup_name, datetime_tag=None):
    """Full energy score pipeline, run in ood_utils.py"""
    model.eval()
    torch.set_grad_enabled(False)

    tag = datetime_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting val scores for threshold calibration.")
    val_scores, val_labels, _ = get_energy_scores(model, val_loader, device)

    id_energy = val_scores[val_labels >= 0].numpy()
    ood_energy = val_scores[val_labels == -1].numpy()
    print(f"ID  energy mean: {id_energy.mean():.4f}")
    print(f"OOD energy mean: {ood_energy.mean():.4f}")

    threshold = calibrate_threshold(val_scores, val_labels)

    print("Extracting test scores.")
    test_scores, test_labels, class_preds = get_energy_scores(
        model, test_loader, device
    )
    # debug prints
    print(f"ID val energy  — mean: {id_energy.mean():.3f}, "
          f"std: {id_energy.std():.3f}, "
          f"min: {id_energy.min():.3f}, "
          f"max: {id_energy.max():.3f}")
    print(f"Threshold: {threshold:.3f}")
    id_pass = (val_scores[val_labels >= 0] >= threshold).float().mean()
    print(
        f"Fraction of ID val passing threshold: {id_pass:.3f}")

    save_run(
        base_dir=output_dir,
        setup_name=setup_name,
        method="energy",
        test_scores=test_scores.numpy(),
        test_labels=test_labels.numpy(),
        class_preds=class_preds.numpy(),
        ood_higher=False,   # higher energy = more ID, so ood_higher=False
        meta={"threshold": threshold},
    )

    plot_score_histogram(test_scores, test_labels, threshold, output_dir, tag)
    plot_roc(test_scores, test_labels, output_dir, tag)

    return test_scores, test_labels, class_preds
