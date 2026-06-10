import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from utils.save_run import save_run


def get_msp_scores(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            msp = probs.max(dim=1).values
            all_scores.append(msp.cpu())
            all_labels.append(y)
            all_preds.append(probs.argmax(dim=1).cpu())
    
    return torch.cat(all_scores), torch.cat(all_labels), torch.cat(all_preds)

def treshhold_tuning(scores, labels):
    id_scores = scores[labels >= 0].numpy()
    threshold = np.percentile(id_scores, 5)
    return threshold

# PLOTS

def score_histogram(scores, labels, threshold, output_dir):
    id_scores  = scores[labels >= 0].numpy()
    ood_scores = scores[labels == -1].numpy()
    
    plt.figure(figsize=(8, 4))
    plt.hist(id_scores,  bins=50, alpha=0.6, label='ID',  density=True, color='steelblue')
    plt.hist(ood_scores, bins=50, alpha=0.6, label='OOD', density=True, color='tomato')
    plt.axvline(threshold, color='k', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.xlabel('MSP score')
    plt.ylabel('Density')
    plt.legend()
    plt.title('MSP score distributions: ID vs OOD')
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(output_dir, f'msp_histogram_{timestamp}.png'), dpi=150)
    plt.close()


def run_all_msp(model, val_loader, test_loader, device, output_dir, setup_name):
    """Full MSP pipeline, run in ood_utils.py"""
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting train scores")
    train_scores, train_labels, class_preds = get_msp_scores(model, val_loader, device)
    threshold = treshhold_tuning(train_scores, train_labels)

    print("Extracting test scores")
    test_scores, test_labels, class_preds = get_msp_scores(model, test_loader, device)

    save_run(
        base_dir=output_dir,
        setup_name=setup_name,          # add this as a parameter to run_ensemble_ood
        method="msp",
        test_scores=test_scores.numpy(),
        test_labels=test_labels.numpy(),
        class_preds=class_preds.numpy() if class_preds is not None else np.zeros(len(test_labels), dtype=np.int32),
        ood_higher=False,               # ensemble MSP: higher = more ID
    )

    score_histogram(test_scores, test_labels, threshold, output_dir)
    return  test_scores, test_labels, class_preds