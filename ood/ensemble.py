import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score
from typing import List, Optional, Tuple
from utils.save_run import save_run

from networks.architectures import TinyNN_NoDropout, TinyWideNN, TinyDeepNN, TinyNN_BatchNorm


# rebuild architecture
def build_model_from_params(params, input_size, num_classes):
    arch = params['architecture']

    hs = params.get(
        'hidden_size',
        params.get(
            'wide_hidden_size',
            params.get(
                'deep_hidden_size',
                params.get('bn_hidden_size')
            )
        )
    )

    nl = params.get('num_hidden_layers',params.get('bn_num_hidden_layers'))

    if arch == 'TinyNN_NoDropout':
        return TinyNN_NoDropout(input_size, num_classes, hs)
    elif arch == 'TinyWideNN':
        return TinyWideNN(input_size, num_classes, hs)
    elif arch == 'TinyDeepNN':
        if nl is None:
            raise ValueError("Missing num_hidden_layers for TinyDeepNN checkpoint.")
        return TinyDeepNN(input_size, num_classes, hs, nl)
    elif arch == 'TinyNN_BatchNorm':
        if nl is None:
            raise ValueError("Missing num_hidden_layers for TinyNN_BatchNorm checkpoint.")
        return TinyNN_BatchNorm(input_size, num_classes, hs, nl)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def train_single_member(model, train_loader, val_loader, device, arch_params, num_epochs=100,
                         lr=1e-3, weight_decay=1e-4, patience=10, min_delta=0.001, save_path=None):
    """Train one ensemble member with standard cross-entropy adn random init"""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    best_val_bal_acc = 0.0
    no_improve       = 0

    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs    = model(data)
                val_loss  += criterion(outputs, targets).item()
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())

        val_loss    /= len(val_loader)
        val_bal_acc  = balanced_accuracy_score(
            torch.cat(all_targets).numpy(),
            torch.cat(all_preds).numpy()
        ) * 100

        scheduler.step(val_loss)

        # save checkpoint
        if val_bal_acc > best_val_bal_acc + min_delta:
            best_val_bal_acc = val_bal_acc
            no_improve       = 0
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'architecture': arch_params['architecture'],
                    'input_size': arch_params['input_size'],
                    'num_classes': arch_params['num_classes'],
                    'hidden_size': arch_params.get(
                        'hidden_size',
                        arch_params.get(
                            'wide_hidden_size',
                            arch_params.get(
                                'deep_hidden_size',
                                arch_params.get('bn_hidden_size')
                            )
                        )
                    ),
                    'num_hidden_layers': arch_params.get(
                        'num_hidden_layers',
                        arch_params.get('bn_num_hidden_layers')
                    ),
                }, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        if epoch % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | val_loss={val_loss:.4f} | "
                  f"val_bal_acc={val_bal_acc:.2f}%")

    return best_val_bal_acc


def train_ensemble(checkpoint_path, train_loader, val_loader, ensemble_dir, device="cuda", M=10, num_epochs=100, lr=1e-3, weight_decay=1e-4):
    """ Train all, arch from checkpoint, no weights, random initialisations from gaussian distribution."""
    os.makedirs(ensemble_dir, exist_ok=True)

    checkpoint  = torch.load(checkpoint_path, map_location="cpu")
    input_size  = checkpoint['input_size']
    num_classes = checkpoint['num_classes']
    arch        = checkpoint['architecture']
    print(f"Architecture : {arch}")
    print(f"Input size   : {input_size}  |  Classes: {num_classes}")
    print(f"Training {M} ensemble members, saving to {ensemble_dir}\n")

    member_paths = []

    for m in range(M):
        print(f"── Member {m + 1}/{M} ──────────────────────────────────────────")
        model     = build_model_from_params(checkpoint, input_size, num_classes)
        save_path = os.path.join(ensemble_dir, f"ensemble_member_{m + 1}.pth")

        best_acc = train_single_member(
            model, train_loader, val_loader,
            arch_params=checkpoint,
            device=device, num_epochs=num_epochs,
            lr=lr, weight_decay=weight_decay,
            save_path=save_path,
        )
        print(f"  Member {m + 1} done — best val bal_acc = {best_acc:.2f}%\n")
        member_paths.append(save_path)

    return member_paths


def load_ensemble_members(ensemble_dir):
    member_paths = sorted(
        glob.glob(os.path.join(ensemble_dir, "ensemble_member_*.pth"))
    )
    if not member_paths:
        raise FileNotFoundError(
            f"No ensemble members found in '{ensemble_dir}'. "
            "Set train_ood=True to train them first."
        )
    print(f"Loaded {len(member_paths)} ensemble members from '{ensemble_dir}'")
    return member_paths


def get_ensemble_scores(member_paths, dataloader, device):
    """Ensemble MSP scores and labels, higher score = more ID"""
    # softmax probabilities for all members
    all_member_probs = []

    for path in member_paths:
        ckpt  = torch.load(path, map_location="cpu")
        model = build_model_from_params(ckpt, ckpt['input_size'], ckpt['num_classes'])
        model.load_state_dict(ckpt['model_state_dict'])

        model.to(device)
        model.eval()

        member_probs = []
        all_labels   = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                probs = F.softmax(model(x), dim=1)
                member_probs.append(probs.cpu())
                all_labels.append(y)

        all_member_probs.append(torch.cat(member_probs))

    labels = torch.cat(all_labels)

    # average over members
    mean_probs    = torch.stack(all_member_probs, dim=0).mean(dim=0)   
    ensemble_msp  = mean_probs.max(dim=1).values                       
    class_preds   = mean_probs.argmax(dim=1)
    return ensemble_msp, labels, class_preds


# tau
def calibrate_threshold(val_scores: torch.Tensor, val_labels: torch.Tensor, tpr_target: float = 0.95):
    id_scores = val_scores[val_labels >= 0].numpy()
    threshold = np.percentile(id_scores, 100 * (1 - tpr_target))
    print(f"[Threshold] {threshold:.4f}  (TPR target {tpr_target:.0%})")
    return threshold


######### PLOTS ###################
def plot_score_histogram(scores, labels, threshold, output_dir, tag):
    id_scores  = scores[labels >= 0].numpy()
    ood_scores = scores[labels == -1].numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(id_scores,  bins=50, alpha=0.6, density=True, color='steelblue', label='ID')
    ax.hist(ood_scores, bins=50, alpha=0.6, density=True, color='tomato',    label='OOD')
    ax.axvline(threshold, color='k', linestyle='--', label=f'Threshold ({threshold:.2f})')
    ax.set_xlabel('Ensemble MSP score')
    ax.set_ylabel('Density')
    ax.set_title('Ensemble MSP score distributions: ID vs OOD')
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"ensemble_histogram_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_ensemble_ood(
    member_paths: List[str],
    val_loader,
    test_loader,
    setup_name:   str,
    device:       str,
    output_dir:   str,
    datetime_tag: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Run in ood_utils.py fo inference, needs trained and loaded ensemble"""
    tag = datetime_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting val scores for threshold calibration")
    val_scores, val_labels, _ = get_ensemble_scores(member_paths, val_loader, device)
    threshold = calibrate_threshold(val_scores, val_labels)

    print("Extracting test scores")
    test_scores, test_labels, class_preds = get_ensemble_scores(member_paths, test_loader, device)

    save_run(
        base_dir=output_dir,
        setup_name=setup_name,
        method="ensemble",
        test_scores=test_scores.numpy(),
        test_labels=test_labels.numpy(),
        class_preds=class_preds.numpy() if class_preds is not None else np.zeros(len(test_labels), dtype=np.int32),
        ood_higher=False,               # ensemble MSP: higher = more ID
        meta={"M": len(member_paths)},
    )

    plot_score_histogram(test_scores, test_labels, threshold, output_dir, tag)

    return test_scores, test_labels, class_preds