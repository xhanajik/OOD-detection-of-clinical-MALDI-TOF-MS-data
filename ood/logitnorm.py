from networks.architectures import TinyNN_NoDropout, TinyWideNN, TinyDeepNN, TinyNN_BatchNorm
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import wandb
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import shutil
import os

"""
# fix w&b issue on the hpc
scratch = os.environ.get("VSC_SCRATCH", os.getcwd())
tmp_dir = os.path.join(scratch, "tmp")
wandb_dir = os.path.join(scratch, "wandb")
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(wandb_dir, exist_ok=True)
os.makedirs(os.path.join(wandb_dir, "cache"), exist_ok=True)
os.makedirs(os.path.join(wandb_dir, "config"), exist_ok=True)
os.environ["TMPDIR"] = tmp_dir
os.environ["WANDB_DIR"] = wandb_dir
os.environ["WANDB_CACHE_DIR"] = os.path.join(wandb_dir, "cache")
os.environ["WANDB_CONFIG_DIR"] = os.path.join(wandb_dir, "config")"""
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04):
        super().__init__()
        self.tau = tau

    def forward(self, logits, labels):
        norms = torch.norm(logits, p=2, dim=1, keepdim=True).clamp(min=1e-7)
        logits_normed = logits / (norms * self.tau)
        return F.cross_entropy(logits_normed, labels)


def build_model_from_params(params, input_size, num_classes):
    """ Reconstruct the best arch, without weights"""
    arch = params['architecture']
    hs = params.get('hidden_size',
                    params.get('wide_hidden_size',
                               params.get('deep_hidden_size',
                                          params.get('bn_hidden_size'))))
    nl = params.get('num_hidden_layers',
                    params.get('bn_num_hidden_layers', None))

    if arch == 'TinyNN_NoDropout':
        return TinyNN_NoDropout(input_size, num_classes, hs)
    elif arch == 'TinyWideNN':
        return TinyWideNN(input_size, num_classes, hs)
    elif arch == 'TinyDeepNN':
        return TinyDeepNN(input_size, num_classes, hs, nl)
    elif arch == 'TinyNN_BatchNorm':
        return TinyNN_BatchNorm(input_size, num_classes, hs, nl)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def compute_metrics(outputs, targets):
    _, preds = outputs.max(1)
    p, t = preds.cpu().numpy(), targets.cpu().numpy()
    return {
        'accuracy':          accuracy_score(t, p) * 100,
        'balanced_accuracy': balanced_accuracy_score(t, p) * 100,
        'f1_score':          f1_score(t, p, average='weighted', zero_division=0) * 100,
        'precision':         precision_score(t, p, average='weighted', zero_division=0) * 100,
        'recall':            recall_score(t, p, average='weighted', zero_division=0) * 100,
    }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_logitnorm(model, train_loader, val_loader, criterion, optimizer,
                    scheduler, num_epochs, device, save_path,
                    early_stopping=None, checkpoint_params=None):
    """optimizes for balanced accuracy"""
    best_val_bal_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_outs, train_tgts = [], []

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_outs.append(outputs.detach())
            train_tgts.append(targets.detach())

        model.eval()
        val_loss = 0.0
        val_outs, val_tgts = [], []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                val_outs.append(outputs)
                val_tgts.append(targets)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_metrics = compute_metrics(
            torch.cat(train_outs), torch.cat(train_tgts))
        val_metrics = compute_metrics(
            torch.cat(val_outs),   torch.cat(val_tgts))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        wandb.log({
            'epoch':                    epoch + 1,
            'train_loss':               train_loss,
            'val_loss':                 val_loss,
            'learning_rate':            optimizer.param_groups[0]['lr'],
            'train_accuracy':           train_metrics['accuracy'],
            'train_balanced_accuracy':  train_metrics['balanced_accuracy'],
            'train_f1_score':           train_metrics['f1_score'],
            'train_recall':             train_metrics['recall'],
            'train_precision':          train_metrics['precision'],
            'val_accuracy':             val_metrics['accuracy'],
            'val_balanced_accuracy':    val_metrics['balanced_accuracy'],
            'val_f1_score':             val_metrics['f1_score'],
            'val_recall':               val_metrics['recall'],
            'val_precision':            val_metrics['precision'],
        })

        # save checkpoint
        if val_metrics['balanced_accuracy'] > best_val_bal_acc:
            best_val_bal_acc = val_metrics['balanced_accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'architecture':     model.__class__.__name__,
                'input_size':       next(iter(train_loader))[0].shape[1],
                'num_classes':      model.output.out_features,
                'hidden_size':      model.hidden[0].out_features,
                'num_hidden_layers': getattr(model, 'num_hidden_layers', None),
                'num_hidden_layers': checkpoint_params.get(
                    'num_hidden_layers',
                    checkpoint_params.get('bn_num_hidden_layers')
                ),
            }, save_path)
            if epoch % 10 == 0 or epoch == 0:
                print(f"  [epoch {epoch+1}] saved → {save_path} "
                      f"(bal_acc={best_val_bal_acc:.2f}%)")

        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
                  f"val_bal_acc={val_metrics['balanced_accuracy']:.2f}%")

        if early_stopping:
            early_stopping(val_metrics['balanced_accuracy'])
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_val_bal_acc


def run_logitnorm_gridsearch(checkpoint_path, train_loader, val_loader, setup_name, device="cuda",
                             taus=(0.01, 0.02, 0.04, 0.07, 0.1), learning_rates=(1e-4, 5e-4, 1e-3), num_epochs=100, wandb_project=None):
    """ to find best model, run a gridsearch to tune for tau (t-in the thesis) and lr"""
    project = wandb_project or f"logitnorm_{RUN_TIMESTAMP}"
    models_dir = os.path.join("./models", "logitnorm", setup_name)
    os.makedirs(models_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_size = checkpoint['input_size']
    num_classes = checkpoint['num_classes']
    architecture = checkpoint['architecture']
    print(f"Architecture : {architecture}")
    print(f"Input size   : {input_size}  |  Classes: {num_classes}")
    print(f"Grid         : {len(taus)} taus × {len(learning_rates)} lrs "
          f"= {len(taus)*len(learning_rates)} runs\n")
    print(f"Saving models to: {models_dir}\n")

    summary_rows = []

    for tau in taus:
        for lr in learning_rates:
            run_name = f"{architecture}_tau{tau}_lr{lr:.0e}"
            save_path = os.path.join(models_dir, f"{run_name}.pth")

            print(f"tau={tau}, lr={lr:.0e}, run={run_name}")
            model = build_model_from_params(
                checkpoint, input_size, num_classes)
            model.to(device)

            criterion = LogitNormLoss(tau=tau)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )
            early_stopping = EarlyStopping(patience=15)

            wandb.init(
                project=project,
                name=run_name,
                config={
                    'architecture':  architecture,
                    'tau':           tau,
                    'learning_rate': lr,
                    'weight_decay':  1e-4,
                    'num_epochs':    num_epochs,
                    'input_size':    input_size,
                    'num_classes':   num_classes,
                    'run_timestamp': RUN_TIMESTAMP,
                    'setup_name':    setup_name,
                },
                reinit=True,
            )

            best_val_bal_acc = train_logitnorm(
                model, train_loader, val_loader,
                criterion, optimizer, scheduler,
                num_epochs=num_epochs, device=device,
                save_path=save_path, early_stopping=early_stopping, checkpoint_params=checkpoint
            )

            wandb.log({'best_val_balanced_accuracy': best_val_bal_acc})
            wandb.finish()

            # add tau and lr info to model
            ckpt = torch.load(save_path, map_location="cpu")
            ckpt['tau'] = tau
            ckpt['lr'] = lr
            torch.save(ckpt, save_path)

            summary_rows.append({
                'tau':              tau,
                'lr':               lr,
                'best_val_bal_acc': best_val_bal_acc,
                'save_path':        save_path,
            })
            print(f"  ✓ best val bal_acc = {best_val_bal_acc:.2f}%")

    print("Grid search summary:")
    summary_rows.sort(key=lambda r: r['best_val_bal_acc'], reverse=True)
    for r in summary_rows:
        print(f"tau={r['tau']}, lr={r['lr']:.0e}"
              f"val_bal_acc={r['best_val_bal_acc']:.2f}%, save_path={r['save_path']}")

    best = summary_rows[0]
    print(f"\n  Best config : tau={best['tau']}, lr={best['lr']:.0e}  "
          f"bal_acc={best['best_val_bal_acc']:.2f}%")
    print(f"Checkpoint : {best['save_path']}")

    best_path = os.path.join(models_dir, "best_model.pth")
    shutil.copy(best['save_path'], best_path)
    print(f" Best model also copied to: {best_path}")

    return summary_rows
