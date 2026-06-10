import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
import optuna
import numpy as np
import pandas as pd
import warnings
import wandb
import os
from datetime import datetime
from networks.architectures import (
    TinyNN_NoDropout, TinyWideNN, TinyDeepNN, TinyNN_BatchNorm
)

warnings.filterwarnings('ignore')

# macros
ARCH_LR_RANGES = {
    "TinyNN_NoDropout": (5e-4, 5e-3),
    "TinyWideNN":       (1e-4, 3e-3),
    "TinyDeepNN":       (5e-5, 1e-3),
    "TinyNN_BatchNorm": (1e-4, 3e-3),
}

ARCH_WARMUP_STEPS = {
    "TinyNN_NoDropout": 0,    # TinyNN doesnt really need warmup
    "TinyWideNN":       200,
    "TinyDeepNN":       300,
    "TinyNN_BatchNorm": 100,
}


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, downstream_scheduler=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.downstream = downstream_scheduler
        self.step_count = 0
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, val_loss=None):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(self.warmup_steps, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = max(base_lr / 100.0, base_lr * scale)
        elif self.downstream is not None:
            if isinstance(self.downstream, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    self.downstream.step(val_loss)
            else:
                self.downstream.step()

    def is_warming_up(self):
        return self.step_count <= self.warmup_steps


# BachNorm didnt reset between runs so manual reset
def reset_bn(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.reset_running_stats()


def calculate_balanced_accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    return balanced_accuracy_score(targets.cpu().numpy(), predicted.cpu().numpy())


def calculate_metrics(outputs, targets):
    _, predicted = outputs.max(1)
    p = predicted.cpu().numpy()
    t = targets.cpu().numpy()
    return {
        'accuracy':          accuracy_score(t, p) * 100,
        'balanced_accuracy': balanced_accuracy_score(t, p) * 100,
        'precision':         precision_score(t, p, average='weighted', zero_division=0) * 100,
        'recall':            recall_score(t, p, average='weighted', zero_division=0) * 100,
        'f1_score':          f1_score(t, p, average='weighted', zero_division=0) * 100,
    }


def calculate_class_weights(train_loader, num_classes):
    class_counts = torch.zeros(num_classes)
    total = 0
    for _, targets in train_loader:
        for c in range(num_classes):
            class_counts[c] += (targets == c).sum().item()
        total += targets.size(0)
    return total / (num_classes * class_counts.clamp(min=1))


def create_weighted_loss(train_loader, num_classes, device, label_smoothing=0.1):
    weights = calculate_class_weights(train_loader, num_classes).to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def create_standard_loss(label_smoothing=0.1):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _arch_params_from_trial(trial, architecture):
    """Specific params for the individual networks."""
    if architecture == 'TinyNN_NoDropout':
        return {
            'hidden_size': trial.suggest_categorical(
                'hidden_size', [64, 96, 128, 192, 256, 384, 512])
        }
    elif architecture == 'TinyWideNN':
        return {
            'hidden_size': trial.suggest_categorical(
                'wide_hidden_size', [256, 384, 512, 768, 1024])
        }
    elif architecture == 'TinyDeepNN':
        return {
            'hidden_size': trial.suggest_categorical(
                'deep_hidden_size', [96, 128, 192, 256, 384]),
            'num_hidden_layers': trial.suggest_categorical(
                'num_hidden_layers', [2, 3, 4, 5]),
        }
    elif architecture == 'TinyNN_BatchNorm':
        return {
            'hidden_size': trial.suggest_categorical(
                'bn_hidden_size', [64, 96, 128, 192, 256, 384, 512]),
            'num_hidden_layers': trial.suggest_categorical(
                'bn_num_hidden_layers', [2, 3, 4]),
        }
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def _build(architecture, input_size, num_classes, arch_params):
    """Make nn.Module from arch_params dict."""
    hs = arch_params['hidden_size']
    nl = arch_params.get('num_hidden_layers', None)
    if architecture == 'TinyNN_NoDropout':
        return TinyNN_NoDropout(input_size, num_classes, hs)
    elif architecture == 'TinyWideNN':
        return TinyWideNN(input_size, num_classes, hs)
    elif architecture == 'TinyDeepNN':
        return TinyDeepNN(input_size, num_classes, hs, nl)
    elif architecture == 'TinyNN_BatchNorm':
        return TinyNN_BatchNorm(input_size, num_classes, hs, nl)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def create_model(trial, input_size, num_classes, architecture):
    """Create model for Optuna."""
    arch_params = _arch_params_from_trial(trial, architecture)
    model = _build(architecture, input_size, num_classes, arch_params)
    print(f"  Built {architecture} with params: {arch_params}")
    return model, arch_params


def build_model_from_params(params, input_size, num_classes):
    """Reconstruct a model from params dict."""
    architecture = params['architecture']
    arch_params = {'hidden_size': params.get(
        'hidden_size',
        params.get('wide_hidden_size',
                   params.get('deep_hidden_size',
                              params.get('bn_hidden_size'))))}

    if architecture in ('TinyDeepNN', 'TinyNN_BatchNorm'):
        arch_params['num_hidden_layers'] = params.get(
            'num_hidden_layers',
            params.get('bn_num_hidden_layers'))

    return _build(architecture, input_size, num_classes, arch_params)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=100, device='cuda', early_stopping=None,
                use_wandb=True, save_path=None, warmup_scheduler=None, hidden_size=None, input_size=None, output_size=None):
    """Training loop, saves best model's state_dict to save_path.
    """

    train_losses, val_losses, val_metrics_history = [], [], []
    best_val_balanced_acc = 0.0
    model.to(device)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(
            save_path) else '.', exist_ok=True)

    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss = 0.0
        all_train_out, all_train_tgt = [], []

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            all_train_out.append(outputs.detach())
            all_train_tgt.append(targets.detach())

            # warmup scheduler steps every batch
            if warmup_scheduler is not None:
                warmup_scheduler.step()

        # val
        model.eval()
        val_loss = 0.0
        all_val_out, all_val_tgt = [], []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                all_val_out.append(outputs)
                all_val_tgt.append(targets)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_metrics = calculate_metrics(
            torch.cat(all_train_out), torch.cat(all_train_tgt))
        val_metrics = calculate_metrics(
            torch.cat(all_val_out), torch.cat(all_val_tgt))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'train_accuracy': train_metrics['accuracy'],
                'train_balanced_accuracy': train_metrics['balanced_accuracy'],
                'train_f1_score': train_metrics['f1_score'],
                'train_recall': train_metrics['recall'],
                'train_precision': train_metrics['precision'],
                'val_accuracy': val_metrics['accuracy'],
                'val_balanced_accuracy': val_metrics['balanced_accuracy'],
                'val_f1_score': val_metrics['f1_score'],
                'val_recall': val_metrics['recall'],
                'val_precision': val_metrics['precision'],
            })
        if warmup_scheduler is not None:
            if not warmup_scheduler.is_warming_up():
                warmup_scheduler.step(val_loss=val_loss)
        elif scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # save best model's state_dic
        if val_metrics['balanced_accuracy'] > best_val_balanced_acc:
            best_val_balanced_acc = val_metrics['balanced_accuracy']
            if save_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "hidden_size": hidden_size,
                }, save_path)
                if epoch % 10 == 0 or epoch == 0:
                    print(f"  [epoch {epoch+1}] New best saved → {save_path} "
                          f"(bal_acc={best_val_balanced_acc:.2f}%)")

        if early_stopping:
            early_stopping(val_metrics['balanced_accuracy'])
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # prints balanced acc every 10 epochs
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"val_bal_acc={val_metrics['balanced_accuracy']:.2f}%")

    return train_losses, val_losses, val_metrics_history, best_val_balanced_acc


def objective(trial, train_loader, val_loader, input_size, num_classes,
              architectures_to_test):
    """Optuna objective."""
    # separate seeding
    torch.manual_seed(trial.number + 42)
    np.random.seed(trial.number + 42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        architecture = trial.suggest_categorical(
            'architecture', architectures_to_test)

        # set lr rates to test from macro
        lr_lo, lr_hi = ARCH_LR_RANGES[architecture]
        learning_rate = trial.suggest_float(
            'learning_rate', lr_lo, lr_hi, log=True)

        weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical(
            'optimizer', ['adam', 'adamw', 'sgd'])

        # set other options to test
        momentum = None
        nesterov = None
        if optimizer_name == 'sgd':
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            nesterov = trial.suggest_categorical('nesterov', [True, False])

        use_scheduler = trial.suggest_categorical(
            'use_scheduler', [True, False])
        scheduler_type = step_size = gamma = T_max = None
        if use_scheduler:
            scheduler_type = trial.suggest_categorical(
                'scheduler_type', ['step', 'cosine', 'reduce_on_plateau'])
            if scheduler_type == 'step':
                step_size = trial.suggest_int('step_size', 10, 50)
                gamma = trial.suggest_float('gamma', 0.1, 0.5)
            elif scheduler_type == 'cosine':
                T_max = trial.suggest_int('T_max', 20, 100)

        use_class_weights = trial.suggest_categorical(
            'use_class_weights', [True, False])
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)
        if use_class_weights:
            criterion = create_weighted_loss(train_loader, num_classes, device,
                                             label_smoothing=label_smoothing)
        else:
            criterion = create_standard_loss(label_smoothing=label_smoothing)

        print(f"\nTrial {trial.number}: {architecture} | "
              f"lr={learning_rate:.2e} | wd={weight_decay:.2e} | "
              f"opt={optimizer_name} | weighted_loss={use_class_weights}")

        # save path per trial in outputs/<timestamp>/trial_{n}_Arch.pth
        save_path = os.path.join(
            OUTPUTS_DIR, f"trial_{trial.number}_{architecture}.pth")

        model, arch_params = create_model(
            trial, input_size, num_classes, architecture)
        model.apply(reset_bn)
        model.to(device)

        wandb.init(
            project=f"bacteria_nn_{RUN_TIMESTAMP}",
            name=f"trial_{trial.number}_{architecture}",
            config={
                "trial_number":      trial.number,
                "architecture":      architecture,
                "learning_rate":     learning_rate,
                "weight_decay":      weight_decay,
                "optimizer":         optimizer_name,
                "use_scheduler":     use_scheduler,
                "use_class_weights": use_class_weights,
                "label_smoothing":   label_smoothing,
                "warmup_steps":      ARCH_WARMUP_STEPS[architecture],
                "run_timestamp":     RUN_TIMESTAMP,
                **arch_params,
            },
            reinit=True,
        )

        # optimizers
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                momentum=momentum, nesterov=nesterov)

        # schedulers
        scheduler = None
        if use_scheduler:
            if scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max)
            elif scheduler_type == 'reduce_on_plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=5)

        early_stopping = EarlyStopping(patience=10)  # maybe 5/10 is enough

        # warmup
        warmup_steps = ARCH_WARMUP_STEPS[architecture]
        warmup_sched = WarmupScheduler(optimizer, warmup_steps,
                                       downstream_scheduler=scheduler) \
            if warmup_steps > 0 else None
        sched_arg = scheduler if warmup_sched is None else None

        _, _, _, best_val_balanced_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, sched_arg,
            num_epochs=35, device=device, early_stopping=early_stopping,
            use_wandb=True, save_path=save_path, warmup_scheduler=warmup_sched, input_size=input_size, output_size=num_classes, hidden_size=arch_params["hidden_size"]
        )

        # update checkpoint
        checkpoint = torch.load(save_path)
        checkpoint.update({
            "architecture": architecture,
            "input_size":   input_size,
            "num_classes":  num_classes,
            **arch_params,
        })
        torch.save(checkpoint, save_path)

        wandb.log({
            "best_val_balanced_accuracy": best_val_balanced_acc,
            "model_path": save_path,
        })

        return best_val_balanced_acc

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0
    finally:
        wandb.finish()


def optimize_architectures(train_loader, val_loader, input_size, num_classes,
                           architectures_to_test=None, n_trials=100):

    if architectures_to_test is None:
        architectures_to_test = [
            "TinyDeepNN", "TinyWideNN", "TinyNN_NoDropout", "TinyNN_BatchNorm"
        ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Architectures:", architectures_to_test)
    print(f"Run ID:   {RUN_TIMESTAMP}")
    print(f"Outputs:  {OUTPUTS_DIR}/")

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective(
            trial, train_loader, val_loader, input_size, num_classes,
            architectures_to_test),
        n_trials=n_trials,
        n_jobs=1,
    )
    return study


def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate using balanced acc. + other metrics (acc, precision, recall,...)"""
    model.eval()
    all_out, all_tgt = [], []
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            all_out.append(model(data))
            all_tgt.append(targets)
    all_out = torch.cat(all_out)
    all_tgt = torch.cat(all_tgt)
    metrics = calculate_metrics(all_out, all_tgt)
    _, preds = all_out.max(1)
    report = classification_report(
        all_tgt.cpu().numpy(), preds.cpu().numpy(),
        target_names=[f'Class_{i}' for i in range(num_classes)],
        output_dict=True)
    return metrics, report, preds.cpu().numpy(), all_tgt.cpu().numpy()


def compare_architectures(study, train_loader, val_loader, test_loader,
                          input_size, num_classes, top_n=5):
    """Compare top-n architectures, retrain and evaluate to choose best."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True)[:top_n]

    results = []
    print(f"\n COMPARING TOP {top_n} ARCHITECTURES")

    for rank, trial in enumerate(top_trials, 1):
        arch = trial.params['architecture']
        print(f"\nRank {rank}: {arch}  (trial {trial.number}, "
              f"optuna val_bal_acc={trial.value:.4f})")

        try:
            # reconstruct not build
            model = build_model_from_params(
                trial.params, input_size, num_classes)
            model.to(device)

            ls = trial.params.get('label_smoothing', 0.1)
            criterion = (create_weighted_loss(train_loader, num_classes, device,
                                              label_smoothing=ls)
                         if trial.params.get('use_class_weights')
                         else create_standard_loss(label_smoothing=ls))

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=trial.params['learning_rate'],
                weight_decay=trial.params['weight_decay'])
            downstream = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10)

            warmup_steps = ARCH_WARMUP_STEPS.get(arch, 0)
            warmup_sched = WarmupScheduler(optimizer, warmup_steps,
                                           downstream_scheduler=downstream) \
                if warmup_steps > 0 else None
            sched_arg = downstream if warmup_sched is None else None

            early_stopping = EarlyStopping(patience=20)

            retrain_path = os.path.join(
                OUTPUTS_DIR, f"retrain_top{rank}_trial{trial.number}_{arch}.pth")

            print("  Retraining …")
            _, _, _, best_bal_acc = train_model(
                model, train_loader, val_loader, criterion, optimizer, sched_arg,
                num_epochs=100, device=device, early_stopping=early_stopping,
                use_wandb=False, save_path=retrain_path,
                warmup_scheduler=warmup_sched)

            # load weights
            checkpoint = torch.load(retrain_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded best weights (val_bal_acc={best_bal_acc:.2f}%)")

            val_metrics, val_report, _, _ = evaluate_model(
                model, val_loader, device, num_classes)

            test_metrics = test_report = None
            if test_loader is not None:
                test_metrics, test_report, _, _ = evaluate_model(
                    model, test_loader, device, num_classes)

            results.append({
                'architecture':  arch,
                'trial_number':  trial.number,
                'parameters':    trial.params,
                'model_path':    retrain_path,
                'val_metrics':   val_metrics,
                'test_metrics':  test_metrics,
                'val_report':    val_report,
                'test_report':   test_report,
            })

            print("  Val  :", {k: f"{v:.2f}%" for k, v in val_metrics.items()})
            if test_metrics:
                print("  Test :", {k: f"{v:.2f}%" for k,
                      v in test_metrics.items()})

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Error: {e}")

    return results


# run in main
def run_architecture_comparison_pipeline(train_loader, val_loader, input_size,
                                         num_classes, test_loader=None,
                                         architectures_to_test=None,
                                         n_trials=50, top_n=5):
    # make a global timestamp and use it for naming
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUTS_DIR = os.path.join("outputs", RUN_TIMESTAMP)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("NEURAL NETWORK ARCHITECTURE COMPARISON")
    print(f"Run ID:       {RUN_TIMESTAMP}")
    print(f"Outputs dir:  {OUTPUTS_DIR}")
    print(f"Input size:   {input_size}  |  Classes: {num_classes}")

    print("\nRUNNING OPTUNA OPTIMIZATION")
    study = optimize_architectures(
        train_loader, val_loader, input_size, num_classes,
        architectures_to_test, n_trials)

    print(f"\nBest trial #{study.best_trial.number}: "
          f"val_bal_acc={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nCOMPARING TOP {top_n} ARCHITECTURES")
    results = compare_architectures(
        study, train_loader, val_loader, test_loader,
        input_size, num_classes, top_n)

    return study, results, df
