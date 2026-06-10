
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np



def save_run(base_dir, setup_name, method, test_scores, test_labels, class_preds, ood_higher, meta=None):
    """Saves info from an OOD detection run, used for future plotting (ROC curves comparisons across setups/methods)"""
    run_dir = Path(base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "scores.npy",      np.asarray(test_scores,  dtype=np.float32))
    np.save(run_dir / "labels.npy",      np.asarray(test_labels,  dtype=np.int32))
    np.save(run_dir / "class_preds.npy", np.asarray(class_preds,  dtype=np.int32))

    run_meta = {
        "setup":      setup_name,
        "method":     method,
        "ood_higher": ood_higher,
        "saved_at":   datetime.now().isoformat(timespec="seconds"),
        **(meta or {}),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[save_run] Saved → {run_dir}")
    return run_dir



def load_run(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "run_meta.json") as f:
        meta = json.load(f)

    return {
        "scores":      np.load(run_dir / "scores.npy"),
        "labels":      np.load(run_dir / "labels.npy"),
        "class_preds": np.load(run_dir / "class_preds.npy"),
        "ood_higher":  meta.pop("ood_higher"),
        "meta":        meta,
    }
