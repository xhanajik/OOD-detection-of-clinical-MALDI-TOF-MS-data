import argparse
import time
import torch
import numpy as np
import random
import psutil
import datetime
import sys


from utils.config_utils import read_config
from utils.dataset_utils import load_dataset_driams
from utils.model_utils import load_network
from utils.print_header import print_header
from utils.ood_utils import run_ood_evaluation
from networks.utils import run_architecture_comparison_pipeline


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_init_info():
    print_header("INITIAL INFO")
    print("Start timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Memory usage (before loading):",
          psutil.virtual_memory().used / 1e9, "GB")

    print("Cuda available?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Forcing CUDA initialization", flush=True)
        torch.cuda.current_device()  # forces init
        a = torch.randn(1).to("cuda")
        print("Tensor moved to GPU successfully.")

    sys.stdout.flush()


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/Debug_MIX.yml"
    )
    args = parser.parse_args()

    main_config, dataset_config, ood_config = read_config(args.config_file)

    return args, main_config, dataset_config, ood_config


def peek_loader(loader, name, n=5):
    batch = next(iter(loader))
    samples = batch[0] if isinstance(batch, (list, tuple)) else batch
    print(f"\n[Repro check] First {n} samples of {name}:")
    print(samples[:n])


#######################
######## Main #########
#######################

if __name__ == "__main__":
    # ---------- init ----------
    print_init_info()
    args, main_config, dataset_config, ood_config = load_config()

    if "seed" in main_config:
        set_seed(main_config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- dataset ----------
    start = time.time()
    if dataset_config["name"] == "DRIAMS":
        data_module = load_dataset_driams(
            main_config, dataset_config, ood_config, train=True)
    elif dataset_config["name"] == "LM-UGent":
        #data_module = load_dataset_lmugent(
        #    main_config, dataset_config, ood_config, train=True)
        pass
    else:
        raise ValueError('Unknown dataset.')

    end = time.time()

    if main_config.get("verbose", False):
        print(f"Dataset loading time {end-start}", flush=True)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # ---------- train or load model ----------

    if ood_config["train"]:
        print_header("NETWORK TRAINING")
        architectures_to_test = [
            "TinyDeepNN", "TinyWideNN", "TinyNN_NoDropout", "TinyNN_BatchNorm"
        ]

        study, results, comparison_df = run_architecture_comparison_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            input_size=data_module.n_features(),
            num_classes=data_module.n_classes(distribution="id"),
            test_loader=test_loader,
            architectures_to_test=architectures_to_test,
            n_trials=30,
            top_n=5
        )

        if main_config.get("verbose", False):
            print("\n=== OPTUNA PIPELINE COMPLETED ===")
            print(f"Total trials run: {len(study.trials)}")
            print(
                f"Best architecture: {study.best_params.get('architecture', 'Unknown')}")
            print(f"Best balanced accuracy: {study.best_value:.4f}")

    elif main_config["name"] == "DRIAMS_all" or main_config["name"] == "LMUGENT_all":
        pass 
        # used for plotting   
    else:
        model = load_network(dataset_config=dataset_config,
                             ood_config=ood_config, device=device)

    # ---------- ood ----------
    run_ood_evaluation(main_config=main_config, dataset_config=dataset_config, ood_config=ood_config, model=model,
                       train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device, dataset=data_module)
