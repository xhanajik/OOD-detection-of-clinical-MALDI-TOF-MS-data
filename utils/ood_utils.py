from ood.knn import run_knn_ood
from ood.mahalonobis import run_mahalanobis_lda
from ood.msp import run_all_msp, get_msp_scores
from ood.gmm import run_gmm_lda
from ood.energy_maha import run_energy_maha_ood
from ood.msp_maha import run_msp_maha_ood
from ood.energy import run_energy_ood
from ood.logitnorm_eval import logitnorm_eval_with_msp
from ood.logitnorm import run_logitnorm_gridsearch
from ood.ensemble import train_ensemble, run_ensemble_ood, load_ensemble_members
from utils.print_header import print_header
from utils.metrics_utils import OODMetrics
from utils.report_utils import OODReport
import os


def run_ood_evaluation(main_config, dataset_config, ood_config, model, train_loader, val_loader, test_loader, device, dataset):
    # "knn", "maha", "msp", "logitnorm", "vim", "deep_ensemble", "gmm"
    print_header("OOD DETECTION METHOD TRAINING")
    output_dir = f"{main_config['output_dir']}/{ood_config['method']}"
    os.makedirs(output_dir, exist_ok=True)

############# KNN ###################

    if ood_config["method"] == "knn":
        test_scores, test_labels, class_preds, figure_paths = run_knn_ood(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            device=device,
            # Options: "raw", "pca", "embedding"
            variant=ood_config["knn_variant"],
            metric="cosine",  # Options: "euclidean", "cosine", "manhattan", "canberra"
            pca_components=0.8,          # or 0.95 for variance-based
            fpr_target=0.05,
            output_dir=output_dir,
            setup_name=ood_config["setup"]
        )
        ood_higher = True
        pass

############# MAHALANOBIS ###################

    elif ood_config["method"] == "maha":
        test_scores, test_labels, class_preds, figure_paths = run_mahalanobis_lda(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=output_dir,
            setup_name=ood_config["setup"],
            cov_type="empirical",   # "ledoitwolf" or "empirical"
        )
        ood_higher = True

############# MSP ###################

    elif ood_config["method"] == "msp":
        test_scores, test_labels, class_preds = run_all_msp(model=model,
                                                            val_loader=val_loader,
                                                            test_loader=test_loader,
                                                            device=device,
                                                            output_dir=output_dir,
                                                            setup_name=ood_config["setup"]
                                                            )
        ood_higher = False
        figure_paths = []

############# LOGITNORM ###################

    elif ood_config["method"] == "logitnorm":
        if ood_config["train_ood"]:
            summary = run_logitnorm_gridsearch(
                checkpoint_path=ood_config["pretrained_model_path"],
                train_loader=train_loader,
                val_loader=val_loader,
                setup_name=main_config["name"],
                device=device,
                taus=ood_config.get("taus", (0.01, 0.02, 0.04, 0.07, 0.1)),
                learning_rates=ood_config.get(
                    "learning_rates", (1e-4, 5e-4, 1e-3)),
                num_epochs=ood_config.get("num_epochs", 100),
            )
            logitnorm_path = summary[0]["save_path"]   # best model
        logitnorm_path = os.path.join(
            "./models", "logitnorm", main_config["name"], "best_model.pth")
        test_scores, test_labels, class_preds = logitnorm_eval_with_msp(logitnorm_path,
                                                                        val_loader=val_loader, test_loader=test_loader, output_dir=output_dir,
                                                                        setup_name=ood_config["setup"], device=device)

        ood_higher = False
        figure_paths = []

############# ENSEMBLE ###################

    elif ood_config["method"] == "deep_ensemble":
        ensemble_dir = os.path.join("models", "ensemble", main_config["name"])
        if ood_config["train_ood"]:
            member_paths = train_ensemble(
                checkpoint_path=ood_config["pretrained_model_path"],
                train_loader=train_loader,
                val_loader=val_loader,
                # output_dir      = output_dir,
                ensemble_dir=ensemble_dir,
                device=device,
                M=ood_config.get("M", 10),
                num_epochs=ood_config.get("num_epochs", 100),
                lr=ood_config.get("lr", 1e-3),
                weight_decay=ood_config.get("weight_decay", 1e-4),
            )
        else:
            member_paths = load_ensemble_members(ensemble_dir)
        test_scores, test_labels, class_preds = run_ensemble_ood(
            member_paths=member_paths,  # list of M checkpoint paths
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            setup_name=main_config["name"],
        )
        ood_higher = False
        figure_paths = []

############# GMM ###################

    elif ood_config["method"] == "gmm":
        test_scores, test_labels, class_preds, figure_paths = run_gmm_lda(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=output_dir,
            setup_name=main_config["name"],
            random_state=24,
        )
        ood_higher = True

############# ENERGY ###################

    elif ood_config["method"] == "energy":
        test_scores, test_labels, class_preds = run_energy_ood(
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            setup_name=ood_config["setup"]
        )
        ood_higher = False  # higher energy = more ID, lower = more OOD
        figure_paths = []

############# ENERGY + MAHA ###################

    elif ood_config["method"] == "energy_maha":
        test_scores, test_labels, class_preds, figure_paths = run_energy_maha_ood(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            setup_name=ood_config["setup"],
            cov_type="empirical",
        )
        ood_higher = True

############# MSP + MAHA ###################

    elif ood_config["method"]=="msp_maha":
        test_scores, test_labels, class_preds, figure_paths = run_msp_maha_ood(
            model = model,
            train_loader = train_loader,
            val_loader=val_loader,
            test_loader = test_loader,
            device = device,
            output_dir=output_dir,
            setup_name = ood_config["setup"],
            cov_type= "empirical",
            )
        ood_higher = True
    else:
        raise ValueError(
            f"Unknown OOD method: '{ood_config['method']}'. "
            "Supported methods are: knn, maha, msp, logitnorm, vim, deep_ensemble, gmm, energy, energy_maha."
        )

    metrics = OODMetrics(
        setup=main_config["name"], abundance_filter=dataset_config["abundance_percentage"],
        dataset_name=dataset_config["name"], model_path="TODO",
        hierarchy_level=ood_config["hierarchy_level"],
    )
    metrics.compute(test_scores, test_labels, class_preds,
                    ood_higher=ood_higher)  # class_names=class_names)
    metrics.print_summary()

    path = main_config["output_dir"]
    report_dir = f"{path}/reports/{ood_config['method']}"

    if ood_config["method"] == "knn":
        report_dir = f"{path}/reports/{ood_config['method']}/{ood_config['knn_variant']}"
    else:
        report_dir = f"{path}/reports/{ood_config['method']}"

    report = OODReport(metrics, output_dir=report_dir)
    report.build(
        dataset=dataset, main_config=main_config,
        dataset_config=dataset_config, ood_config=ood_config,
        method_name=ood_config["method"],
        extra_figures=figure_paths,  # optional
    )
    print(f"OOD method ran: results saved to {output_dir} and {report_dir}")
