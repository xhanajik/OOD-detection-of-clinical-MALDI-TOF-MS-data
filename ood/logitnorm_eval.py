import torch
from ood.logitnorm import build_model_from_params
from ood.msp import get_msp_scores, treshhold_tuning, score_histogram
from utils.save_run import save_run


def logitnorm_eval_with_msp(logitnorm_path, val_loader, test_loader, output_dir, setup_name, device):
    """Load best logitnorm model and score test samples with MSP"""
    checkpoint = torch.load(logitnorm_path, map_location=device)
    ln_model = build_model_from_params(
        checkpoint, checkpoint["input_size"], checkpoint["num_classes"]
    )
    ln_model.load_state_dict(checkpoint["model_state_dict"])
    ln_model.to(device)

    val_scores, val_labels, _ = get_msp_scores(ln_model, val_loader, device)
    threshold = treshhold_tuning(val_scores, val_labels)
    test_scores, test_labels, class_preds = get_msp_scores(
        ln_model, test_loader, device)

    evaluate_msp(test_scores, test_labels, threshold, output_dir)
    score_histogram(test_scores, test_labels, threshold, output_dir)

    save_run(
        base_dir=output_dir,
        setup_name=setup_name,
        method="logitnorm",
        test_scores=test_scores.numpy(),
        test_labels=test_labels.numpy(),
        class_preds=class_preds.numpy(),
        ood_higher=False,
        meta={
            "tau":            checkpoint.get("tau", "unknown"),
            "lr":             checkpoint.get("lr", "unknown"),
            "architecture":   checkpoint.get("architecture"),
            "logitnorm_path": logitnorm_path,
        },
    )

    return test_scores, test_labels, class_preds
