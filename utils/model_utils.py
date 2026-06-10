import torch
import numpy as np
from utils.print_header import print_header
from networks.utils import build_model_from_params
from tqdm import tqdm
from torchinfo import summary
from networks.architectures import TinyNN_NoDropout


def load_network(dataset_config, ood_config, device):
    print_header("MODEL LOADING")

    dataset = dataset_config["name"]
    setup = ood_config["setup"]
    abundance_percentage = dataset_config["abundance_percentage"]
    level = ood_config["hierarchy_level"]

    if dataset == "DRIAMS":
        if setup == "mix":
            model_file = "models/driams_mix_trial_10_TinyWideNN.pth"
        elif setup == "abundance":
            if abundance_percentage == 2:
                model_file = "models/abund_2_driams_trial_22_TinyNN_NoDropout.pth"
            elif abundance_percentage == 5:
                model_file = "models/abund_5_driams_trial_29_TinyWideNN.pth"
            elif abundance_percentage == 10:
                model_file = "models/abund_10_driams_trial_6_TinyWideNN.pth"
            else:
                raise ValueError(
                    f"Unknown abundance percentage: '{dataset_config['abundance_percentage']}'. "
                    "Supported percentages for DRIAMS dataset are: 2, 5, 10."
                )
        elif setup == "hierarchy":
            if level == "y":
                model_file = "models/driams_hierarchy_species_trial_24_TinyNN_NoDropout.pth"
            elif level == "genus":
                model_file = "models/driams_hierarchy_genus_trial_14_TinyWideNN.pth"
            elif level == "family":
                model_file = "models/driams_hierarchy_family_trial_27_TinyWideNN.pth"
            else:
                raise ValueError(
                    f"Unknown taxonomy level: '{level}'. "
                    "Supported levels for DRIAMS dataset are: y, genus, family."
                )
        else:
            raise ValueError(
                f"Unknown experimental setup: '{ood_config['setup']}'. "
                "Supported methods for DRIAMS dataset are: mix, abundance, hierarchy."
            )
    elif dataset == "LM-UGent":
        if setup == "abundance":
            if abundance_percentage == 2:
                model_file = "models/abund_2_lmugent_trial_12_TinyNN_NoDropout.pth"
            elif abundance_percentage == 5:
                model_file = "models/abund_5_lmugent_trial_20_TinyNN_NoDropout.pth"
            elif abundance_percentage == 10:
                model_file = "models/abund_10_lmugent_trial_28_TinyNN_NoDropout.pth"
            else:
                raise ValueError(
                    f"Unknown abundance percentage: '{dataset_config['abundance_percentage']}'. "
                    "Supported percentages for LM-UGent dataset are: 2, 5, 10."
                )
        elif setup == "hierarchy":
            if level == "y":
                model_file = "models/lmugent_hierarchy_species_trial_28_TinyNN_NoDropout.pth"
            elif level == "genus":
                model_file = "models/lmugnet_hierarchy_genus_trial_25_TinyNN_NoDropout.pth"
            elif level == "family":
                model_file = "models/lmugent_hierarchical_family_trial_8_TinyNN_BatchNorm.pth"
            else:
                raise ValueError(
                    f"Unknown taxonomy level: '{level}'. "
                    "Supported levels for LM-UGent dataset are: y, genus, family."
                )
        else:
            raise ValueError(
                f"Unknown experimental setup: '{ood_config['setup']}'. "
                "Supported methods for VanDamme dataset are: abundance, hierarchy."
            )
    else:
        raise ValueError(
            f"Unknown dataset name: '{ood_config['setup']}'. "
            "Supported datasets are: VanDamme, DRIAMS."
        )

    checkpoint = torch.load(model_file, map_location=device)
    model = build_model_from_params(
        checkpoint, checkpoint['input_size'], checkpoint['num_classes'])
    print(model.hidden)
    print(model.output)

    # some older versions of networks had a different structure
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('network.') for k in state_dict.keys()):
        key_map = {
            'network.0.weight': 'hidden.0.weight',
            'network.0.bias':   'hidden.0.bias',
            'network.2.weight': 'output.weight',
            'network.2.bias':   'output.bias',
        }
        state_dict = {key_map.get(k, k): v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    summary(model, input_size=(1, checkpoint['input_size']))

    return model


def extract_embeddings(loader, model, device="cpu", text="Extracting embeddings"):
    """extract embeddings for KNN embedded features"""
    # model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=text):
            x = x.to(device)

            if hasattr(model, "hidden"):
                feats = model.hidden(x)

            # old version of networks
            elif hasattr(model, "network"):
                # remove last Linear layer automatically
                feats = model.network[:-1](x)

            else:
                raise ValueError("Unknown model architecture")

            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"{text}: {X.shape}")
    model.train()

    return X, y
