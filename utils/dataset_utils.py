from utils.config_utils import read_config
from utils.print_header import print_header
from dataset.dataset_driams import LitDriamsDataModule
#from dataset.dataset_vandamme import LitDriamsDataModuleLMUGENT
import pandas as pd
import datetime
import os


def load_dataset_driams(main_config, dataset_config, ood_config, train = True, preprocess = False):
    print_header("DATASET LOADING")
    dataset = LitDriamsDataModule(
        data_dir=dataset_config["data_dir"],
        data_file=dataset_config["data_file"],
        output_dir=main_config["output_dir"],
        batch_size=dataset_config["batch_size"],
        val_size=dataset_config["validation_size"],
        test_size=dataset_config["test_size"],
        setup=ood_config["setup"],
        min_samples_per_class=dataset_config["min_samples_per_class"],
        abundance_percentage=dataset_config["abundance_percentage"],
        verbose=main_config["verbose"],
        cpus= main_config["cpus"],
        hierarchy_level=ood_config["hierarchy_level"],
        hierarchy_leaveout=ood_config["hierarchy_leaveout"]
    )

    dataset.prepare_dataset(preprocess)
    if main_config["verbose"]:
        print_infos(dataset, main_config, dataset_config, ood_config)
    return dataset

"""
def load_dataset_lmugent(main_config, dataset_config, ood_config, train = True, preprocess = False):
    print_header("DATASET LOADING")
    dataset = LitDriamsDataModuleLMUGENT(
        data_dir=dataset_config["data_dir"],
        data_file=dataset_config["data_file"],
        output_dir=main_config["output_dir"],
        batch_size=dataset_config["batch_size"],
        val_size=dataset_config["validation_size"],
        test_size=dataset_config["test_size"],
        setup=ood_config["setup"],
        min_samples_per_class=dataset_config["min_samples_per_class"],
        abundance_percentage=dataset_config["abundance_percentage"],
        verbose=main_config["verbose"],
        cpus= main_config["cpus"],
        hierarchy_level=ood_config["hierarchy_level"],
        hierarchy_leaveout=ood_config["hierarchy_leaveout"]
    )

    dataset.prepare_dataset(preprocess)
    if main_config["verbose"]:
        print_infos(dataset, main_config, dataset_config, ood_config)
    return dataset
"""

def print_infos(dataset, main_config, dataset_config, ood_config):
    samples = dataset.df.shape[0] 

    print(f"Data: {main_config['name']} at {os.path.join(dataset_config['data_dir'], dataset_config['data_file'])}")
    print(f"\tTotal sample count: {samples}")
    print(f"Setup : {ood_config['setup']}")
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

    print("SPLIT info:")
    print(f"\tTrain set size: {len(dataset.data_train)} / {samples}")
    print(f"\tValidation set size: {len(dataset.data_val)} / {samples}")
    print(f"\tTest set size: {len(dataset.data_test)} / {samples}")
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

    print("OOD/ID info:")
    print(f"\tTotal OOD sample count {dataset.df['OOD_flag'].sum()} / {samples}")
    print(f"\tTotal ID sample count {samples - dataset.df['OOD_flag'].sum()} / {samples}")
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

    print("CLASS info:")
    print(f"\tCLASSES total: {dataset.n_classes('all')}")
    print(f"\tCLASSES in OOD: {dataset.n_classes('ood')}")
    print(f"\tCLASSES in ID: {dataset.n_classes('id')}\n")
    print("\tTrain classes:", len(dataset.data_train.get_labels().unique()))
    print("\tVal classes:", len(dataset.data_val.get_labels().unique()))
    print("\tTest classes:", len(dataset.data_test.get_labels().unique()))
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")