import pytorch_lightning as L
import os
import h5py
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from dataset.macros import incomplete_labels_driams as incomplete_labels
from sklearn.preprocessing import LabelEncoder
from dataset.taxonomy import build_ncbi_taxonomy_table, save_taxonomy_snapshot


# Lightning
class BaseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values, dtype=torch.float32) if isinstance(
            data, pd.DataFrame) else torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long) if isinstance(
            labels, pd.Series) else torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def n_classes(self):
        return len(torch.unique(self.labels))

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


class LitDriamsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        data_file,
        output_dir,
        batch_size,
        val_size,
        test_size,
        setup,
        min_samples_per_class,
        abundance_percentage,
        verbose,
        cpus,
        hierarchy_level,
        hierarchy_leaveout
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        self.output_dir = output_dir
        self.verbose = verbose
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.setup = setup
        self.min_samples_per_class = min_samples_per_class
        self.abundance_percentage = abundance_percentage
        self.cpus = cpus
        self.hierarchy_leaveout = hierarchy_leaveout
        self.hierarchy_level = hierarchy_level
        self.taxonomy_dbfile = "./data/taxa.sqlite"
        self.taxonomy_file = "./data/ncbi_taxonomy_DRIAMS.tsv"

    def encode_test_labels(self, df, label_encoder):
        encoded = []
        for label, ood_flag in zip(df["y"], df["OOD_flag"]):
            if ood_flag == 0:
                encoded.append(label_encoder.transform([label])[0])
            else:
                encoded.append(-1)  # OOD label
        return encoded

    def prepare_dataset(self, preprocess):
        self.load_HDF5_file_as_pandas()
        self.filter_low_abundant()
        self.define_OOD()

        train_df, val_df, test_df = self.create_split()

        label_encoder = LabelEncoder()
        train_df["y_encoded"] = label_encoder.fit_transform(train_df["y"])
        val_df["y_encoded"] = label_encoder.transform(val_df["y"])
        self.label_encoder = label_encoder
        test_df["y_encoded"] = self.encode_test_labels(test_df, label_encoder)

        n_features = self.n_features()

        self.data_train = BaseDataset(
            train_df.iloc[:, :n_features], train_df['y_encoded']
        )
        self.data_val = BaseDataset(
            val_df.iloc[:, :n_features], val_df['y_encoded']
        )
        self.data_test = BaseDataset(
            test_df.iloc[:, :n_features], test_df['y_encoded']
        )

    def load_HDF5_file_as_pandas(self):
        path = (self.data_dir + self.data_file)
        with h5py.File(path, "r") as f:
            data_X = f["X"][:]  # Load data as a NumPy array
            X = pd.DataFrame(data_X.astype(float))
        with h5py.File(path, "r") as f:
            data_y = f["y"][:]
            y = pd.DataFrame(data_y)
            y.rename(columns={0: "y"}, inplace=True)
            y["y"] = y["y"].astype(str)
        self.df = pd.concat([X, y], axis=1)
        with h5py.File(path, "r") as f:
            if "source" in f:
                source = f["source"][:]
                source = pd.DataFrame(source)
                self.df["source"] = source.values
                self.df["source"] = self.df["source"].astype(
                    str).str.extract(r"(DRIAMS-[A-Z])")

    def filter_low_abundant(self):
        """Remove classes with fewer than min_samples_per_class samples (not including incomplete or MIX labels)"""
        samples = self.df.shape[0]
        df = self.df.copy()

        # Count abundance only from normal samples (exclude MIX and incomplete)
        normal_samples = df[(~df["y"].str.startswith("MIX!"))
                            & (~df["y"].isin(incomplete_labels))]
        class_counts = normal_samples["y"].value_counts()
        low_abundant_classes = class_counts[class_counts <
                                            self.min_samples_per_class].index

        # Create masks for what to keep
        is_mix = df["y"].str.startswith("MIX!")
        is_incomplete = df["y"].isin(incomplete_labels)
        is_abundant = ~df["y"].isin(low_abundant_classes)

        # Keep samples if they are: MIX OR incomplete OR abundant
        keep_mask = is_mix | is_incomplete | is_abundant
        df = df[keep_mask]

        self.df = df.reset_index(drop=True)

        if self.verbose:
            print(f"All normal classes: {len(class_counts)}")
            print(f"Low abundant classes: {len(low_abundant_classes)}")
            print(f"All samples: {samples}")
            print(f"After filtering: {self.df.shape[0]}")

    def define_OOD(self):
        """
        Creates an OOD flag column depicting if the sample is considered ID or OOD
        based on the setup parameter.
        """
        df = self.df.copy()
        df["OOD_flag"] = 0

        if self.setup == "mix":
            print("\nCreating MIXED Dataset setup\n")
            df = df[~df["y"].isin(incomplete_labels)]
            mix_mask = df["y"].str.startswith("MIX!")
            df.loc[mix_mask, "OOD_flag"] = 1
            n_ood = df["OOD_flag"].sum()
            print(f"Flagged {n_ood} samples ({100*n_ood/len(df):.1f}%)")

        elif self.setup == "hierarchy":
            print("\nCreating HIERARCHY Dataset setup\n")
            valid_mask = (~df["y"].str.startswith("MIX!")) & (
                ~df["y"].isin(incomplete_labels))
            df = df[valid_mask]

            # Taxonomy columns
            if not os.path.exists(self.taxonomy_file):
                print("Building NCBI taxonomy table (first run, needs taxa.sqlite)...")
                taxonomy = build_ncbi_taxonomy_table(
                    df=df,
                    dbfile=self.taxonomy_dbfile,
                    output_file=self.taxonomy_file
                )
            else:
                print("Loading cached NCBI taxonomy table...")
                taxonomy = pd.read_csv(self.taxonomy_file, sep="\t")

            df = df.merge(
                taxonomy.rename(columns={"species": "y"}),
                on="y",
                how="left"
            )
            save_taxonomy_snapshot(df)

            # Chceck for taxonomy levels in dataset
            if self.hierarchy_level not in df.columns:
                available = [c for c in df.columns
                             if c in ["y", "domain", "phylum", "class",
                                      "order", "family", "genus"]]
                raise ValueError(
                    f"hierarchy_level '{self.hierarchy_level}' not found. "
                    f"Available taxonomy columns: {available}"
                )

            leaveout = self.hierarchy_leaveout
            if leaveout is None:
                leaveout = []

            if len(leaveout) == 0:
                print("WARNING: hierarchy_leaveout is empty")

            ood_mask = df[self.hierarchy_level].isin(leaveout)
            df.loc[ood_mask, "OOD_flag"] = 1

            n_ood = ood_mask.sum()
            n_groups = df.loc[ood_mask, self.hierarchy_level].nunique()
            print(f"OOD level:  {self.hierarchy_level}")
            print(f"OOD groups: {sorted(leaveout)}")
            print(f"Flagged {n_ood} samples ({100*n_ood/len(df):.1f}%) "
                  f"across {n_groups} taxonomic groups")

        elif self.setup == "abundance":
            print("\nCreating ABUNDANCE Dataset setup\n")
            valid_mask = (~df["y"].str.startswith("MIX!")) & (
                ~df["y"].isin(incomplete_labels))
            df = df[valid_mask]

            if len(df) > 0:
                class_counts = df["y"].value_counts()
                total_samples = len(df)
                target_ood_samples = total_samples * \
                    (self.abundance_percentage / 100)
                ood_sample_count = 0
                low_abundant_classes = []

                # Iterate rarest first
                for rarest_class in class_counts.index[::-1]:
                    if ood_sample_count >= target_ood_samples:
                        break
                    low_abundant_classes.append(rarest_class)
                    ood_sample_count += class_counts[rarest_class]

                low_abundant_mask = df["y"].isin(low_abundant_classes)
                df.loc[low_abundant_mask, "OOD_flag"] = 1
            n_ood = df["OOD_flag"].sum()
            print(f"Flagged {n_ood} samples ({100*n_ood/len(df):.1f}%)")
        self.df = df.reset_index(drop=True)

    def create_split(self, random_seed=24):
        df_id = self.df[self.df["OOD_flag"] == 0].copy()
        df_ood = self.df[self.df["OOD_flag"] == 1].copy()

        split_size = self.val_size + self.test_size
        train_df, temp_id = train_test_split(
            df_id, test_size=split_size, stratify=df_id['y'], random_state=random_seed)

        split_size = self.test_size / (self.val_size + self.test_size)
        val_df, test_id = train_test_split(
            temp_id, test_size=split_size, stratify=temp_id['y'], random_state=random_seed)

        test_df = pd.concat([test_id, df_ood], ignore_index=True)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        return train_df, val_df, test_df

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=0)

    def n_classes(self, distribution="all"):
        """
        Returns the number of unique classes in the dataset
        for either in-distribution (id), out-of-distribution (ood) or all (all) samples.
        """
        if distribution == "all":
            filtered = self.df
        elif distribution == "ood":
            filtered = self.df[self.df["OOD_flag"] == 1]
        elif distribution == "id":
            filtered = self.df[self.df["OOD_flag"] == 0]
        unique_labels = filtered["y"].unique()
        return len(unique_labels)

    def n_features(self):
        """
        Returns the number of features — columns in self.df that start with a digit.
        """
        feature_cols = [col for col in self.df.columns if str(col)[
            0].isdigit()]
        print(f"n_features: {len(feature_cols)}")
        return len(feature_cols)

    def get_data(self):
        X_train = self.data_train.get_data()
        y_train = self.data_train.get_labels()

        X_val = self.data_val.get_data()
        y_val = self.data_val.get_labels()

        X_test = self.data_test.get_data()
        y_test = self.data_test.get_labels()
        return X_train, y_train, X_val, y_val, X_test, y_test
