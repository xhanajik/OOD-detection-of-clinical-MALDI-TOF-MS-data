import os
import pandas as pd
from ete3 import NCBITaxa
from dataset.macros import manual_taxonomy


def save_taxonomy_snapshot(df, filepath="taxonomy_snapshot.txt", n=50):
    """
    Make a snapshot to see how taxa gets filled in
    """
    rank_cols = ["domain", "phylum", "class", "order", "family", "genus"]
    available = [c for c in ["y"] + rank_cols if c in df.columns]
    tax_cols = [c for c in rank_cols if c in df.columns]

    with open(filepath, "w") as f:
        f.write("="*80 + "\n")
        f.write("TAXONOMY COLUMN INSPECTION (NCBI)\n")
        f.write("="*80 + "\n\n")

        f.write("--- FILL RATES ---\n")
        for col in tax_cols:
            filled = df[col].notna().sum()
            missing = df[col].isna().sum()
            f.write(f"  {col:<15} {filled}/{len(df)} filled "
                    f"({100*filled/len(df):.1f}%) | {missing} missing\n")

        f.write("\n--- SPECIES NOT FOUND IN NCBI (family is None) ---\n")
        if "family" in df.columns:
            missing_mask = df["family"].isna()
            missing_species = df.loc[missing_mask, "y"].value_counts()
            f.write(f"  Total: {len(missing_species)} unique species, "
                    f"{missing_mask.sum()} samples\n\n")
            for sp, count in missing_species.items():
                f.write(f"  {sp:<60} ({count} samples)\n")

        f.write("\n--- UNIQUE VALUES PER RANK ---\n")
        for col in tax_cols:
            unique_vals = sorted(df[col].dropna().unique())
            f.write(f"\n  [{col}] — {len(unique_vals)} unique:\n")
            for v in unique_vals:
                count = (df[col] == v).sum()
                f.write(f"    {str(v):<50} {count:>6} samples\n")

        if "family" in df.columns:
            f.write("\n--- ONE REPRESENTATIVE PER FAMILY ---\n")
            representative = (
                df[df["family"].notna()]
                .groupby("family")
                .first()
                .reset_index()[available]
            )
            f.write(representative.to_string(index=False))
            f.write("\n")

        f.write(f"\n--- RAW DATA (first {n} rows) ---\n")
        f.write(df[available].head(n).to_string(index=True))
        f.write("\n")

    print(f"Taxonomy snapshot saved to {filepath}")


def build_ncbi_taxonomy_table(df,  dbfile, output_file="ncbi_taxonomy.tsv",):
    """
    Takes your dataset df, queries NCBI
    taxonomy db, makes TSV with rank columns.
    """
    ncbi = NCBITaxa(dbfile=dbfile)
    ranks_wanted = ["domain", "phylum", "class", "order", "family", "genus"]
    species_list = [
        s for s in df["y"].unique()  # y is species lvl
    ]

    print(f"Processing {len(species_list)} unique species...")

    rows = []
    not_found = []
    for species in species_list:
        row = {"species": species}
        if species in manual_taxonomy:
            row.update(manual_taxonomy[species])
            rows.append(row)
            continue
        try:
            taxa_from_name = ncbi.get_name_translator([species])

            if not taxa_from_name:
                genus = species.split()[0]
                taxa_from_name = ncbi.get_name_translator([genus])  # try genus

            if not taxa_from_name:
                not_found.append(species)
                for r in ranks_wanted:
                    row[r] = None
            else:
                taxid = list(taxa_from_name.values())[0][0]
                lineage = ncbi.get_lineage(taxid)
                rank_map = ncbi.get_rank(lineage)
                name_map = ncbi.get_taxid_translator(lineage)

                for r in ranks_wanted:
                    match = [name_map[t] for t in lineage if rank_map[t] == r]
                    row[r] = match[0] if match else None

        except Exception as e:
            print(f"Error on {species}: {e}")
            not_found.append(species)
            for r in ranks_wanted:
                row[r] = None

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(output_file, sep="\t", index=False)

    print(f"\nResolved:  {len(rows) - len(not_found)}/{len(rows)}")
    print(f"Not found: {len(not_found)}")
    if not_found:
        print("Not found list:")
        for s in not_found:
            print(f"  {s}")
    print(f"Saved to {output_file}")
    return result
