# Macros
incomplete_labels_driams = [
    "Pandoraea sp[2]", "Acinetobacter sp", "Pseudomonas sp", "Salmonella spp",
    "Corynebacterium sp", "Achromobacter sp[3]", "Fusobacterium sp[2]", "Gardnerella sp",
    "Megamonas sp[2]", "Staphylococcus sp[1]", "Propionibacterium sp[2]", "Fusobacterium sp",
    "Propionibacterium sp", "Clostridium sp", "Anaerococcus sp", "Salmonella sp",
    "Kocuria sp", "Moraxella sp", "Capnocytophaga sp", "Peptoniphilus sp",
    "Pseudomonas sp[2]", "Leptotrichia sp", "Mobiluncus sp", "Nocardia sp",
    "Eubacterium sp[2]", "Ochrobactrum sp[3]", "Chryseobacterium sp", "Paenibacillus sp",
    "Actinomyces sp", "Atopobium sp", "Sphingomonas sp", "Erwinia sp", "Stenotrophomonas sp",
    "Penicillium sp", "Cardiobacterium sp", "Neisseria sp[2]", "Pandoraea sp[2]"
]

incomplete_labels_lmugent = [
    "Acetobacter sp.", "Acinetobacter sp.", "Aerococcus sp.",
    "Agrobacterium sp.", "Aliivibrio sp.", "Alteromonas sp.",
    "Ancylobacter sp.", "Aquimarina sp.", "Arcobacter sp.",
    "Aurantimonas sp.", "Bacillus sp. (in: Bacteria)",
    "Bifidobacterium sp.", "Brachybacterium sp.", "Bradyrhizobium sp.",
    "Burkholderia sp.", "Caballeronia sp.", "Campylobacter sp.",
    "Chromohalobacter sp.", "Cobetia sp.", "Cupriavidus sp.",
    "Cyclobacterium sp.", "Devosia sp.", "Ensifer sp.",
    "Enterococcus sp.", "Erysipelothrix sp.", "Georgenia sp.",
    "Gluconacetobacter sp.", "Gluconobacter sp.", "Komagataeibacter sp.",
    "Lactobacillus sp.", "Leuconostoc sp.", "Limosilactobacillus sp.",
    "Lysinibacillus sp.", "Micromonospora sp.", "Microbacterium sp.", "Neokomagataea sp.",
    "Paenibacillus sp.", "Paracoccus sp.", "Paraburkholderia sp.",
    "Planctomyces sp.", "Planomicrobium sp.", "Propionibacterium sp.",
    "Pseudidiomarina sp.", "Pseudoalteromonas sp.", "Pseudoclavibacter sp.",
    "Pseudomonas sp.", "Ralstonia sp.", "Rhizobium sp.",
    "Rosenbergiella sp.", "Roseomonas sp.", "Sporolactobacillus sp.",
    "Staphylococcus sp.", "Tatumella sp.", "Vagococcus sp.",
    "Variovorax sp.", "Vibrio sp.", "Weissella sp.",
    "Winogradskyella sp.",
    "'Chitinophaga terrae' An et al. 2007",
    "[Propionibacterium] namnetense",
]

manual_taxonomy = {
    # Moraxella subgenus Branhamella — clinical name for Moraxella catarrhalis
    "Moraxella_sg_Branhamella catarrhalis": {
        "phylum": "Pseudomonadota", "class": "Gammaproteobacteria",
        "order": "Moraxellales", "family": "Moraxellaceae", "genus": "Moraxella"
    },
    # Moraxella subgenus Moraxella osloensis
    "Moraxella_sg_Moraxella osloensis": {
        "phylum": "Pseudomonadota", "class": "Gammaproteobacteria",
        "order": "Moraxellales", "family": "Moraxellaceae", "genus": "Moraxella"
    },
    # Capnocytophaga sputigena — valid species, just not in your NCBI version
    "Capnocytophaga sputigena": {
        "phylum": "Bacteroidota", "class": "Flavobacteriia",
        "order": "Flavobacteriales", "family": "Flavobacteriaceae",
        "genus": "Capnocytophaga"
    },
    # Haemophilus sputorum — reclassified, NCBI may not resolve old name
    "Haemophilus sputorum": {
        "phylum": "Pseudomonadota", "class": "Gammaproteobacteria",
        "order": "Pasteurellales", "family": "Pasteurellaceae",
        "genus": "Haemophilus"
    },
    "Rhodobacter blasticus": {
        "phylum": "Pseudomonadota", "class": "Alphaproteobacteria",
        "order": "Rhodobacterales", "family": "Roseobacteraceae", "genus": "Rhodobacter"
    },
}
