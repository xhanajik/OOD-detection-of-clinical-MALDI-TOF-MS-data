# **Making clinical infection prediction from MALDI-TOF MS data more reliable with out-of-distribution detection methods**
Supplementary code to the UGent dissertation.

## Abstract
Accurate bacterial species identification is essential for the diagnosis and treatment of infectious diseases. In clinical microbiology laboratories, matrix-assisted laser desorption/ionization time-of-flight mass spectrometry (MALDI-TOF MS) is routinely used to identify bacterial species from characteristic spectral fingerprints. While existing commercial systems achieve high accuracy for species represented in their databases, they are often unable to reliably recognise samples originating from previously unseen species. This limitation poses a challenge in clinical practice, where novel, rare, or underrepresented organisms may be encountered. Recent advances in machine learning have shown that out-of-distribution (OOD) detection methods can identify inputs that differ from the data observed during training. Although these approaches have been extensively studied in domains such as computer vision, their applicability to MALDI-TOF MS-based bacterial species identification remains largely unexplored. This thesis investigates the potential of OOD detection to improve the reliability of bacterial species identification from MALDI-TOF MS data. Eight OOD detection methods were implemented and benchmarked across two large-scale datasets and five experimental setups, each focusing on a different aspect of OOD detection. The results demonstrate that established OOD detection methods can be successfully transferred to the domain of MALDI-TOF MS and are capable of identifying spectra originating from previously unseen bacterial species, with several methods achieving strong performance. These findings highlight the potential of OOD detection to enhance the reliability of machine learning-based prediction models and provide a foundation for the integration of novelty detection into future MALDI-TOF MS-based identification workflows.

---
## Repository Structure

```text
├── configs/          # YAML configuration files
├── data/             # Place downloaded dataset files here
├── dataset/          # Dataset loading and preprocessing logic
├── models/           # Trained models
├── networks/         # Network architectures, optimization and training code
├── ood/              # OOD detection method implementations
├── utils/            # Utility/helper functions
├── main.py
├── run_myjob.sh      # Example job script for the Accelgor HPC UGent cluster
├── modules.txt       # HPC module list
└── requirements.txt  # Python dependencies
```
---

## Data
Although the benchmark in this project involves two datasets, LM-UGent is private and is therefore not included in this repository. Additionally, all models trained on the LM-UGent dataset have been removed to prevent reverse engineering of the private data. 

To run the benchmark, please download the required data files from the link below and place them in the `data/` folder: <br>
**Download DRIAMS dataset:** https://drive.google.com/file/d/14XnSWr9ibANwqiKbH2VO1nLvR9ucjHCN/view?usp=sharing <br>
**Download NCBI data:** https://drive.google.com/file/d/1Ll0rIi-pZEpqiKf0foXeiB_9AfSwiejG/view?usp=sharing

---

## Running the Code

Experiments can be executed using:
```python main.py --config_file <config_file.yml>```

---

## Requirements
The code was developed and tested on the Accelgor HPC UGent cluster. The loaded modules can be found in modules.txt and the requirements in requirements.txt.

---

## Notes
This repository contains the research code used during the development of the thesis. The codebase evolved throughout the project and is provided primarily to support reproducibility of the reported experiments rather than as a production-ready software package. Additional plotting scripts are available on request.
