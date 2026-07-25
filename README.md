# QSAR Modeling for Alzheimer's Drug Discovery

[![Build Status](https://img.shields.io/badge/build-passing-lightgrey?style=flat-square)](#)
[![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-N%2FA-lightgrey?style=flat-square)](#)

## Overview

This repository implements a Quantitative Structure–Activity Relationship (QSAR)
pipeline for the prediction of acetylcholinesterase (AChE) inhibitory activity, a
principal molecular target in the pharmacological treatment of Alzheimer's disease.
The pipeline combines Random Forest regression and classification, SHAP-based
model interpretation, and molecular docking to identify and rationalize candidate
inhibitors from bioactivity data sourced from ChEMBL and PubChem.

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)
- [Author](#author)

## Methodology

The pipeline proceeds through the following stages:

1. **Data Collection** — Bioactivity records for AChE inhibitors, retrieved from ChEMBL and PubChem.
2. **Data Cleaning and Transformation** — Normalization of bioactivity values and conversion to pIC50.
3. **Exploratory Data Analysis** — Assessment of drug-likeness via Lipinski's Rule of Five.
4. **Feature Extraction** — Generation of 1024-bit Morgan (circular) molecular fingerprints.
5. **Model Training** — Random Forest regression and classification models.
6. **Model Evaluation** — 10-fold cross-validation.
7. **Model Interpretation** — SHAP (SHapley Additive exPlanations) value analysis.
8. **Molecular Docking** — Structure-based validation using AutoDock Vina.

## Results

| Metric   | Value  |
|----------|--------|
| MAPE     | 11.53% |
| R²       | 0.7236 |
| AUC-ROC  | 0.8169 |
| F1 Score | 0.7652 |

These figures correspond to the regression and classification models described
in the Methodology section above and are reported here for reference; they
should be reproduced independently before being relied upon for downstream
research decisions.

## Repository Structure

```
QSAR-Alzheimer-Drug-Discovery/
├── outputs/                                                  # Generated figures, models, and reports
├── acetylcholinesterase_bioactivity_data_3class_pIC50.csv    # Source bioactivity dataset
├── main.py                                                    # Pipeline entry point
├── qsar_completes.py                                          # Core QSAR modeling pipeline
├── addon_features.py                                          # SHAP force plots and substructure diagrams
├── docking_meeko.py                                           # Molecular docking with AutoDock Vina / Meeko
├── Pymol script.txt                                           # PyMOL visualization commands
├── Output file.txt                                            # Reference pipeline output log
├── requirements.txt                                           # Python dependencies
└── README.md
```

## Tech Stack

| Component          | Technology                        |
|---------------------|------------------------------------|
| Language            | Python 3.9 or later                |
| Data handling        | pandas, numpy                      |
| Machine learning     | scikit-learn                       |
| Cheminformatics      | RDKit                              |
| Model interpretation | SHAP                               |
| Visualization        | matplotlib, seaborn, Pillow        |
| Statistics           | scipy                              |
| Molecular docking     | Meeko, AutoDock Vina (external)    |
| Structural visualization | PyMOL (external, optional)     |

## Installation

The following instructions assume a Unix-like shell with Python 3.9 or later
installed. RDKit is most reliably installed via conda; a pip-based fallback
is also provided.

**Option A — conda (recommended for RDKit compatibility):**

```bash
git clone https://github.com/Mandar-K010/QSAR-Alzheimer-Drug-Discovery.git
cd QSAR-Alzheimer-Drug-Discovery

conda create -n qsar-ache python=3.10 -y
conda activate qsar-ache
conda install -c conda-forge rdkit -y

pip install -r requirements.txt
```

**Option B — pip / venv:**

```bash
git clone https://github.com/Mandar-K010/QSAR-Alzheimer-Drug-Discovery.git
cd QSAR-Alzheimer-Drug-Discovery

python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

AutoDock Vina must be installed separately and made available on the system
PATH for the docking stage (`docking_meeko.py`) to function. Refer to the
[AutoDock Vina documentation](https://vina.scripps.edu/) for platform-specific
installation instructions.

## Usage

Execute the complete pipeline from the repository root:

```bash
python main.py
```

Individual stages may also be invoked directly for development or debugging
purposes:

```bash
# Run the core QSAR modeling pipeline (data processing, training, evaluation)
python qsar_completes.py

# Generate SHAP interpretation plots and substructure diagrams
python addon_features.py

# Perform molecular docking of candidate ligands
python docking_meeko.py
```

Generated artifacts (trained models, figures, and evaluation reports) are
written to the `outputs/` directory.

```
[Placeholder: insert a terminal screenshot or sample console output here]
```

## Testing

An automated test suite is not yet included in this repository. Contributors
are encouraged to add unit and integration tests under a `tests/` directory
using `pytest`. Once introduced, tests should be run as follows:

```bash
pip install pytest pytest-cov
pytest --cov=. tests/
```

Refer to `CONTRIBUTING.md` for guidance on adding new tests alongside code
contributions.

## Citation

If this repository informs your research, please cite the underlying
methodological reference:

> Imani et al., *International Journal of Computing and Digital Systems*, 2025, Vol. 17, No. 1.

## License

This project is distributed under the terms of the MIT License. See
[`LICENSE`](LICENSE) for the full text.

## Author

Maintained by [Mandar-K010](https://github.com/Mandar-K010).
