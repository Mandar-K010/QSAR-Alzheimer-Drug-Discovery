# QSAR Modeling for Alzheimer's Drug Discovery

## Overview
This project implements QSAR (Quantitative Structure-Activity 
Relationship) modeling to predict AChE inhibitors for 
Alzheimer's disease using Random Forest and SHAP interpretation.

## Project Pipeline
1. Data Collection (ChEMBL + PubChem)
2. Data Cleaning & Transformation
3. Exploratory Data Analysis (Lipinski's Rule of Five)
4. Feature Extraction (1024-bit Morgan Fingerprints)
5. Model Training (Random Forest Regression + Classification)
6. Model Evaluation (10-Fold Cross Validation)
7. SHAP Interpretation
8. Molecular Docking (AutoDock Vina)

## Results
| Metric    | Our Result | 
|-----------|-----------|
| MAPE      |  11.53%   |
| R²        |  0.7236   |
| AUC-ROC   |  0.8169   |
| F1 Score  |  0.7652   |

## Requirements

pip install pandas numpy scikit-learn rdkit shap
matplotlib seaborn scipy meeko pillow

## Files
| File | Description |
|------|-------------|
| qsar_completes.py | Main QSAR pipeline |
| addon_features.py | SHAP force plots & Molecular substructure diagrams |
| docking_meeko.py | Molecular docking |

## Reference
Imani et al., International Journal of Computing and 
Digital Systems, 2025, VOL. 17, NO.1
