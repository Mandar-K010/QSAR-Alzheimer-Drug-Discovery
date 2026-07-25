# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
-

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

### Security
-

## [0.1.0] - 2026-07-25

### Added
- Initial QSAR pipeline (`qsar_completes.py`) covering data cleaning,
  Lipinski's Rule of Five analysis, Morgan fingerprint extraction, Random
  Forest regression and classification, and 10-fold cross-validation.
- SHAP-based model interpretation and molecular substructure visualization
  (`addon_features.py`).
- Molecular docking module using AutoDock Vina and Meeko
  (`docking_meeko.py`).
- Source bioactivity dataset for acetylcholinesterase inhibitors
  (`acetylcholinesterase_bioactivity_data_3class_pIC50.csv`).
- Project documentation (`README.md`).
