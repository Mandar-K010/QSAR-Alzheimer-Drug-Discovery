# CLAUDE.md

This document provides project context, architectural notes, and coding
guidelines for AI assistants (including Claude) operating on this
repository. It is intended to be read before any code modification,
documentation change, or dependency update is proposed.

## Project Summary

This repository implements a QSAR (Quantitative Structure–Activity
Relationship) pipeline for predicting acetylcholinesterase (AChE)
inhibitory activity relevant to Alzheimer's disease drug discovery. The
system integrates cheminformatics feature extraction, supervised machine
learning, model interpretability, and structure-based molecular docking.

## Domain Context

- **Target**: Acetylcholinesterase (AChE), a validated pharmacological
  target for Alzheimer's disease.
- **Endpoint**: Bioactivity expressed as pIC50, derived from IC50 values
  sourced from ChEMBL and PubChem, and categorized into three activity
  classes.
- **Descriptors**: 1024-bit Morgan (circular) fingerprints, computed via
  RDKit.
- **Drug-likeness filtering**: Lipinski's Rule of Five is applied during
  exploratory data analysis; assistants should not remove this filtering
  step without explicit instruction, as it is central to the scientific
  validity of the candidate set.

## Architecture

| Module                | Responsibility |
|------------------------|----------------|
| `main.py`               | Pipeline entry point; orchestrates the stages below. |
| `qsar_completes.py`     | Core pipeline: data ingestion, cleaning, feature extraction, Random Forest training (regression and classification), 10-fold cross-validation. |
| `addon_features.py`     | Post-hoc model interpretation: SHAP value computation, force plots, molecular substructure visualization. |
| `docking_meeko.py`      | Structure preparation (via Meeko) and molecular docking (via AutoDock Vina) for top-ranked candidates. |
| `outputs/`               | Destination for generated models, figures, and reports. Assistants should not hand-author files into this directory; it is a pipeline artifact directory. |

Data flows linearly: raw bioactivity data → cleaned/transformed data →
molecular descriptors → trained models → interpretation artifacts → docking
inputs for top candidates. Assistants modifying one stage should verify
that the expected input/output contract with adjacent stages is preserved.

## Coding Guidelines

1. **Determinism**: Any change to model training code must preserve or
   explicitly document changes to random seed handling. Reproducibility of
   the reported metrics (MAPE, R², AUC-ROC, F1) depends on this.
2. **Descriptor integrity**: Do not silently alter fingerprint length,
   radius, or descriptor type in `qsar_completes.py` without updating the
   corresponding documentation and, where relevant, retraining and
   re-reporting evaluation metrics.
3. **Validation discipline**: Cross-validation folds and train/test splits
   must not be modified in ways that introduce data leakage between
   training and evaluation sets.
4. **Style**: Follow PEP 8. Format with Black. All new functions require
   docstrings stating purpose, parameters, and return values, consistent
   with `CONTRIBUTING.md`.
5. **Scientific claims**: Do not introduce comparative or superlative
   claims about model performance (e.g., "state-of-the-art") without a
   citation or reproducible benchmark. This project favors precise,
   reported metrics over qualitative claims.
6. **Dependencies**: New dependencies must be added to `requirements.txt`
   with a pinned or minimum version, and justified in the pull request
   description per the protocol in `CONTRIBUTING.md`.
7. **External tools**: AutoDock Vina and PyMOL are invoked as external
   processes, not Python libraries. Assistants should not assume their
   availability in the execution environment and should guard docking-
   related code paths accordingly.
8. **Testing**: New logic in `qsar_completes.py`, `addon_features.py`, or
   `docking_meeko.py` should be accompanied by unit tests under `tests/`
   once the test suite is established, per `CONTRIBUTING.md`.

## Non-Goals

- This repository is not intended to serve as a production inference
  service. Assistants should not introduce web servers, APIs, or
  deployment infrastructure unless explicitly requested.
- This repository does not currently support descriptor types other than
  Morgan fingerprints; extension to other descriptor families (e.g.,
  MACCS keys, RDKit 2D descriptors) should be treated as a feature
  addition requiring its own evaluation, not a silent substitution.

## Reference

Imani et al., *International Journal of Computing and Digital Systems*,
2025, Vol. 17, No. 1. Consult this reference before altering the modeling
methodology, as the current pipeline is designed to align with it.
