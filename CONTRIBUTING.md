# Contributing Guidelines

Contributions to this project are welcome. This document describes the
process by which changes are proposed, reviewed, and incorporated.

## Development Environment Setup

1. Fork the repository and clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/QSAR-Alzheimer-Drug-Discovery.git
   cd QSAR-Alzheimer-Drug-Discovery
   ```

2. Create an isolated environment and install dependencies, including
   development tooling:

   ```bash
   conda create -n qsar-ache-dev python=3.10 -y
   conda activate qsar-ache-dev
   conda install -c conda-forge rdkit -y
   pip install -r requirements.txt
   pip install black flake8 pytest pytest-cov
   ```

3. Create a branch for your change:

   ```bash
   git checkout -b feature/<short-description>
   ```

## Code Style

This project adheres to the following conventions:

- Formatting is enforced with **Black** (default line length).
- Static analysis is performed with **flake8**.
- All public functions and modules must include docstrings describing their
  purpose, parameters, and return values.
- Variable and function names must be descriptive and follow `snake_case`
  convention, consistent with PEP 8.
- Data processing steps that alter chemical structures or bioactivity labels
  must include inline comments explaining the scientific rationale.

Before committing, run:

```bash
black .
flake8 .
```

## Testing Requirements

Any change that modifies data processing, model training, or evaluation
logic must be accompanied by a corresponding test under `tests/`. Run the
test suite prior to submitting a change:

```bash
pytest --cov=. tests/
```

## Commit Messages

Commit messages should be written in the imperative mood and should
summarize the intent of the change concisely, for example:

```
Add cross-validation fold reporting to qsar_completes.py
```

## Pull Request Protocol

1. Ensure your branch is rebased against the latest `main` branch prior to
   submission.
2. Open a pull request against `main` using the provided pull request
   template, completing all required sections.
3. Ensure all automated checks pass.
4. A maintainer will review the submission. Reviewers may request changes;
   please address feedback through additional commits rather than force
   pushes, to preserve review history.
5. Once approved, the pull request will be merged by a maintainer using a
   squash merge, unless otherwise specified.

## Reporting Issues

Bugs and feature proposals should be filed using the issue templates
provided under `.github/ISSUE_TEMPLATE/`. Security vulnerabilities must
**not** be reported through public issues; refer to `SECURITY.md`.

## Code of Conduct

All contributors are expected to adhere to the project's
[Code of Conduct](CODE_OF_CONDUCT.md).
