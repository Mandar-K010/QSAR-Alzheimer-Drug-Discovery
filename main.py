"""
QSAR Drug Discovery Pipeline — Main Runner
============================================
Runs all three scripts in order:
  1. qsar_completes.py     → QSAR modeling, cross-validation, SHAP
  2. addon_features.py     → Substructures, force plots, docking prep
  3. docking_meeko.py       → Molecular docking with AutoDock Vina

Usage:
  python main.py              Run all steps
  python main.py 1            Run only Step 1
  python main.py 2            Run only Step 2
  python main.py 3            Run only Step 3
  python main.py 1 2          Run Steps 1 and 2
  python main.py --check      Check dependencies only
"""

import os
import sys
import time
import subprocess

SCRIPTS = [
    ("Step 1", "qsar_completes.py",
     "QSAR modeling, Lipinski, Mann-Whitney, Morgan FP, RF training, SHAP, feature importance"),
    ("Step 2", "addon_features.py",
     "Molecular substructures, SHAP force plots, docking preparation"),
    ("Step 3", "docking_meeko.py",
     "Receptor/ligand prep, AutoDock Vina docking, results visualization"),
]

REQUIRED_FILES = {
    "qsar_completes.py": ["acetylcholinesterase_bioactivity_data_3class_pIC50.csv"],
    "addon_features.py": ["acetylcholinesterase_bioactivity_data_3class_pIC50.csv",
                          "outputs/models/regression_model.pkl",
                          "outputs/models/classification_model.pkl"],
    "docking_meeko.py":  ["4EY7.pdb",
                          "outputs/docking/selected_compounds.json"],
}


def check_dependencies():
    """Verify all required Python packages are installed."""
    print("\n  Checking Python packages...")
    packages = {
        "pandas":       "import pandas",
        "numpy":        "import numpy",
        "matplotlib":   "import matplotlib",
        "seaborn":      "import seaborn",
        "scipy":        "from scipy.stats import mannwhitneyu",
        "scikit-learn": "from sklearn.ensemble import RandomForestClassifier",
        "rdkit":        "from rdkit import Chem",
        "shap":         "import shap",
        "Pillow":       "from PIL import Image",
    }

    optional = {
        "meeko":     "from meeko import MoleculePreparation",
        "openbabel": "from openbabel import pybel",
    }

    all_ok = True
    for name, test in packages.items():
        try:
            exec(test)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ❌ {name} — pip install {name}")
            all_ok = False

    print("\n  Optional packages (for docking):")
    for name, test in optional.items():
        try:
            exec(test)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ⚠️  {name} — not installed (manual fallback will be used)")

    return all_ok


def check_input_files(script):
    """Verify required input files exist before running a script."""
    missing = []
    for f in REQUIRED_FILES.get(script, []):
        if not os.path.exists(f):
            missing.append(f)
    return missing


def run_script(step_name, script, description):
    """Run a single pipeline script and return success/failure."""
    print(f"\n{'━' * 70}")
    print(f"  {step_name}: {script}")
    print(f"  {description}")
    print(f"{'━' * 70}")

    if not os.path.exists(script):
        print(f"\n  ❌ ERROR: {script} not found in {os.getcwd()}")
        return False

    missing = check_input_files(script)
    if missing:
        print(f"\n  ❌ Missing required files:")
        for f in missing:
            print(f"     → {f}")
        if script == "addon_features.py":
            print(f"\n  Run Step 1 (qsar_completes.py) first!")
        elif script == "docking_meeko.py":
            print(f"\n  Run Step 2 (addon_features.py) first!")
        return False

    start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if result.returncode == 0:
        print(f"\n  ✅ {step_name} completed in {minutes}m {seconds}s")
        return True
    else:
        print(f"\n  ❌ {step_name} FAILED (exit code {result.returncode}) after {minutes}m {seconds}s")
        return False


def main():
    print("=" * 70)
    print("  QSAR Drug Discovery Pipeline — Alzheimer's (AChE Inhibitors)")
    print("=" * 70)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Working : {os.getcwd()}")

    args = sys.argv[1:]

    # --check mode
    if "--check" in args:
        ok = check_dependencies()
        print(f"\n  {'All required packages installed!' if ok else 'Install missing packages first.'}")
        print(f"\n  Required data files:")
        all_files = set()
        for files in REQUIRED_FILES.values():
            all_files.update(files)
        for f in sorted(all_files):
            exists = "✅" if os.path.exists(f) else "❌"
            print(f"    {exists} {f}")
        sys.exit(0 if ok else 1)

    # Determine which steps to run
    if args:
        try:
            steps = [int(a) for a in args if a.isdigit()]
        except ValueError:
            print(f"  Usage: python main.py [1] [2] [3] [--check]")
            sys.exit(1)
    else:
        steps = [1, 2, 3]

    # Validate
    for s in steps:
        if s not in [1, 2, 3]:
            print(f"  Invalid step: {s}. Use 1, 2, or 3.")
            sys.exit(1)

    # Quick dependency check
    ok = check_dependencies()
    if not ok:
        print(f"\n  ❌ Install missing packages first: pip install -r requirements.txt")
        sys.exit(1)

    # Run selected steps
    total_start = time.time()
    results = {}

    for step_num in steps:
        step_name, script, description = SCRIPTS[step_num - 1]
        success = run_script(step_name, script, description)
        results[step_name] = success
        if not success:
            print(f"\n  ⚠️  {step_name} failed — stopping pipeline.")
            break

    # Summary
    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE SUMMARY — Total time: {total_min}m {total_sec}s")
    print(f"{'=' * 70}")

    for step_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"    {step_name}: {status}")

    if all(results.values()):
        print(f"""
  All steps completed successfully!

  Output files in: outputs/
  ├── lipinski_descriptors.png
  ├── mannwhitney_distributions.png
  ├── fingerprint_analysis.png
  ├── evaluation_metrics.png
  ├── roc_curve.png
  ├── actual_vs_predicted.png
  ├── shap_global_summary.png
  ├── shap_dot_plot.png
  ├── shap_force_active.html
  ├── shap_force_inactive.html
  ├── feature_importance_classification.png
  ├── feature_importance_regression.png
  ├── feature_importance_comparison.png
  ├── molecular_substructures_top20.png
  ├── shap_force_plots_summary.png
  ├── morgan_fingerprints.csv
  ├── models/
  │   ├── regression_model.pkl
  │   └── classification_model.pkl
  ├── substructures/
  │   └── bit_XXX.png (individual)
  ├── shap_force_plots/
  │   ├── active/active_rank01-10.png
  │   └── inactive/inactive_rank01-10.png
  └── docking/
      ├── selected_compounds.json
      ├── docking_results.png
      ├── docking_results.txt
      ├── docking_compounds_2d.png
      ├── docking_instructions.txt
      ├── receptor.pdbqt
      ├── active_ligand.sdf / .pdbqt
      ├── inactive_ligand.sdf / .pdbqt
      ├── active_docked.pdbqt
      └── inactive_docked.pdbqt
""")
    else:
        print(f"\n  Fix the errors above and re-run.")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
