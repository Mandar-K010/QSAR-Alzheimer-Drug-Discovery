"""
Addon Features for QSAR Project
================================
Run AFTER qsar_completes.py

Part A: Molecular Substructure Diagrams (Top 20, using OLD API for DrawMorganBit)
Part B: SHAP Force Plots (10 active + 10 inactive)
Part C: Molecular Docking Prep (saves JSON for docking_meeko.py)

Requirements: pip install pandas numpy scikit-learn rdkit matplotlib shap
"""

import os
import io
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

RDLogger.DisableLog('rdApp.*')

CSV_FILE     = "acetylcholinesterase_bioactivity_data_3class_pIC50.csv"
OUTPUT_DIR   = "outputs"
MODELS_DIR   = "outputs/models"
FP_RADIUS    = 2
FP_BITS      = 1024
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/substructures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/shap_force_plots/active", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/shap_force_plots/inactive", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/docking", exist_ok=True)

print("=" * 65)
print("  ADDON FEATURES — QSAR Alzheimer Project")
print("=" * 65)


# ── Load Data & Models ────────────────────────────────────────────
print("\n[Loading Data & Models...]")

df = pd.read_csv(CSV_FILE)
df = df[df['bioactivity_class'].isin(['active', 'inactive'])].copy()
df = df.dropna(subset=['canonical_smiles', 'pIC50']).reset_index(drop=True)


def smiles_to_morgan_new(smiles, radius=FP_RADIUS, n_bits=FP_BITS):
    """Morgan fingerprint using NEW API (matches saved models)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp  = gen.GetFingerprint(mol)
    except Exception:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)


fingerprints_new = []
valid_indices    = []
for i, smi in enumerate(df['canonical_smiles']):
    fp = smiles_to_morgan_new(str(smi))
    if fp is not None:
        fingerprints_new.append(fp)
        valid_indices.append(i)

df      = df.iloc[valid_indices].reset_index(drop=True)
X_new   = np.array(fingerprints_new)
y_reg   = df['pIC50'].values
y_class = (df['bioactivity_class'] == 'active').astype(int).values

reg_model   = pickle.load(open(f"{MODELS_DIR}/regression_model.pkl", 'rb'))
class_model = pickle.load(open(f"{MODELS_DIR}/classification_model.pkl", 'rb'))

print(f"  Loaded {len(X_new)} compounds, models OK")


# ══════════════════════════════════════════════════════════════════
# PART A: MOLECULAR SUBSTRUCTURE DIAGRAMS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART A: Molecular Substructure Diagrams")
print("=" * 60)

print("\n[A.1] Generating OLD API fingerprints...")

fingerprints_old = []
valid_old        = []
all_bit_infos    = []

for i, smi in enumerate(df['canonical_smiles']):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        continue
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, FP_RADIUS, nBits=FP_BITS, bitInfo=bit_info
    )
    arr = np.zeros(FP_BITS, dtype=np.uint8)
    for b in bit_info.keys():
        arr[b] = 1
    fingerprints_old.append(arr.tolist())
    valid_old.append(i)
    all_bit_infos.append(bit_info)

X_old = np.array(fingerprints_old)
y_old = (df.iloc[valid_old]['bioactivity_class'] == 'active').astype(int).values
bit_counts = X_old.sum(axis=0)

print(f"  {len(X_old)} compounds, {(bit_counts > 0).sum()} non-zero bits")

print("\n[A.2] Training RF on OLD API fingerprints...")
quick_rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
quick_rf.fit(X_old, y_old)
importances = quick_rf.feature_importances_

valid_bits   = np.where(bit_counts >= 5)[0]
valid_imps   = importances[valid_bits]
top20_sorted = valid_bits[np.argsort(valid_imps)[::-1][:20]]
print(f"  Top 20 bits: {list(top20_sorted)}")

print("\n[A.3] Drawing substructures...")

def draw_bit(smiles, bit_idx, bit_info, size=(250, 250)):
    """Draw Morgan bit substructure as PIL image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or int(bit_idx) not in bit_info:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = False
        Draw.DrawMorganBit(mol, int(bit_idx), bit_info, drawer=drawer)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')
    except Exception:
        try:
            env_atoms = set()
            for atom_idx, radius in bit_info[int(bit_idx)]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    env_atoms.add(bond.GetBeginAtomIdx())
                    env_atoms.add(bond.GetEndAtomIdx())
                env_atoms.add(atom_idx)
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.DrawMolecule(mol, highlightAtoms=list(env_atoms))
            drawer.FinishDrawing()
            return Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')
        except Exception:
            return None


bit_images = []
bit_found  = []

for bit_idx in top20_sorted:
    bit_idx = int(bit_idx)
    found   = False
    candidates = np.where(X_old[:, bit_idx] == 1)[0]

    for comp_idx in candidates[:100]:
        smi = str(df['canonical_smiles'].iloc[valid_old[comp_idx]])
        img = draw_bit(smi, bit_idx, all_bit_infos[comp_idx])
        if img is not None:
            bit_images.append(img)
            bit_found.append(True)
            found = True
            break

    if not found:
        bit_images.append(Image.new('RGB', (250, 250), (235, 235, 235)))
        bit_found.append(False)

print(f"  Drew {sum(bit_found)}/20 substructures")

print("\n[A.4] Creating 4x5 grid...")

fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor('white')
fig.suptitle("Top 20 Most Important Morgan Fingerprints\nMolecular Substructure Visualization",
             fontsize=16, fontweight='bold', y=0.99)

for i, (img, bit_idx, found) in enumerate(zip(bit_images, top20_sorted, bit_found)):
    ax = fig.add_subplot(5, 4, i + 1)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Bit {bit_idx} (Rank #{i+1})", fontsize=12, fontweight='bold',
                 color='#1a252f', pad=5)
    ax.text(0.5, -0.06,
            f"n={int(bit_counts[int(bit_idx)])} compounds | importance={importances[int(bit_idx)]:.4f}",
            transform=ax.transAxes, ha='center', fontsize=8, color='#555555')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#27ae60' if found else '#e74c3c')
        spine.set_linewidth(2.5)

legend_patches = [
    mpatches.Patch(edgecolor='#27ae60', facecolor='white', linewidth=2.5,
                   label='Substructure drawn successfully'),
    mpatches.Patch(edgecolor='#e74c3c', facecolor='#ebebeb', linewidth=2.5,
                   label='Could not draw (placeholder)')
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=11,
           frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f"{OUTPUT_DIR}/molecular_substructures_top20.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/molecular_substructures_top20.png")

for img, bit_idx in zip(bit_images, top20_sorted):
    img.save(f"{OUTPUT_DIR}/substructures/bit_{int(bit_idx)}.png")


# ══════════════════════════════════════════════════════════════════
# PART B: SHAP FORCE PLOTS FOR TOP 20 COMPOUNDS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART B: SHAP Force Plots (10 Active + 10 Inactive)")
print("=" * 60)

print("  Computing SHAP values...")
df_X        = pd.DataFrame(X_new, columns=[f"FP_{i}" for i in range(FP_BITS)])
explainer   = shap.TreeExplainer(class_model)
shap_values = explainer.shap_values(df_X, check_additivity=False)

if isinstance(shap_values, list):
    shap_class1   = shap_values[1]
    expected_val1 = explainer.expected_value[1]
else:
    shap_class1   = shap_values[:, :, 1]
    expected_val1 = (explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__')
                     else explainer.expected_value)
print("  SHAP done!")

y_pred_prob  = class_model.predict_proba(X_new)[:, 1]
y_pred_pic50 = reg_model.predict(X_new)

active_indices   = np.where(y_class == 1)[0]
inactive_indices = np.where(y_class == 0)[0]
top10_active     = active_indices[np.argsort(y_pred_pic50[active_indices])[::-1][:10]]
top10_inactive   = inactive_indices[np.argsort(y_pred_pic50[inactive_indices])[:10]]

for group_label, indices, folder in [("Active", top10_active, "active"),
                                      ("Inactive", top10_inactive, "inactive")]:
    print(f"  Generating {group_label} force plots...")
    for rank, idx in enumerate(indices, 1):
        try:
            plt.figure(figsize=(14, 3))
            shap.force_plot(expected_val1, shap_class1[idx], df_X.iloc[idx],
                            matplotlib=True, show=False)
            plt.title(
                f"SHAP Force Plot — {group_label} #{rank}\n"
                f"pIC50={y_reg[idx]:.3f} | Predicted={y_pred_pic50[idx]:.3f} | "
                f"P(Active)={y_pred_prob[idx]:.1%}\n"
                f"SMILES: {df['canonical_smiles'].iloc[idx][:60]}...",
                fontsize=9, pad=5)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/shap_force_plots/{folder}/{folder}_rank{rank:02d}.png",
                        dpi=130, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning rank {rank}: {e}")

print("  Creating combined summary...")
all_force_images = []
labels = []

for rank, idx in enumerate(top10_active, 1):
    path = f"{OUTPUT_DIR}/shap_force_plots/active/active_rank{rank:02d}.png"
    if os.path.exists(path):
        all_force_images.append(path)
        labels.append(f"Active #{rank} | pIC50={y_reg[idx]:.2f} | P(Act)={y_pred_prob[idx]:.0%}")

for rank, idx in enumerate(top10_inactive, 1):
    path = f"{OUTPUT_DIR}/shap_force_plots/inactive/inactive_rank{rank:02d}.png"
    if os.path.exists(path):
        all_force_images.append(path)
        labels.append(f"Inactive #{rank} | pIC50={y_reg[idx]:.2f} | P(Act)={y_pred_prob[idx]:.0%}")

if all_force_images:
    n     = len(all_force_images)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
    fig.suptitle("SHAP Force Plots — Top 10 Active + Top 10 Inactive Compounds\n"
                 "Red bars = features pushing toward Active | Blue bars = features pushing toward Inactive",
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for ax, img_path, label in zip(axes, all_force_images, labels):
        ax.imshow(Image.open(img_path))
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.axis('off')
    for ax in axes[len(all_force_images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_force_plots_summary.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/shap_force_plots_summary.png")


# ══════════════════════════════════════════════════════════════════
# PART C: MOLECULAR DOCKING PREPARATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART C: Molecular Docking Preparation")
print("=" * 60)

best_active_idx      = top10_active[0]
best_inactive_idx    = top10_inactive[0]
best_active_smiles   = str(df['canonical_smiles'].iloc[best_active_idx])
best_inactive_smiles = str(df['canonical_smiles'].iloc[best_inactive_idx])
best_active_pic50    = float(y_reg[best_active_idx])
best_inactive_pic50  = float(y_reg[best_inactive_idx])
best_active_prob     = float(y_pred_prob[best_active_idx])
best_inactive_prob   = float(y_pred_prob[best_inactive_idx])

print(f"\n  Active   : pIC50={best_active_pic50:.4f}, P(Active)={best_active_prob:.2%}")
print(f"  Inactive : pIC50={best_inactive_pic50:.4f}, P(Active)={best_inactive_prob:.2%}")

selected_compounds = {
    "active": {
        "smiles":      best_active_smiles,
        "pIC50":       best_active_pic50,
        "prob_active": best_active_prob,
        "chembl_id":   str(df['molecule_chembl_id'].iloc[best_active_idx])
                       if 'molecule_chembl_id' in df.columns else "unknown",
    },
    "inactive": {
        "smiles":      best_inactive_smiles,
        "pIC50":       best_inactive_pic50,
        "prob_active": best_inactive_prob,
        "chembl_id":   str(df['molecule_chembl_id'].iloc[best_inactive_idx])
                       if 'molecule_chembl_id' in df.columns else "unknown",
    },
    "grid_box": {
        "center_x": -8.471, "center_y": -41.155, "center_z": -37.05,
        "size_x": 70, "size_y": 46, "size_z": 40,
    },
    "protein_pdb": "4EY7.pdb",
}

json_path = f"{OUTPUT_DIR}/docking/selected_compounds.json"
with open(json_path, 'w') as f:
    json.dump(selected_compounds, f, indent=2)
print(f"  Saved: {json_path}")


def smiles_to_3d_sdf(smiles, filename):
    """Convert SMILES to optimized 3D structure as SDF."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            return False
    AllChem.UFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(filename)
    writer.write(mol)
    writer.close()
    return True


for label, smi, fname in [("Active", best_active_smiles, "active_ligand.sdf"),
                           ("Inactive", best_inactive_smiles, "inactive_ligand.sdf")]:
    path = f"{OUTPUT_DIR}/docking/{fname}"
    if smiles_to_3d_sdf(smi, path):
        print(f"  Saved: {path}")
    else:
        print(f"  Failed: {label} 3D generation")

docking_info = f"""MOLECULAR DOCKING INFORMATION
==============================
Target: Acetylcholinesterase (AChE) — PDB: 4EY7
Grid Box: center=(-8.471, -41.155, -37.05), size=(70, 46, 40)

Active:   SMILES={best_active_smiles}
          pIC50={best_active_pic50:.4f}, P(Active)={best_active_prob:.2%}

Inactive: SMILES={best_inactive_smiles}
          pIC50={best_inactive_pic50:.4f}, P(Active)={best_inactive_prob:.2%}

Next: python docking_meeko.py (reads selected_compounds.json)

Paper Reference: Active=-13.18 kcal/mol, Inactive=-5.845 kcal/mol
"""
with open(f"{OUTPUT_DIR}/docking/docking_instructions.txt", 'w') as f:
    f.write(docking_info)

# 2D structure diagram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Compounds Selected for Molecular Docking",
             fontsize=13, fontweight='bold')

compounds = [
    (best_active_smiles,   "ACTIVE",   '#2ecc71', best_active_pic50,   best_active_prob),
    (best_inactive_smiles, "INACTIVE", '#e74c3c', best_inactive_pic50, best_inactive_prob),
]

for ax, (smi, label, color, pic50, prob) in zip(axes, compounds):
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        ax.imshow(Image.open(io.BytesIO(drawer.GetDrawingText())))
    ax.set_title(f"{label}\npIC50 = {pic50:.3f} | P(Active) = {prob:.1%}",
                 fontsize=11, fontweight='bold', color=color, pad=8)
    ax.axis('off')
    display_smi = f"SMILES: {smi[:55]}..." if len(smi) > 55 else f"SMILES: {smi}"
    ax.text(0.5, -0.05, display_smi, transform=ax.transAxes,
            ha='center', fontsize=7, color='#555555')

legend_patches = [
    mpatches.Patch(color='#2ecc71', label='Active — expected binding: -10 to -13 kcal/mol'),
    mpatches.Patch(color='#e74c3c', label='Inactive — expected binding: -4 to -6 kcal/mol'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/docking/docking_compounds_2d.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/docking/docking_compounds_2d.png")


print(f"\n{'='*65}")
print(f"  ADDON FEATURES COMPLETE!")
print(f"{'='*65}")
print(f"""
  Part A: {OUTPUT_DIR}/molecular_substructures_top20.png
  Part B: {OUTPUT_DIR}/shap_force_plots_summary.png
  Part C: {OUTPUT_DIR}/docking/selected_compounds.json

  Next: python docking_meeko.py
""")
