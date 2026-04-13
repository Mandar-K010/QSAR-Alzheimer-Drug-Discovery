"""
Molecular Docking with AutoDock Vina
=====================================
Run AFTER addon_features.py — reads outputs/docking/selected_compounds.json

Steps: Prepare Receptor → Prepare Ligands → Run Vina → Plot Results

Requirements: pip install rdkit meeko (optional: openbabel)
External: AutoDock Vina executable (https://github.com/ccsb-scripps/AutoDock-Vina/releases)
"""

import os
import sys
import glob
import json
import platform
import shutil
import subprocess
import sysconfig
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

OUTPUT_DIR     = "outputs/docking"
COMPOUNDS_JSON = "outputs/docking/selected_compounds.json"
DEFAULT_GRID   = {
    'center_x': -8.471, 'center_y': -41.155, 'center_z': -37.05,
    'size_x': 70, 'size_y': 46, 'size_z': 40,
}
AD_TYPES = {'C':'C', 'N':'NA', 'O':'OA', 'S':'SA', 'H':'HD',
            'P':'P', 'F':'F', 'CL':'Cl', 'BR':'Br', 'I':'I'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  MOLECULAR DOCKING — AutoDock Vina")
print("=" * 60)


# ── Load Compounds ────────────────────────────────────────────────
print("\n[Loading Compounds...]")

if os.path.exists(COMPOUNDS_JSON):
    with open(COMPOUNDS_JSON, 'r') as f:
        compounds_data = json.load(f)
    ACTIVE_SMILES   = compounds_data['active']['smiles']
    INACTIVE_SMILES = compounds_data['inactive']['smiles']
    ACTIVE_PIC50    = compounds_data['active']['pIC50']
    INACTIVE_PIC50  = compounds_data['inactive']['pIC50']
    GRID            = compounds_data.get('grid_box', DEFAULT_GRID)
    PROTEIN_PDB     = compounds_data.get('protein_pdb', '4EY7.pdb')
    print(f"  Loaded from: {COMPOUNDS_JSON}")
    print(f"  Active   : pIC50={ACTIVE_PIC50:.4f} | {ACTIVE_SMILES[:50]}...")
    print(f"  Inactive : pIC50={INACTIVE_PIC50:.4f} | {INACTIVE_SMILES[:50]}...")
else:
    print(f"  WARNING: {COMPOUNDS_JSON} not found — run addon_features.py first!")
    print(f"  Falling back to hardcoded SMILES")
    ACTIVE_SMILES   = "COc1ccccc1C(C)NS(N)(=O)=O"
    INACTIVE_SMILES = "COc1ccccc1CNCCCCCCNCCSSCCNCCCCCCNCc1ccccc1OC"
    ACTIVE_PIC50    = 10.5711
    INACTIVE_PIC50  = 1.0000
    GRID            = DEFAULT_GRID
    PROTEIN_PDB     = "4EY7.pdb"


# ── Detect Vina ───────────────────────────────────────────────────
print("\n[Detecting Vina...]")

def find_vina():
    """Find Vina executable across platforms."""
    is_win  = platform.system() == "Windows"
    patterns = ["vina*.exe", "**/vina*.exe"] if is_win else ["vina*", "**/vina*", "vina*.exe"]
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if os.path.isfile(f):
                if not is_win and not os.access(f, os.X_OK):
                    try: os.chmod(f, 0o755)
                    except Exception: pass
                return f
    return shutil.which("vina")

VINA_PATH = find_vina()
if VINA_PATH:
    print(f"  Found: {VINA_PATH} ({platform.system()})")
else:
    print(f"  Not found — download from: https://github.com/ccsb-scripps/AutoDock-Vina/releases")
    VINA_PATH = "vina"


# ── Helper Functions ──────────────────────────────────────────────

def find_script(script_name):
    """Find a Python script in Scripts directory."""
    scripts_dir = sysconfig.get_path('scripts')
    for path in [os.path.join(scripts_dir, script_name),
                 os.path.join(os.path.dirname(sys.executable), script_name),
                 os.path.join(os.path.dirname(sys.executable), 'Scripts', script_name),
                 script_name]:
        if os.path.exists(path):
            return path
    return None


def pdb_to_pdbqt_manual(pdb_path, pdbqt_path):
    """Manual PDB → PDBQT conversion with correct column positions."""
    with open(pdb_path, 'r') as f:
        pdb_lines = f.readlines()

    pdbqt_lines = []
    for line in pdb_lines:
        if not line.startswith('ATOM'):
            continue
        try:
            serial  = int(line[6:11])
            name    = line[12:16]
            altloc  = line[16]
            resname = line[17:20]
            chain   = line[21]
            resseq  = int(line[22:26])
            icode   = line[26]
            x       = float(line[30:38])
            y       = float(line[38:46])
            z       = float(line[46:54])
            occ     = float(line[54:60]) if len(line) > 54 else 1.00
            bfac    = float(line[60:66]) if len(line) > 60 else 0.00
            element = line[76:78].strip() if len(line) > 76 else ''
        except (ValueError, IndexError):
            continue

        if not element:
            element = ''.join(c for c in name.strip() if c.isalpha())[:2]
        atype = AD_TYPES.get(element.upper(),
                AD_TYPES.get(element[:1].upper() if element else 'C', 'C'))

        std_part = (f"ATOM  {serial:5d} {name}{altloc}"
                    f"{resname:3s} {chain}{resseq:4d}{icode}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}")
        std_part = f"{std_part:<66}"
        pdbqt_lines.append(std_part + f"     {0.000:5.3f} {atype:>2}\n")

    with open(pdbqt_path, 'w') as f:
        f.writelines(pdbqt_lines)
    return len(pdbqt_lines) > 0


# ══════════════════════════════════════════════════════════════════
# STEP 1: PREPARE RECEPTOR
# ══════════════════════════════════════════════════════════════════
print(f"\n[STEP 1] Preparing Receptor ({PROTEIN_PDB})...")

if not os.path.exists(PROTEIN_PDB):
    print(f"  ERROR: {PROTEIN_PDB} not found!")
    sys.exit(1)

receptor_pdbqt  = f"{OUTPUT_DIR}/receptor.pdbqt"
receptor_script = find_script("mk_prepare_receptor.py")
receptor_ok     = False

if receptor_script:
    result = subprocess.run(
        [sys.executable, receptor_script, '-i', PROTEIN_PDB,
         '-o', f"{OUTPUT_DIR}/receptor", '--add_H'],
        capture_output=True, text=True, timeout=120)
    if os.path.exists(receptor_pdbqt) and os.path.getsize(receptor_pdbqt) > 0:
        print(f"  Done (meeko): {receptor_pdbqt}")
        receptor_ok = True

if not receptor_ok:
    try:
        from openbabel import pybel
        mol = next(pybel.readfile("pdb", PROTEIN_PDB))
        mol.write("pdbqt", receptor_pdbqt, overwrite=True)
        print(f"  Done (openbabel): {receptor_pdbqt}")
        receptor_ok = True
    except Exception:
        pass

if not receptor_ok:
    if pdb_to_pdbqt_manual(PROTEIN_PDB, receptor_pdbqt):
        print(f"  Done (manual): {receptor_pdbqt}")
        receptor_ok = True


# ══════════════════════════════════════════════════════════════════
# STEP 2: PREPARE LIGANDS
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 2] Preparing Ligands...")

ligand_script = find_script("mk_prepare_ligand.py")


def prepare_ligand(smiles, name):
    """Generate 3D structure and convert to PDBQT."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ERROR: Invalid SMILES for {name}")
        return None

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            print(f"  ERROR: Cannot generate 3D for {name}")
            return None

    AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

    sdf_path   = f"{OUTPUT_DIR}/{name}.sdf"
    pdbqt_path = f"{OUTPUT_DIR}/{name}.pdbqt"

    w = Chem.SDWriter(sdf_path)
    mol.SetProp('_Name', name)
    w.write(mol)
    w.close()

    # Try meeko script
    if ligand_script:
        result = subprocess.run(
            [sys.executable, ligand_script, '-i', sdf_path, '-o', pdbqt_path],
            capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(pdbqt_path):
            print(f"  {name}: OK (meeko script)")
            return pdbqt_path

    # Try meeko API
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setup = prep.prepare(mol)
        if isinstance(setup, (list, tuple)):
            setup = setup[0]
        pdbqt_str = PDBQTWriterLegacy.write_string(setup)
        if isinstance(pdbqt_str, tuple):
            pdbqt_str = pdbqt_str[0]
        with open(pdbqt_path, 'w') as f:
            f.write(pdbqt_str)
        print(f"  {name}: OK (meeko API)")
        return pdbqt_path
    except Exception:
        pass

    # Manual fallback
    pdb_path = f"{OUTPUT_DIR}/{name}_3d.pdb"
    Chem.MolToPDBFile(mol, pdb_path)

    with open(pdb_path) as f:
        pdb_lines = f.readlines()

    lines_out = [f"REMARK  {name}\n"]
    count = 0
    for line in pdb_lines:
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        try:
            name_a  = line[12:16]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else ''
            if not element:
                element = ''.join(c for c in name_a.strip() if c.isalpha())[:1]
            atype = AD_TYPES.get(element.upper(), 'C')
        except (ValueError, IndexError):
            continue
        count += 1
        std = f"HETATM{count:5d} {name_a:<4} LIG     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
        std = f"{std:<66}"
        lines_out.append(std + f"     {0.000:5.3f} {atype:>2}\n")

    lines_out.append("TORSDOF 0\n")
    with open(pdbqt_path, 'w') as f:
        f.writelines(lines_out)
    print(f"  {name}: OK (manual)")
    return pdbqt_path


active_pdbqt   = prepare_ligand(ACTIVE_SMILES,   "active_ligand")
inactive_pdbqt = prepare_ligand(INACTIVE_SMILES, "inactive_ligand")


# ══════════════════════════════════════════════════════════════════
# STEP 3: RUN VINA
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 3] Running AutoDock Vina...")


def parse_vina(text):
    """Parse Vina output table into list of dicts."""
    results = []
    in_table = False
    for line in text.split('\n'):
        s = line.strip()
        if not in_table:
            if 'mode' in s.lower() and 'affinity' in s.lower():
                in_table = True
            continue
        if '-----' in s or not s:
            continue
        parts = s.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        try:
            results.append({
                'mode': int(parts[0]), 'affinity': float(parts[1]),
                'rmsd_lb': float(parts[2]), 'rmsd_ub': float(parts[3]),
            })
        except ValueError:
            if results:
                break
    return results


def dock(ligand_pdbqt, out_pdbqt, label):
    """Run Vina docking and return parsed results."""
    if not ligand_pdbqt or not os.path.exists(ligand_pdbqt):
        print(f"  ERROR: {ligand_pdbqt} not found")
        return None

    cmd = [VINA_PATH, '--receptor', receptor_pdbqt, '--ligand', ligand_pdbqt,
           '--center_x', str(GRID['center_x']), '--center_y', str(GRID['center_y']),
           '--center_z', str(GRID['center_z']), '--size_x', str(GRID['size_x']),
           '--size_y', str(GRID['size_y']), '--size_z', str(GRID['size_z']),
           '--out', out_pdbqt, '--exhaustiveness', '8', '--num_modes', '9',
           '--scoring', 'vina']

    print(f"\n  Docking {label}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        print(f"  ERROR: Vina not found at {VINA_PATH}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timed out after 10 minutes")
        return None

    with open(out_pdbqt.replace('.pdbqt', '.log'), 'w', encoding='utf-8') as f:
        f.write(result.stdout + "\n" + result.stderr)

    print(f"  Return code: {result.returncode}")
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        return None

    res = parse_vina(result.stdout)
    if res:
        print(f"  Best: {res[0]['affinity']:.3f} kcal/mol")
    return res


active_res   = dock(active_pdbqt,   f"{OUTPUT_DIR}/active_docked.pdbqt",   "ACTIVE")
inactive_res = dock(inactive_pdbqt, f"{OUTPUT_DIR}/inactive_docked.pdbqt", "INACTIVE")


# ══════════════════════════════════════════════════════════════════
# STEP 4: RESULTS PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 4] Generating Results...")

best_a = active_res[0]['affinity']   if active_res   else None
best_i = inactive_res[0]['affinity'] if inactive_res else None

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Molecular Docking Results — AChE (PDB: 4EY7)\n"
             "More negative binding affinity = stronger binding = better drug candidate",
             fontsize=13, fontweight='bold')

# Graph 1: Affinity per mode
ax = axes[0]
if active_res:
    ax.plot([r['mode'] for r in active_res], [r['affinity'] for r in active_res],
            'o-', color='#2ecc71', lw=2.5, ms=8,
            label=f'Active (pIC50={ACTIVE_PIC50:.2f})')
    for r in active_res:
        ax.annotate(f"{r['affinity']:.2f}", (r['mode'], r['affinity']),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
if inactive_res:
    ax.plot([r['mode'] for r in inactive_res], [r['affinity'] for r in inactive_res],
            's-', color='#e74c3c', lw=2.5, ms=8,
            label=f'Inactive (pIC50={INACTIVE_PIC50:.2f})')
    for r in inactive_res:
        ax.annotate(f"{r['affinity']:.2f}", (r['mode'], r['affinity']),
                    textcoords='offset points', xytext=(4, -10), fontsize=7)
ax.set_xlabel("Docking Mode (pose number)")
ax.set_ylabel("Binding Affinity (kcal/mol)")
ax.set_title("Binding Affinity per Mode", fontweight='bold')
ax.legend(fontsize=9, loc='best', title='Compound', title_fontsize=9)
ax.grid(alpha=0.3)
if active_res or inactive_res:
    ax.invert_yaxis()

# Graph 2: Comparison bar chart
ax = axes[1]
labels_bar, values_bar, colors_bar = [], [], []
if best_a is not None:
    labels_bar.append(f'Your Active\npIC50={ACTIVE_PIC50:.2f}')
    values_bar.append(best_a)
    colors_bar.append('#2ecc71')
if best_i is not None:
    labels_bar.append(f'Your Inactive\npIC50={INACTIVE_PIC50:.2f}')
    values_bar.append(best_i)
    colors_bar.append('#e74c3c')
labels_bar += ['Paper Active\n(reference)', 'Paper Inactive\n(reference)']
values_bar += [-13.18, -5.845]
colors_bar += ['#27ae60', '#c0392b']

bars = ax.bar(labels_bar, values_bar, color=colors_bar, edgecolor='black',
              alpha=0.85, width=0.6)
for bar, val in zip(bars, values_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.3,
            f'{val:.3f}', ha='center', va='top', fontsize=10,
            fontweight='bold', color='white')

legend_patches = [
    mpatches.Patch(color='#2ecc71', label='Your active compound'),
    mpatches.Patch(color='#e74c3c', label='Your inactive compound'),
    mpatches.Patch(color='#27ae60', label='Paper active reference (-13.18)'),
    mpatches.Patch(color='#c0392b', label='Paper inactive reference (-5.845)'),
]
ax.legend(handles=legend_patches, fontsize=7, loc='lower right',
          title='Legend', title_fontsize=8)
ax.set_ylabel("Best Binding Affinity (kcal/mol)")
ax.set_title("Your Results vs Paper", fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Graph 3: RMSD
ax = axes[2]
if active_res:
    ax.bar([r['mode']-0.2 for r in active_res], [r['rmsd_ub'] for r in active_res],
           width=0.35, color='#2ecc71', alpha=0.8,
           label=f'Active (pIC50={ACTIVE_PIC50:.2f})')
if inactive_res:
    ax.bar([r['mode']+0.2 for r in inactive_res], [r['rmsd_ub'] for r in inactive_res],
           width=0.35, color='#e74c3c', alpha=0.8,
           label=f'Inactive (pIC50={INACTIVE_PIC50:.2f})')
ax.axhline(2.0, color='gray', linestyle='--', linewidth=1,
           label='RMSD=2.0 Å (reliable threshold)')
ax.set_xlabel("Docking Mode (pose number)")
ax.set_ylabel("RMSD Upper Bound (Å)")
ax.set_title("Conformational Variability", fontweight='bold')
ax.legend(fontsize=8, loc='best', title='Compound', title_fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/docking_results.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/docking_results.png")

# Save text results
with open(f"{OUTPUT_DIR}/docking_results.txt", 'w', encoding='utf-8') as f:
    f.write(f"""DOCKING RESULTS
===============
Active:   {f"{best_a:.3f} kcal/mol" if best_a else "FAILED"} (pIC50={ACTIVE_PIC50:.4f})
Inactive: {f"{best_i:.3f} kcal/mol" if best_i else "FAILED"} (pIC50={INACTIVE_PIC50:.4f})

Reference: Active=-13.18, Inactive=-5.845 kcal/mol
""")

print(f"\n{'='*60}")
print(f"  DONE!")
print(f"  Active   : {f'{best_a:.3f} kcal/mol' if best_a else 'FAILED'}")
print(f"  Inactive : {f'{best_i:.3f} kcal/mol' if best_i else 'FAILED'}")
print(f"  Paper    : Active=-13.18, Inactive=-5.845")
print(f"\n  3D visualization in PyMOL:")
print(f"    load {OUTPUT_DIR}/receptor.pdbqt")
print(f"    load {OUTPUT_DIR}/active_docked.pdbqt")
print(f"    show surface, receptor; show sticks, active_docked")
