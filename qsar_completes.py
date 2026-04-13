"""
QSAR Modeling for Alzheimer's Drug Discovery
=============================================
Pipeline: Load → Lipinski → Mann-Whitney → Morgan FP → RF Models → SHAP → Feature Importance

Requirements: pip install pandas numpy scikit-learn rdkit matplotlib seaborn scipy shap
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

CSV_FILE     = "acetylcholinesterase_bioactivity_data_3class_pIC50.csv"
OUTPUT_DIR   = "outputs"
MODELS_DIR   = "outputs/models"
N_TREES      = 100
N_FOLDS      = 10
FP_RADIUS    = 2
FP_BITS      = 1024
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("  QSAR Alzheimer Drug Discovery Pipeline")
print("=" * 65)


# ── Helper Functions ──────────────────────────────────────────────

def lipinski_descriptors(smiles):
    """Calculate Lipinski Rule of Five descriptors: MW, LogP, HBD, HBA."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None
    return (
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    )


def smiles_to_morgan(smiles, radius=FP_RADIUS, n_bits=FP_BITS):
    """Convert SMILES to Morgan fingerprint binary vector."""
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


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_r2(y_true, y_pred):
    """R-squared coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def plot_bar_cv(ax, values, title, ylabel, color, edgecolor, decimals=4, suffix=""):
    """Bar chart of cross-validation results with average line and std band."""
    folds = [f"F{i+1}" for i in range(len(values))]
    bars  = ax.bar(folds, values, color=color, edgecolor=edgecolor, alpha=0.85)

    mean_val = np.mean(values)
    std_val  = np.std(values)

    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Avg = {mean_val:.{decimals}f}{suffix}')
    ax.fill_between(range(len(values)), mean_val - std_val, mean_val + std_val,
                    alpha=0.15, color='red', label=f'±1 Std = {std_val:.{decimals}f}')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(values) * 0.01),
                f'{val:.{decimals}f}{suffix}',
                ha='center', va='bottom', fontsize=7.5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc='best')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)


COLORS = {'active': '#2ecc71', 'inactive': '#e74c3c'}


# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading and Cleaning Data...")
print("-" * 45)

df = pd.read_csv(CSV_FILE)
print(f"  Raw dataset shape    : {df.shape}")
print(f"  Columns              : {list(df.columns)}")
print(f"  Bioactivity classes  : {df['bioactivity_class'].value_counts().to_dict()}")

df = df[df['bioactivity_class'].isin(['active', 'inactive'])].copy()
df = df.dropna(subset=['canonical_smiles', 'pIC50']).reset_index(drop=True)

print(f"\n  After cleaning:")
print(f"    Total compounds    : {len(df)}")
print(f"    Active compounds   : {len(df[df['bioactivity_class']=='active'])}")
print(f"    Inactive compounds : {len(df[df['bioactivity_class']=='inactive'])}")


# ══════════════════════════════════════════════════════════════════
# STEP 2: LIPINSKI'S RULE OF FIVE
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 2] Calculating Lipinski Descriptors...")
print("-" * 45)

df[['MW','LogP','NumHDonors','NumHAcceptors']] = df['canonical_smiles'].apply(
    lambda s: pd.Series(lipinski_descriptors(str(s)))
)
df['lipinski_pass'] = (
    (df['MW'] < 500) & (df['LogP'] < 5) &
    (df['NumHDonors'] < 5) & (df['NumHAcceptors'] < 10)
)

print(f"  Drug-like (pass) : {df['lipinski_pass'].sum()}")
print(f"  Not drug-like    : {(~df['lipinski_pass']).sum()}")
print(f"\n  Means by class:")
print(df.groupby('bioactivity_class')[['MW','LogP','NumHDonors','NumHAcceptors']].mean().round(2).to_string())

# ── Plot ──
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Lipinski's Rule of Five Descriptors by Bioactivity Class",
             fontsize=14, fontweight='bold')

features = ['MW',    'LogP', 'NumHDonors', 'NumHAcceptors']
limits   = [500,     5,      5,            10            ]
xlabels  = ['Molecular Weight (Da)', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors']

for ax, feat, limit, xlabel in zip(axes.flatten(), features, limits, xlabels):
    for cls, grp in df.groupby('bioactivity_class'):
        ax.hist(grp[feat].dropna(), bins=30, alpha=0.6,
                label=f'{cls.capitalize()} (n={len(grp)})',
                color=COLORS[cls], edgecolor='white')
    ax.axvline(limit, color='black', linestyle='--', linewidth=2,
               label=f'Lipinski limit = {limit}')
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Compounds')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lipinski_descriptors.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {OUTPUT_DIR}/lipinski_descriptors.png")


# ══════════════════════════════════════════════════════════════════
# STEP 3: MANN-WHITNEY U TEST
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 3] Mann-Whitney U Test...")
print("-" * 45)

active_df   = df[df['bioactivity_class'] == 'active']
inactive_df = df[df['bioactivity_class'] == 'inactive']

print(f"\n  {'Feature':<20} {'Statistic':>12} {'P-value':>12} {'Significant?':>15}")
print("  " + "─" * 62)

mw_results = {}
for feat in ['pIC50', 'MW', 'NumHDonors', 'NumHAcceptors', 'LogP']:
    a = active_df[feat].dropna()
    b = inactive_df[feat].dropna()
    stat, pval = mannwhitneyu(a, b, alternative='two-sided')
    significant = pval < 0.05
    mw_results[feat] = {'stat': stat, 'pval': pval, 'significant': significant}
    result_str = "YES (p<0.05)" if significant else "NO"
    print(f"  {feat:<20} {stat:>12.1f} {pval:>12.6f} {result_str:>15}")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Active vs Inactive Compound Distributions", fontsize=13, fontweight='bold')

for ax, feat in zip(axes, ['pIC50', 'MW']):
    for cls, grp in df.groupby('bioactivity_class'):
        ax.hist(grp[feat].dropna(), bins=30, alpha=0.6,
                label=f'{cls.capitalize()} (n={len(grp)})',
                color=COLORS[cls], edgecolor='white')
    pval = mw_results[feat]['pval']
    sig_text = "significant" if mw_results[feat]['significant'] else "not significant"
    ax.set_title(f"{feat} Distribution\n(Mann-Whitney p={pval:.2e}, {sig_text})", fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mannwhitney_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/mannwhitney_distributions.png")


# ══════════════════════════════════════════════════════════════════
# STEP 4: MORGAN FINGERPRINTS
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 4] Morgan Fingerprints...")
print("-" * 45)
print(f"  Settings: radius={FP_RADIUS}, n_bits={FP_BITS}")

fingerprints  = []
valid_indices = []

for i, smiles in enumerate(df['canonical_smiles']):
    fp = smiles_to_morgan(str(smiles))
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(i)

df      = df.iloc[valid_indices].reset_index(drop=True)
X       = np.array(fingerprints)
y_reg   = df['pIC50'].values
y_class = (df['bioactivity_class'] == 'active').astype(int).values

print(f"  Feature matrix : {X.shape}")
print(f"  Active={y_class.sum()}, Inactive={(y_class==0).sum()}")

avg_bits_on = X.sum(axis=1).mean()
print(f"  Avg bits ON    : {avg_bits_on:.1f} / {FP_BITS} ({avg_bits_on/10.24:.1f}%)")

df_fp = pd.DataFrame(X, columns=[f"FP_{i}" for i in range(FP_BITS)])
df_fp.insert(0, 'molecule_chembl_id', df['molecule_chembl_id'].values)
df_fp.insert(1, 'canonical_smiles',   df['canonical_smiles'].values)
df_fp.insert(2, 'bioactivity_class',  df['bioactivity_class'].values)
df_fp.insert(3, 'pIC50',              df['pIC50'].values)
df_fp.to_csv(f"{OUTPUT_DIR}/morgan_fingerprints.csv", index=False)
print(f"  Saved: {OUTPUT_DIR}/morgan_fingerprints.csv ({df_fp.shape})")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Morgan Fingerprint Analysis", fontsize=13, fontweight='bold')

bits_on = X.sum(axis=1)
for cls_label, cls_val in [('Active', 1), ('Inactive', 0)]:
    mask = y_class == cls_val
    axes[0].hist(bits_on[mask], bins=30, alpha=0.6,
                 label=f'{cls_label} (n={mask.sum()})',
                 color=COLORS[cls_label.lower()], edgecolor='white')
axes[0].axvline(bits_on.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Overall mean = {bits_on.mean():.1f}')
axes[0].set_title("Bits ON per Compound", fontweight='bold')
axes[0].set_xlabel(f"Number of bits = 1 (out of {FP_BITS})")
axes[0].set_ylabel("Number of Compounds")
axes[0].legend(fontsize=8, loc='best')
axes[0].grid(alpha=0.3)

bit_freq = X.mean(axis=0)
axes[1].plot(range(FP_BITS), bit_freq, color='#3498db', alpha=0.7, linewidth=0.8,
             label='Bit frequency')
axes[1].fill_between(range(FP_BITS), 0, bit_freq, alpha=0.3, color='#3498db')
axes[1].axhline(bit_freq.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean frequency = {bit_freq.mean():.4f}')
axes[1].set_title("Frequency of Each Bit Across All Compounds", fontweight='bold')
axes[1].set_xlabel(f"Bit Position (0 to {FP_BITS-1})")
axes[1].set_ylabel("Fraction of Compounds with Bit=1")
axes[1].legend(fontsize=8, loc='best')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fingerprint_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/fingerprint_analysis.png")


# ══════════════════════════════════════════════════════════════════
# STEP 5 & 6: MODEL TRAINING + 10-FOLD CV
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 5 & 6] Training + 10-Fold Cross Validation...")
print("-" * 45)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── Regression ──
print(f"\n  [A] Random Forest Regression")
print(f"  {'Fold':<8} {'MAPE':>10} {'R2':>10}")
print("  " + "─" * 30)

mape_list = []
r2_list   = []
reg_model = RandomForestRegressor(
    n_estimators=N_TREES, criterion='squared_error',
    random_state=RANDOM_STATE, n_jobs=-1
)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    reg_model.fit(X[train_idx], y_reg[train_idx])
    y_pred = reg_model.predict(X[val_idx])
    mape = calculate_mape(y_reg[val_idx], y_pred)
    r2   = calculate_r2(y_reg[val_idx], y_pred)
    mape_list.append(mape)
    r2_list.append(r2)
    print(f"  Fold-{fold+1:<3} {mape:>9.4f}% {r2:>10.4f}")

print(f"  {'─'*30}")
print(f"  {'Avg':<8} {np.mean(mape_list):>9.4f}% {np.mean(r2_list):>10.4f}")
reg_model.fit(X, y_reg)

# ── Classification ──
print(f"\n  [B] Random Forest Classification")
print(f"  {'Fold':<8} {'AUC-ROC':>10} {'F1':>10}")
print("  " + "─" * 30)

auc_list  = []
f1_list   = []
class_model = RandomForestClassifier(
    n_estimators=N_TREES, criterion='gini',
    random_state=RANDOM_STATE, n_jobs=-1
)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    class_model.fit(X[train_idx], y_class[train_idx])
    y_pred = class_model.predict(X[val_idx])
    y_prob = class_model.predict_proba(X[val_idx])[:, 1]
    auc = roc_auc_score(y_class[val_idx], y_prob)
    f1  = f1_score(y_class[val_idx], y_pred)
    auc_list.append(auc)
    f1_list.append(f1)
    print(f"  Fold-{fold+1:<3} {auc:>10.4f} {f1:>10.4f}")

print(f"  {'─'*30}")
print(f"  {'Avg':<8} {np.mean(auc_list):>10.4f} {np.mean(f1_list):>10.4f}")
class_model.fit(X, y_class)


# ── Evaluation Graphs ──
print("\n  Generating evaluation graphs...")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("10-Fold Cross Validation Results — QSAR Alzheimer Drug Discovery",
             fontsize=15, fontweight='bold')
plot_bar_cv(axes[0,0], mape_list, "MAPE per Fold (lower = better)",
            "MAPE (%)", '#3498db', 'navy', decimals=2, suffix="%")
plot_bar_cv(axes[0,1], r2_list,   "R² per Fold (higher = better)",
            "R²",       '#2ecc71', 'darkgreen')
plot_bar_cv(axes[1,0], auc_list,  "AUC-ROC per Fold (higher = better)",
            "AUC-ROC",  '#e67e22', 'darkorange')
plot_bar_cv(axes[1,1], f1_list,   "F1 Score per Fold (higher = better)",
            "F1 Score", '#9b59b6', 'purple')
for ax in [axes[0,1], axes[1,0], axes[1,1]]:
    ax.set_ylim(0, 1.1)
axes[0,0].set_ylim(0, max(mape_list) * 1.25)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/evaluation_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/evaluation_metrics.png")

# ── ROC Curve ──
plt.figure(figsize=(9, 7))
colors_roc = plt.cm.tab10(np.linspace(0, 1, N_FOLDS))
roc_auc_per_fold = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    class_model.fit(X[train_idx], y_class[train_idx])
    y_prob = class_model.predict_proba(X[val_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y_class[val_idx], y_prob)
    auc = roc_auc_score(y_class[val_idx], y_prob)
    roc_auc_per_fold.append(auc)
    plt.plot(fpr, tpr, color=colors_roc[fold], alpha=0.75, linewidth=1.5,
             label=f'Fold-{fold+1} (AUC={auc:.3f})')

plt.plot([0,1], [0,1], 'k--', linewidth=1.5, label='Random (AUC=0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve — 10-Fold CV (Mean AUC = {np.mean(roc_auc_per_fold):.4f})',
          fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=8, title='Fold Performance',
           title_fontsize=9, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
class_model.fit(X, y_class)
print(f"  Saved: {OUTPUT_DIR}/roc_curve.png")

# ── Actual vs Predicted pIC50 ──
y_pred_all = np.zeros(len(y_reg))
for train_idx, val_idx in kf.split(X):
    reg_model.fit(X[train_idx], y_reg[train_idx])
    y_pred_all[val_idx] = reg_model.predict(X[val_idx])
reg_model.fit(X, y_reg)

plt.figure(figsize=(8, 7))
scatter = plt.scatter(y_reg, y_pred_all, alpha=0.3, s=15, c=y_class,
                      cmap='RdYlGn', label='Compounds')
min_val = min(y_reg.min(), y_pred_all.min())
max_val = max(y_reg.max(), y_pred_all.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
         label='Perfect prediction (y=x)')
cbar = plt.colorbar(scatter, shrink=0.8)
cbar.set_label('Class (0=Inactive, 1=Active)', fontsize=10)
plt.xlabel('Actual pIC50', fontsize=12)
plt.ylabel('Predicted pIC50', fontsize=12)
plt.title(f'Actual vs Predicted pIC50\n(R²={np.mean(r2_list):.4f}, MAPE={np.mean(mape_list):.2f}%)',
          fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/actual_vs_predicted.png")


# ══════════════════════════════════════════════════════════════════
# STEP 7A: SHAP INTERPRETATION
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 7A] SHAP Interpretation...")
print("-" * 45)

import shap

df_X = pd.DataFrame(X, columns=[f"FP_{i}" for i in range(FP_BITS)])

print("  Computing SHAP values...")
explainer   = shap.TreeExplainer(class_model)
shap_values = explainer.shap_values(df_X, check_additivity=False)

if isinstance(shap_values, list):
    shap_class0   = shap_values[0]
    shap_class1   = shap_values[1]
    expected_val0 = explainer.expected_value[0]
    expected_val1 = explainer.expected_value[1]
    shap_for_plot = shap_values
else:
    shap_class0   = shap_values[:, :, 0]
    shap_class1   = shap_values[:, :, 1]
    if hasattr(explainer.expected_value, "__len__"):
        expected_val0 = explainer.expected_value[0]
        expected_val1 = explainer.expected_value[1]
    else:
        expected_val0 = explainer.expected_value
        expected_val1 = explainer.expected_value
    shap_for_plot = [shap_class0, shap_class1]

print("  SHAP values computed!")

# SHAP Global Bar
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_for_plot, df_X, plot_type="bar", max_display=20, show=False)
plt.title("Top 20 Morgan Fingerprints — SHAP Global Summary (Classification)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_global_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_global_summary.png")

# SHAP Dot Plot
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_class1, df_X, max_display=20, show=False)
plt.title("SHAP Dot Plot — Feature Impact Direction\n"
          "Red = high feature value | Blue = low feature value",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_dot_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_dot_plot.png")

# SHAP Force Plots (HTML)
active_idx   = np.where(y_class == 1)[0][0]
inactive_idx = np.where(y_class == 0)[0][0]
shap.initjs()

for label, idx in [("active", active_idx), ("inactive", inactive_idx)]:
    try:
        force = shap.force_plot(expected_val1, shap_class1[idx],
                                df_X.iloc[idx], show=False)
        html = f"<html><head>{shap.getjs()}</head><body>{force.html()}</body></html>"
        with open(f"{OUTPUT_DIR}/shap_force_{label}.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Saved: {OUTPUT_DIR}/shap_force_{label}.html")
    except Exception as e:
        print(f"  Warning ({label} force plot): {e}")

# Top 10 SHAP features
mean_shap      = np.abs(shap_class1).mean(axis=0)
top10_shap_idx = np.argsort(mean_shap)[::-1][:10]
print(f"\n  Top 10 SHAP Features:")
for rank, idx in enumerate(top10_shap_idx, 1):
    print(f"    {rank}. FP_{idx} = {mean_shap[idx]:.6f}")


# ══════════════════════════════════════════════════════════════════
# STEP 7B: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 7B] Feature Importance Analysis...")
print("-" * 45)

clf_importances = class_model.feature_importances_
clf_top20_idx   = np.argsort(clf_importances)[::-1][:20]
clf_top20_vals  = clf_importances[clf_top20_idx]
clf_top20_names = [f"FP_{i}" for i in clf_top20_idx]

reg_importances = reg_model.feature_importances_
reg_top20_idx   = np.argsort(reg_importances)[::-1][:20]
reg_top20_vals  = reg_importances[reg_top20_idx]
reg_top20_names = [f"FP_{i}" for i in reg_top20_idx]

print(f"\n  Top 10 — Classification:")
for rank, (name, val) in enumerate(zip(clf_top20_names[:10], clf_top20_vals[:10]), 1):
    print(f"    {rank}. {name} = {val:.6f}")
print(f"\n  Top 10 — Regression:")
for rank, (name, val) in enumerate(zip(reg_top20_names[:10], reg_top20_vals[:10]), 1):
    print(f"    {rank}. {name} = {val:.6f}")

# ── Classification Feature Importance ──
plt.figure(figsize=(14, 6))
bars = plt.bar(range(20), clf_top20_vals, color='#3498db', edgecolor='navy',
               alpha=0.85, label='Gini importance (classification)')
for bar, val in zip(bars, clf_top20_vals):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)
plt.axhline(clf_top20_vals.mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'Mean top-20 = {clf_top20_vals.mean():.4f}')
plt.xticks(range(20), clf_top20_names, rotation=45, ha='right', fontsize=9)
plt.title("Top 20 Important Morgan Fingerprints — Classification Model (Active vs Inactive)",
          fontsize=13, fontweight='bold')
plt.xlabel("Morgan Fingerprint Bit Index")
plt.ylabel("Feature Importance (Gini)")
plt.legend(fontsize=9, loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_classification.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/feature_importance_classification.png")

# ── Regression Feature Importance ──
plt.figure(figsize=(14, 6))
bars = plt.bar(range(20), reg_top20_vals, color='#2ecc71', edgecolor='darkgreen',
               alpha=0.85, label='MSE importance (regression)')
for bar, val in zip(bars, reg_top20_vals):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)
plt.axhline(reg_top20_vals.mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'Mean top-20 = {reg_top20_vals.mean():.4f}')
plt.xticks(range(20), reg_top20_names, rotation=45, ha='right', fontsize=9)
plt.title("Top 20 Important Morgan Fingerprints — Regression Model (pIC50 Prediction)",
          fontsize=13, fontweight='bold')
plt.xlabel("Morgan Fingerprint Bit Index")
plt.ylabel("Feature Importance (MSE Reduction)")
plt.legend(fontsize=9, loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_regression.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/feature_importance_regression.png")

# ── Side-by-Side Comparison ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature Importance Comparison — Classification vs Regression",
             fontsize=14, fontweight='bold')

axes[0].barh(range(20)[::-1], clf_top20_vals, color='#3498db', alpha=0.85,
             edgecolor='navy', label='Classification (Gini)')
axes[0].axvline(clf_top20_vals.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {clf_top20_vals.mean():.4f}')
axes[0].set_yticks(range(20)[::-1])
axes[0].set_yticklabels(clf_top20_names, fontsize=9)
axes[0].set_title("Classification — Active vs Inactive", fontweight='bold')
axes[0].set_xlabel("Importance Score")
axes[0].legend(fontsize=8, loc='lower right')
axes[0].grid(axis='x', alpha=0.3)
for i, val in enumerate(clf_top20_vals[::-1]):
    axes[0].text(val + 0.0001, i, f'{val:.4f}', va='center', fontsize=7.5)

axes[1].barh(range(20)[::-1], reg_top20_vals, color='#2ecc71', alpha=0.85,
             edgecolor='darkgreen', label='Regression (MSE)')
axes[1].axvline(reg_top20_vals.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {reg_top20_vals.mean():.4f}')
axes[1].set_yticks(range(20)[::-1])
axes[1].set_yticklabels(reg_top20_names, fontsize=9)
axes[1].set_title("Regression — pIC50 Prediction", fontweight='bold')
axes[1].set_xlabel("Importance Score")
axes[1].legend(fontsize=8, loc='lower right')
axes[1].grid(axis='x', alpha=0.3)
for i, val in enumerate(reg_top20_vals[::-1]):
    axes[1].text(val + 0.0001, i, f'{val:.4f}', va='center', fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/feature_importance_comparison.png")


# ══════════════════════════════════════════════════════════════════
# STEP 8: SAVE MODELS & PREDICTIONS
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 8] Saving Models...")
print("-" * 45)

with open(f"{MODELS_DIR}/regression_model.pkl", 'wb') as f:
    pickle.dump(reg_model, f)
with open(f"{MODELS_DIR}/classification_model.pkl", 'wb') as f:
    pickle.dump(class_model, f)
print(f"  Saved: {MODELS_DIR}/regression_model.pkl")
print(f"  Saved: {MODELS_DIR}/classification_model.pkl")

# Verify
with open(f"{MODELS_DIR}/regression_model.pkl", 'rb') as f:
    loaded_reg = pickle.load(f)
with open(f"{MODELS_DIR}/classification_model.pkl", 'rb') as f:
    loaded_clf = pickle.load(f)
print(f"  Models verified!")

# ── Example Predictions ──
new_smiles_list = [
    "CCOc1nn(-c2cccc(OCc3ccccc3)c2)c(=O)o1",
    "O=C(N1CCCCC1)n1nc(-c2ccc(Cl)cc2)nc1SCC1CC1",
    "CN(C(=O)n1nc(-c2ccc(Cl)cc2)nc1SCC(F)(F)F)c1ccccc1",
    "CSc1nc(-c2ccc(OC(F)(F)F)cc2)nn1C(=O)N(C)C",
    "CCCCCCSc1nc(-c2ccc(Cl)cc2)nn1C(=O)N1CCOCC1",
]

new_fps   = [fp for smi in new_smiles_list if (fp := smiles_to_morgan(smi)) is not None]
valid_new = [smi for smi in new_smiles_list if smiles_to_morgan(smi) is not None]
new_fps   = np.array(new_fps)

pic50_pred = loaded_reg.predict(new_fps)
class_pred = loaded_clf.predict(new_fps)
class_prob = loaded_clf.predict_proba(new_fps)[:, 1]

print(f"\n  Predictions:")
print(f"  {'#':<3} {'pIC50':>7} {'Class':<10} {'P(Active)':>10}  SMILES")
print("  " + "─" * 75)
for i, (smi, pic50, cls, prob) in enumerate(
        zip(valid_new, pic50_pred, class_pred, class_prob), 1):
    label = "Active" if cls == 1 else "Inactive"
    print(f"  {i:<3} {pic50:>7.4f} {label:<10} {prob:>9.2%}  {smi[:50]}")


# ── Final Summary ──
print(f"\n{'='*65}")
print(f"  PIPELINE COMPLETE")
print(f"{'='*65}")
print(f"""
  Regression  : MAPE={np.mean(mape_list):.2f}%, R²={np.mean(r2_list):.4f}
  Classification: AUC={np.mean(auc_list):.4f}, F1={np.mean(f1_list):.4f}

  Output files in: {OUTPUT_DIR}/
  Models in      : {MODELS_DIR}/
  Next step      : python addon_features.py
""")
