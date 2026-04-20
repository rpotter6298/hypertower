"""Compare fold assignments between v3 PatientFirstSplitManager and v4 SplitManager."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3.classes.split_manager import PatientFirstSplitManager
from v4.classes.split_manager import SplitManager
from v4.classes.profiles.v4papila import build_data

args = {
    "image_dir":       "Papila/FundusImages",
    "clinical_dir":    "Papila/ClinicalData",
    "label_col":       "Diagnosis",
    "iop_corr_method": "ratio",
    "iop_drop_raw":    True,
    "exclude_cols":    ["Axial_Length"],
}
# Resolve relative paths
root = Path(__file__).resolve().parents[2]
args["image_dir"]    = str(root / args["image_dir"])
args["clinical_dir"] = str(root / args["clinical_dir"])

data = build_data(args)

label_col   = data.label_col
patient_col = data.patient_col
df_mode     = data.df[data.df[label_col].isin([0, 1])].reset_index(drop=True)

# ── v3 splits ────────────────────────────────────────────────────────────────
split_mgr_v3  = PatientFirstSplitManager(patient_col=patient_col, label_col=label_col)
split_args_v3 = SimpleNamespace(eval_mode="binary", n_splits=5, fold_seed=100)
clinical_ns   = SimpleNamespace(df=df_mode, label_col=label_col)
splits_v3     = split_mgr_v3.build_plans(clinical=clinical_ns, args=split_args_v3, profile=None)

# ── v4 splits ────────────────────────────────────────────────────────────────
splits_v4 = SplitManager(group_col=patient_col).build_plans(
    df_mode, label_col=label_col, n_splits=5, seed=100,
)

# ── Compare ──────────────────────────────────────────────────────────────────
print(f"{'Fold':<6} {'Set':<6} {'v3 patients':<8} {'v4 patients':<8} {'Match'}")
print("-" * 50)

all_match = True
for fold in range(5):
    s3, s4 = splits_v3[fold], splits_v4[fold]
    for label, df3, df4 in [
        ("train", s3.train, s4.train),
        ("val",   s3.val,   s4.val),
        ("test",  s3.test,  s4.test),
    ]:
        ids3 = set(df3[patient_col].unique()) if df3 is not None else set()
        ids4 = set(df4[patient_col].unique()) if df4 is not None else set()
        match = ids3 == ids4
        if not match:
            all_match = False
        print(f"{fold+1:<6} {label:<6} {len(ids3):<8} {len(ids4):<8} {'✓' if match else '✗ DIFF'}")
        if not match:
            print(f"       only in v3: {sorted(ids3 - ids4)[:10]}")
            print(f"       only in v4: {sorted(ids4 - ids3)[:10]}")

print()
print("All folds match!" if all_match else "SPLITS DIFFER — fold assignments changed.")
