"""
Run this from anywhere inside (or above) the repo, e.g.:

    python find_fold3_inputs.py
    python find_fold3_inputs.py /path/to/repo/root

It walks the given root (default: current directory) looking for every file
07_fold3_final_evaluation.ipynb will need, prints the full relative path,
size, and last-modified time for each match, and previews the top-level
contents of the three pickles Section 1 depends on (winner_decision_fold2.pkl,
regime_thresholds.pkl, conformal_residuals_fold2.pkl) so paths/keys can be
confirmed without sending the actual data.

Paste the full printed output back — that's what's needed to wire up real
paths in Section 1 (and later sections) instead of assumed ones.
"""

import os
import sys
import pickle
from datetime import datetime

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

# Exact filenames this pipeline is known to depend on (Fold 2 artifacts Fold 3 reuses,
# plus the Fold 3 outputs it will eventually write — checked so we know if any already
# exist and would be silently overwritten).
TARGET_FILES = [
    "winner_decision_fold2.pkl",
    "tweedie_optimized_fold2.txt",
    "tweedie_best_params.pkl",
    "features_train_v2.parquet",
    "features_val_v2.parquet",
    "feature_cols_v2.pkl",
    "item_mean_price_lookup.pkl",
    "regime_thresholds.pkl",
    "sku_regimes_fold2.parquet",
    "conformal_residuals_fold2.pkl",
    "unified_lumpy_intermittent_fold2.parquet",
    "lumpy_policy_params.pkl",
    "croston_predictions_fold2.parquet",
    "final_reorder_params_fold2.parquet",
    "dynamic_policy_full_population_fold2.parquet",
    "dynamic_cost_sensitivity_fold2.parquet",
    "static_vs_dynamic_headtohead_fold2.parquet",
    # Fold 3 outputs — existence here means a prior partial run, not expected on a fresh pass
    "tweedie_optimized_fold3.txt",
    "sku_regimes_fold3.parquet",
    "conformal_residuals_fold3.pkl",
    "final_predictions_fold3.parquet",
    "simulation_results_fold3.parquet",
    "dynamic_cost_sensitivity_fold3.parquet",
    "static_vs_dynamic_headtohead_fold3.parquet",
]

# Notebooks, for confirming relative-path assumptions (06g used '../data/processed')
TARGET_NOTEBOOKS = [
    "06g_inventory_simulation.ipynb",
    "07_fold3_final_evaluation.ipynb",
]

def find_all(root, filenames):
    hits = {name: [] for name in filenames}
    for dirpath, dirnames, filenames_here in os.walk(root):
        # skip heavy/irrelevant dirs
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__", ".ipynb_checkpoints")]
        for name in filenames_here:
            if name in hits:
                full = os.path.join(dirpath, name)
                hits[name].append(full)
    return hits


def describe(path):
    size = os.path.getsize(path)
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    return f"{path}  ({size:,} bytes, modified {mtime})"


def main():
    print(f"Searching under: {os.path.abspath(ROOT)}\n")

    hits = find_all(ROOT, TARGET_FILES + TARGET_NOTEBOOKS)

    print("=" * 80)
    print("FOLD 2 INPUT ARTIFACTS")
    print("=" * 80)
    for name in TARGET_FILES:
        if not name.endswith("_fold3.txt") and not name.endswith("_fold3.pkl") and "fold3" not in name:
            paths = hits[name]
            if paths:
                for p in paths:
                    print(f"[FOUND]   {describe(p)}")
            else:
                print(f"[MISSING] {name}")

    print()
    print("=" * 80)
    print("FOLD 3 OUTPUT PATHS (should be empty on a fresh run)")
    print("=" * 80)
    for name in TARGET_FILES:
        if "fold3" in name:
            paths = hits[name]
            if paths:
                for p in paths:
                    print(f"[EXISTS]  {describe(p)}  <-- already present, confirm before overwriting")
            else:
                print(f"[clear]   {name}")

    print()
    print("=" * 80)
    print("NOTEBOOKS")
    print("=" * 80)
    for name in TARGET_NOTEBOOKS:
        paths = hits[name]
        if paths:
            for p in paths:
                print(f"[FOUND]   {describe(p)}")
        else:
            print(f"[MISSING] {name}")

    print()
    print("=" * 80)
    print("PICKLE CONTENTS PREVIEW (Section 1 depends on these three)")
    print("=" * 80)
    for name in ["winner_decision_fold2.pkl", "regime_thresholds.pkl", "conformal_residuals_fold2.pkl"]:
        paths = hits[name]
        if not paths:
            print(f"\n-- {name}: not found, cannot preview --")
            continue
        path = paths[0]
        print(f"\n-- {name} @ {path} --")
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    v_repr = repr(v)
                    if len(v_repr) > 200:
                        v_repr = v_repr[:200] + f"... <{type(v).__name__}, len={len(v) if hasattr(v, '__len__') else '?'}>"
                    print(f"  {k!r}: {v_repr}")
            else:
                print(f"  <non-dict object of type {type(obj).__name__}, repr truncated>")
                print(f"  {repr(obj)[:300]}")
        except Exception as e:
            print(f"  ERROR loading pickle: {e}")

    print()
    print("Paste everything above back to Claude.")


if __name__ == "__main__":
    main()
