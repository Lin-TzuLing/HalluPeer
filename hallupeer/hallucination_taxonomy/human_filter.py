"""
Apply synonym-filter decisions to a taxonomy tree pickle.

Modes
-----
--mode uuid   Read a text file (one UUID per line); remove each node from tree.
--mode csv    Read a review CSV; for each row:
                keep=A  → remove uuid_B
                keep=B  → remove uuid_A
                keep=both / skip / empty → do nothing

Usage
-----
    # UUID mode
    python final_filter.py --pkl ensemble.pkl --mode uuid \
        --uuids uuids_to_remove.txt --output filtered.pkl

    # CSV mode
    python final_filter.py --pkl ensemble.pkl --mode csv \
        --csv ensemble_review.csv --output filtered.pkl
        
    # CSV human mode (uses 'human_keep' column instead of 'keep')
    python human_filter.py --pkl ensemble.pkl --mode csv_human \
      --csv ensemble_all_review_human.csv --output filtered_human.pkl
"""

import argparse
import importlib.util
import pickle
import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap: load tree module without triggering hallupeer/__init__.py
# ---------------------------------------------------------------------------
def _bootstrap_tree_module(src_root: Path):
    if "hallupeer.hallucination_taxonomy.hallucination_tree" in sys.modules:
        return sys.modules["hallupeer.hallucination_taxonomy.hallucination_tree"]

    for name, path in [
        ("hallupeer", src_root),
        ("hallupeer.hallucination_taxonomy", src_root / "hallucination_taxonomy"),
    ]:
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            pkg.__package__ = name
            sys.modules[name] = pkg

    mod_path = src_root / "hallucination_taxonomy" / "hallucination_tree.py"
    spec = importlib.util.spec_from_file_location(
        "hallupeer.hallucination_taxonomy.hallucination_tree", mod_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hallupeer.hallucination_taxonomy.hallucination_tree"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_tree(pkl_path: str, src_root: Path):
    _bootstrap_tree_module(src_root)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_tree(tree, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(tree, f)


def _collect_subtree_uuids(node) -> set:
    uuids = {node.uuid}
    for child in node.children:
        uuids |= _collect_subtree_uuids(child)
    return uuids


def apply_removals(tree, uuids: list[str], ht_mod) -> tuple[int, int, int]:
    removed, skipped_cascade, skipped_missing = 0, 0, 0
    implicitly_removed: set[str] = set()

    for uid in uuids:
        if uid in implicitly_removed:
            print(f"  [warn] cascade-removed (parent already deleted): {uid}")
            skipped_cascade += 1
            continue

        node = tree.get_node_by_uuid(uid)
        if node is None:
            print(f"  [warn] not found in tree: {uid}")
            skipped_missing += 1
            continue

        subtree_uuids = _collect_subtree_uuids(node)  # includes node itself

        try:
            ok = ht_mod.remove_node(tree, uid)
            if ok:
                implicitly_removed |= subtree_uuids  # node + all descendants
                removed += 1
            else:
                print(f"  [warn] not found in tree: {uid}")
                skipped_missing += 1
        except ValueError as e:
            print(f"  [warn] {e} (uuid={uid})")
            skipped_missing += 1

    return removed, skipped_cascade, skipped_missing


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_uuid(args, tree, ht_mod):
    with open(args.uuids, encoding="utf-8") as f:
        uuids = [line.strip() for line in f if line.strip()]
    print(f"UUIDs to remove: {len(uuids)}")
    return apply_removals(tree, uuids, ht_mod)


def mode_csv(args, tree, ht_mod):
    import pandas as pd
    df = pd.read_csv(args.csv, encoding="utf-8-sig", dtype=str).fillna("")

    use_col = "majority_keep" if "majority_keep" in df.columns else "keep"
    uuids = []
    for _, row in df.iterrows():
        k = row.get(use_col, "").strip()
        if k == "A":
            uuids.append(row["uuid_B"])
        elif k == "B":
            uuids.append(row["uuid_A"])

    print(f"Rows in CSV: {len(df)}  |  nodes to remove: {len(uuids)}")
    return apply_removals(tree, uuids, ht_mod)


def mode_csv_human(args, tree, ht_mod):
    import pandas as pd
    df = pd.read_csv(args.csv, encoding="utf-8-sig", dtype=str).fillna("")

    if "human_keep" not in df.columns:
        sys.exit("csv_human mode requires 'human_keep' column in CSV.")

    uuids = []
    skipped_empty = 0
    for _, row in df.iterrows():
        k = row["human_keep"].strip()
        if k == "A":
            uuids.append(row["uuid_B"])
        elif k == "B":
            uuids.append(row["uuid_A"])
        else:
            skipped_empty += 1

    print(f"Rows in CSV: {len(df)}  |  nodes to remove: {len(uuids)}  |  skipped (empty/both): {skipped_empty}")
    return apply_removals(tree, uuids, ht_mod)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Input taxonomy tree pickle")
    parser.add_argument("--mode", choices=["uuid", "csv", "csv_human"], required=True)
    parser.add_argument("--uuids", default="uuids_to_remove.txt",
                        help="[uuid mode] text file with one UUID per line")
    parser.add_argument("--csv", default="ensemble_review.csv",
                        help="[csv mode] review CSV with keep column")
    parser.add_argument("--output", default=None,
                        help="Output pkl path (default: <input>_filtered.pkl)")
    parser.add_argument("--src", default=None,
                        help="Path to hallupeer/ directory (auto-detected if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()

    src_root = Path(args.src) if args.src else Path(__file__).resolve().parents[1]
    if not src_root.exists():
        sys.exit(f"hallupeer/ not found at {src_root}. Pass --src explicitly.")

    print(f"Loading tree: {args.pkl}")
    tree = load_tree(args.pkl, src_root)
    ht_mod = sys.modules["hallupeer.hallucination_taxonomy.hallucination_tree"]
    print(f"Tree loaded  — root: '{tree.concept}'  size: {tree.get_size()}")

    if args.mode == "uuid":
        removed, skipped_cascade, skipped_missing = mode_uuid(args, tree, ht_mod)
    elif args.mode == "csv":
        removed, skipped_cascade, skipped_missing = mode_csv(args, tree, ht_mod)
    else:
        removed, skipped_cascade, skipped_missing = mode_csv_human(args, tree, ht_mod)

    print(f"Removed: {removed}  |  cascade-skipped: {skipped_cascade}  |  not-found: {skipped_missing}")
    print(f"New tree size: {tree.get_size()}")

    output = args.output or args.pkl.replace(".pkl", "_filtered.pkl")
    save_tree(tree, output)
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()
