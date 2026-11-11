"""
Evaluate similar_filter output JSON.

Usage:
    python ensemble_similar_filter.py --input_path similar_filter_Qwen_Qwen3-32B.json
    python ensemble_similar_filter.py --input_path similar_filter_Qwen_Qwen3-32B.json --output_csv review.csv

    # Ensemble-all: keep only pairs where is_synonym=True in ALL inputs
    python ensemble_similar_filter.py --ensemble-all a.json b.json c.json
    python ensemble_similar_filter.py --ensemble-all a.json b.json --output_csv ensemble_all_review.csv

    # Ensemble-any: keep only pairs where is_synonym=True in ANY input
    python ensemble_similar_filter.py --ensemble-any a.json b.json c.json
    python ensemble_similar_filter.py --ensemble-any a.json b.json --output_csv ensemble_any_review.csv
"""

import argparse
import json
from collections import Counter

import pandas as pd


def path_to_str(path: list) -> str:
    return " > ".join(p["concept"] for p in path)


def pair_key(r: dict) -> tuple:
    a, b = r["node_A"]["uuid"], r["node_B"]["uuid"]
    return (a, b) if a < b else (b, a)


COLS_SINGLE = [
    "is_synonym", "keep",
    "uuid_A", "concept_A", "concept_desc_A", "depth_A", "parent_A", "path_A",
    "uuid_B", "concept_B", "concept_desc_B", "depth_B", "parent_B", "path_B",
    "explanation", "raw_response",
]

COLS_ENSEMBLE = [
    "is_synonym",
    "keep_A_votes", "keep_B_votes", "keep_A_files", "keep_B_files", "majority_keep",
    "uuid_A", "concept_A", "concept_desc_A", "depth_A", "parent_A", "path_A",
    "uuid_B", "concept_B", "concept_desc_B", "depth_B", "parent_B", "path_B",
    "explanation", "raw_response",
]


def make_row(r: dict) -> dict:
    na, nb = r["node_A"], r["node_B"]
    return {
        "is_synonym":     r.get("is_synonym"),
        "keep":           r.get("keep", "NA"),
        "uuid_A":         na["uuid"],
        "concept_A":      na["concept"],
        "concept_desc_A": na.get("concept_desc", ""),
        "depth_A":        na["depth"],
        "parent_A":       na["path"][-2]["concept"] if len(na["path"]) >= 2 else "",
        "path_A":         path_to_str(na["path"]),
        "uuid_B":         nb["uuid"],
        "concept_B":      nb["concept"],
        "concept_desc_B": nb.get("concept_desc", ""),
        "depth_B":        nb["depth"],
        "parent_B":       nb["path"][-2]["concept"] if len(nb["path"]) >= 2 else "",
        "path_B":         path_to_str(nb["path"]),
        "explanation":    r.get("explanation", "NA"),
        "raw_response":   r.get("raw_response") or "",
    }


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["is_synonym", "concept_A"],
        key=lambda col: col.map(lambda x: (x is not True, x is None, str(x))) if col.name == "is_synonym" else col,
        ascending=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_path", type=str)
    group.add_argument("--ensemble-all", nargs="+", metavar="JSON", dest="ensemble_all",
                       help="Multiple input JSONs; keep pairs where is_synonym=True in ALL")
    group.add_argument("--ensemble-any", nargs="+", metavar="JSON", dest="ensemble_any",
                       help="Multiple input JSONs; keep pairs where is_synonym=True in ANY")
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def run_single(input_path: str, output_csv: str | None):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total pairs: {len(data)}")

    counts = {"True": 0, "False": 0, "None (parse failed)": 0}
    for r in data:
        v = r.get("is_synonym")
        if v is True:
            counts["True"] += 1
        elif v is False:
            counts["False"] += 1
        else:
            counts["None (parse failed)"] += 1

    print("\n--- is_synonym statistics ---")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    df = sort_df(pd.DataFrame(make_row(r) for r in data))[COLS_SINGLE]

    if output_csv is None:
        output_csv = f"{input_path.rsplit('.', 1)[0]}_review.csv"

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nCSV saved to: {output_csv}  ({len(df)} rows)")


def run_ensemble(input_paths: list[str], output_csv: str | None, mode: str):
    all_data: list[list[dict]] = []
    for p in input_paths:
        with open(p, encoding="utf-8") as f:
            all_data.append(json.load(f))
        print(f"  {p}: {len(all_data[-1])} pairs")

    labels = [p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in input_paths]

    # Build per-file synonym sets (canonical key = sorted uuid tuple)
    synonym_sets = [
        {pair_key(r) for r in data if r.get("is_synonym") is True}
        for data in all_data
    ]
    if mode == "all":
        agreed_keys = synonym_sets[0].intersection(*synonym_sets[1:])
    else:
        agreed_keys = synonym_sets[0].union(*synonym_sets[1:])
    print(f"\nFiles: {len(input_paths)}  |  mode={mode}  |  is_synonym=True pairs: {len(agreed_keys)}")

    # Index every file by pair_key for explanation lookup
    indices = [
        {pair_key(r): r for r in data}
        for data in all_data
    ]

    rows = []
    for r in all_data[0]:
        key = pair_key(r)
        if key not in agreed_keys:
            continue
        row = make_row(r)

        keep_A_files, keep_B_files, parts = [], [], []
        for label, idx in zip(labels, indices):
            rec = idx.get(key, {})
            k = rec.get("keep", "NA")
            if k == "A":
                keep_A_files.append(label)
            elif k == "B":
                keep_B_files.append(label)
            expl = rec.get("explanation", "NA")
            parts.append(f"[{label}] {expl}")

        va, vb = len(keep_A_files), len(keep_B_files)
        if va > vb:
            majority = "A"
        elif vb > va:
            majority = "B"
        else:
            majority = "tie"

        row.pop("keep", None)
        row["keep_A_votes"]  = va
        row["keep_B_votes"]  = vb
        row["keep_A_files"]  = "; ".join(keep_A_files)
        row["keep_B_files"]  = "; ".join(keep_B_files)
        row["majority_keep"] = majority
        row["explanation"]   = "\n\n".join(parts)
        rows.append(row)

    # --- Keep consensus among agreed synonym pairs ---
    n_files = len(input_paths)
    keep_consensus_A = sum(1 for row in rows if row["keep_A_votes"] == n_files)
    keep_consensus_B = sum(1 for row in rows if row["keep_B_votes"] == n_files)
    keep_divergent   = len(rows) - keep_consensus_A - keep_consensus_B
    print(f"\n--- Keep consensus (mode={mode}, {len(rows)} synonym pairs) ---")
    print(f"  All agree keep A: {keep_consensus_A}")
    print(f"  All agree keep B: {keep_consensus_B}")
    print(f"  Divergent:        {keep_divergent}")

    # --- Node survival under majority_keep (nodes visible in similarity pairs) ---
    all_nodes: dict[str, int] = {}
    for r in all_data[0]:
        for node in (r["node_A"], r["node_B"]):
            all_nodes[node["uuid"]] = node["depth"]

    removed: set[str] = set()
    for row in rows:
        mk = row["majority_keep"]
        if mk == "A":
            removed.add(row["uuid_B"])
        elif mk == "B":
            removed.add(row["uuid_A"])

    before = Counter(all_nodes.values())
    after  = Counter(d for uid, d in all_nodes.items() if uid not in removed)
    print(f"\n--- Node survival under majority_keep ---")
    print(f"  depth 2: {before[2]:>4} → {after[2]:>4}  (removed {before[2] - after[2]})")
    print(f"  depth 3: {before[3]:>4} → {after[3]:>4}  (removed {before[3] - after[3]})")

    df = sort_df(pd.DataFrame(rows))[COLS_ENSEMBLE]

    if output_csv is None:
        output_csv = f"ensemble_{mode}_review.csv"
        output_csv = f"{input_paths[0].rsplit('.', 1)[0].rsplit('/', 1)[0]}/{output_csv}"

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"CSV saved to: {output_csv}  ({len(df)} rows)")


def main():
    args = parse_args()
    if args.ensemble_all:
        run_ensemble(args.ensemble_all, args.output_csv, mode="all")
    elif args.ensemble_any:
        run_ensemble(args.ensemble_any, args.output_csv, mode="any")
    else:
        run_single(args.input_path, args.output_csv)


if __name__ == "__main__":
    main()
