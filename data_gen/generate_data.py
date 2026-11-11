#!/usr/bin/env python3
"""End-to-end hallucination benchmark builder.

Naming: the taxonomy depth-1 node (second segment of "A > B > C") is called depth1
throughout this script's output and in upstream `template_meta` (emits `label_depth1`).

Phase 1: Paper profiling — per-paper sentence count and depth1 hallucination inventory.
Phase 2: Stratified paper-level split (Train/Dev/Test) — greedy iterative stratification on
         the 9-class (depth1) inventory, preventing review-level leakage across splits.
Phase 3: Per-set task instantiation & labeling
         * Group A (review-level Task 1/2/3): one minimal pair per review,
           selection biased toward under-represented (depth1, position) within each set.
             - Task 1: label=0 (clean) | label=1 (hallu)
             - Task 2: hallu only, label=<depth1 string> (pure 9-class)
             - Task 3: hallu only, label={start_char_idx,end_char_idx,gold_span_text}
         * Group B (sentence-level Task 1/2): full expansion, <=1 random template per sentence
           (no down-sampling, long-tail preserved).
             - Task 1: label=0 (orig) | label=1 (hallu)
             - Task 2: hallu only, label=<depth1 string>

Usage:
    python generate_data.py \
        --input /path/to/HalluPeer_ICLR2019 \
        --out   /path/to/output_root \
        --datasets all
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import json
import random
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.io import list_json_files, read_json, write_jsonl
from utils.schema import position_bucket, get_second_node_label
from utils.span_utils import build_review_with_offsets


DATASETS = [
    "task1_review", "task2_review", "task3_review",
    "task1_sent",   "task2_sent",
]


# ===================== Phase 1: profiling =====================

def profile_paper(p: Dict[str, Any]) -> Dict[str, Any]:
    """Per-paper sentence count + per-depth1 hallucination inventory."""
    pid = p.get("paper_id")
    n_sents = 0
    inv: Counter = Counter()
    for r in p.get("reviews", []) or []:
        sents = r.get("review_sentences", []) or []
        n_sents += len(sents)
        for s in sents:
            for t in s.get("hallucination_injection_templates", []) or []:
                full = t.get("hallucination_labels") or t.get("hallucination_label") or "UNKNOWN"
                inv[get_second_node_label(full)] += 1
    return {
        "paper_id": pid,
        "n_sents": n_sents,
        "inventory": dict(inv),
        "total_templates": sum(inv.values()),
    }


# ===================== Phase 2: stratified split =====================

def stratified_split(profiles: List[Dict[str, Any]],
                     ratios: Tuple[float, float, float],
                     rng: random.Random) -> Dict[str, str]:
    """Iterative stratification (Sechidis et al., 2011) over depth1 inventory.

    Strategy: process papers in descending total_templates order. For each paper,
    pick a *key label* — argmax over `paper.inv[L] / remaining_demand[L]`, which
    favors rare-but-still-needed labels while letting heavy single-class papers
    dominate. Assign the paper to the still-open split with the largest deficit
    on that key label. Ties broken by paper-count deficit, then randomly. Papers
    with empty inventory are routed purely by paper-count deficit.

    Hard caps on paper counts (`paper_cap`) enforce the requested ratios at
    paper level; rounding drift goes to the largest-ratio split first.

    Returns: paper_id -> "train" | "dev" | "test".
    """
    splits = ["train", "dev", "test"]
    r_map = dict(zip(splits, ratios))
    n_papers = len(profiles)

    label_total: Counter = Counter()
    for prof in profiles:
        for lab, c in prof["inventory"].items():
            label_total[lab] += c
    label_target = {s: {lab: r_map[s] * c for lab, c in label_total.items()} for s in splits}

    paper_cap = {s: int(round(r_map[s] * n_papers)) for s in splits}
    drift = n_papers - sum(paper_cap.values())
    if drift != 0:
        order = sorted(splits, key=lambda s: -r_map[s])
        step = 1 if drift > 0 else -1
        for i in range(abs(drift)):
            paper_cap[order[i % len(order)]] += step

    ordered = list(profiles)
    rng.shuffle(ordered)
    ordered.sort(key=lambda x: x["total_templates"], reverse=True)

    cur_label = {s: Counter() for s in splits}
    cur_papers = {s: 0 for s in splits}
    assignment: Dict[str, str] = {}

    def open_splits() -> List[str]:
        return [s for s in splits if cur_papers[s] < paper_cap[s]]

    for prof in ordered:
        pid = prof["paper_id"]
        inv = prof["inventory"]
        avail = open_splits() or splits  # fallback should not trigger

        if inv:
            remaining_per_label = {
                lab: sum(max(0.0, label_target[s].get(lab, 0.0) - cur_label[s][lab]) for s in avail)
                for lab in inv
            }
            key_lab = max(
                inv,
                key=lambda L: inv[L] / max(1.0, remaining_per_label[L] + 1e-9),
            )
            best_s = max(
                avail,
                key=lambda s: (
                    label_target[s].get(key_lab, 0.0) - cur_label[s][key_lab],
                    paper_cap[s] - cur_papers[s],
                    rng.random(),
                ),
            )
        else:
            best_s = max(avail, key=lambda s: (paper_cap[s] - cur_papers[s], rng.random()))

        assignment[pid] = best_s
        cur_papers[best_s] += 1
        for lab, c in inv.items():
            cur_label[best_s][lab] += c
    return assignment


# ===================== Phase 3 Group A: review-level =====================

def _position_pct(sent_idx: int, n_sents: int) -> int:
    """Integer percentage position of sent_idx in a review of n_sents sentences.
    Uses (sent_idx / n_sents) * 100 rounded to nearest int. Range: 0..100.
    """
    if n_sents <= 0:
        return 0
    return int(round(sent_idx / n_sents * 100))


def _pick_review_template(review: Dict[str, Any],
                          rng: random.Random,
                          class_counter: Counter,
                          pos_counter: Counter,
                          ) -> Optional[Tuple[int, Dict[str, Any], int, Dict[str, Any], str, int, str]]:
    """Choose one (sent_idx, sent_dict, template_idx, template, pos_bucket, pos_pct, depth1)
    biased toward under-represented (depth1, position bucket) within the current set.
    Position bucket (front/mid/back) is used only for balancing; row emits integer percentage."""
    sents = review.get("review_sentences", []) or []
    n = len(sents)
    cand = []
    for i, s in enumerate(sents):
        bucket = position_bucket(i, n)
        if bucket not in ("front", "mid", "back"):
            continue
        pct = _position_pct(i, n)
        for t_idx, t in enumerate(s.get("hallucination_injection_templates", []) or []):
            full = t.get("hallucination_labels") or t.get("hallucination_label") or "UNKNOWN"
            depth1 = get_second_node_label(full)
            cand.append((i, s, t_idx, t, bucket, pct, depth1))
    if not cand:
        return None
    weights = [1.0 / (1 + class_counter[depth1]) + 1.0 / (1 + pos_counter[bucket])
               for _, _, _, _, bucket, _, depth1 in cand]
    pick = rng.choices(cand, weights=weights, k=1)[0]
    i, s, t_idx, t, bucket, pct, depth1 = pick
    class_counter[depth1] += 1
    pos_counter[bucket] += 1
    return i, s, t_idx, t, bucket, pct, depth1


def build_group_a(papers_by_split: Dict[str, List[Dict[str, Any]]],
                  rng: random.Random,
                  include_task1_review: bool,
                  include_task2_review: bool,
                  include_task3_review: bool,
                  ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out = {ds: {"train": [], "dev": [], "test": []}
           for ds in ("task1_review", "task2_review", "task3_review")}

    for split_name, papers in papers_by_split.items():
        class_counter: Counter = Counter()
        pos_counter: Counter = Counter()

        for p in papers:
            pid = p.get("paper_id")

            for review_idx, r in enumerate(p.get("reviews", []) or []):
                rid = r.get("review_id")
                sents = r.get("review_sentences", []) or []
                clean_sent_texts = [s.get("review_sentence", "") for s in sents]
                clean_review, _ = build_review_with_offsets(clean_sent_texts, sep=" ")

                base = {"paper_id": pid}
                review_ref = {"review_idx": review_idx, "review_id": rid}

                # Pick first; skip the whole review if no template available
                # so task1_review stays a strict (clean, hallu) pair.
                picked = _pick_review_template(r, rng, class_counter, pos_counter)
                if picked is None:
                    continue
                sent_idx, sent_dict, template_idx, tmpl, _, pos_pct, depth1 = picked

                if include_task1_review:
                    out["task1_review"][split_name].append({
                        "id": "task1_review_level", **base,
                        "text": clean_review, "label": 0,
                        "metadata": dict(review_ref),
                    })
                # Task 2 is pure 9-class — clean version dropped.
                sid = sent_dict.get("sent_id")
                tid = tmpl.get("template_id")
                hallu_sent_texts = clean_sent_texts.copy()
                hallu_sent_texts[sent_idx] = tmpl.get("hallucination_sentence", "")
                hallu_review, hallu_spans = build_review_with_offsets(hallu_sent_texts, sep=" ")
                sent_span = hallu_spans[sent_idx]

                injection_metadata = {
                    **review_ref,
                    "sent_idx": sent_idx,
                    "sent_id": sid,
                    "template_idx": template_idx,
                    "template_id": tid,
                    "label_depth1": depth1,
                    "position_label": pos_pct,
                }

                if include_task1_review:
                    out["task1_review"][split_name].append({
                        "id": "task1_review_level", **base,
                        "text": hallu_review, "label": 1,
                        "metadata": dict(injection_metadata),
                    })
                if include_task2_review:
                    # label is depth1 — skip label_depth1 in metadata to avoid duplication.
                    out["task2_review"][split_name].append({
                        "id": "task2_review_classification", **base,
                        "text": hallu_review, "label": depth1,
                        "metadata": {k: v for k, v in injection_metadata.items() if k != "label_depth1"},
                    })
                if include_task3_review:
                    out["task3_review"][split_name].append({
                        "id": "task3_span_localization", **base,
                        "text": hallu_review,
                        "label": {
                            "start_char_idx": sent_span[0],
                            "end_char_idx": sent_span[1],
                            "gold_span_text": hallu_review[sent_span[0]:sent_span[1]],
                        },
                        "metadata": dict(injection_metadata),
                    })
    return out


# ===================== Phase 3 Group B: sentence-level =====================

def build_group_b(papers_by_split: Dict[str, List[Dict[str, Any]]],
                  rng: random.Random,
                  include_task1_sent: bool,
                  include_task2_sent: bool,
                  ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out = {ds: {"train": [], "dev": [], "test": []} for ds in ("task1_sent", "task2_sent")}

    for split_name, papers in papers_by_split.items():
        # split-scoped depth1 frequency tracker — used for inverse-frequency template selection.
        class_counter: Counter = Counter()

        for p in papers:
            pid = p.get("paper_id")
            base = {"paper_id": pid}

            for review_idx, r in enumerate(p.get("reviews", []) or []):
                rid = r.get("review_id")
                for sent_idx, s in enumerate(r.get("review_sentences", []) or []):
                    sid = s.get("sent_id")
                    sent_ref = {
                        "review_idx": review_idx,
                        "review_id": rid,
                        "sent_idx": sent_idx,
                        "sent_id": sid,
                    }

                    if include_task1_sent:
                        out["task1_sent"][split_name].append({
                            "id": "task1_sent_level", **base,
                            "text": s.get("review_sentence", ""), "label": 0,
                            "metadata": dict(sent_ref),
                        })
                    # Task 2 is pure 9-class — clean version dropped.

                    templates = s.get("hallucination_injection_templates", []) or []
                    if not templates:
                        continue
                    # Inverse-frequency weighting across depth1 — pushes long-tail classes
                    # whenever a sentence offers multiple depth1 options.
                    depth1_per = [
                        get_second_node_label(
                            t.get("hallucination_labels") or t.get("hallucination_label") or "UNKNOWN"
                        )
                        for t in templates
                    ]
                    weights = [1.0 / (1 + class_counter[d]) for d in depth1_per]
                    template_idx = rng.choices(range(len(templates)), weights=weights, k=1)[0]
                    t = templates[template_idx]
                    depth1 = depth1_per[template_idx]
                    class_counter[depth1] += 1
                    hallu_text = t.get("hallucination_sentence", "")

                    injection_metadata = {
                        **sent_ref,
                        "template_idx": template_idx,
                        "template_id": t.get("template_id"),
                    }

                    if include_task1_sent:
                        out["task1_sent"][split_name].append({
                            "id": "task1_sent_level", **base,
                            "text": hallu_text, "label": 1,
                            "metadata": {**injection_metadata, "label_depth1": depth1},
                        })
                    if include_task2_sent:
                        # label is depth1 — skip label_depth1 in metadata to avoid duplication.
                        out["task2_sent"][split_name].append({
                            "id": "task2_sent_classification", **base,
                            "text": hallu_text, "label": depth1,
                            "metadata": dict(injection_metadata),
                        })
    return out


# ===================== Stats =====================

def _count_with_pct(counter: Counter) -> Dict[str, str]:
    """Convert a Counter to `{key: "count (pct%)"}` ordered by descending count.

    `pct_int = round(count / total * 100)` — useful for at-a-glance distribution shape.
    """
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: f"{v} ({int(round(v / total * 100))}%)" for k, v in counter.most_common()}


def _num_summary(values: Iterable[float]) -> Dict[str, float]:
    """Five-number summary + count/sum/mean/stdev. Mirrors the upstream stats format."""
    vs = list(values)
    if not vs:
        return {"count": 0}
    out: Dict[str, float] = {
        "count": len(vs),
        "sum": sum(vs),
        "min": min(vs),
        "max": max(vs),
        "mean": statistics.mean(vs),
        "median": statistics.median(vs),
        "stdev": statistics.stdev(vs) if len(vs) > 1 else 0.0,
    }
    if len(vs) >= 4:
        q = statistics.quantiles(vs, n=4)
        out["p25"], out["p75"] = q[0], q[2]
    else:
        out["p25"], out["p75"] = float(min(vs)), float(max(vs))
    return out


def _stats_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-(dataset, split) stats: row count, label distribution, text length,
    plus dataset-specific extras (depth1 dist, position_label summary, span lengths)."""
    if not rows:
        return {"total_rows": 0}

    entry: Dict[str, Any] = {"total_rows": len(rows)}
    labels = [r["label"] for r in rows]

    if all(isinstance(l, int) for l in labels):
        # binary detection
        entry["label_counts"] = dict(Counter(labels))
        depth1s = [r["metadata"].get("label_depth1") for r in rows
                   if r.get("metadata", {}).get("label_depth1") is not None]
        if depth1s:
            entry["depth1_distribution_hallu"] = _count_with_pct(Counter(depth1s))
    elif all(isinstance(l, str) for l in labels):
        # 9-class classification — label IS depth1
        entry["label_distribution"] = _count_with_pct(Counter(labels))
    elif all(isinstance(l, dict) for l in labels):
        # span localization
        entry["span_char_lengths"] = _num_summary(
            [r["label"]["end_char_idx"] - r["label"]["start_char_idx"] for r in rows]
        )
        depth1s = [r["metadata"].get("label_depth1") for r in rows
                   if r.get("metadata", {}).get("label_depth1") is not None]
        if depth1s:
            entry["depth1_distribution"] = _count_with_pct(Counter(depth1s))

    pos_labels = [r["metadata"].get("position_label") for r in rows
                  if r.get("metadata", {}).get("position_label") is not None]
    if pos_labels:
        entry["position_label_distribution_hallu"] = _num_summary(pos_labels)

    entry["text_chars"] = _num_summary([len(r["text"]) for r in rows])
    return entry


def build_dataset_stats(ds_data: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    """Build a {dataset_name: {split: stats, total: stats}} report.

    `ds_data` is the merged group_a/group_b output: {ds_name: {split_name: [rows]}}.
    """
    report: Dict[str, Any] = {}
    for ds_name, splits in ds_data.items():
        ds_entry: Dict[str, Any] = {}
        for split_name in ("train", "dev", "test"):
            ds_entry[split_name] = _stats_for_rows(splits.get(split_name, []))
        all_rows = [r for sp in ("train", "dev", "test") for r in splits.get(sp, [])]
        ds_entry["total"] = _stats_for_rows(all_rows)
        report[ds_name] = ds_entry
    return report


# ===================== Main =====================

def parse_dataset_arg(s: str) -> List[str]:
    s = s.strip().lower()
    if s == "all":
        return list(DATASETS)
    items = [x.strip() for x in s.split(",") if x.strip()]
    invalid = [x for x in items if x not in DATASETS]
    if invalid:
        raise SystemExit(f"Unknown datasets: {invalid}. Allowed: {DATASETS} or 'all'.")
    return items


def main():
    ap = argparse.ArgumentParser(description="End-to-end hallucination benchmark builder.")
    ap.add_argument("--input", required=True,
                    help="Directory of paper *.json (e.g. dataset/template_injection/HalluPeer_ICLR2019) or a single json file.")
    ap.add_argument("--out", required=True, help="Output root directory.")
    ap.add_argument("--datasets", default="all",
                    help=f"Comma list or 'all'. Allowed: {','.join(DATASETS)}")
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--dev_ratio",   type=float, default=0.10)
    ap.add_argument("--test_ratio",  type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_rows", action=argparse.BooleanOptionalAction, default=False,
                    help="Shuffle rows within each split before writing.")
    args = ap.parse_args()

    rsum = args.train_ratio + args.dev_ratio + args.test_ratio
    if rsum <= 0:
        raise SystemExit("Sum of split ratios must be > 0.")
    ratios = (args.train_ratio / rsum, args.dev_ratio / rsum, args.test_ratio / rsum)

    requested = parse_dataset_arg(args.datasets)
    rng = random.Random(args.seed)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_json_files(args.input)
    papers: List[Dict[str, Any]] = []

    for fp in files:
        data = read_json(fp)
        papers_iter = data if isinstance(data, list) else [data]
        for p in papers_iter:
            pid = p.get("paper_id")
            if not pid:
                continue
            papers.append(p)

    if not papers:
        raise SystemExit(f"No papers found under {args.input}")

    # Phase 1
    profiles = [profile_paper(p) for p in papers]
    all_labels = sorted({lab for prof in profiles for lab in prof["inventory"].keys()})
    if len(all_labels) != 9:
        print(f"[WARN] Found {len(all_labels)} depth1 classes (spec assumes 9).")

    # Phase 2
    assignment = stratified_split(profiles, ratios, rng)
    papers_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for p in papers:
        papers_by_split[assignment[p.get("paper_id")]].append(p)

    meta_path = out_root / "splits.json"
    meta = {
        "input": str(Path(args.input).resolve()),
        "ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
        "seed": args.seed,
        "datasets": requested,
        "depth1_classes": all_labels,
        "split_counts": {k: len(v) for k, v in papers_by_split.items()},
        "assignment": assignment,
        "profiles": profiles,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Phase 2] split_counts={meta['split_counts']}")

    include_task1_review = "task1_review" in requested
    include_task2_review = "task2_review" in requested
    include_task3_review = "task3_review" in requested
    include_task1_sent = "task1_sent" in requested
    include_task2_sent = "task2_sent" in requested

    emitted: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def _assign_task_ids(splits: Dict[str, List[Dict[str, Any]]], ds_name: str) -> None:
        """Re-key the `id` field to global IDs `{ds_name}_{gid}` where gid is a
        per-dataset counter shared across train -> dev -> test (assigned in that order,
        before any row shuffle)."""
        gid = 0
        for sp in ("train", "dev", "test"):
            for row in splits.get(sp, []):
                row["id"] = f"{ds_name}_{gid}"
                gid += 1

    # Phase 3 — Group A
    if include_task1_review or include_task2_review or include_task3_review:
        ga = build_group_a(papers_by_split, rng,
                           include_task1_review, include_task2_review, include_task3_review)
        for ds_name, splits in ga.items():
            if ds_name == "task1_review" and not include_task1_review: continue
            if ds_name == "task2_review" and not include_task2_review: continue
            if ds_name == "task3_review" and not include_task3_review: continue
            _assign_task_ids(splits, ds_name)
            for sp, rows in splits.items():
                if args.shuffle_rows:
                    rng.shuffle(rows)
                out_file = out_root / ds_name / f"{sp}.jsonl"
                write_jsonl(out_file, rows)
                print(f"[{ds_name}][{sp}] {len(rows)} rows -> {out_file}")
            emitted[ds_name] = splits

    # Phase 3 — Group B
    if include_task1_sent or include_task2_sent:
        gb = build_group_b(papers_by_split, rng,
                           include_task1_sent, include_task2_sent)
        for ds_name, splits in gb.items():
            if ds_name == "task1_sent" and not include_task1_sent: continue
            if ds_name == "task2_sent" and not include_task2_sent: continue
            _assign_task_ids(splits, ds_name)
            for sp, rows in splits.items():
                if args.shuffle_rows:
                    rng.shuffle(rows)
                out_file = out_root / ds_name / f"{sp}.jsonl"
                write_jsonl(out_file, rows)
                print(f"[{ds_name}][{sp}] {len(rows)} rows -> {out_file}")
            emitted[ds_name] = splits

    # Stats — per-(dataset, split) summary written alongside splits.json
    stats_report = {
        "split_counts_papers": meta["split_counts"],
        "datasets": build_dataset_stats(emitted),
    }
    stats_path = out_root / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stats] -> {stats_path}")

    print(f"Done. Output root: {out_root}")


if __name__ == "__main__":
    main()
