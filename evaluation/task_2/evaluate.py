"""
Task 2 Multi-class Classification Evaluator
=====================================================

Reads a ground-truth JSONL (``--source``) and a predictions JSONL (``--pred``),
matches records by ``id``, and computes:

Overall metrics:
  accuracy, macro-F1, micro-F1, weighted-F1, Cohen's kappa (multiclass),
  Matthews Correlation Coefficient (multiclass MCC).

Per-label metrics (one-vs-rest for each class):
  TP, FP, FN, TN, precision, recall, F1, support.

Samples with pred==None, pred=="-1", or pred=="" (unparseable model output) are excluded
from all metric computations but their count is reported.

Output:
    <pred_dir>/metrics/<pred_stem>.json

Usage:
    python -m evaluation.task_2.evaluate \\
        --source dataset/task_split/HalluPeer_NeurIPS2024/{task2_review|task2_sent}/test.jsonl \\
        --pred   evaluation/evaluation_results/task2/zeroshot/HalluPeer_NeurIPS2024/{task2_review|task2_sent}/[test_result_{exp_name}].jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den != 0 else default


def _per_label_metrics(
    labels: List[str],
    pairs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Compute one-vs-rest binary metrics for each label."""
    per_label: Dict[str, Any] = {}
    for cls in sorted(labels):
        tp = fp = fn = tn = 0
        for gt, pred in pairs:
            if gt == cls and pred == cls:
                tp += 1
            elif gt != cls and pred == cls:
                fp += 1
            elif gt == cls and pred != cls:
                fn += 1
            else:
                tn += 1
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = tp + fn
        per_label[cls] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return per_label


def _confusion_matrix(
    labels: List[str],
    pairs: List[Tuple[str, str]],
) -> Dict[str, Dict[str, int]]:
    """Return confusion matrix as {true_label: {pred_label: count}}."""
    label_set = sorted(labels)
    cm: Dict[str, Dict[str, int]] = {c: {d: 0 for d in label_set} for c in label_set}
    for gt, pred in pairs:
        if gt in cm and pred in cm:
            cm[gt][pred] += 1
        elif gt in cm:
            # pred is an unseen class — count under a catch-all key
            cm[gt].setdefault(pred, 0)
            cm[gt][pred] += 1
    return cm


def _multiclass_mcc(
    labels: List[str],
    pairs: List[Tuple[str, str]],
) -> float:
    """
    Multiclass MCC (Gorodkin 2004 / Jurman et al. 2012):
        MCC = (c*N - sum_k s_k*t_k)
              / sqrt((N^2 - sum_k t_k^2) * (N^2 - sum_k s_k^2))
    where:
        c   = correct predictions (trace of confusion matrix)
        N   = total samples
        s_k = actual count of class k (row sum)
        t_k = predicted count of class k (column sum)
    """
    n = len(pairs)
    if n == 0:
        return 0.0

    gt_counts: Dict[str, int] = defaultdict(int)   # s_k
    pred_counts: Dict[str, int] = defaultdict(int)  # t_k
    correct = 0
    for gt, pred in pairs:
        gt_counts[gt] += 1
        pred_counts[pred] += 1
        if gt == pred:
            correct += 1

    all_classes = set(gt_counts) | set(pred_counts)
    sum_sk_tk = sum(gt_counts[c] * pred_counts[c] for c in all_classes)
    sum_tk_sq = sum(v ** 2 for v in pred_counts.values())
    sum_sk_sq = sum(v ** 2 for v in gt_counts.values())

    numerator = correct * n - sum_sk_tk
    denom_sq = (n ** 2 - sum_tk_sq) * (n ** 2 - sum_sk_sq)
    return _safe_div(numerator, math.sqrt(denom_sq), default=0.0)


def _cohen_kappa(pairs: List[Tuple[str, str]]) -> float:
    """Multiclass Cohen's kappa."""
    n = len(pairs)
    if n == 0:
        return 0.0
    gt_counts: Dict[str, int] = defaultdict(int)
    pred_counts: Dict[str, int] = defaultdict(int)
    correct = 0
    for gt, pred in pairs:
        gt_counts[gt] += 1
        pred_counts[pred] += 1
        if gt == pred:
            correct += 1
    p_o = correct / n
    p_e = sum(gt_counts[c] * pred_counts[c] for c in gt_counts) / (n ** 2)
    return _safe_div(p_o - p_e, 1.0 - p_e, default=0.0)


def evaluate(
    source_path: str,
    pred_path: str,
) -> Dict[str, Any]:
    source_records = read_jsonl(source_path)
    pred_records = read_jsonl(pred_path)

    src_map: Dict[Any, Dict[str, Any]] = {r["id"]: r for r in source_records}
    pred_map: Dict[Any, Dict[str, Any]] = {r["id"]: r for r in pred_records}

    matched_ids = [r["id"] for r in pred_records if r["id"] in src_map]
    missing_in_source = [r["id"] for r in pred_records if r["id"] not in src_map]
    missing_in_pred = [rid for rid in src_map if rid not in pred_map]

    pairs: List[Tuple[str, str]] = []
    skipped_unparseable = 0
    for rid in matched_ids:
        src = src_map[rid]
        pred_val = pred_map[rid]["pred"]
        if pred_val is None or pred_val == "-1" or pred_val == -1 or pred_val == "":
            skipped_unparseable += 1
            continue
        gt = str(src["label"])
        pairs.append((gt, str(pred_val)))

    # Collect all true classes (from source) for consistent per-label reporting
    all_labels: List[str] = sorted({str(r["label"]) for r in source_records})

    # ── Overall metrics ──
    n = len(pairs)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    accuracy = _safe_div(correct, n)

    per_label = _per_label_metrics(all_labels, pairs)

    # Macro-F1: unweighted mean of per-label F1
    macro_f1 = _safe_div(
        sum(v["f1"] for v in per_label.values()), len(per_label)
    ) if per_label else 0.0

    # Micro-F1: equals accuracy for multiclass (all TP/FP/FN counts sum symmetrically)
    micro_tp = sum(v["TP"] for v in per_label.values())
    micro_fp = sum(v["FP"] for v in per_label.values())
    micro_fn = sum(v["FN"] for v in per_label.values())
    micro_precision = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )

    # Weighted-F1: F1 weighted by support
    total_support = sum(v["support"] for v in per_label.values())
    weighted_f1 = _safe_div(
        sum(v["f1"] * v["support"] for v in per_label.values()), total_support
    ) if total_support > 0 else 0.0

    # Macro-precision / macro-recall (unweighted)
    macro_precision = _safe_div(
        sum(v["precision"] for v in per_label.values()), len(per_label)
    ) if per_label else 0.0
    macro_recall = _safe_div(
        sum(v["recall"] for v in per_label.values()), len(per_label)
    ) if per_label else 0.0

    kappa = _cohen_kappa(pairs)
    mcc = _multiclass_mcc(all_labels, pairs)

    result = {
        "meta": {
            "source": str(Path(source_path).resolve()),
            "pred": str(Path(pred_path).resolve()),
            "n_source": len(source_records),
            "n_pred": len(pred_records),
            "n_matched": len(matched_ids),
            "n_evaluated": n,
            "n_skipped_unparseable": skipped_unparseable,
            "n_missing_in_source": len(missing_in_source),
            "n_missing_in_pred": len(missing_in_pred),
            "label_set": all_labels,
        },
        "overall": {
            "accuracy": round(accuracy, 6),
            "micro_f1": round(micro_f1, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "cohen_kappa": round(kappa, 6),
            "mcc": round(mcc, 6),
        },
        "per_label": per_label,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Task 2 multi-class hallucination classification predictions."
    )
    ap.add_argument("--source", required=True, help="Ground-truth JSONL path")
    ap.add_argument("--pred", required=True, help="Predictions JSONL path")
    args = ap.parse_args()

    result = evaluate(args.source, args.pred)

    pred_path = Path(args.pred)
    out_dir = pred_path.parent / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (pred_path.stem + ".json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Metrics written to {out_path}")
    ov = result["overall"]
    print(
        f"  accuracy={ov['accuracy']:.4f}  macro_f1={ov['macro_f1']:.4f}"
        f"  micro_f1={ov['micro_f1']:.4f}  weighted_f1={ov['weighted_f1']:.4f}"
        f"  mcc={ov['mcc']:.4f}  kappa={ov['cohen_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
