"""
Task 1 Binary Classification Evaluator
=====================================================

Reads a ground-truth JSONL (``--source``) and a predictions JSONL (``--pred``),
matches records by ``id``, and computes:

Overall binary metrics (positive class = label 1, i.e. hallucinated):
  accuracy, precision, recall, F1, MCC, balanced accuracy, Cohen's kappa,
  specificity (TNR), threat score (CSI), positive/negative predictive value.

Per-class metrics (one-vs-rest, keyed by ``metadata.label_depth1``):
  TP, FP, FN, TN, accuracy, precision, recall, F1.
  "Positive" for class C = label=1 AND label_depth1=C.
  "Negative" = all other samples (label=0 or label=1 with a different depth1 class).

Samples with pred==None, pred==-1, pred=="-1", or pred=="" (unparseable model output)
are excluded from all metric computations but their count is reported.

Output:
    <pred_dir>/metrics/<pred_stem>.json

Usage:
    python -m evaluation.task_1.evaluate \\
        --source dataset/task_split/HalluPeer_NeurIPS2024/{task1_review|task1_sent}/test.jsonl \\
        --pred   evaluation/evaluation_results/task1/zeroshot/HalluPeer_NeurIPS2024/{task1_review|task1_sent}/[test_result_{exp_name}].jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    total = tp + fp + fn + tn
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)   # sensitivity / TPR
    specificity = _safe_div(tn, tn + fp)  # TNR
    f1 = _safe_div(2 * precision * recall, precision + recall)
    ppv = precision
    npv = _safe_div(tn, tn + fn)
    threat_score = _safe_div(tp, tp + fp + fn)  # CSI / Jaccard for binary

    # MCC
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = _safe_div(tp * tn - fp * fn, math.sqrt(denom_sq), default=0.0)

    # Balanced accuracy = (TPR + TNR) / 2
    balanced_acc = (recall + specificity) / 2.0

    # Cohen's kappa
    p_o = accuracy
    p_pos = _safe_div((tp + fp) * (tp + fn), total ** 2)
    p_neg = _safe_div((tn + fn) * (tn + fp), total ** 2)
    p_e = p_pos + p_neg
    kappa = _safe_div(p_o - p_e, 1.0 - p_e, default=0.0) if total > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(balanced_acc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "ppv": round(ppv, 6),
        "npv": round(npv, 6),
        "mcc": round(mcc, 6),
        "cohen_kappa": round(kappa, 6),
        "threat_score_csi": round(threat_score, 6),
    }


def evaluate(
    source_path: str,
    pred_path: str,
) -> Dict[str, Any]:
    source_records = read_jsonl(source_path)
    pred_records = read_jsonl(pred_path)

    # Build id→record maps
    src_map: Dict[Any, Dict[str, Any]] = {r["id"]: r for r in source_records}
    pred_map: Dict[Any, Dict[str, Any]] = {r["id"]: r for r in pred_records}

    # Align by id (inner join on ids present in pred file, keeping pred order)
    matched_ids = [r["id"] for r in pred_records if r["id"] in src_map]
    missing_in_source = [r["id"] for r in pred_records if r["id"] not in src_map]
    missing_in_pred = [rid for rid in src_map if rid not in pred_map]

    pairs: List[Tuple[int, int, Optional[str]]] = []  # (gt, pred, depth1)
    skipped_unparseable = 0
    for rid in matched_ids:
        src = src_map[rid]
        pred_val = pred_map[rid]["pred"]
        if pred_val is None or pred_val == -1 or pred_val == "-1" or pred_val == "":
            skipped_unparseable += 1
            continue
        gt = int(src["label"])
        depth1: Optional[str] = src.get("metadata", {}).get("label_depth1")
        pairs.append((gt, int(pred_val), depth1))

    # ── Overall binary metrics ──
    tp = fp = fn = tn = 0
    for gt, pred_val, _ in pairs:
        if gt == 1 and pred_val == 1:
            tp += 1
        elif gt == 0 and pred_val == 1:
            fp += 1
        elif gt == 1 and pred_val == 0:
            fn += 1
        else:
            tn += 1

    overall = _binary_metrics(tp, fp, fn, tn)

    # ── Per-class one-vs-rest ──
    depth1_classes: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for gt, pred_val, depth1 in pairs:
        if depth1 is not None:
            depth1_classes[depth1].append((gt, pred_val))

    per_class: Dict[str, Any] = {}
    for cls in sorted(depth1_classes.keys()):
        cls_tp = cls_fp = cls_fn = cls_tn = 0
        for gt, pred_val, depth1 in pairs:
            is_positive = (gt == 1 and depth1 == cls)
            if is_positive and pred_val == 1:
                cls_tp += 1
            elif not is_positive and pred_val == 1:
                cls_fp += 1
            elif is_positive and pred_val == 0:
                cls_fn += 1
            else:
                cls_tn += 1
        per_class[cls] = _binary_metrics(cls_tp, cls_fp, cls_fn, cls_tn)
        per_class[cls]["n_positive_gt"] = sum(
            1 for gt, _, d in pairs if gt == 1 and d == cls
        )

    result = {
        "meta": {
            "source": str(Path(source_path).resolve()),
            "pred": str(Path(pred_path).resolve()),
            "n_source": len(source_records),
            "n_pred": len(pred_records),
            "n_matched": len(matched_ids),
            "n_evaluated": len(pairs),
            "n_skipped_unparseable": skipped_unparseable,
            "n_missing_in_source": len(missing_in_source),
            "n_missing_in_pred": len(missing_in_pred),
        },
        "overall": overall,
        "per_depth1_class": per_class,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Task 1 binary hallucination detection predictions."
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
        f"  accuracy={ov['accuracy']:.4f}  precision={ov['precision']:.4f}"
        f"  recall={ov['recall']:.4f}  f1={ov['f1']:.4f}"
        f"  mcc={ov['mcc']:.4f}  kappa={ov['cohen_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
