"""
Task 3 Hallucination Sentence/Span Localization Evaluator
=========================================================

Reads a ground-truth JSONL (``--source``) and a predictions JSONL (``--pred``),
matches records by ``id``, projects gold and predicted spans onto the review
text, and computes span-localization metrics.

Input shapes:
  --source row:
      {"id": "...", "text": "<full review>",
       "label": {"start_char_idx": int, "end_char_idx": int,
                 "gold_span_text": "<exact substring of text>"}}
  --pred row (output of evaluation/task_3/zeroshot.py):
      {"id": "...", "pred": ["<sentence-or-span string>", ...],
       "raw_response": "..."}

Methodology
-----------
For every matched record (skipping records that cannot be aligned):

1. **Gold span resolution.** Take ``(start_char_idx, end_char_idx)`` and check
   that ``text[start:end] == gold_span_text``. If the indices are out of range
   or do not match the gold text, fall back to ``text.find(gold_span_text)``
   (and a whitespace-normalized retry). If the gold span still cannot be
   located, the record is skipped and counted under
   ``n_gold_unresolvable``.
2. **Predicted span resolution.** For each string ``s`` in ``pred``, call
   ``text.find(s)``; on miss, retry against a whitespace-normalized form of
   ``s``. Strings that cannot be located are counted under
   ``n_pred_unlocatable`` and dropped.
3. **Span merging.** Both gold and predicted spans are merged into a list of
   non-overlapping ``(start, end)`` intervals over the character offsets of
   ``text``.
4. **BIO projection.** Each token of ``text`` (whitespace-delimited
   ``\\S+`` matches, with character offsets retained) is assigned a BIO tag:
   ``B`` if the token's offset range is the first inside a span, ``I`` for
   subsequent tokens inside the same span, ``O`` otherwise. A token is "inside"
   if ``[tok_start, tok_end)`` intersects any span. The same projection is run
   for both gold and predicted span lists, producing two parallel BIO
   sequences. For token-level F1 the BIO sequence is collapsed to a binary
   in-span / out-of-span tag (``B``∪``I`` -> 1, ``O`` -> 0), which is the
   standard "tagging F1" reduction.
5. **Character mask.** A parallel binary mask over the characters of ``text``
   is computed by the same span lists for character-level F1.
6. **Metric accumulation (all micro, summed over the corpus):**

   - ``char_f1``      — over character-level masks.
   - ``token_f1``     — over the in-span binary tag sequences derived from
                         BIO.
   - ``span_exact_f1`` — set-equality on merged span boundaries
                         ``(start_char, end_char)``. ``TP = |gold ∩ pred|``,
                         ``FP = |pred \\ gold|``, ``FN = |gold \\ pred|``.
   - ``span_partial_f1`` — predicted spans are greedily matched to gold spans
                            maximizing IoU; a match with any positive overlap
                            counts as a TP for that pair.
   - ``span_iou50_f1`` — same greedy matching but a TP requires
                          ``IoU >= 0.5``.
   - ``mean_iou``     — average IoU over greedily matched ``(pred, gold)``
                         pairs (records with no overlap contribute 0 unless
                         no pred span exists, in which case nothing is added
                         to the average).

Records with pred parsed as ``None`` / ``"-1"`` / empty list are still
included: they produce zero TP / zero FP / and one FN per unmatched gold
span, which is the correct behaviour for a "no detection" output.

Output:
    <pred_dir>/metrics/<pred_stem>.json

Usage:
    python -m evaluation.task_3.evaluate \\
        --source dataset/task_split/HalluPeer_NeurIPS2024/task3_review/test.jsonl \\
        --pred   evaluation/evaluation_results/task3/zeroshot/HalluPeer_NeurIPS2024/task3_review/test_result_model-<...>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── I/O ──────────────────────────────────────────────────────────────────────

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


def _pr_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r)
    return p, r, f1


# ─── Tokenization / span helpers ──────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\S+")


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Whitespace tokens with character offsets ``(token, start, end)``."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping / touching character spans; returned sorted."""
    if not spans:
        return []
    s_sorted = sorted((a, b) for a, b in spans if b > a)
    if not s_sorted:
        return []
    out: List[Tuple[int, int]] = [s_sorted[0]]
    for s, e in s_sorted[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def spans_to_char_mask(spans: List[Tuple[int, int]], n: int) -> List[int]:
    mask = [0] * n
    for s, e in spans:
        s = max(0, s)
        e = min(n, e)
        for i in range(s, e):
            mask[i] = 1
    return mask


def spans_to_bio(
    tokens: List[Tuple[str, int, int]],
    spans: List[Tuple[int, int]],
) -> List[str]:
    """Return BIO tag per token. A token is in-span if its char range overlaps any span."""
    if not spans:
        return ["O"] * len(tokens)
    tags: List[str] = []
    prev_in = False
    prev_span_idx = -1
    for _, ts, te in tokens:
        cur_span_idx = -1
        for j, (ss, se) in enumerate(spans):
            if not (te <= ss or ts >= se):
                cur_span_idx = j
                break
        if cur_span_idx < 0:
            tags.append("O")
            prev_in = False
            prev_span_idx = -1
        else:
            if prev_in and cur_span_idx == prev_span_idx:
                tags.append("I")
            else:
                tags.append("B")
            prev_in = True
            prev_span_idx = cur_span_idx
    return tags


def bio_to_binary(bio: List[str]) -> List[int]:
    return [0 if t == "O" else 1 for t in bio]


def locate_substring(text: str, needle: str) -> Optional[Tuple[int, int]]:
    """Find ``needle`` inside ``text``; fall back to whitespace-normalized retry.

    Returns the char span (start, end) into the *original* text, or None.
    Normalized retry maps any whitespace run in both the haystack and needle
    to a single space; if found there, the original char span is recovered
    by re-scanning ``text`` for the same content with flexible whitespace.
    """
    if not needle:
        return None
    pos = text.find(needle)
    if pos >= 0:
        return pos, pos + len(needle)
    needle_norm = re.sub(r"\s+", " ", needle).strip()
    if not needle_norm:
        return None
    # Build a regex matching the normalized needle with any whitespace runs.
    parts = needle_norm.split(" ")
    pattern = r"\s+".join(re.escape(p) for p in parts if p)
    if not pattern:
        return None
    m = re.search(pattern, text)
    if m:
        return m.start(), m.end()
    return None


def verify_gold_span(
    text: str, start: Any, end: Any, gold_text: str,
) -> bool:
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if not (0 <= start < end <= len(text)):
        return False
    return text[start:end] == gold_text


# ─── Core evaluation ──────────────────────────────────────────────────────────

def evaluate(source_path: str, pred_path: str) -> Dict[str, Any]:
    source_records = read_jsonl(source_path)
    pred_records = read_jsonl(pred_path)

    src_map: Dict[Any, Dict[str, Any]] = {r["id"]: r for r in source_records}
    pred_map: Dict[Any, Dict[str, Any]] = {r["id"]: r for r in pred_records}

    matched_ids = [r["id"] for r in pred_records if r["id"] in src_map]
    missing_in_source = [r["id"] for r in pred_records if r["id"] not in src_map]
    missing_in_pred = [rid for rid in src_map if rid not in pred_map]

    # Micro accumulators
    char_tp = char_fp = char_fn = 0
    tok_tp = tok_fp = tok_fn = 0
    span_exact_tp = span_exact_fp = span_exact_fn = 0
    span_partial_tp = span_partial_fp = span_partial_fn = 0
    span_iou50_tp = span_iou50_fp = span_iou50_fn = 0
    iou_sum = 0.0
    iou_count = 0

    n_gold_idx_valid = 0
    n_gold_idx_recovered = 0
    n_gold_unresolvable = 0
    n_pred_none = 0
    n_pred_empty_list = 0
    n_pred_unlocatable = 0
    n_evaluated = 0

    for rid in matched_ids:
        src = src_map[rid]
        pred_rec = pred_map[rid]
        text = str(src.get("text") or "")
        label = src.get("label") or {}
        gs_idx = label.get("start_char_idx")
        ge_idx = label.get("end_char_idx")
        gold_text = str(label.get("gold_span_text") or "")

        # Resolve gold span
        if verify_gold_span(text, gs_idx, ge_idx, gold_text):
            gold_span: Optional[Tuple[int, int]] = (int(gs_idx), int(ge_idx))
            n_gold_idx_valid += 1
        else:
            recov = locate_substring(text, gold_text) if gold_text else None
            if recov is not None:
                gold_span = recov
                n_gold_idx_recovered += 1
            else:
                gold_span = None
                n_gold_unresolvable += 1
                continue  # cannot score this record

        gold_spans = merge_spans([gold_span]) if gold_span else []

        # Resolve pred spans
        pred_val = pred_rec.get("pred")
        if pred_val is None or pred_val == "-1" or pred_val == -1:
            n_pred_none += 1
            preds: List[str] = []
        elif isinstance(pred_val, str):
            preds = [pred_val] if pred_val.strip() else []
        elif isinstance(pred_val, list):
            preds = [s for s in pred_val if isinstance(s, str) and s.strip()]
        else:
            preds = []

        if not preds:
            n_pred_empty_list += 1

        pred_spans_raw: List[Tuple[int, int]] = []
        for s in preds:
            loc = locate_substring(text, s)
            if loc is None:
                n_pred_unlocatable += 1
                continue
            pred_spans_raw.append(loc)
        pred_spans = merge_spans(pred_spans_raw)

        n_evaluated += 1

        # Char-level F1
        n_chars = len(text)
        gold_mask = spans_to_char_mask(gold_spans, n_chars)
        pred_mask = spans_to_char_mask(pred_spans, n_chars)
        for gm, pm in zip(gold_mask, pred_mask):
            if gm and pm:
                char_tp += 1
            elif pm:
                char_fp += 1
            elif gm:
                char_fn += 1

        # Token-level F1 (collapsed from BIO)
        tokens = tokenize_with_offsets(text)
        gold_bio = spans_to_bio(tokens, gold_spans)
        pred_bio = spans_to_bio(tokens, pred_spans)
        gold_bin = bio_to_binary(gold_bio)
        pred_bin = bio_to_binary(pred_bio)
        for gt, pt in zip(gold_bin, pred_bin):
            if gt and pt:
                tok_tp += 1
            elif pt:
                tok_fp += 1
            elif gt:
                tok_fn += 1

        # Span exact F1 (set equality)
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)
        tp_e = len(gold_set & pred_set)
        span_exact_tp += tp_e
        span_exact_fp += len(pred_set) - tp_e
        span_exact_fn += len(gold_set) - tp_e

        # Span partial / IoU>=0.5 (greedy matching, max IoU)
        matched_gold = [False] * len(gold_spans)
        tp_p = tp_iou50 = 0
        for ps, pe in pred_spans:
            best_iou = 0.0
            best_j = -1
            for j, (gs2, ge2) in enumerate(gold_spans):
                if matched_gold[j]:
                    continue
                inter = max(0, min(pe, ge2) - max(ps, gs2))
                if inter == 0:
                    continue
                union = (pe - ps) + (ge2 - gs2) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                matched_gold[best_j] = True
                tp_p += 1
                iou_sum += best_iou
                iou_count += 1
                if best_iou >= 0.5:
                    tp_iou50 += 1
        span_partial_tp += tp_p
        span_partial_fp += len(pred_spans) - tp_p
        span_partial_fn += len(gold_spans) - tp_p
        span_iou50_tp += tp_iou50
        span_iou50_fp += len(pred_spans) - tp_iou50
        span_iou50_fn += len(gold_spans) - tp_iou50

    char_p, char_r, char_f1 = _pr_f1(char_tp, char_fp, char_fn)
    tok_p, tok_r, tok_f1 = _pr_f1(tok_tp, tok_fp, tok_fn)
    ex_p, ex_r, ex_f1 = _pr_f1(span_exact_tp, span_exact_fp, span_exact_fn)
    pa_p, pa_r, pa_f1 = _pr_f1(span_partial_tp, span_partial_fp, span_partial_fn)
    io_p, io_r, io_f1 = _pr_f1(span_iou50_tp, span_iou50_fp, span_iou50_fn)
    mean_iou = _safe_div(iou_sum, iou_count)

    result: Dict[str, Any] = {
        "meta": {
            "source": str(Path(source_path).resolve()),
            "pred": str(Path(pred_path).resolve()),
            "n_source": len(source_records),
            "n_pred": len(pred_records),
            "n_matched": len(matched_ids),
            "n_evaluated": n_evaluated,
            "n_missing_in_source": len(missing_in_source),
            "n_missing_in_pred": len(missing_in_pred),
            "n_gold_idx_valid": n_gold_idx_valid,
            "n_gold_idx_recovered": n_gold_idx_recovered,
            "n_gold_unresolvable": n_gold_unresolvable,
            "n_pred_none": n_pred_none,
            "n_pred_empty_list": n_pred_empty_list,
            "n_pred_unlocatable": n_pred_unlocatable,
        },
        "char_level": {
            "TP": char_tp, "FP": char_fp, "FN": char_fn,
            "precision": round(char_p, 6),
            "recall": round(char_r, 6),
            "f1": round(char_f1, 6),
        },
        "token_level": {
            "TP": tok_tp, "FP": tok_fp, "FN": tok_fn,
            "precision": round(tok_p, 6),
            "recall": round(tok_r, 6),
            "f1": round(tok_f1, 6),
        },
        "span_exact": {
            "TP": span_exact_tp, "FP": span_exact_fp, "FN": span_exact_fn,
            "precision": round(ex_p, 6),
            "recall": round(ex_r, 6),
            "f1": round(ex_f1, 6),
        },
        "span_partial": {
            "TP": span_partial_tp, "FP": span_partial_fp, "FN": span_partial_fn,
            "precision": round(pa_p, 6),
            "recall": round(pa_r, 6),
            "f1": round(pa_f1, 6),
        },
        "span_iou50": {
            "TP": span_iou50_tp, "FP": span_iou50_fp, "FN": span_iou50_fn,
            "precision": round(io_p, 6),
            "recall": round(io_r, 6),
            "f1": round(io_f1, 6),
        },
        "mean_iou_over_matched_pairs": round(mean_iou, 6),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Task 3 hallucination span/sentence localization predictions."
    )
    ap.add_argument("--source", required=True, help="Ground-truth JSONL (task3_review/test.jsonl)")
    ap.add_argument("--pred", required=True, help="Predictions JSONL (from zeroshot.py)")
    args = ap.parse_args()

    result = evaluate(args.source, args.pred)

    pred_path = Path(args.pred)
    out_dir = pred_path.parent / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (pred_path.stem + ".json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Metrics written to {out_path}")
    cl = result["char_level"]
    tl = result["token_level"]
    ex = result["span_exact"]
    pa = result["span_partial"]
    io = result["span_iou50"]
    print(
        f"  char_f1={cl['f1']:.4f}  token_f1={tl['f1']:.4f}"
        f"  span_exact_f1={ex['f1']:.4f}  span_partial_f1={pa['f1']:.4f}"
        f"  span_iou50_f1={io['f1']:.4f}  mean_iou={result['mean_iou_over_matched_pairs']:.4f}"
    )


if __name__ == "__main__":
    main()
