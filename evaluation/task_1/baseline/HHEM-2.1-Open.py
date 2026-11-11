"""
Task 1 baseline — HHEM-2.1-Open
================================

Vectara hallucination_evaluation_model. For each input record (review- or
sentence-level), evidence is built from {paper_id}.json (no_retrieval or BM25,
identical to zero_shot.py), the record text is split into sentences, and each
sentence is scored against every evidence chunk via ``model.predict([(premise,
hypothesis), ...])``. Per-sentence consistency = max over chunks; review-level
truth = aggregate (mean/min/median) of sentence consistencies; pred = 0 if
truth >= threshold else 1.

Threshold mode (``--threshold_mode``):
    * ``fixed``       use ``scoring.fixed_threshold`` (or ``--fixed_threshold``).
    * ``precomputed`` look up ``scoring.precompute_threshold_split{T|F}_agg{...}_level{review|sent}``.
                      If absent: score the matching ``train.jsonl`` (auto-derived
                      from --input by replacing the stem with "train"), sweep
                      thresholds on (truth_score, label) pairs to pick max F1-macro,
                      write the result back into the YAML, and EXIT. The user must
                      then re-run to perform the actual test prediction.

Config:
    Defaults in ``./config_baseline.yaml`` under section ``hhem``.

Usage (no_retrieval, precomputed threshold flow — run twice):
    python -m evaluation.task_1.baseline.HHEM-2.1-Open \\
        --input dataset/task_split/HalluPeer_ICLR2019/task1_review/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_ICLR2019 \\
        --out evaluation/evaluation_results/task1/hhem \\
        --evidence no_retrieval \\
        --threshold_mode precomputed

Output:
    ``<out>/<sub-path>/<stem>_result_model-<model>_evidence-<type>_thr-<mode>-<value>.jsonl``
    one JSON object per line: {"id", "pred", "truth_score", "halluc_score"}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_baseline.yaml"
EVAL_DATASET_MARKER = "task_split"
CONFIG_SECTION = "hhem"


# =============================================================================
# Config / IO helpers (mirrors zero_shot.py)
# =============================================================================

def _resolve_config_path(path: Optional[str]) -> Path:
    return Path(path) if path else DEFAULT_CONFIG


def load_full_config(path: Optional[str]) -> Dict[str, Any]:
    p = _resolve_config_path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    raw = load_full_config(path)
    section = raw.get(CONFIG_SECTION)
    return section if isinstance(section, dict) else {}


def cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def update_config_scoring(config_path: Optional[str], key: str, value: float,
                          stats: Optional[Dict[str, Any]] = None) -> Path:
    p = _resolve_config_path(config_path)
    raw: Dict[str, Any] = {}
    if p.exists():
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded
    section = raw.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        section = {}
        raw[CONFIG_SECTION] = section
    scoring = section.get("scoring")
    if not isinstance(scoring, dict):
        scoring = {}
        section["scoring"] = scoring
    scoring[key] = float(value)
    if stats is not None:
        scoring[f"{key}__stats"] = stats
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def build_precompute_key(sentence_level: bool, aggregate_how: str, data_level: str) -> str:
    sl = "T" if sentence_level else "F"
    agg = (aggregate_how or "mean") if sentence_level else "none"
    lvl = data_level if data_level in {"review", "sent"} else "review"
    return f"precompute_threshold_split{sl}_agg{agg}_level{lvl}"


def derive_data_level(input_path: Path, override: Optional[str] = None) -> str:
    if override in {"review", "sent"}:
        return override
    parts = {p.lower() for p in input_path.parts}
    if "task1_review" in parts:
        return "review"
    if "task1_sent" in parts:
        return "sent"
    return "review"


def derive_train_path(test_input: Path) -> Path:
    """Replace the stem with 'train' in the same directory.

    Falls back to <parent>/train.jsonl if the stem-swap target does not exist.
    """
    candidate = test_input.with_name(f"train{test_input.suffix or '.jsonl'}")
    if candidate.exists():
        return candidate
    fallback = test_input.parent / "train.jsonl"
    return fallback


def find_best_threshold(scores: Sequence[float], labels: Sequence[int],
                        step: float = 0.01) -> Tuple[float, Dict[str, Any]]:
    """Sweep thresholds on truth_score; pick max F1-macro.

    Convention: ``pred = 0 if truth_score >= threshold else 1``
    where label 1 = hallucinated, label 0 = truthful.
    """
    if not scores or not labels or len(scores) != len(labels):
        return 0.5, {"reason": "empty/mismatched", "n": len(scores)}
    step = max(1e-4, float(step))
    candidates: List[float] = []
    x = 0.0
    while x <= 1.0 + 1e-9:
        candidates.append(round(x, 6))
        x += step
    best_thr = 0.5
    best_f1m = -1.0
    best_stats: Dict[str, Any] = {}
    for thr in candidates:
        tp = fp = tn = fn = 0
        for s, l in zip(scores, labels):
            pred = 0 if s >= thr else 1
            if pred == 1 and l == 1:
                tp += 1
            elif pred == 1 and l == 0:
                fp += 1
            elif pred == 0 and l == 0:
                tn += 1
            else:
                fn += 1
        p1 = tp / (tp + fp) if (tp + fp) else 0.0
        r1 = tp / (tp + fn) if (tp + fn) else 0.0
        f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) else 0.0
        p0 = tn / (tn + fn) if (tn + fn) else 0.0
        r0 = tn / (tn + fp) if (tn + fp) else 0.0
        f1_0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) else 0.0
        f1m = (f1_1 + f1_0) / 2.0
        if f1m > best_f1m:
            best_f1m = f1m
            best_thr = thr
            n = len(labels)
            best_stats = {
                "threshold": float(thr),
                "f1_macro": float(f1m),
                "f1_hallu": float(f1_1),
                "f1_truthful": float(f1_0),
                "accuracy": float((tp + tn) / n) if n else 0.0,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "n_total": n,
                "n_hallu": int(sum(1 for l in labels if l == 1)),
                "n_truthful": int(sum(1 for l in labels if l == 0)),
            }
    return float(best_thr), best_stats


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def sanitize_model_name(model: str) -> str:
    return model.split("/")[-1]


def derive_output_path(input_path: Path, out_root: Path, suffix: str) -> Path:
    parts = input_path.parts
    if EVAL_DATASET_MARKER in parts:
        idx = parts.index(EVAL_DATASET_MARKER)
        sub = Path(*parts[idx + 1 :])
    else:
        sub = Path(input_path.name)
    stem = sub.stem
    ext = sub.suffix or ".jsonl"
    new_name = f"{stem}_{suffix}{ext}"
    return out_root / sub.parent / new_name


def build_output_suffix(model: str, evidence: str, *, sentence_level: bool,
                        aggregate_how: str, threshold_mode: str,
                        threshold: float) -> str:
    sl_tag = "split-T" if sentence_level else "split-F"
    agg_tag = f"agg-{aggregate_how}" if sentence_level else "agg-none"
    thr_tag = f"thr-{threshold_mode}-{float(threshold):.3f}"
    return "_".join([
        "result",
        f"model-{sanitize_model_name(model)}",
        f"evidence-{evidence}",
        sl_tag,
        agg_tag,
        thr_tag,
    ])


# =============================================================================
# Evidence extraction (mirrors zero_shot.py)
# =============================================================================

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


def join_paper_content(paper_content: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in paper_content.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}\n{v.strip()}")
        elif isinstance(v, (list, tuple)):
            joined = "\n".join(str(x) for x in v if x)
            if joined.strip():
                parts.append(f"{k}\n{joined.strip()}")
        elif isinstance(v, dict):
            sub = join_paper_content(v)
            if sub:
                parts.append(f"{k}\n{sub}")
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]
    n = len(text)
    step = max(1, chunk_chars - max(0, overlap))
    chunks: List[str] = []
    i = 0
    while i < n:
        chunks.append(text[i : i + chunk_chars])
        if i + chunk_chars >= n:
            break
        i += step
    return chunks


def bm25_evidence(paper_content: Dict[str, Any], query: str, top_k: int, chunk_chars: int, overlap: int) -> str:
    sections: List[tuple] = []
    for k, v in paper_content.items():
        if isinstance(v, str) and v.strip():
            sections.append((k, v.strip()))
        elif isinstance(v, dict):
            sub = join_paper_content(v)
            if sub:
                sections.append((k, sub))
    chunks: List[str] = []
    for k, v in sections:
        for ch in chunk_text(v, chunk_chars, overlap):
            chunks.append(f"{k}\n{ch}")
    if not chunks:
        return ""
    tokenized = [_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    q_tokens = _tokenize(query)
    if not q_tokens:
        return "\n\n".join(chunks[:top_k])
    scores = bm25.get_scores(q_tokens)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    order.sort()
    return "\n\n".join(chunks[i] for i in order)


def load_paper(source_root: Path, paper_id: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if paper_id in cache:
        return cache[paper_id]
    p = source_root / f"{paper_id}.json"
    pc: Dict[str, Any] = {}
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("paper_content"), dict):
                pc = d["paper_content"]
        except Exception:
            pc = {}
    cache[paper_id] = pc
    return pc


# =============================================================================
# Sentence split / aggregate (mirrors baseline/previous/evaluation/base.py)
# =============================================================================

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def split_into_sentences(text: str, *, min_len: int = 10, max_len: int = 400) -> List[str]:
    if not text:
        return []
    raw = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    out: List[str] = []
    for s in raw:
        if len(s) < min_len:
            continue
        if len(s) <= max_len:
            out.append(s)
            continue
        subs = re.split(r"[;；,，]\s+", s)
        for sub in subs:
            sub = sub.strip()
            if min_len <= len(sub) <= max_len:
                out.append(sub)
            elif len(sub) > max_len:
                for i in range(0, len(sub), max_len):
                    chunk = sub[i : i + max_len].strip()
                    if len(chunk) >= min_len:
                        out.append(chunk)
    return out


def chunk_text_by_tokens(text: str, tokenizer: Any, *, max_tokens: int, stride: int = 200) -> List[str]:
    if max_tokens <= 0:
        return [text]
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if not ids:
        return [""]
    chunks: List[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunk_ids = ids[start:end]
        c = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if c:
            chunks.append(c)
        if end == len(ids):
            break
        start = max(0, end - stride)
        if start == end:
            start = end
    seen: set = set()
    uniq: List[str] = []
    for c in chunks:
        key = c[:200]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def aggregate(values: Sequence[float], how: str = "mean") -> float:
    if not values:
        return 0.0
    if how == "mean":
        return float(sum(values) / len(values))
    if how == "min":
        return float(min(values))
    if how == "median":
        st = sorted(values)
        mid = len(st) // 2
        return float(st[mid] if len(st) % 2 == 1 else (st[mid - 1] + st[mid]) / 2)
    raise ValueError(f"Unknown aggregate: {how}")


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# =============================================================================
# HHEM scorer
# =============================================================================

class HHEMScorer:
    """Vectara hallucination_evaluation_model wrapper.

    ``model.predict(pairs)`` returns a [0,1] consistency score per pair (high =
    hypothesis consistent with premise = NOT hallucinated).
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str],
        chunk_paper: bool,
        tokenizer_id: str,
        premise_max_tokens: int,
        premise_stride: int,
        batch_size: int,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # transformers ≥4.38 renamed all_tied_weights_keys → _tied_weights_keys;
        # HHEM's trust_remote_code class still uses the old name.
        # import transformers as _tf
        # if not hasattr(_tf.PreTrainedModel, "all_tied_weights_keys"):
        #     _tf.PreTrainedModel.all_tied_weights_keys = property(
        #         lambda self: getattr(self, "_tied_weights_keys", None) or []
        #     )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        )
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
        self.model.eval()

        if not hasattr(self.model, "predict"):
            raise RuntimeError(
                "HHEM model is expected to provide a .predict(pairs) method. "
                "Check transformers version and trust_remote_code=True."
            )

        self.chunk_paper = bool(chunk_paper)
        self.premise_max_tokens = int(premise_max_tokens)
        self.premise_stride = int(premise_stride)
        self.batch_size = int(batch_size)
        self._chunk_tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
            if self.chunk_paper else None
        )

    def paper_chunks(self, paper_text: str) -> List[str]:
        if not self.chunk_paper or self._chunk_tokenizer is None:
            return [paper_text] if paper_text else [""]
        chunks = chunk_text_by_tokens(
            paper_text, self._chunk_tokenizer,
            max_tokens=self.premise_max_tokens, stride=self.premise_stride,
        )
        return chunks or [""]

    def _predict_scores(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = list(pairs[start : start + self.batch_size])
            out = self.model.predict(batch)
            if isinstance(out, torch.Tensor):
                batch_scores = out.detach().float().cpu().tolist()
            else:
                try:
                    batch_scores = list(out)
                except Exception:
                    batch_scores = [float(out)]
            scores.extend([clamp01(float(s)) for s in batch_scores])
        return scores

    def score_record(
        self,
        evidence_text: str,
        record_text: str,
        *,
        sentence_level: bool,
        aggregate_how: str,
        threshold: float,
        sentence_min_len: int,
        sentence_max_len: int,
    ) -> Dict[str, Any]:
        chunks = self.paper_chunks(evidence_text)
        if sentence_level:
            sentences = split_into_sentences(
                record_text, min_len=sentence_min_len, max_len=sentence_max_len
            )
        else:
            sentences = [record_text.strip()] if record_text and record_text.strip() else []

        if not sentences:
            return {"pred": 1, "truth_score": 0.0, "halluc_score": 1.0,
                    "num_sentences": 0, "num_chunks": len(chunks)}

        sent_truths: List[float] = []
        for sent in sentences:
            pairs = [(c, sent) for c in chunks]
            scores = self._predict_scores(pairs)
            sent_truths.append(float(max(scores)) if scores else 0.0)

        overall_truth = clamp01(aggregate(sent_truths, aggregate_how))
        pred = 0 if overall_truth >= threshold else 1
        return {
            "pred": int(pred),
            "truth_score": float(overall_truth),
            "halluc_score": float(1.0 - overall_truth),
            "num_sentences": len(sentences),
            "num_chunks": len(chunks),
        }


# =============================================================================
# Main loop
# =============================================================================

def _resolve_settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_id = args.model or cfg_get(cfg, ["model", "model_id"], "vectara/hallucination_evaluation_model")
    device = args.device or cfg_get(cfg, ["model", "device"])
    chunk_paper = (
        args.chunk_paper if args.chunk_paper is not None
        else bool(cfg_get(cfg, ["model", "chunk_paper"], True))
    )
    tokenizer_id = cfg_get(cfg, ["model", "tokenizer_id"], "google/flan-t5-base")
    premise_max_tokens = int(cfg_get(cfg, ["model", "premise_max_tokens"], 1200))
    premise_stride = int(cfg_get(cfg, ["model", "premise_stride"], 200))
    batch_size = int(args.batch_size or cfg_get(cfg, ["model", "batch_size"], 32))

    evidence_type = args.evidence or cfg_get(cfg, ["evidence", "type"], "no_retrieval")
    bm25_top_k = int(args.bm25_top_k or cfg_get(cfg, ["evidence", "bm25_top_k"], 5))
    bm25_chunk_chars = int(args.bm25_chunk_chars or cfg_get(cfg, ["evidence", "bm25_chunk_chars"], 2000))
    bm25_chunk_overlap = int(
        args.bm25_chunk_overlap if args.bm25_chunk_overlap is not None
        else cfg_get(cfg, ["evidence", "bm25_chunk_overlap"], 300)
    )

    sentence_level = (
        bool(args.sentence_level) if args.sentence_level is not None
        else bool(cfg_get(cfg, ["scoring", "sentence_level"], True))
    )
    aggregate_how = str(args.aggregate or cfg_get(cfg, ["scoring", "aggregate"], "mean"))
    sentence_min_len = int(cfg_get(cfg, ["scoring", "sentence_min_len"], 10))
    sentence_max_len = int(cfg_get(cfg, ["scoring", "sentence_max_len"], 400))

    return {
        "model_id": model_id, "device": device, "chunk_paper": bool(chunk_paper),
        "tokenizer_id": tokenizer_id, "premise_max_tokens": premise_max_tokens,
        "premise_stride": premise_stride, "batch_size": batch_size,
        "evidence_type": evidence_type, "bm25_top_k": bm25_top_k,
        "bm25_chunk_chars": bm25_chunk_chars, "bm25_chunk_overlap": bm25_chunk_overlap,
        "sentence_level": bool(sentence_level), "aggregate_how": aggregate_how,
        "sentence_min_len": sentence_min_len, "sentence_max_len": sentence_max_len,
    }


def _build_scorer(s: Dict[str, Any]) -> "HHEMScorer":
    return HHEMScorer(
        model_id=s["model_id"], device=s["device"],
        chunk_paper=s["chunk_paper"], tokenizer_id=s["tokenizer_id"],
        premise_max_tokens=s["premise_max_tokens"], premise_stride=s["premise_stride"],
        batch_size=s["batch_size"],
    )


def _build_evidence(s: Dict[str, Any], paper_content: Dict[str, Any], query: str) -> str:
    if s["evidence_type"] == "BM25":
        return bm25_evidence(
            paper_content, query, s["bm25_top_k"],
            s["bm25_chunk_chars"], s["bm25_chunk_overlap"],
        )
    return join_paper_content(paper_content)


def precompute_threshold_on_train(
    args: argparse.Namespace, settings: Dict[str, Any], cfg: Dict[str, Any],
    train_path: Path, precompute_key: str,
) -> None:
    if not train_path.exists():
        raise SystemExit(f"[ERROR] train file not found: {train_path}")
    train_data = read_jsonl(str(train_path))
    if args.max_samples:
        train_data = train_data[: args.max_samples]
    if not train_data:
        raise SystemExit(f"[ERROR] train file is empty: {train_path}")

    print(f"[PRECOMPUTE] Scoring {len(train_data)} train records on {train_path}")
    source_root = Path(args.source_data)
    paper_cache: Dict[str, Dict[str, Any]] = {}
    scorer = _build_scorer(settings)

    truth_scores: List[float] = []
    labels: List[int] = []
    for r in tqdm(train_data, desc="HHEM-precompute", dynamic_ncols=True):
        sentence = str(r.get("text") or "").strip()
        if not sentence:
            continue
        if r.get("label") not in (0, 1):
            continue
        paper_id = str(r.get("paper_id") or "").strip()
        paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}
        evidence_text = _build_evidence(settings, paper_content, sentence)
        res = scorer.score_record(
            evidence_text, sentence,
            sentence_level=settings["sentence_level"],
            aggregate_how=settings["aggregate_how"],
            threshold=0.5,  # ignored — we use truth_score directly
            sentence_min_len=settings["sentence_min_len"],
            sentence_max_len=settings["sentence_max_len"],
        )
        truth_scores.append(float(res["truth_score"]))
        labels.append(int(r["label"]))

    if not truth_scores:
        raise SystemExit("[ERROR] no usable train records (need label in {0,1} + non-empty text)")

    step = float(cfg_get(cfg, ["scoring", "train_sweep_step"], 0.01))
    best_thr, stats = find_best_threshold(truth_scores, labels, step=step)
    print(f"[PRECOMPUTE] Best threshold {best_thr:.4f} | stats={stats}")

    out_root = Path(args.out).resolve()
    log_dir = out_root / "precompute"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{precompute_key}.json"
    log_payload = {
        "key": precompute_key,
        "train_path": str(train_path),
        "config_path": str(_resolve_config_path(args.config)),
        "best_threshold": best_thr,
        "stats": stats,
        "n_scored": len(truth_scores),
        "settings": settings,
        "scores_labels": [
            {"truth_score": ts, "label": lb} for ts, lb in zip(truth_scores, labels)
        ],
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    written_to = update_config_scoring(args.config, precompute_key, best_thr, stats=stats)
    print(f"[PRECOMPUTE] Wrote {precompute_key} = {best_thr:.4f} → {written_to}")
    print(f"[PRECOMPUTE] Sweep log → {log_path}")
    print("[PRECOMPUTE] Re-run the same command to perform the actual test prediction.")


def run_test_predict(
    args: argparse.Namespace, settings: Dict[str, Any],
    threshold: float, threshold_mode: str,
) -> None:
    input_path = Path(args.input).resolve()
    data = read_jsonl(str(input_path))
    if args.max_samples:
        data = data[: args.max_samples]

    source_root = Path(args.source_data)
    paper_cache: Dict[str, Dict[str, Any]] = {}

    out_root = Path(args.out).resolve()
    out_suffix = build_output_suffix(
        settings["model_id"], settings["evidence_type"],
        sentence_level=settings["sentence_level"],
        aggregate_how=settings["aggregate_how"],
        threshold_mode=threshold_mode,
        threshold=threshold,
    )
    out_path = derive_output_path(input_path, out_root, out_suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Resume: load existing results; only skip pred != -1 -----
    existing: Dict[Any, Dict] = {}
    done_ids: set = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as fr:
            for ln in fr:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                rid = obj.get("id")
                if rid is not None and obj.get("pred") != -1:
                    existing[rid] = obj
                    done_ids.add(rid)
        print(f"[RESUME] {len(done_ids)} done; re-running pred=-1 / new.")

    # ----- Sidecar config (write up front) -----
    sidecar_path = out_path.with_suffix(".config.json")
    bm25_block = (
        {"bm25_top_k": settings["bm25_top_k"],
         "bm25_chunk_chars": settings["bm25_chunk_chars"],
         "bm25_chunk_overlap": settings["bm25_chunk_overlap"]}
        if settings["evidence_type"] == "BM25" else {}
    )
    sidecar = {
        "input": str(input_path),
        "source_data": str(source_root),
        "output": str(out_path),
        "evidence": {"type": settings["evidence_type"], **bm25_block},
        "model": {
            "model_id": settings["model_id"], "device": settings["device"],
            "chunk_paper": settings["chunk_paper"], "tokenizer_id": settings["tokenizer_id"],
            "premise_max_tokens": settings["premise_max_tokens"],
            "premise_stride": settings["premise_stride"],
            "batch_size": settings["batch_size"],
        },
        "scoring": {
            "sentence_level": settings["sentence_level"],
            "aggregate": settings["aggregate_how"],
            "threshold": float(threshold),
            "threshold_mode": threshold_mode,
            "sentence_min_len": settings["sentence_min_len"],
            "sentence_max_len": settings["sentence_max_len"],
        },
        "max_samples": args.max_samples,
        "num_examples": len(data),
        "resume": bool(args.resume),
        "done_at_start": len(done_ids),
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----- Build scorer (lazy: skip model load if nothing to do) -----
    scorer: Optional[HHEMScorer] = None

    def _ensure_scorer() -> HHEMScorer:
        nonlocal scorer
        if scorer is None:
            scorer = _build_scorer(settings)
        return scorer

    # ----- Sequential per-record (torch inference is in-process; order trivial) -----
    fw = out_path.open("w", encoding="utf-8")
    n_processed = 0
    try:
        for r in tqdm(data, desc="HHEM", dynamic_ncols=True):
            rid = r.get("id")
            if rid in done_ids:
                fw.write(json.dumps(existing[rid], ensure_ascii=False) + "\n")
                fw.flush()
                continue

            sentence = str(r.get("text") or "").strip()
            paper_id = str(r.get("paper_id") or "").strip()
            paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}
            evidence_text = _build_evidence(settings, paper_content, sentence)

            if not sentence:
                row = {"id": rid, "pred": -1, "truth_score": None, "halluc_score": None,
                       "num_sentences": 0, "num_chunks": 0}
            else:
                s = _ensure_scorer()
                res = s.score_record(
                    evidence_text, sentence,
                    sentence_level=settings["sentence_level"],
                    aggregate_how=settings["aggregate_how"],
                    threshold=threshold,
                    sentence_min_len=settings["sentence_min_len"],
                    sentence_max_len=settings["sentence_max_len"],
                )
                row = {"id": rid, **res}

            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
            fw.flush()
            n_processed += 1
    finally:
        fw.close()

    print(f"[DONE] Wrote {n_processed} new + {len(done_ids)} resumed → {out_path}")
    print(f"[DONE] Run config: {sidecar_path}")


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    settings = _resolve_settings(args, cfg)

    threshold_mode = (args.threshold_mode or "fixed").lower()
    input_path = Path(args.input).resolve()
    data_level = derive_data_level(input_path, args.data_level)

    if threshold_mode == "precomputed":
        precompute_key = build_precompute_key(
            settings["sentence_level"], settings["aggregate_how"], data_level
        )
        precomputed = cfg_get(cfg, ["scoring", precompute_key])
        if precomputed is None:
            print(
                f"[INFO] Precomputed threshold for key '{precompute_key}' not found in "
                f"{_resolve_config_path(args.config)}.\n"
                "[INFO] Computing it on train.jsonl now. You MUST re-run this same "
                "command afterwards to perform the actual test prediction."
            )
            train_path = (
                Path(args.train_input).resolve() if args.train_input
                else derive_train_path(input_path)
            )
            precompute_threshold_on_train(args, settings, cfg, train_path, precompute_key)
            return
        print(f"[INFO] Using precomputed threshold '{precompute_key}' = {float(precomputed):.4f}")
        run_test_predict(args, settings, float(precomputed), threshold_mode="precomputed")
        return

    if threshold_mode == "fixed":
        thr = float(
            args.fixed_threshold if args.fixed_threshold is not None
            else cfg_get(cfg, ["scoring", "fixed_threshold"], 0.5)
        )
        run_test_predict(args, settings, thr, threshold_mode="fixed")
        return

    raise SystemExit(f"[ERROR] unknown --threshold_mode: {threshold_mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task1 baseline: HHEM-2.1-Open (Vectara)")
    ap.add_argument("--config", default=None, help="YAML config (default: ./config_baseline.yaml)")
    ap.add_argument("--input", required=True, help="Task1 JSONL under task_split/...")
    ap.add_argument("--source_data", required=True, help="Root dir of {paper_id}.json files")
    ap.add_argument("--out", required=True, help="Output ROOT dir")
    ap.add_argument("--evidence", choices=["no_retrieval", "BM25"], default=None)
    ap.add_argument("--bm25_top_k", type=int, default=None)
    ap.add_argument("--bm25_chunk_chars", type=int, default=None)
    ap.add_argument("--bm25_chunk_overlap", type=int, default=None)
    ap.add_argument("--model", default=None, help="Override model.model_id")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk_paper", type=lambda x: x.lower() in {"1", "true", "yes"}, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--sentence_level",
                    type=lambda x: x.lower() in {"1", "true", "yes"}, default=None,
                    help="Override scoring.sentence_level "
                         "(true = split + aggregate; false = whole record as one hypothesis)")
    ap.add_argument("--aggregate", choices=["mean", "min", "median"], default=None)
    ap.add_argument("--threshold_mode", choices=["fixed", "precomputed"], default="fixed",
                    help="fixed: use --fixed_threshold / scoring.fixed_threshold. "
                         "precomputed: look up scoring.precompute_threshold_split{T|F}_agg{...}_level{...}; "
                         "if missing, score train.jsonl, write threshold to config, exit.")
    ap.add_argument("--fixed_threshold", type=float, default=None,
                    help="Threshold value when --threshold_mode fixed.")
    ap.add_argument("--data_level", choices=["review", "sent"], default=None,
                    help="Override the level tag in the precompute key "
                         "(auto-derived from --input path otherwise).")
    ap.add_argument("--train_input", default=None,
                    help="Override the train.jsonl path used by --threshold_mode precomputed.")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
