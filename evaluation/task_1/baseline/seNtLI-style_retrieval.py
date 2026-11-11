"""
Task 1 baseline — seNtLI-style retrieval
==========================================

3-way NLI (default ``roberta-large-mnli``) with sentence-level retrieval:

  1. Build evidence text from {paper_id}.json (no_retrieval or BM25, identical
     to zero_shot.py), then split it into sentences (``doc_sents``).
  2. Split the record text into sentences (``hyp_sents``).
  3. Run NLI over the full ``hyp × doc`` matrix → (entail, contra) per cell.
  4. For each hypothesis sentence build a context from the top-k doc sentences:
       - mode=topk    : top-k by entailment
       - mode=sentli  : top-ke by entailment ∪ top-kc by contradiction
  5. Re-rank with NLI on (context, hyp_sent) to get the final entailment score.
  6. Record truth = aggregate(entailment per sentence); pred = 0 if truth >=
     threshold else 1.

Config:
    Defaults in ``./config_baseline.yaml`` under section ``sentli``.

Usage:
    python -m evaluation.task_1.baseline.\\
        seNtLI-style_retrieval \\
        --input  dataset/task_split/.../task1_review/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_ICLR2019 \\
        --out    evaluation/outputs_May/task1_sentli \\
        --evidence no_retrieval --mode sentli --k 5
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
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_baseline.yaml"
EVAL_DATASET_MARKER = "task_split"
CONFIG_SECTION = "sentli"


# =============================================================================
# Config / IO
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


def cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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
    return out_root / sub.parent / f"{sub.stem}_{suffix}{sub.suffix or '.jsonl'}"


def build_output_suffix(model: str, evidence: str, mode: str, k: int,
                        threshold_mode: str, threshold: float) -> str:
    return "_".join([
        "result",
        f"model-{sanitize_model_name(model)}",
        f"evidence-{evidence}",
        f"mode-{mode}",
        f"k-{k}",
        f"thr-{threshold_mode}-{float(threshold):.3f}",
    ])


def build_precompute_key(mode: str, k: int, data_level: str) -> str:
    lvl = data_level if data_level in {"review", "sent"} else "review"
    return f"precompute_threshold_mode{mode}_k{k}_level{lvl}"


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
    candidate = test_input.with_name(f"train{test_input.suffix or '.jsonl'}")
    if candidate.exists():
        return candidate
    return test_input.parent / "train.jsonl"


def find_best_threshold(scores: Sequence[float], labels: Sequence[int],
                        step: float = 0.01) -> Tuple[float, Dict[str, Any]]:
    """Sweep truth_score thresholds; pick max F1-macro.

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


# =============================================================================
# Evidence extraction
# =============================================================================

_TOKEN_RE = re.compile(r"\w+")


def _bm25_tokenize(s: str) -> List[str]:
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
    tokenized = [_bm25_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    q_tokens = _bm25_tokenize(query)
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
# Sentence / aggregate
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
                    c = sub[i : i + max_len].strip()
                    if len(c) >= min_len:
                        out.append(c)
    return out


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
# Sentli scorer
# =============================================================================

class SentliScorer:
    def __init__(self, model_id: str, device: Optional[str], batch_size: int) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = int(batch_size)

        cfg = self.model.config
        label2id = {str(k).lower(): int(v) for k, v in (cfg.label2id or {}).items()}
        id2label = {int(k): str(v).lower() for k, v in (cfg.id2label or {}).items()}

        def find(cands: Sequence[str]) -> Optional[int]:
            for c in cands:
                if c.lower() in label2id:
                    return label2id[c.lower()]
            for idx, name in id2label.items():
                if any(c.lower() == name or c.lower() in name for c in cands):
                    return idx
            return None

        self.entail_id = find(["entailment", "entailed"])
        self.contra_id = find(["contradiction", "contradict"])
        self.neutral_id = find(["neutral"])
        if self.entail_id is None or self.contra_id is None or self.neutral_id is None:
            self.contra_id, self.neutral_id, self.entail_id = 0, 1, 2

        max_len = getattr(self.tokenizer, "model_max_length", 512)
        if not isinstance(max_len, int) or max_len > 4096:
            max_len = 512
        self.max_len = int(max_len)

    @torch.inference_mode()
    def _probs(self, premises: List[str], hyps: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            premises, hyps,
            truncation="only_first", padding=True,
            max_length=self.max_len, return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return F.softmax(self.model(**enc).logits, dim=-1)

    def ent_con_matrix(
        self, doc_sents: List[str], hyp_sents: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not doc_sents or not hyp_sents:
            shape = (len(hyp_sents), len(doc_sents))
            return torch.empty(shape), torch.empty(shape)

        ent = torch.zeros((len(hyp_sents), len(doc_sents)), dtype=torch.float32)
        con = torch.zeros((len(hyp_sents), len(doc_sents)), dtype=torch.float32)
        bs = self.batch_size
        for hi, hyp in enumerate(hyp_sents):
            ent_row: List[float] = []
            con_row: List[float] = []
            for st in range(0, len(doc_sents), bs):
                en = min(st + bs, len(doc_sents))
                probs = self._probs(doc_sents[st:en], [hyp] * (en - st)).detach().cpu()
                ent_row.extend(probs[:, self.entail_id].tolist())
                con_row.extend(probs[:, self.contra_id].tolist())
            ent[hi, :] = torch.tensor(ent_row, dtype=torch.float32)
            con[hi, :] = torch.tensor(con_row, dtype=torch.float32)
        return ent, con

    def rerank(self, contexts: List[str], hyps: List[str]) -> List[float]:
        results: List[float] = []
        if not hyps:
            return results
        bs = self.batch_size
        for st in range(0, len(hyps), bs):
            en = min(st + bs, len(hyps))
            probs = self._probs(contexts[st:en], hyps[st:en]).detach().cpu()
            for i in range(probs.shape[0]):
                results.append(float(probs[i, self.entail_id]))
        return results

    @staticmethod
    def _topk_idxs(values: torch.Tensor, k: int) -> List[int]:
        if values.numel() == 0:
            return []
        k = max(1, min(k, values.numel()))
        return [int(i) for i in torch.topk(values, k=k, largest=True).indices.tolist()]

    def build_context(
        self,
        doc_sents: List[str],
        ent_row: torch.Tensor,
        con_row: torch.Tensor,
        *,
        k: int,
        mode: str,
        ke: Optional[int],
        kc: Optional[int],
    ) -> str:
        if not doc_sents:
            return ""
        if mode == "topk":
            idxs = self._topk_idxs(ent_row, k)
            return "\n".join(doc_sents[i] for i in idxs)
        if mode == "sentli":
            ke_eff = ke if ke is not None else k
            kc_eff = kc if kc is not None else k
            ent_idxs = self._topk_idxs(ent_row, ke_eff)
            con_idxs = self._topk_idxs(con_row, kc_eff)
            idxs: List[int] = []
            seen: set = set()
            for i in ent_idxs + con_idxs:
                if i in seen:
                    continue
                seen.add(i)
                idxs.append(i)
            return "\n".join(doc_sents[i] for i in idxs)
        raise ValueError(f"Unknown mode: {mode}")

    def score_record(
        self,
        evidence_text: str,
        record_text: str,
        *,
        mode: str,
        k: int,
        ke: Optional[int],
        kc: Optional[int],
        aggregate_how: str,
        threshold: float,
        sentence_min_len: int,
        sentence_max_len: int,
        doc_sentence_min_len: int,
        doc_sentence_max_len: int,
    ) -> Dict[str, Any]:
        doc_sents = split_into_sentences(
            evidence_text, min_len=doc_sentence_min_len, max_len=doc_sentence_max_len
        )
        hyp_sents = split_into_sentences(
            record_text, min_len=sentence_min_len, max_len=sentence_max_len
        )
        if not hyp_sents:
            return {"pred": 1, "truth_score": 0.0, "halluc_score": 1.0,
                    "num_sentences": 0, "num_doc_sentences": len(doc_sents)}
        if not doc_sents:
            return {"pred": 1, "truth_score": 0.0, "halluc_score": 1.0,
                    "num_sentences": len(hyp_sents), "num_doc_sentences": 0}

        ent_mat, con_mat = self.ent_con_matrix(doc_sents, hyp_sents)
        contexts = [
            self.build_context(
                doc_sents, ent_mat[i, :], con_mat[i, :],
                k=k, mode=mode, ke=ke, kc=kc,
            )
            for i in range(len(hyp_sents))
        ]
        entail_scores = self.rerank(contexts, hyp_sents)
        overall_truth = clamp01(aggregate(entail_scores, aggregate_how))
        pred = 0 if overall_truth >= threshold else 1
        return {
            "pred": int(pred),
            "truth_score": float(overall_truth),
            "halluc_score": float(1.0 - overall_truth),
            "num_sentences": len(hyp_sents),
            "num_doc_sentences": len(doc_sents),
        }


# =============================================================================
# Main loop
# =============================================================================

def _resolve_settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_id = args.model or cfg_get(cfg, ["model", "model_id"], "roberta-large-mnli")
    device = args.device or cfg_get(cfg, ["model", "device"])
    batch_size = int(args.batch_size or cfg_get(cfg, ["model", "batch_size"], 32))

    evidence_type = args.evidence or cfg_get(cfg, ["evidence", "type"], "no_retrieval")
    bm25_top_k = int(args.bm25_top_k or cfg_get(cfg, ["evidence", "bm25_top_k"], 5))
    bm25_chunk_chars = int(args.bm25_chunk_chars or cfg_get(cfg, ["evidence", "bm25_chunk_chars"], 1400))
    bm25_chunk_overlap = int(
        args.bm25_chunk_overlap if args.bm25_chunk_overlap is not None
        else cfg_get(cfg, ["evidence", "bm25_chunk_overlap"], 150)
    )

    mode = args.mode or cfg_get(cfg, ["scoring", "mode"], "topk")
    if mode not in ("topk", "sentli"):
        raise SystemExit(f"--mode must be topk or sentli, got: {mode}")
    k = int(args.k or cfg_get(cfg, ["scoring", "k"], 3))
    ke = args.ke if args.ke is not None else cfg_get(cfg, ["scoring", "ke"])
    kc = args.kc if args.kc is not None else cfg_get(cfg, ["scoring", "kc"])
    ke = int(ke) if ke is not None else None
    kc = int(kc) if kc is not None else None
    aggregate_how = str(cfg_get(cfg, ["scoring", "aggregate"], "mean"))
    sentence_min_len = int(cfg_get(cfg, ["scoring", "sentence_min_len"], 10))
    sentence_max_len = int(cfg_get(cfg, ["scoring", "sentence_max_len"], 400))
    doc_sentence_min_len = int(cfg_get(cfg, ["scoring", "doc_sentence_min_len"], 10))
    doc_sentence_max_len = int(cfg_get(cfg, ["scoring", "doc_sentence_max_len"], 400))

    return {
        "model_id": model_id, "device": device, "batch_size": batch_size,
        "evidence_type": evidence_type, "bm25_top_k": bm25_top_k,
        "bm25_chunk_chars": bm25_chunk_chars, "bm25_chunk_overlap": bm25_chunk_overlap,
        "mode": mode, "k": k, "ke": ke, "kc": kc,
        "aggregate_how": aggregate_how,
        "sentence_min_len": sentence_min_len, "sentence_max_len": sentence_max_len,
        "doc_sentence_min_len": doc_sentence_min_len, "doc_sentence_max_len": doc_sentence_max_len,
    }


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
    scorer = SentliScorer(
        model_id=settings["model_id"], device=settings["device"],
        batch_size=settings["batch_size"],
    )

    truth_scores: List[float] = []
    labels: List[int] = []
    for r in tqdm(train_data, desc="sentli-precompute", dynamic_ncols=True):
        sentence = str(r.get("text") or r.get("sentence") or r.get("review_sentence") or "").strip()
        if not sentence:
            continue
        if r.get("label") not in (0, 1):
            continue
        paper_id = str(r.get("paper_id") or "").strip()
        paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}
        evidence_text = _build_evidence(settings, paper_content, sentence)
        res = scorer.score_record(
            evidence_text, sentence,
            mode=settings["mode"], k=settings["k"],
            ke=settings["ke"], kc=settings["kc"],
            aggregate_how=settings["aggregate_how"],
            threshold=0.5,  # ignored — we use truth_score directly
            sentence_min_len=settings["sentence_min_len"],
            sentence_max_len=settings["sentence_max_len"],
            doc_sentence_min_len=settings["doc_sentence_min_len"],
            doc_sentence_max_len=settings["doc_sentence_max_len"],
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
        settings["mode"], settings["k"],
        threshold_mode, threshold,
    )
    out_path = derive_output_path(input_path, out_root, out_suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    sidecar_path = out_path.with_suffix(".config.json")
    sidecar = {
        "input": str(input_path),
        "source_data": str(source_root),
        "output": str(out_path),
        "evidence": {
            "type": settings["evidence_type"],
            **({"bm25_top_k": settings["bm25_top_k"],
                "bm25_chunk_chars": settings["bm25_chunk_chars"],
                "bm25_chunk_overlap": settings["bm25_chunk_overlap"]}
               if settings["evidence_type"] == "BM25" else {}),
        },
        "model": {
            "model_id": settings["model_id"], "device": settings["device"],
            "batch_size": settings["batch_size"],
        },
        "scoring": {
            "mode": settings["mode"], "k": settings["k"],
            "ke": settings["ke"], "kc": settings["kc"],
            "aggregate": settings["aggregate_how"],
            "threshold": float(threshold), "threshold_mode": threshold_mode,
            "sentence_min_len": settings["sentence_min_len"],
            "sentence_max_len": settings["sentence_max_len"],
            "doc_sentence_min_len": settings["doc_sentence_min_len"],
            "doc_sentence_max_len": settings["doc_sentence_max_len"],
        },
        "max_samples": args.max_samples,
        "num_examples": len(data),
        "resume": bool(args.resume),
        "done_at_start": len(done_ids),
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    scorer: Optional[SentliScorer] = None

    def _ensure_scorer() -> SentliScorer:
        nonlocal scorer
        if scorer is None:
            scorer = SentliScorer(
                model_id=settings["model_id"], device=settings["device"],
                batch_size=settings["batch_size"],
            )
        return scorer

    fw = out_path.open("w", encoding="utf-8")
    n_processed = 0
    try:
        for r in tqdm(data, desc="sentli", dynamic_ncols=True):
            rid = r.get("task") or r.get("id") or r.get("sent_id")
            if rid in done_ids:
                fw.write(json.dumps(existing[rid], ensure_ascii=False) + "\n")
                fw.flush()
                continue

            sentence = str(r.get("text") or r.get("sentence") or r.get("review_sentence") or "").strip()
            paper_id = str(r.get("paper_id") or "").strip()
            paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}
            evidence_text = _build_evidence(settings, paper_content, sentence)

            if not sentence:
                row = {"id": rid, "pred": -1, "truth_score": None, "halluc_score": None,
                       "num_sentences": 0, "num_doc_sentences": 0}
            else:
                s = _ensure_scorer()
                res = s.score_record(
                    evidence_text, sentence,
                    mode=settings["mode"], k=settings["k"],
                    ke=settings["ke"], kc=settings["kc"],
                    aggregate_how=settings["aggregate_how"],
                    threshold=threshold,
                    sentence_min_len=settings["sentence_min_len"],
                    sentence_max_len=settings["sentence_max_len"],
                    doc_sentence_min_len=settings["doc_sentence_min_len"],
                    doc_sentence_max_len=settings["doc_sentence_max_len"],
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
        precompute_key = build_precompute_key(settings["mode"], settings["k"], data_level)
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
    ap = argparse.ArgumentParser(description="Task1 baseline: seNtLI-style retrieval")
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--source_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", choices=["no_retrieval", "BM25"], default=None)
    ap.add_argument("--bm25_top_k", type=int, default=None)
    ap.add_argument("--bm25_chunk_chars", type=int, default=None)
    ap.add_argument("--bm25_chunk_overlap", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--mode", choices=["topk", "sentli"], default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--ke", type=int, default=None)
    ap.add_argument("--kc", type=int, default=None)
    ap.add_argument("--threshold_mode", choices=["fixed", "precomputed"], default="fixed",
                    help="fixed: use --fixed_threshold / scoring.fixed_threshold. "
                         "precomputed: look up scoring.precompute_threshold_mode{mode}_k{k}_level{...}; "
                         "if missing, score train.jsonl, write threshold to config, exit.")
    ap.add_argument("--fixed_threshold", type=float, default=None,
                    help="Threshold value when --threshold_mode fixed (default 0.5).")
    ap.add_argument("--data_level", choices=["review", "sent"], default=None,
                    help="Override level tag in precompute key (auto-derived from --input otherwise).")
    ap.add_argument("--train_input", default=None,
                    help="Override train.jsonl path used by --threshold_mode precomputed.")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
