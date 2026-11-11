"""
Task 1 baseline — RefChecker (claim-level NLI hallucination judge)
==================================================================

For each input record (review- or sentence-level):
  1. Build evidence from {paper_id}.json (no_retrieval / BM25, identical to
     RA-LLM.py / TrueNLI.py).
  2. Extract claim triples from the record text using a Mistral claim
     extractor served via vLLM (OpenAI-compatible).
  3. Check each claim against the evidence with one of two checkers:
       * ``alignscore``  — local AlignScore (roberta-large) NLI checker.
       * ``llm``         — LLMChecker backed by a vLLM (e.g. Qwen/Qwen3-32B).
  4. Per-record truth_score = fraction of claims labelled "Entailment".
     pred = 0 if truth_score >= threshold else 1.  If no claims extracted,
     pred = -1.

Threshold mode (``--threshold_mode``):
    * ``fixed``       use ``scoring.fixed_threshold`` (or ``--fixed_threshold``).
    * ``precomputed`` look up ``scoring.precompute_threshold_chker{mode}_level{review|sent}``.
                      If absent: score the matching ``train.jsonl`` (auto-derived
                      from --input by replacing the stem with "train"), sweep
                      thresholds on (truth_score, label) pairs to pick max F1-macro,
                      write the result back into the YAML, and EXIT. Re-run the
                      same command to perform the actual test prediction.

Config:
    Defaults in ``./config_baseline.yaml`` under section ``refchecker``.

Usage:
    # extractor server (Mistral)
    vllm serve dongyru/Mistral-7B-Claim-Extractor --port 5001
    # (LLM-checker mode only) checker server
    vllm serve Qwen/Qwen3-32B --port 5002

    python evaluation/task_1/baseline/Refchecker.py \\
        --input  dataset/task_split/.../task1_sent/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_NeurIPS2024 \\
        --out    evaluation/evaluation_results/task1/refchecker \\
        --checker_mode alignscore --evidence BM25 --threshold_mode precomputed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm
from rank_bm25 import BM25Okapi


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_baseline.yaml"
EVAL_DATASET_MARKER = "task_split"
CONFIG_SECTION = "refchecker"


# =============================================================================
# Config / IO helpers
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


def build_precompute_key(checker_mode: str, data_level: str) -> str:
    lvl = data_level if data_level in {"review", "sent"} else "review"
    return f"precompute_threshold_chker{checker_mode}_level{lvl}"


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
    return out_root / sub.parent / f"{stem}_{suffix}{ext}"


def build_output_suffix(
    *,
    extractor_model: str,
    checker_mode: str,
    checker_model: str,
    evidence: str,
) -> str:
    return "_".join([
        "result",
        f"ext-{sanitize_model_name(extractor_model)}",
        f"chk-{checker_mode}",
        f"chkmodel-{sanitize_model_name(checker_model)}",
        f"evidence-{evidence}",
    ])


# =============================================================================
# Paper / evidence helpers (mirrored from RA-LLM.py / TrueNLI.py)
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


def bm25_evidence(
    paper_content: Dict[str, Any],
    query: str,
    top_k: int,
    chunk_chars: int,
    overlap: int,
) -> str:
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


def build_evidence(
    evidence_type: str,
    paper_content: Dict[str, Any],
    query: str,
    bm25_top_k: int,
    bm25_chunk_chars: int,
    bm25_chunk_overlap: int,
) -> str:
    if evidence_type == "BM25":
        return bm25_evidence(paper_content, query, bm25_top_k, bm25_chunk_chars, bm25_chunk_overlap)
    return join_paper_content(paper_content)


# =============================================================================
# Claim / label helpers
# =============================================================================

def _claim_to_list(c: Any) -> List[str]:
    """Coerce a claim's ``content`` attribute into a list of strings.

    LLMExtractor returns claim objects whose ``content`` is typically a
    (subject, predicate, object) triple, but can also be a plain string.
    """
    if isinstance(c, str):
        return [c]
    if isinstance(c, (list, tuple)):
        return [str(x) for x in c]
    return [str(c)]


def _normalize_claim_label(raw: Any) -> str:
    """Flatten the per-claim label returned by refchecker to a single string."""
    val = raw
    while isinstance(val, (list, tuple)) and val:
        val = val[0]
    return str(val) if val is not None else "Neutral"


def aggregate_entailment_ratio(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    ent = sum(1 for l in labels if str(l).strip().lower() == "entailment")
    return ent / len(labels)


# =============================================================================
# Settings + model builders
# =============================================================================

def _resolve_settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Extractor (always Mistral via vLLM OpenAI endpoint)
    extractor_model = args.extractor_model or cfg_get(
        cfg, ["extractor", "model_name"], "openai/dongyru/Mistral-7B-Claim-Extractor"
    )
    extractor_port = int(args.extractor_port or cfg_get(cfg, ["extractor", "port"], 5001))
    extractor_api_key = str(
        args.extractor_api_key
        if args.extractor_api_key is not None
        else cfg_get(cfg, ["extractor", "api_key"], "abc")
    )
    extractor_batch_size = int(cfg_get(cfg, ["extractor", "batch_size"], 8))

    # Checker mode
    checker_mode = (args.checker_mode or cfg_get(cfg, ["checker", "mode"], "alignscore")).lower()
    if checker_mode not in {"alignscore", "llm"}:
        raise SystemExit(f"--checker_mode must be alignscore|llm, got: {checker_mode}")

    # AlignScore checker
    alignscore_model = cfg_get(cfg, ["checker", "alignscore", "model"], "roberta-large")
    alignscore_ckpt = args.alignscore_ckpt or cfg_get(
        cfg, ["checker", "alignscore", "ckpt_path"], "alignscore.ckpt"
    )
    alignscore_device = args.device or cfg_get(cfg, ["checker", "alignscore", "device"], "cuda")
    alignscore_batch_size = int(cfg_get(cfg, ["checker", "alignscore", "batch_size"], 32))
    alignscore_eval_mode = cfg_get(cfg, ["checker", "alignscore", "evaluation_mode"], "nli_sp")

    # LLM checker
    llm_model = args.checker_model or cfg_get(
        cfg, ["checker", "llm", "model_name"], "openai/Qwen/Qwen3-32B"
    )
    llm_port = int(args.checker_port or cfg_get(cfg, ["checker", "llm", "port"], 5002))
    llm_api_key = str(
        args.checker_api_key
        if args.checker_api_key is not None
        else cfg_get(cfg, ["checker", "llm", "api_key"], "abc")
    )
    llm_batch_size = int(cfg_get(cfg, ["checker", "llm", "batch_size"], 8))

    # Evidence
    evidence_type = args.evidence or cfg_get(cfg, ["evidence", "type"], "BM25")
    bm25_top_k = int(args.bm25_top_k or cfg_get(cfg, ["evidence", "bm25_top_k"], 5))
    bm25_chunk_chars = int(args.bm25_chunk_chars or cfg_get(cfg, ["evidence", "bm25_chunk_chars"], 2000))
    bm25_chunk_overlap = int(
        args.bm25_chunk_overlap if args.bm25_chunk_overlap is not None
        else cfg_get(cfg, ["evidence", "bm25_chunk_overlap"], 300)
    )

    # Pipeline batching
    pipeline_batch_size = int(args.batch_size or cfg_get(cfg, ["pipeline_batch_size"], 8))

    return {
        "extractor_model": extractor_model,
        "extractor_port": extractor_port,
        "extractor_api_key": extractor_api_key,
        "extractor_batch_size": extractor_batch_size,
        "checker_mode": checker_mode,
        "alignscore_model": alignscore_model,
        "alignscore_ckpt": alignscore_ckpt,
        "alignscore_device": alignscore_device,
        "alignscore_batch_size": alignscore_batch_size,
        "alignscore_eval_mode": alignscore_eval_mode,
        "llm_model": llm_model,
        "llm_port": llm_port,
        "llm_api_key": llm_api_key,
        "llm_batch_size": llm_batch_size,
        "evidence_type": evidence_type,
        "bm25_top_k": bm25_top_k,
        "bm25_chunk_chars": bm25_chunk_chars,
        "bm25_chunk_overlap": bm25_chunk_overlap,
        "pipeline_batch_size": pipeline_batch_size,
    }


def _build_extractor(settings: Dict[str, Any]):
    from refchecker import LLMExtractor  # local import: heavy dep
    # NOTE: LLMExtractor.__init__ does NOT accept api_key; it is forwarded
    # through extract(**kwargs) → litellm.batch_completion(..., api_key=...).
    return LLMExtractor(
        model=settings["extractor_model"],
        api_base=f"http://localhost:{settings['extractor_port']}/v1",
        batch_size=settings["extractor_batch_size"],
    )


def _build_checker(settings: Dict[str, Any]):
    mode = settings["checker_mode"]
    if mode == "alignscore":
        # AlignScoreChecker lives in ./utils/refchecker (handles ckpt download).
        rebuttal_utils = Path(__file__).resolve().parent
        if str(rebuttal_utils) not in sys.path:
            sys.path.insert(0, str(rebuttal_utils))
        from utils.refchecker.alignscore_checker import AlignScoreChecker  # type: ignore
        return AlignScoreChecker(
            ckpt_path=settings["alignscore_ckpt"],
            device=settings["alignscore_device"],
            batch_size=settings["alignscore_batch_size"],
        )
    # llm
    from refchecker import LLMChecker
    # NOTE: LLMChecker.__init__ does NOT accept api_key; it is forwarded
    # through check(**kwargs) → litellm.batch_completion(..., api_key=...).
    return LLMChecker(
        model=settings["llm_model"],
        api_base=f"http://localhost:{settings['llm_port']}/v1",
        batch_size=settings["llm_batch_size"],
    )


def _extract_claims_batch(
    extractor,
    reviews: List[str],
    *,
    api_key: Optional[str] = None,
) -> List[List[List[str]]]:
    """Returns ``batch_claims[i][j]`` = list-form claim (e.g. [s, p, o]).

    ``api_key`` is forwarded via ``**kwargs`` to litellm.batch_completion."""
    extra: Dict[str, Any] = {}
    if api_key:
        extra["api_key"] = api_key
    results = extractor.extract(batch_responses=reviews, **extra)
    out: List[List[List[str]]] = []
    for res in results:
        claims = getattr(res, "claims", []) or []
        out.append([_claim_to_list(c.content) for c in claims])
    return out


def _check_claims_batch(
    checker,
    batch_claims: List[List[List[str]]],
    batch_references: List[str],
    *,
    api_key: Optional[str] = None,
) -> List[List[str]]:
    """Returns ``batch_labels[i][j]`` = single normalized label per claim.

    ``api_key`` is forwarded via ``**kwargs`` to litellm.batch_completion
    (only meaningful in LLM checker mode)."""
    extra: Dict[str, Any] = {}
    if api_key:
        extra["api_key"] = api_key
    raw = checker.check(
        batch_claims=batch_claims,
        batch_references=batch_references,
        is_joint=False,
        **extra,
    )
    out: List[List[str]] = []
    for per_record in raw:
        out.append([_normalize_claim_label(x) for x in (per_record or [])])
    return out


def _score_one_batch(
    extractor,
    checker,
    refs: List[str],
    reviews: List[str],
    *,
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run extract + check on a batch. Returns one dict per input row containing
    ``{truth_score, num_claims, claims, labels}`` (truth_score=None when no
    claims could be checked)."""
    n = len(reviews)
    non_empty = [i for i, s in enumerate(reviews) if s]
    batch_claims: List[List[List[str]]] = [[] for _ in range(n)]
    batch_labels: List[List[str]] = [[] for _ in range(n)]

    extractor_api_key = settings.get("extractor_api_key")
    checker_api_key = (
        settings.get("llm_api_key") if settings.get("checker_mode") == "llm" else None
    )

    if non_empty:
        sub_reviews = [reviews[i] for i in non_empty]
        try:
            sub_claims = _extract_claims_batch(
                extractor, sub_reviews, api_key=extractor_api_key
            )
        except Exception as e:
            tqdm.write(f"[WARN] extractor failed on batch: {e}", file=sys.stderr)
            sub_claims = [[] for _ in sub_reviews]
        for k, src_idx in enumerate(non_empty):
            batch_claims[src_idx] = sub_claims[k]

        check_idx = [i for i, cs in enumerate(batch_claims) if cs]
        if check_idx:
            try:
                sub_labels = _check_claims_batch(
                    checker,
                    [batch_claims[i] for i in check_idx],
                    [refs[i] for i in check_idx],
                    api_key=checker_api_key,
                )
            except Exception as e:
                tqdm.write(f"[WARN] checker failed on batch: {e}", file=sys.stderr)
                sub_labels = [[] for _ in check_idx]
            for k, src_idx in enumerate(check_idx):
                batch_labels[src_idx] = sub_labels[k]

    out: List[Dict[str, Any]] = []
    for i in range(n):
        claims = batch_claims[i]
        labels = batch_labels[i]
        nc = len(claims)
        if nc == 0 or len(labels) != nc:
            out.append({
                "truth_score": None,
                "num_claims": nc,
                "claims": claims,
                "labels": labels,
            })
        else:
            out.append({
                "truth_score": float(aggregate_entailment_ratio(labels)),
                "num_claims": nc,
                "claims": claims,
                "labels": labels,
            })
    return out


def _iter_batches(seq: List[Any], batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield i, seq[i : i + batch_size]


# =============================================================================
# Precompute / predict
# =============================================================================

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

    usable = [r for r in train_data
              if str(r.get("text") or "").strip() and r.get("label") in (0, 1)]
    if not usable:
        raise SystemExit("[ERROR] no usable train records (need label in {0,1} + non-empty text)")

    print(f"[PRECOMPUTE] Scoring {len(usable)} train records on {train_path}")
    source_root = Path(args.source_data)
    paper_cache: Dict[str, Dict[str, Any]] = {}

    print(f"[INIT] Loading extractor {settings['extractor_model']} "
          f"@ port {settings['extractor_port']}")
    extractor = _build_extractor(settings)
    print(f"[INIT] Loading checker (mode={settings['checker_mode']})")
    checker = _build_checker(settings)

    truth_scores: List[float] = []
    labels: List[int] = []
    batch_size = settings["pipeline_batch_size"]
    with tqdm(total=len(usable), desc="RefChecker-precompute", dynamic_ncols=True) as pbar:
        for _, batch in _iter_batches(usable, batch_size):
            refs: List[str] = []
            reviews: List[str] = []
            lbls: List[int] = []
            for r in batch:
                sentence = str(r.get("text") or "").strip()
                paper_id = str(r.get("paper_id") or "").strip()
                paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}
                refs.append(build_evidence(
                    settings["evidence_type"], paper_content, sentence,
                    settings["bm25_top_k"], settings["bm25_chunk_chars"],
                    settings["bm25_chunk_overlap"],
                ))
                reviews.append(sentence)
                lbls.append(int(r["label"]))

            results = _score_one_batch(extractor, checker, refs, reviews, settings=settings)
            for res, lb in zip(results, lbls):
                if res["truth_score"] is None:
                    continue
                truth_scores.append(float(res["truth_score"]))
                labels.append(int(lb))
            pbar.update(len(batch))

    if not truth_scores:
        raise SystemExit("[ERROR] no usable train records produced claims to score")

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

    checker_model_display = (
        settings["llm_model"] if settings["checker_mode"] == "llm"
        else settings["alignscore_model"]
    )
    out_root = Path(args.out).resolve()
    base_suffix = build_output_suffix(
        extractor_model=settings["extractor_model"],
        checker_mode=settings["checker_mode"],
        checker_model=checker_model_display,
        evidence=settings["evidence_type"],
    )
    thr_tag = f"thr-{threshold_mode}-{float(threshold):.3f}"
    out_suffix = f"{base_suffix}_{thr_tag}"
    out_path = derive_output_path(input_path, out_root, out_suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Resume -----
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
        print(f"[RESUME] {len(done_ids)} done (pred≠-1); re-running pred=-1 and new.")

    # ----- Sidecar config -----
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
        "extractor": {
            "model_name": settings["extractor_model"],
            "port": settings["extractor_port"],
            "batch_size": settings["extractor_batch_size"],
        },
        "checker": {
            "mode": settings["checker_mode"],
            **(
                {
                    "model_name": settings["llm_model"],
                    "port": settings["llm_port"],
                    "batch_size": settings["llm_batch_size"],
                }
                if settings["checker_mode"] == "llm"
                else {
                    "model": settings["alignscore_model"],
                    "ckpt_path": settings["alignscore_ckpt"],
                    "device": settings["alignscore_device"],
                    "batch_size": settings["alignscore_batch_size"],
                    "evaluation_mode": settings["alignscore_eval_mode"],
                }
            ),
        },
        "scoring": {
            "threshold": float(threshold),
            "threshold_mode": threshold_mode,
        },
        "pipeline_batch_size": settings["pipeline_batch_size"],
        "max_samples": args.max_samples,
        "num_examples": len(data),
        "resume": bool(args.resume),
        "done_at_start": len(done_ids),
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    pending_indices = [i for i, r in enumerate(data) if r.get("id") not in done_ids]
    print(f"[INFO] To process: {len(pending_indices)} / {len(data)}")

    # ----- Lazy load extractor + checker -----
    extractor = None
    checker = None

    def _ensure_models():
        nonlocal extractor, checker
        if extractor is None:
            print(f"[INIT] Loading extractor {settings['extractor_model']} "
                  f"@ port {settings['extractor_port']}")
            extractor = _build_extractor(settings)
        if checker is None:
            print(f"[INIT] Loading checker (mode={settings['checker_mode']})")
            checker = _build_checker(settings)

    batch_size = settings["pipeline_batch_size"]

    fw = out_path.open("w", encoding="utf-8")
    n_new = 0
    n_resumed = 0
    n_pred_minus_one = 0
    try:
        idx_in_data = 0
        with tqdm(total=len(data), desc="RefChecker", dynamic_ncols=True) as pbar:
            while idx_in_data < len(data):
                # Emit any consecutive resumed rows in input order.
                while idx_in_data < len(data) and data[idx_in_data].get("id") in done_ids:
                    rid = data[idx_in_data]["id"]
                    fw.write(json.dumps(existing[rid], ensure_ascii=False) + "\n")
                    fw.flush()
                    n_resumed += 1
                    pbar.update(1)
                    idx_in_data += 1
                if idx_in_data >= len(data):
                    break

                # Collect the next batch of pending records.
                batch_rows: List[Tuple[int, Dict[str, Any]]] = []
                cursor = idx_in_data
                while cursor < len(data) and len(batch_rows) < batch_size:
                    r = data[cursor]
                    if r.get("id") in done_ids:
                        break
                    batch_rows.append((cursor, r))
                    cursor += 1

                refs: List[str] = []
                reviews: List[str] = []
                rids: List[Any] = []
                for _, r in batch_rows:
                    sentence = str(r.get("text") or r.get("sentence") or r.get("review_sentence") or "").strip()
                    paper_id = str(r.get("paper_id") or "").strip()
                    paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}
                    refs.append(build_evidence(
                        settings["evidence_type"], paper_content, sentence,
                        settings["bm25_top_k"], settings["bm25_chunk_chars"],
                        settings["bm25_chunk_overlap"],
                    ))
                    reviews.append(sentence)
                    rids.append(r.get("id"))

                # Ensure models loaded only when there is real work.
                if any(s for s in reviews):
                    _ensure_models()
                    results = _score_one_batch(extractor, checker, refs, reviews, settings=settings)
                else:
                    results = [{"truth_score": None, "num_claims": 0, "claims": [], "labels": []}
                               for _ in reviews]

                for rid, res in zip(rids, results):
                    if res["truth_score"] is None:
                        row = {
                            "id": rid,
                            "pred": -1,
                            "truth_score": None,
                            "halluc_score": None,
                            "num_claims": res["num_claims"],
                            "claims": res["claims"],
                            "labels": res["labels"],
                        }
                        n_pred_minus_one += 1
                    else:
                        truth = float(res["truth_score"])
                        pred = 0 if truth >= threshold else 1
                        row = {
                            "id": rid,
                            "pred": int(pred),
                            "truth_score": truth,
                            "halluc_score": float(1.0 - truth),
                            "num_claims": res["num_claims"],
                            "claims": res["claims"],
                            "labels": res["labels"],
                        }
                    fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                fw.flush()
                n_new += len(results)
                pbar.update(len(results))
                idx_in_data = cursor
    finally:
        fw.close()

    print(f"[DONE] Wrote {n_new} new + {n_resumed} resumed "
          f"({n_pred_minus_one} pred=-1) → {out_path}")
    print(f"[DONE] Run config: {sidecar_path}")


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    settings = _resolve_settings(args, cfg)

    threshold_mode = (args.threshold_mode or "fixed").lower()
    input_path = Path(args.input).resolve()
    data_level = derive_data_level(input_path, args.data_level)

    if threshold_mode == "precomputed":
        precompute_key = build_precompute_key(settings["checker_mode"], data_level)
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
    ap = argparse.ArgumentParser(description="Task1 baseline: RefChecker (claim-level NLI)")
    ap.add_argument("--config", default=None, help="YAML config (default: ./config_baseline.yaml)")
    ap.add_argument("--input", required=True, help="Task1 JSONL under task_split/...")
    ap.add_argument("--source_data", required=True, help="Root dir of {paper_id}.json files")
    ap.add_argument("--out", required=True, help="Output ROOT dir")

    ap.add_argument("--evidence", choices=["no_retrieval", "BM25"], default=None)
    ap.add_argument("--bm25_top_k", type=int, default=None)
    ap.add_argument("--bm25_chunk_chars", type=int, default=None)
    ap.add_argument("--bm25_chunk_overlap", type=int, default=None)

    ap.add_argument("--checker_mode", choices=["alignscore", "llm"], default=None,
                    help="Which claim checker to use (alignscore | llm).")
    ap.add_argument("--extractor_model", default=None, help="Override extractor model name.")
    ap.add_argument("--extractor_port", type=int, default=None)
    ap.add_argument("--extractor_api_key", default=None,
                    help="API key for the extractor vLLM endpoint.")
    ap.add_argument("--checker_model", default=None, help="LLM-checker model (mode=llm only).")
    ap.add_argument("--checker_port", type=int, default=None, help="LLM-checker port (mode=llm only).")
    ap.add_argument("--checker_api_key", default=None,
                    help="API key for the LLM-checker vLLM endpoint (mode=llm only).")
    ap.add_argument("--alignscore_ckpt", default=None,
                    help="AlignScore checkpoint path (mode=alignscore only).")
    ap.add_argument("--device", default=None,
                    help="Device for AlignScore (e.g. cuda, cuda:0).")

    ap.add_argument("--threshold_mode", choices=["fixed", "precomputed"], default="fixed",
                    help="fixed: use --fixed_threshold / scoring.fixed_threshold. "
                         "precomputed: look up scoring.precompute_threshold_chker{mode}_level{...}; "
                         "if missing, score train.jsonl, write threshold to config, exit.")
    ap.add_argument("--fixed_threshold", type=float, default=None,
                    help="Entailment-ratio threshold; pred=0 if truth_score >= threshold.")
    ap.add_argument("--data_level", choices=["review", "sent"], default=None,
                    help="Override level tag in precompute key (auto-derived from --input otherwise).")
    ap.add_argument("--train_input", default=None,
                    help="Override train.jsonl path used by --threshold_mode precomputed.")

    ap.add_argument("--batch_size", type=int, default=None,
                    help="Pipeline batch size for extract+check.")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
