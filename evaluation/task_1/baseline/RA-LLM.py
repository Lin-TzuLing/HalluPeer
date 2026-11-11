"""
Task 1 baseline — RA-LLM (hallucination judge w/ BM25 / LCS retrieval)
=========================================================================

For each input record:
  1. Build paper text from {paper_id}.json (full sections).
  2. Retrieve evidence:
       no_retrieval → full paper_content as one block
       BM25         → Okapi BM25 top-k chunks (rank_bm25)
       LCS          → top-k chunks ranked by word-level LCS with the query
  3. Fill prompt template ({evidence}, {sentence}); call LLM.
  4. Parse JSON {"hallucination": bool} → pred 1/0/-1.

Config:
    Defaults in ``./config_baseline.yaml`` under section ``ra_llm``.

Usage:
    vllm serve Qwen/Qwen3-32B --port 8003
    python evaluation/task_1/baseline/RA-LLM.py \\
        --input  dataset/task_split/.../task1_sent/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_NeurIPS2024 \\
        --out    evaluation/evaluation_results/task1/ra_llm \\
        --evidence BM25 --mode kr --model Qwen/Qwen3-32B --port 8003
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_baseline.yaml"
EVAL_DATASET_MARKER = "task_split"
CONFIG_SECTION = "ra_llm"

PROMPTS_DIR = SCRIPT_DIR.parents[2] / "evaluation" / "prompts" / "task1"
_PROMPT_FILE = {
    "cot":      "RA-LLM_CoT.txt",
    "contrast": "RA-LLM_Contrast.txt",
}


# =============================================================================
# Config / IO helpers
# =============================================================================

def load_config(path: Optional[str]) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    section = raw.get(CONFIG_SECTION)
    return section if isinstance(section, dict) else {}


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
    stem = sub.stem
    ext = sub.suffix or ".jsonl"
    return out_root / sub.parent / f"{stem}_{suffix}{ext}"


def build_output_suffix(model: str, evidence: str, mode: str, top_k: int) -> str:
    return "_".join([
        "result",
        f"model-{sanitize_model_name(model)}",
        f"evidence-{evidence}",
        f"mode-{mode}",
        f"retk-{top_k}",
    ])


# =============================================================================
# Paper helpers  (mirrored from zero_shot.py)
# =============================================================================

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


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


def _build_chunks(paper_content: Dict[str, Any], chunk_chars: int, overlap: int) -> List[str]:
    """Chunk paper_content into section-labelled sliding windows."""
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
    return chunks


def bm25_evidence(
    paper_content: Dict[str, Any],
    query: str,
    top_k: int,
    chunk_chars: int,
    overlap: int,
) -> str:
    chunks = _build_chunks(paper_content, chunk_chars, overlap)
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


def lcs_evidence(
    paper_content: Dict[str, Any],
    query: str,
    top_k: int,
    chunk_chars: int,
    overlap: int,
) -> str:
    """Top-k chunks ranked by word-level LCS length with the query sentence.

    Uses difflib.SequenceMatcher (C-backed) to sum matching-block sizes,
    which equals the LCS length of the two token sequences.
    Ties broken by original document order; result chunks re-sorted by
    original order before joining so the returned evidence is contiguous.
    """
    chunks = _build_chunks(paper_content, chunk_chars, overlap)
    if not chunks:
        return ""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return "\n\n".join(chunks[:top_k])
    scored: List[tuple] = []
    for i, ch in enumerate(chunks):
        ch_tokens = _tokenize(ch)
        lcs = sum(
            m.size
            for m in SequenceMatcher(None, q_tokens, ch_tokens, autojunk=False).get_matching_blocks()
        )
        scored.append((lcs, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    order = sorted(idx for _, idx in scored[:top_k])
    return "\n\n".join(chunks[i] for i in order)


# =============================================================================
# Prompt / response helpers
# =============================================================================

def parse_pred(text: str) -> int:
    """Parse JSON {"hallucination": bool}. Returns 1/0/-1."""
    if not text:
        return -1
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "hallucination" in obj:
            val = obj["hallucination"]
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, str):
                return 1 if val.lower() == "true" else (0 if val.lower() == "false" else -1)
    except Exception:
        pass
    m = re.search(r'"hallucination"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return 1 if m.group(1).lower() == "true" else 0
    if re.search(r"\btrue\b", text, re.IGNORECASE):
        return 1
    if re.search(r"\bfalse\b", text, re.IGNORECASE):
        return 0
    return -1


# =============================================================================
# Async LLM agent
# =============================================================================

class RAAgent:
    def __init__(
        self,
        model_name: str,
        port: int,
        api_key: str,
        temperature: float,
        max_new_tokens: int,
        max_retries: int,
        retry_delay: float,
        timeout: int,
        n: int,
        think_mode: bool,
        reasoning_model: bool,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.n = n
        self.think_mode = think_mode
        self.reasoning_model = reasoning_model
        self.client = AsyncOpenAI(api_key=api_key, base_url=f"http://localhost:{port}/v1")

    async def get_query(self, messages: List[Dict[str, str]]) -> str:
        for attempt in range(self.max_retries):
            try:
                if self.reasoning_model:
                    req: Dict[str, Any] = dict(
                        model=self.model_name, messages=messages,
                        temperature=0.0, max_completion_tokens=8192,
                        timeout=self.timeout,
                    )
                else:
                    req = dict(
                        model=self.model_name, messages=messages,
                        temperature=self.temperature, max_tokens=self.max_new_tokens,
                        n=self.n, timeout=self.timeout,
                    )
                if "Qwen3" in self.model_name:
                    req["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": self.think_mode}
                    }
                response = await self.client.chat.completions.create(**req)
                msg = response.choices[0].message
                return msg.content or getattr(msg, "reasoning_content", None) or ""
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error: {e}, attempt {attempt}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    return f"Error: {e}"


# =============================================================================
# Main async loop
# =============================================================================

async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    model_name = args.model or cfg_get(cfg, ["agent", "model_name"])
    if not model_name:
        raise SystemExit("Missing --model (or ra_llm.agent.model_name in config).")
    port = int(args.port or cfg_get(cfg, ["agent", "port"], 8003))
    api_key = str(cfg_get(cfg, ["agent", "api_key"], "abc"))
    temperature = float(
        args.temperature if args.temperature is not None
        else cfg_get(cfg, ["agent", "temperature"], 0.0)
    )
    max_new_tokens = int(args.max_tokens or cfg_get(cfg, ["agent", "max_new_tokens"], 256))
    max_retries = int(cfg_get(cfg, ["agent", "max_retries"], 10))
    retry_delay = float(cfg_get(cfg, ["agent", "retry_delay"], 1.0))
    timeout = int(cfg_get(cfg, ["agent", "timeout"], 120))
    n = int(cfg_get(cfg, ["agent", "n"], 1))
    think_mode = bool(cfg_get(cfg, ["agent", "think_mode"], False))
    reasoning_model = bool(args.reasoning_model or cfg_get(cfg, ["agent", "reasoning_model"], False))

    evidence_type = args.evidence or cfg_get(cfg, ["evidence", "type"], "BM25")
    bm25_top_k = int(args.bm25_top_k or cfg_get(cfg, ["evidence", "bm25_top_k"], 5))
    bm25_chunk_chars = int(args.bm25_chunk_chars or cfg_get(cfg, ["evidence", "bm25_chunk_chars"], 2000))
    bm25_chunk_overlap = int(
        args.bm25_chunk_overlap if args.bm25_chunk_overlap is not None
        else cfg_get(cfg, ["evidence", "bm25_chunk_overlap"], 200)
    )

    mode = args.mode or cfg_get(cfg, ["scoring", "mode"], "kr")
    if mode not in ("kr", "cot", "contrast"):
        raise SystemExit(f"--mode must be kr|cot|contrast, got: {mode}")

    max_concurrent = int(args.max_concurrent or cfg_get(cfg, ["max_concurrent"], 80))

    input_path = Path(args.input).resolve()

    if mode == "kr":
        input_str = str(input_path)
        if "task1_review" in input_str:
            prompt_path = PROMPTS_DIR / "RA-LLM_KR_review.txt"
        elif "task1_sent" in input_str:
            prompt_path = PROMPTS_DIR / "RA-LLM_KR_sent.txt"
        else:
            raise SystemExit("--mode kr expects a task1_review or task1_sent input")
    else:
        prompt_path = PROMPTS_DIR / _PROMPT_FILE[mode]
    if not prompt_path.exists():
        raise SystemExit(f"Prompt template not found: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")

    data = read_jsonl(str(input_path))
    if args.max_samples:
        data = data[: args.max_samples]

    source_root = Path(args.source_data)
    paper_cache: Dict[str, Dict[str, Any]] = {}

    out_root = Path(args.out).resolve()
    out_suffix = build_output_suffix(model_name, evidence_type, mode, bm25_top_k)
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

    # ----- Build prompts -----
    messages_list: List[List[Dict[str, str]]] = []
    ids: List[Any] = []
    for r in tqdm(data, desc="Build prompts"):
        rid = r.get("task") or r.get("id") or r.get("sent_id")
        sentence = str(r.get("text") or r.get("sentence") or r.get("review_sentence") or "").strip()
        paper_id = str(r.get("paper_id") or "").strip()
        paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}

        if evidence_type == "BM25":
            evidence = bm25_evidence(paper_content, sentence, bm25_top_k, bm25_chunk_chars, bm25_chunk_overlap)
        elif evidence_type == "LCS":
            evidence = lcs_evidence(paper_content, sentence, bm25_top_k, bm25_chunk_chars, bm25_chunk_overlap)
        else:  # no_retrieval
            evidence = join_paper_content(paper_content)

        prompt_text = prompt_template.replace("{evidence}", evidence).replace("{sentence}", sentence)
        messages_list.append([{"role": "user", "content": prompt_text}])
        ids.append(rid)

    # ----- Sidecar config -----
    sidecar_path = out_path.with_suffix(".config.json")
    sidecar = {
        "input": str(input_path),
        "source_data": str(source_root),
        "output": str(out_path),
        "evidence": {
            "type": evidence_type,
            **({"bm25_top_k": bm25_top_k, "bm25_chunk_chars": bm25_chunk_chars,
                "bm25_chunk_overlap": bm25_chunk_overlap}
               if evidence_type != "no_retrieval" else {}),
        },
        "agent": {
            "model_name": model_name, "port": port,
            "temperature": temperature, "max_new_tokens": max_new_tokens,
            "max_retries": max_retries, "retry_delay": retry_delay,
            "timeout": timeout, "n": n, "think_mode": think_mode,
            "reasoning_model": reasoning_model,
        },
        "scoring": {"mode": mode},
        "prompt_path": str(prompt_path),
        "max_concurrent": max_concurrent,
        "max_samples": args.max_samples,
        "num_examples": len(ids),
        "resume": bool(args.resume),
        "done_at_start": len(done_ids),
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    pending_indices = [i for i, rid in enumerate(ids) if rid not in done_ids]
    print(f"[INFO] To process: {len(pending_indices)} / {len(ids)}")

    if not pending_indices:
        fw = out_path.open("w", encoding="utf-8")
        for i, rid in enumerate(ids):
            if rid in existing:
                fw.write(json.dumps(existing[rid], ensure_ascii=False) + "\n")
        fw.close()
        print(f"[DONE] Nothing new. Output: {out_path}")
        return

    # ----- Dispatch -----
    agent = RAAgent(
        model_name=model_name, port=port, api_key=api_key,
        temperature=temperature, max_new_tokens=max_new_tokens,
        max_retries=max_retries, retry_delay=retry_delay,
        timeout=timeout, n=n, think_mode=think_mode,
        reasoning_model=reasoning_model,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(idx: int, messages: List[Dict[str, str]]):
        async with semaphore:
            content = await agent.get_query(messages)
        return idx, content or ""

    buffer: Dict[int, tuple] = {}
    for i, rid in enumerate(ids):
        if rid in done_ids:
            row = existing[rid]
            buffer[i] = (row["pred"], row.get("raw_response", ""))

    tasks = [_one(i, messages_list[i]) for i in pending_indices]
    pbar = tqdm(total=len(tasks), desc=f"RA-LLM {mode}", unit="req",
                dynamic_ncols=True, mininterval=0.2)

    fw = out_path.open("w", encoding="utf-8")
    next_global = 0
    try:
        for coro in asyncio.as_completed(tasks):
            idx, content = await coro
            pbar.update(1)
            raw = content if isinstance(content, str) else ""
            buffer[idx] = (parse_pred(raw), raw)
            while next_global < len(ids) and next_global in buffer:
                i = next_global
                pred, raw_response = buffer.pop(i)
                fw.write(json.dumps({"id": ids[i], "pred": pred, "raw_response": raw_response},
                                    ensure_ascii=False) + "\n")
                fw.flush()
                next_global += 1
    finally:
        pbar.close()
        fw.close()

    print(f"[DONE] Wrote {next_global} rows → {out_path}")
    print(f"[DONE] Run config: {sidecar_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task1 baseline: RA-LLM (JSON hallucination judge)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--source_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", choices=["no_retrieval", "BM25", "LCS"], default=None)
    ap.add_argument("--mode", choices=["kr", "cot", "contrast"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--bm25_top_k", type=int, default=None)
    ap.add_argument("--bm25_chunk_chars", type=int, default=None)
    ap.add_argument("--bm25_chunk_overlap", type=int, default=None)
    ap.add_argument("--max_concurrent", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reasoning_model", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
