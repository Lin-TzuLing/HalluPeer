"""
Task 2 Zero-Shot Hallucination Classification
==============================================

Runs zero-shot hallucination TYPE classification on Task 2 review-level (or
sentence-level) JSONL data by querying an OpenAI-compatible chat endpoint
served by an externally-started vLLM instance.

Pipeline overview:
    1. Read the input JSONL (one record per review / sentence) under
       ``dataset/task_split/<dataset>/<task2_*>/``.
    2. Load the label space from ``predefined_nodes.json`` (first_level entries).
    3. For each record, locate ``{paper_id}.json`` under ``--source_data`` and
       extract evidence from its ``paper_content`` field:
         - ``no_retrieval``: concatenate every section (key + value) as one block.
         - ``BM25``: chunk every section, score chunks against the review text
           with Okapi BM25, and keep the top-k.
    4. Render the prompt template (``task2.txt``) by
       substituting ``{label_space}``, ``{evidence}``, and ``{sentence}``
       placeholders.
    5. Issue concurrent chat-completion requests under an ``asyncio.Semaphore``.
    6. Parse each response for a JSON ``"label": "<concept>"`` field and
       write a JSONL of ``{"id", "pred"}`` rows to a path derived from the
       input by preserving the sub-path that follows ``task_split``.

Config:
    Defaults live in ``./config.yaml`` (section ``zero_shot``). CLI flags
    override config values when supplied.

Usage:
    # Start vLLM externally:
    vllm serve Qwen/Qwen3-32B --port 8003

    # No-retrieval evidence + plain prompt (task2.txt):
    python -m evaluation.task_2.zeroshot \\
        --input dataset/task_split/HalluPeer_ICLR2019/task2_review/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_ICLR2019 \\
        --out evaluation/evaluation_results/task2/zeroshot \\
        --evidence no_retrieval \\
        --port 8003

    # BM25 retrieval + plain prompt (task2.txt):
    python -m evaluation.task_2.zeroshot \\
        --input dataset/task_split/HalluPeer_ICLR2019/task2_review/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_ICLR2019 \\
        --out evaluation/evaluation_results/task2/zeroshot \\
        --evidence BM25 --bm25_top_k 5 \\
        --port 8003

Output:
    Two files per run, under
    ``<--out>/<sub-path-after-task_split-with-stem-suffixed>``:

    * ``<stem>_result_CoT-{bool}_evidence-{type}_model-{Qwen3-32B}[_bm25-topk{k}-chunk{c}-ovl{o}].jsonl``
      — one JSON object per line: ``{"id": "task2_review_3203", "pred": "Attribution Failure"}``
      (``pred``: label string, or ``""`` when unparseable).
    * ``<same-stem>.config.json`` — sidecar with the full resolved run config.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import os

import yaml
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI


def _is_openai_gpt(model_name: str) -> bool:
    name = model_name.lower()
    return "gpt" in name and "gpt-oss" not in name


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
EVAL_DATASET_MARKER = "task_split"
CONFIG_SECTION = "zero_shot"

DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources/hallucination_taxonomy/predefined_nodes.json"
)


class Task2ZeroShotAgent:
    """OpenAI-compatible chat agent for Task 2 zero-shot classification."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        port: Optional[int],
        max_retries: int = 60,
        retry_delay: float = 0.1,
        think_mode: bool = False,
        timeout: int = 120,
        n: int = 1,
        api_key: str = "",
        reasoning_model: bool = False,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.think_mode = think_mode
        self.timeout = timeout
        self.n = n
        self.reasoning_model = reasoning_model
        if port is not None:
            self.client = AsyncOpenAI(api_key=api_key, base_url=f"http://localhost:{port}/v1")
        else:
            self.client = AsyncOpenAI(api_key=api_key)

    async def get_query(self, messages: List[Dict[str, str]]):
        for attempt in range(self.max_retries):
            try:
                if self.reasoning_model:
                    req_kwargs: Dict[str, Any] = dict(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_completion_tokens=self.max_new_tokens,
                        timeout=self.timeout,
                    )
                else:
                    req_kwargs = dict(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                        n=self.n,
                        timeout=self.timeout,
                    )
                if "Qwen3" in self.model_name:
                    req_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": self.think_mode}
                    }
                response = await self.client.chat.completions.create(**req_kwargs)
                if self.n > 1:
                    content = [
                        c.message.content or getattr(c.message, "reasoning_content", None) or ""
                        for c in response.choices
                    ]
                else:
                    msg = response.choices[0].message
                    content = msg.content or getattr(msg, "reasoning_content", None)
                return content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error: {e}, attempt {attempt}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    return f"Error: {e}"


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


def load_label_space(taxonomy_path: Path) -> tuple[List[Dict[str, str]], Set[str]]:
    """Load first_level taxonomy entries and return (nodes, valid_concepts set)."""
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    nodes: List[Dict[str, str]] = data.get("first_level", [])
    valid = {n["concept"] for n in nodes if "concept" in n}
    return nodes, valid


def format_label_space(nodes: List[Dict[str, str]]) -> str:
    """Render taxonomy nodes as a numbered list with concept + description."""
    lines: List[str] = []
    for i, n in enumerate(nodes, 1):
        concept = n.get("concept", "")
        desc = n.get("concept_desc", "")
        lines.append(f'{i}. concept: "{concept}"\n   concept_desc: "{desc}"')
    return "\n".join(lines)


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


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


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


def load_paper(
    source_root: Path,
    paper_id: str,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
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


def parse_pred(text: str, valid_labels: Set[str]) -> str:
    """Parse model output for the classification label.

    Searches for ``"label": "<value>"`` in the JSON response.
    Returns the matched label if it is in ``valid_labels``, the raw extracted
    string if not recognized, or ``""`` when nothing can be parsed.
    """
    if not text:
        return ""
    m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    return ""


def sanitize_model_name(model: str) -> str:
    return model.split("/")[-1]


def build_output_suffix(
    cot: bool,
    evidence: str,
    model: str,
    bm25_top_k: int,
    bm25_chunk_chars: int,
    bm25_chunk_overlap: int,
) -> str:
    parts = [
        "result",
        f"model-{sanitize_model_name(model)}",
        f"evidence-{evidence}",
        f"CoT-{cot}",
    ]
    return "_".join(parts)


def derive_output_path(input_path: Path, out_root: Path, suffix: str) -> Path:
    parts = input_path.parts
    idx = next(
        (
            i
            for i, p in enumerate(parts)
            if p == EVAL_DATASET_MARKER or p.startswith(EVAL_DATASET_MARKER + "_")
        ),
        None,
    )
    if idx is not None:
        sub = Path(*parts[idx + 1 :])
    else:
        sub = Path(input_path.name)
    stem = sub.stem
    ext = sub.suffix or ".jsonl"
    new_name = f"{stem}_{suffix}{ext}"
    return out_root / sub.parent / new_name


def build_prompt(template: str, label_space: str, evidence: str, sentence: str) -> str:
    """Substitute ``{label_space}``, ``{evidence}``, and ``{sentence}`` in the template."""
    return (
        template
        .replace("{label_space}", label_space)
        .replace("{evidence}", evidence)
        .replace("{sentence}", sentence)
    )


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    model_name = args.model or cfg_get(cfg, ["agent", "model_name"])
    if not model_name:
        raise SystemExit("Missing --model (or zero_shot.agent.model_name in config).")

    if _is_openai_gpt(model_name):
        port = None
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("GPT model requires --api_key or OPENAI_API_KEY env var.")
    else:
        port = int(args.port or cfg_get(cfg, ["agent", "port"], 8003))
        api_key = args.api_key or ""
        if not api_key:
            raise SystemExit("--api_key is required.")
    max_new_tokens = int(
        args.max_tokens if args.max_tokens is not None else cfg_get(cfg, ["agent", "max_new_tokens"], 512)
    )
    temperature = float(
        args.temperature if args.temperature is not None else cfg_get(cfg, ["agent", "temperature"], 0.0)
    )
    max_retries = int(cfg_get(cfg, ["agent", "max_retries"], 60))
    retry_delay = float(cfg_get(cfg, ["agent", "retry_delay"], 0.1))
    think_mode = bool(cfg_get(cfg, ["agent", "think_mode"], False))
    timeout = int(cfg_get(cfg, ["agent", "timeout"], 120))
    n = int(cfg_get(cfg, ["agent", "n"], 1))
    reasoning_model = bool(args.reasoning_model or cfg_get(cfg, ["agent", "reasoning_model"], False))

    evidence_type = args.evidence or cfg_get(cfg, ["evidence", "type"], "no_retrieval")
    bm25_top_k = int(args.bm25_top_k or cfg_get(cfg, ["evidence", "bm25_top_k"], 5))
    bm25_chunk_chars = int(args.bm25_chunk_chars or cfg_get(cfg, ["evidence", "bm25_chunk_chars"], 2000))
    bm25_chunk_overlap = int(
        args.bm25_chunk_overlap if args.bm25_chunk_overlap is not None
        else cfg_get(cfg, ["evidence", "bm25_chunk_overlap"], 200)
    )

    if args.prompt:
        prompt_path = args.prompt
    else:
        prompt_path = cfg_get(cfg, ["prompt", "path"])
    if not prompt_path:
        raise SystemExit("Missing prompt path (--prompt / prompt.path in config).")
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    cot_flag = ("cot" in Path(prompt_path).stem.lower())

    taxonomy_path = Path(args.taxonomy) if args.taxonomy else DEFAULT_TAXONOMY_PATH
    label_nodes, valid_labels = load_label_space(taxonomy_path)
    label_space_str = format_label_space(label_nodes)

    max_concurrent = int(args.max_concurrent or cfg_get(cfg, ["max_concurrent"], 80))

    input_path = Path(args.input).resolve()
    data = read_jsonl(str(input_path))
    if args.max_samples:
        data = data[: args.max_samples]

    source_root = Path(args.source_data)
    paper_cache: Dict[str, Dict[str, Any]] = {}

    messages_list: List[List[Dict[str, str]]] = []
    ids: List[Any] = []
    for r in tqdm(data, desc="Build prompts"):
        rid = r.get("task") or r.get("id") or r.get("sent_id")
        sentence = str(r.get("text") or r.get("sentence") or r.get("review_sentence") or "").strip()
        paper_id = str(r.get("paper_id") or "").strip()
        paper_content = load_paper(source_root, paper_id, paper_cache) if paper_id else {}
        if evidence_type == "BM25":
            evidence = bm25_evidence(paper_content, sentence, bm25_top_k, bm25_chunk_chars, bm25_chunk_overlap)
        else:
            evidence = join_paper_content(paper_content)
        prompt_text = build_prompt(prompt_template, label_space_str, evidence, sentence)
        messages_list.append([{"role": "user", "content": prompt_text}])
        ids.append(rid)

    agent = Task2ZeroShotAgent(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        port=port,
        max_retries=max_retries,
        retry_delay=retry_delay,
        think_mode=think_mode,
        timeout=timeout,
        n=n,
        api_key=api_key,
        reasoning_model=reasoning_model,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(idx: int, messages: List[Dict[str, str]]):
        async with semaphore:
            content = await agent.get_query(messages)
        if isinstance(content, list):
            content = content[0] if content else ""
        return idx, content or ""

    out_suffix = build_output_suffix(
        cot=cot_flag,
        evidence=evidence_type,
        model=model_name,
        bm25_top_k=bm25_top_k,
        bm25_chunk_chars=bm25_chunk_chars,
        bm25_chunk_overlap=bm25_chunk_overlap,
    )
    out_path = derive_output_path(input_path, Path(args.out).resolve(), out_suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results; only skip pred != "" so failures are retried.
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
                if rid is not None and obj.get("pred"):
                    existing[rid] = obj
                    done_ids.add(rid)
        print(f"[RESUME] {len(done_ids)} done (pred≠''); re-running pred='' and new.")

    pending_indices = [i for i, rid in enumerate(ids) if rid not in done_ids]
    print(f"[INFO] To process: {len(pending_indices)} / {len(ids)}")

    sidecar_path = out_path.with_suffix(".config.json")
    sidecar = {
        "input": str(input_path),
        "source_data": str(source_root),
        "output": str(out_path),
        "taxonomy": str(taxonomy_path),
        "label_space": [n.get("concept") for n in label_nodes],
        "evidence": {
            "type": evidence_type,
            **({"bm25_top_k": bm25_top_k, "bm25_chunk_chars": bm25_chunk_chars, "bm25_chunk_overlap": bm25_chunk_overlap}
               if evidence_type == "BM25" else {}),
        },
        "prompt": {"path": str(prompt_path), "cot": cot_flag},
        "agent": {
            "model_name": model_name,
            "port": port,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "think_mode": think_mode,
            "n": n,
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
        },
        "max_concurrent": max_concurrent,
        "max_samples": args.max_samples,
        "num_examples": len(ids),
        "resume": bool(args.resume),
        "done_at_start": len(done_ids),
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    if not pending_indices:
        # Rewrite clean: deduplicate, drop stale pred="" rows, preserve input order.
        fw = out_path.open("w", encoding="utf-8")
        for i, rid in enumerate(ids):
            if rid in existing:
                fw.write(json.dumps(existing[rid], ensure_ascii=False) + "\n")
        fw.close()
        print(f"[DONE] Nothing new to process. Output: {out_path}")
        print(f"[DONE] Run config: {sidecar_path}")
        return

    tasks = [_one(i, messages_list[i]) for i in pending_indices]
    pbar = tqdm(
        total=len(tasks),
        desc="Task2 zero-shot",
        unit="req",
        dynamic_ncols=True,
        mininterval=0.2,
    )

    # Write all ids in input order (existing + newly computed) to a fresh file.
    # Pre-populate buffer with existing (pred≠"") results; new results arrive via
    # asyncio.as_completed. Flush the longest contiguous prefix over ALL indices.
    buffer: Dict[int, tuple] = {}
    for i, rid in enumerate(ids):
        if rid in done_ids:
            row = existing[rid]
            buffer[i] = (row["pred"], row.get("raw_response", ""))

    fw = out_path.open("w", encoding="utf-8")
    next_global = 0
    try:
        for coro in asyncio.as_completed(tasks):
            idx, content = await coro
            pbar.update(1)
            raw = content if isinstance(content, str) else ""
            buffer[idx] = (parse_pred(raw, valid_labels), raw)
            while next_global < len(ids) and next_global in buffer:
                i = next_global
                pred, raw_response = buffer.pop(i)
                fw.write(json.dumps({"id": ids[i], "pred": pred, "raw_response": raw_response}, ensure_ascii=False) + "\n")
                fw.flush()
                next_global += 1
    finally:
        pbar.close()
        fw.close()

    print(f"[DONE] Wrote {next_global} rows total to {out_path}")
    print(f"[DONE] Run config: {sidecar_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2 zero-shot classification via OpenAI-compatible vLLM endpoint")
    ap.add_argument("--config", default=None, help="YAML config (default: ./config.yaml)")
    ap.add_argument("--input", required=True, help="Path to test JSONL (under task_split)")
    ap.add_argument("--source_data", required=True, help="Root dir of {paper_id}.json files")
    ap.add_argument("--out", required=True, help="Output ROOT dir; subpath after task_split is appended")
    ap.add_argument("--taxonomy", default=None, help="Path to predefined_nodes.json (default: resources/hallucination_taxonomy/predefined_nodes.json)")
    ap.add_argument("--prompt", default=None, help="Prompt template path (overrides config)")
    ap.add_argument("--evidence", choices=["no_retrieval", "BM25"], default=None)
    ap.add_argument("--model", default=None, help="Override agent.model_name")
    ap.add_argument("--port", type=int, default=None, help="Override agent.port")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_tokens", type=int, default=None, help="Override agent.max_new_tokens")
    ap.add_argument("--bm25_top_k", type=int, default=None)
    ap.add_argument("--bm25_chunk_chars", type=int, default=None)
    ap.add_argument("--bm25_chunk_overlap", type=int, default=None)
    ap.add_argument("--max_concurrent", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip already-done ids found in the output file (if any)")
    ap.add_argument("--reasoning_model", action="store_true", help="Use max_completion_tokens + temperature=1 for OpenAI reasoning models (o-series)")
    ap.add_argument("--api_key", default=None, help="OpenAI API key for GPT models (fallback: OPENAI_API_KEY env var)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
