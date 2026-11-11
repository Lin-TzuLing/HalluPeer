"""
Task 1 Zero-Shot Hallucination Detection
========================================

Runs zero-shot binary hallucination detection on Task 1 review-level (or
sentence-level) JSONL data by querying an OpenAI-compatible chat endpoint
served by an externally-started vLLM instance.

Pipeline overview:
    1. Read the input JSONL (one record per review / sentence) under
       ``dataset/task_split/<dataset>/<task1_*>/``.
    2. For each record, locate ``{paper_id}.json`` under ``--source_data`` and
       extract evidence from its ``paper_content`` field:
         - ``no_retrieval``: concatenate every section (key + value) as one block.
         - ``BM25``: chunk every section, score chunks against the review text
           with Okapi BM25, and keep the top-k.
    3. Render the prompt template (``task1.txt``) by
       substituting ``{evidence}`` and ``{sentence}`` placeholders.
    4. Issue concurrent chat-completion requests via :class:`Task1ZeroShotAgent`
       (an inline AsyncOpenAI client) under an ``asyncio.Semaphore``.
    5. Parse each response for a JSON ``"hallucination": true|false`` field and
       write a JSONL of ``{"id", "pred"}`` rows to a path derived from the
       input by preserving the sub-path that follows ``task_split``.

Config:
    Defaults live in ``./config.yaml`` (section ``zero_shot``). CLI flags
    override config values when supplied. Multiple baselines can share this
    YAML file by adding sibling top-level sections.

Usage:
    # ---------------------------------------------------------------------
    # Step 1. Start a vLLM OpenAI-compatible server externally (separate
    # process / terminal). Pick the same model name + port you will pass
    # to this script (or set them in config.yaml under zero_shot.agent).
    # ---------------------------------------------------------------------
    vllm serve Qwen/Qwen3-32B --port 8003

    # ---------------------------------------------------------------------
    # Step 2. Run zero-shot inference. CLI flags override config.yaml.
    # ---------------------------------------------------------------------

    # (a) No-retrieval evidence + plain prompt (task1.txt):
    python -m evaluation.task_1.zeroshot \\
        --input dataset/task_split/HalluPeer_ICLR2019/task1_review/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_ICLR2019 \\
        --out evaluation/evaluation_results/task1/zeroshot \\
        --evidence no_retrieval \\
        --port 8003

    # (b) BM25 retrieval + plain prompt (task1.txt):
    python -m evaluation.task_1.zeroshot \\
        --input dataset/task_split/HalluPeer_ICLR2019/task1_review/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_ICLR2019 \\
        --out evaluation/evaluation_results/task1/zeroshot \\
        --evidence BM25 --bm25_top_k 5 \\
        --port 8003

    # (c) Override model / config / sample budget for a quick smoke test:
    python -m evaluation.task_1.zeroshot \\
        --config evaluation/task_1/config.yaml \\
        --model Qwen/Qwen2.5-7B-Instruct --port 8004 \\
        --input  dataset/task_split/HalluPeer_ICLR2019/task1_review/test.jsonl \\
        --source_data dataset/template_injection/HalluPeer_ICLR2019 \\
        --out    evaluation/evaluation_results/task1/zeroshot \\
        --max_samples 10 --max_concurrent 8

Output:
    Two files per run, under
    ``<--out>/<sub-path-after-task_split-with-stem-suffixed>``:

    * ``<stem>_result_CoT-{bool}_evidence-{type}_model-{Qwen3-32B}[_bm25-topk{k}-chunk{c}-ovl{o}].jsonl``
      — one JSON object per line: ``{"id": "task1_review_5638", "pred": 1}``
      (``pred``: ``1`` hallucinated / ``0`` not / ``-1`` unparseable).
    * ``<same-stem>.config.json`` — sidecar with the full resolved run config
      (input, source_data, evidence + BM25 knobs, prompt + cot flag, agent
      params, max_concurrent, max_samples, num_examples) for reproducibility.

    The BM25 ``_bm25-topk{k}-chunk{c}-ovl{o}`` segment is appended only when
    ``--evidence BM25``. The model name has the HF org prefix stripped
    (``Qwen/Qwen3-32B`` → ``Qwen3-32B``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class Task1ZeroShotAgent:
    """OpenAI-compatible chat agent for Task1 zero-shot inference.

    Talks to a locally-served vLLM endpoint at http://localhost:{port}/v1.
    Mirrors the interface pattern used by hallupeer/dataset_construction/.../inject_template.py
    (AsyncOpenAI + chat.completions + retry + Qwen3 enable_thinking toggle).
    """

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
        """Chat-completion with simple fixed-delay retry. Returns content or 'Error: ...' string on exhaustion."""
        for attempt in range(self.max_retries):
            try:
                if self.reasoning_model:
                    req_kwargs: Dict[str, Any] = dict(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_completion_tokens=8192,
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
                    return [
                        c.message.content or getattr(c.message, "reasoning_content", None) or ""
                        for c in response.choices
                    ]
                msg = response.choices[0].message
                return msg.content or getattr(msg, "reasoning_content", None)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error: {e}, attempt {attempt}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    return f"Error: {e}"


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load and return the ``zero_shot`` section of the YAML config.

    Args:
        path: YAML file path. Falls back to :data:`DEFAULT_CONFIG`
            (``./config.yaml`` next to this script) when ``None``.

    Returns:
        The dict under ``zero_shot:``, or an empty dict if the file or
        section is missing.
    """
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    section = raw.get(CONFIG_SECTION)
    return section if isinstance(section, dict) else {}


def cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Nested-dict getter. ``cfg_get(cfg, ["a", "b"], d)`` ≡ ``cfg["a"]["b"]`` with default ``d``."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Blank lines are skipped."""
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def join_paper_content(paper_content: Dict[str, Any]) -> str:
    """Flatten a ``paper_content`` dict into a single evidence string.

    Each top-level entry becomes a ``"{key}\\n{value}"`` block, blocks joined
    by blank lines. Lists are joined line-by-line; nested dicts recurse.
    Used as the no-retrieval evidence and as the section-value source for
    BM25 chunking.
    """
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
    """Split ``text`` into sliding-window chunks of ``chunk_chars`` with ``overlap``.

    Character-based (not token-based) so it has no tokenizer dependency.
    Returns ``[text]`` when ``chunk_chars <= 0`` and ``[]`` when ``text`` is empty.
    """
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
    """Lowercase word tokenizer for BM25 (matches ``\\w+``)."""
    return _TOKEN_RE.findall(s.lower())


def bm25_evidence(
    paper_content: Dict[str, Any],
    query: str,
    top_k: int,
    chunk_chars: int,
    overlap: int,
) -> str:
    """Retrieve the top-k BM25 (Okapi) chunks from ``paper_content``.

    Each section value is chunked independently, then all chunks are scored
    against ``query`` (the review text). The top-k chunks (by descending
    score) are returned in their original document order, joined by blank
    lines. Falls back to the first ``top_k`` chunks when the query has no
    tokens, and to ``""`` when the paper has no content.
    """
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
    """Load ``source_root/{paper_id}.json`` and return its ``paper_content`` dict.

    Results are memoised in ``cache``. Missing files, JSON errors, or a
    non-dict ``paper_content`` all resolve to an empty dict (which still gets
    cached so we don't retry on every record).
    """
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


def parse_pred(text: str) -> int:
    """Parse the model output to a binary label.

    Tries in order:
      1. Full JSON parse — extract ``hallucination`` key.
      2. Regex ``"hallucination": true|false`` — handles markdown-wrapped JSON.
      3. Bare ``true``/``false`` token — last resort for non-JSON responses.
    Returns ``1`` hallucinated / ``0`` not / ``-1`` unparseable.
    Never raises.
    """
    if not text:
        return -1

    # Strip markdown code fences so json.loads works on ```json ... ``` outputs.
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "hallucination" in obj:
            val = obj["hallucination"]
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, str):
                if val.lower() == "true":
                    return 1
                if val.lower() == "false":
                    return 0
    except Exception:
        pass

    # Regex fallback: catches JSON embedded in prose / partial output.
    m = re.search(r'"hallucination"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return 1 if m.group(1).lower() == "true" else 0

    # Last resort: bare token — only safe when no reasoning text is present.
    if re.search(r"\btrue\b", text, re.IGNORECASE):
        return 1
    if re.search(r"\bfalse\b", text, re.IGNORECASE):
        return 0

    return -1


def sanitize_model_name(model: str) -> str:
    """Strip the HF org prefix so the model name is filesystem-safe.

    e.g. ``"Qwen/Qwen3-32B"`` -> ``"Qwen3-32B"``.
    """
    return model.split("/")[-1]


def build_output_suffix(
    cot: bool,
    evidence: str,
    model: str,
    bm25_top_k: int,
    bm25_chunk_chars: int,
    bm25_chunk_overlap: int,
) -> str:
    """Compose the run-identifying filename suffix from the main run axes.

    Format: ``result_CoT-{cot}_evidence-{evidence}_model-{model}`` with an
    appended ``_bm25-topk{k}-chunk{chars}-ovl{overlap}`` segment only when
    ``evidence == "BM25"``. Model name is sanitised via
    :func:`sanitize_model_name`.
    """
    parts = [
        "result",
        f"model-{sanitize_model_name(model)}",
        f"evidence-{evidence}",
        f"CoT-{cot}",
    ]
    return "_".join(parts)


def derive_output_path(input_path: Path, out_root: Path, suffix: str) -> Path:
    """Mirror the input's sub-path under ``out_root`` and tag with ``suffix``.

    Anything after the ``task_split`` segment of ``input_path`` is
    appended verbatim to ``out_root``; the input stem is then suffixed with
    ``_{suffix}`` while the extension is preserved. Example::

        input  = .../task_split/HalluPeer_ICLR2019/task1_review/test.jsonl
        out    = .../outputs_May/task1_zero_shot
        suffix = "result_CoT-True_evidence-BM25_model-Qwen3-32B_bm25-topk5-chunk2000-ovl200"
        result = .../outputs_May/task1_zero_shot/HalluPeer_ICLR2019/task1_review/
                   test_result_CoT-True_evidence-BM25_model-Qwen3-32B_bm25-topk5-chunk2000-ovl200.jsonl

    Falls back to ``out_root/<basename>`` if the marker is absent.
    """
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


def build_prompt(template: str, evidence: str, sentence: str) -> str:
    """Substitute ``{evidence}`` and ``{sentence}`` in the prompt template.

    Uses ``str.replace`` rather than ``str.format`` because the templates
    contain literal ``{`` / ``}`` from the required JSON output schema, which
    would otherwise break Python format strings.
    """
    return template.replace("{evidence}", evidence).replace("{sentence}", sentence)


async def run(args: argparse.Namespace) -> None:
    """End-to-end async entry point.

    Merges YAML config with CLI overrides, builds one prompt per input row,
    fans out chat-completion requests via :class:`Task1ZeroShotAgent` under
    a semaphore, parses each response, and writes ``{"id", "pred"}`` rows
    to the derived output JSONL path.
    """
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
        args.bm25_chunk_overlap if args.bm25_chunk_overlap is not None else cfg_get(cfg, ["evidence", "bm25_chunk_overlap"], 200)
    )

    if args.prompt:
        prompt_path = args.prompt
    else:
        prompt_path = cfg_get(cfg, ["prompt", "path"])
    if not prompt_path:
        raise SystemExit("Missing prompt path (--prompt / prompt.path in config).")
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    # CoT flag for filename: sniff prompt stem.
    cot_flag = ("cot" in Path(prompt_path).stem.lower())

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
        prompt_text = build_prompt(prompt_template, evidence, sentence)
        messages_list.append([{"role": "user", "content": prompt_text}])
        ids.append(rid)

    agent = Task1ZeroShotAgent(
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
        """Run one bounded request and tag it with its original index."""
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

    # Resume: load existing results; only skip pred != -1 so failures are retried.
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

    pending_indices = [i for i, rid in enumerate(ids) if rid not in done_ids]
    print(f"[INFO] To process: {len(pending_indices)} / {len(ids)}")

    # Sidecar config — written up front so an interrupted run still leaves a record.
    sidecar_path = out_path.with_suffix(".config.json")
    sidecar = {
        "input": str(input_path),
        "source_data": str(source_root),
        "output": str(out_path),
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
        # Rewrite clean: deduplicate, drop stale pred=-1 rows, preserve input order.
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
        desc="Task1 zero-shot",
        unit="req",
        dynamic_ncols=True,
        mininterval=0.2,
    )

    # Write all ids in input order (existing + newly computed) to a fresh file.
    # Pre-populate buffer with existing (pred≠-1) results; new results arrive via
    # asyncio.as_completed. Flush the longest contiguous prefix over ALL indices
    # (not just pending) so existing rows are written in the correct position.
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
            buffer[idx] = (parse_pred(raw), raw)
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
    """CLI entry point. Parses arguments and dispatches to :func:`run`."""
    ap = argparse.ArgumentParser(description="Task 1 zero-shot inference via OpenAI-compatible vLLM endpoint")
    ap.add_argument("--config", default=None, help="YAML config (default: ./config.yaml)")
    ap.add_argument("--input", required=True, help="Path to test JSONL (under task_split)")
    ap.add_argument("--source_data", required=True, help="Root dir of {paper_id}.json files")
    ap.add_argument("--out", required=True, help="Output ROOT dir; subpath after task_split is appended")
    ap.add_argument("--prompt", default=None, help="Prompt template path (overrides config)")
    ap.add_argument("--evidence", choices=["no_retrieval", "BM25"], default=None)
    ap.add_argument("--model", default=None, help="Override agent.model_name")
    ap.add_argument("--port", type=int, default=None, help="Override agent.port (OpenAI-compatible endpoint)")
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
