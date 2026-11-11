"""Build fewshot examples for RA-LLM_KR prompt.

For each granularity (review-level / sent-level) we:
  1. Load train.jsonl.
  2. Pick one example per label_depth1 category (None == non-hallucinated label=0;
     plus the 9 hallucination types) to keep the fewshot set balanced.
  3. For each picked example, chunk the corresponding paper content (small word
     chunks so several fewshots still fit) and use Okapi BM25 to retrieve the
     top-1 chunk against the example's review text -> use it as the evidence.
  4. Render an aligned fewshot block (paper evidence + review sentence + judgement)
     and write the final prompts to RA-LLM_KR_review.txt / RA-LLM_KR_sent.txt.

Run:
    python RA-LLM_KR_fewshot_construct.py
"""
import json
import random
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "dataset/task_split/HalluPeer_NeurIPS2024"
# Paper text = "\n".join(f"{section}: {text}") over `paper_content` of {paper_id}.json
PAPER_DIR = ROOT / "dataset/template_injection/HalluPeer_NeurIPS2024"
PROMPT_DIR = ROOT / "evaluation/prompts/task1"

# Balanced sampling over label_depth1 (None = non-hallucinated, label==0).
DEPTH1_TYPES = [
    None,
    "Reasoning Error",
    "Context-based meaning error",
    "False Concatenation",
    "Entity",
    "Temporal",
    "Attribution Failure",
    "Hyperbole",
    "Number",
    "Overgeneralization",
]

# Small chunks so several fewshots still fit in the final prompt.
CHUNK_WORDS = 80
CHUNK_STRIDE = 60
SEED = 42

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    return [w.lower() for w in _WORD_RE.findall(text)]


def chunk_paper(text: str, size: int = CHUNK_WORDS, stride: int = CHUNK_STRIDE):
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
        i += stride
    return chunks


def load_jsonl(fp: Path):
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_paper_cache: dict = {}


def load_paper(paper_id: str) -> str:
    if paper_id in _paper_cache:
        return _paper_cache[paper_id]
    content = json.loads((PAPER_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))["paper_content"]
    text = "\n".join(f"{k}: {v}" for k, v in content.items())
    _paper_cache[paper_id] = text
    return text


def bm25_top_chunk(paper_text: str, query: str) -> str:
    chunks = chunk_paper(paper_text)
    if not chunks:
        return ""
    tokenized = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))
    best = max(range(len(scores)), key=lambda i: scores[i])
    return chunks[best]


def pick_example(rows, depth1, rng):
    """Pick one example for the given label_depth1 bucket with a usable paper file."""
    if depth1 is None:
        pool = [r for r in rows if r["label"] == 0]
    else:
        pool = [r for r in rows if r.get("metadata", {}).get("label_depth1") == depth1]
    rng.shuffle(pool)
    for r in pool:
        if (PAPER_DIR / f"{r['paper_id']}.json").exists():
            return r
    return None


def build_fewshots(rows, rng):
    examples = []
    for d1 in DEPTH1_TYPES:
        ex = pick_example(rows, d1, rng)
        if ex is None:
            print(f"  [warn] no example available for depth1={d1!r}")
            continue
        evidence = bm25_top_chunk(load_paper(ex["paper_id"]), ex["text"])
        examples.append(
            {
                "id": ex["id"],
                "paper_id": ex["paper_id"],
                "label": ex["label"],
                "label_depth1": ex.get("metadata", {}).get("label_depth1"),
                "review_text": ex["text"],
                "evidence": evidence,
                "judgement": {"hallucination": bool(ex["label"])},
            }
        )
    return examples


def render_fewshot_block(idx: int, fs: dict) -> str:
    return (
        f"### Example {idx} (label_depth1: {fs['label_depth1']})\n"
        f"#Paper Evidence#: \"\"\"{fs['evidence']}\"\"\"\n"
        f"#Review Sentence#: \"\"\"{fs['review_text']}\"\"\"\n"
        f"#Your Judgement#: {json.dumps(fs['judgement'])}"
    )


PROMPT_HEADER = """You are a scientific peer-review hallucination detector. Given paper evidence and a {unit}, your objective is to determine if the {unit} contains non-factual or hallucinated information relative to the paper. You SHOULD give your judgement based on the following hallucination types and the paper evidence.

## Definitions & Constraints
- A review claim is hallucinated if it is factually unsupported by, or incorrect relative to, the paper content.
- EXCLUSIONS: Do not evaluate tone (e.g., harshness). Do NOT treat subjective judgments, critiques, or opinions (e.g., whether novelty is high) as hallucinations. Focus strictly on factual grounding.
- Hallucination types to consider: Reasoning Error, Context-based meaning error, False Concatenation, Entity, Temporal, Attribution Failure, Hyperbole, Number, Overgeneralization.

## Few-shot Examples
Each example shows a retrieved paper-evidence chunk, the {unit}, and the gold judgement.

"""

PROMPT_TAIL = """

## Inputs
### PAPER EVIDENCE
\"\"\"{{evidence}}\"\"\"

### {section_label}
\"\"\"{{sentence}}\"\"\"

## Output Format
Return a valid JSON object ONLY, using EXACTLY the following template:

{{
  "hallucination": <true or false>
}}

## Instructions
- "true" = the {unit} contains a factual claim that is hallucinated (unsupported or incorrect relative to the evidence).
- "false" = the {unit} is factually supported by and consistent with the evidence, OR the {unit} is purely a subjective judgment/critique.
- Output ONLY the JSON object. Do NOT include markdown tags like ```json.
"""


def render_prompt(fewshots, unit: str, section_label: str) -> str:
    header = PROMPT_HEADER.format(unit=unit)
    body = "\n\n".join(render_fewshot_block(i, fs) for i, fs in enumerate(fewshots, 1))
    tail = PROMPT_TAIL.format(unit=unit, section_label=section_label)
    return header + body + tail


def main():
    rng = random.Random(SEED)

    print("[sent-level]")
    sent_rows = load_jsonl(DATA_DIR / "task1_sent/train.jsonl")
    sent_fs = build_fewshots(sent_rows, rng)

    print("[review-level]")
    review_rows = load_jsonl(DATA_DIR / "task1_review/train.jsonl")
    review_fs = build_fewshots(review_rows, rng)

    (PROMPT_DIR / "RA-LLM_KR_sent.txt").write_text(
        render_prompt(sent_fs, unit="review sentence", section_label="REVIEW SENTENCE"),
        encoding="utf-8",
    )
    (PROMPT_DIR / "RA-LLM_KR_review.txt").write_text(
        render_prompt(review_fs, unit="review", section_label="REVIEW"),
        encoding="utf-8",
    )

    print(f"sent fewshots picked  : {len(sent_fs)} -> RA-LLM_KR_sent.txt")
    print(f"review fewshots picked: {len(review_fs)} -> RA-LLM_KR_review.txt")


if __name__ == "__main__":
    main()
