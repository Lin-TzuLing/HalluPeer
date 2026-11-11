"""
Similar Node Filter for Hallucination Taxonomy

Reads a taxonomy pickle, collects all nodes at depth > 1,
pairs them, sends each pair to an LLM using similar_filter.txt,
and outputs a list of dicts sorted with is_synonym=True first.

Usage:
    python -m hallupeer.hallucination_taxonomy.similar_filter \
        --input_pkl PATH_TO_ENSEMBLE.pkl \
        --output_path OUTPUT.json \
        [--min_depth 2] \
        [--concurrency 8] \
        [--model_name MODEL] \
        [--port PORT]
"""

import argparse
import asyncio
import json
import pickle
import re
from itertools import combinations

from tqdm.asyncio import tqdm

from hallupeer.agents import load_agent
from hallupeer.utils import load_configs, load_prompt_template


# ============================================================================
# Helpers
# ============================================================================


def _format_prompt(template: str, **kwargs) -> str:
    """Safe substitution: replaces only known {key} patterns, leaves other braces intact."""
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def get_all_nodes(root, min_depth: int = 2) -> list:
    """BFS to collect all nodes at depth >= min_depth."""
    nodes, queue = [], [root]
    while queue:
        node = queue.pop(0)
        if node.depth >= min_depth:
            nodes.append(node)
        queue.extend(node.children)
    return nodes


def get_ancestor_path(node) -> list:
    path, current = [], node
    while current is not None:
        path.append({"concept": current.concept, "description": current.concept_desc})
        current = current.parent
    return list(reversed(path))


def _repair_truncated_json(text: str):
    """Attempt to close a truncated JSON object by appending missing terminators."""
    for suffix in ('"}', '"]}', '}', ']}'):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue
    return None


def _extract_fields_regex(text: str):
    """Last-resort: extract is_synonym / keep / explanation individually via regex."""
    is_syn = re.search(r'"is_synonym"\s*:\s*(true|false)', text, re.IGNORECASE)
    keep = re.search(r'"keep"\s*:\s*"([^"]*)"', text)
    expl = re.search(r'"explanation"\s*:\s*"(.*?)(?:"|$)', text, re.DOTALL)
    if is_syn is None:
        return None
    return {
        "is_synonym": is_syn.group(1).lower() == "true",
        "keep": keep.group(1) if keep else "NA",
        "explanation": expl.group(1).strip() if expl else "NA",
    }


def extract_json_object(text: str):
    """Extract first valid JSON object from LLM response, handling Python bool literals."""
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)

    cleaned = re.sub(r"```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)

    # Strategy 1: find all {...} blobs, pick last valid
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL))):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue

    # Strategy 2: bracket-matched scan from first {
    pos = cleaned.find("{")
    while pos >= 0:
        depth = 0
        for i in range(pos, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[pos : i + 1])
                    except json.JSONDecodeError:
                        break
        pos = cleaned.find("{", pos + 1)

    # Strategy 3: repair truncated JSON (missing closing " and/or })
    start = cleaned.find("{")
    if start >= 0:
        repaired = _repair_truncated_json(cleaned[start:])
        if repaired is not None:
            return repaired

    # Strategy 4: regex field extraction
    return _extract_fields_regex(cleaned)


# ============================================================================
# Pipeline
# ============================================================================


class SimilarFilterPipeline:
    def __init__(self, generator, prompt_template: str, concurrency: int = 8):
        self.generator = generator
        self.prompt_template = prompt_template
        self.semaphore = asyncio.Semaphore(concurrency)

    def _prepare_prompt(self, node_a, node_b) -> str:
        return _format_prompt(
            self.prompt_template,
            concept_A=node_a.concept,
            description_A=node_a.concept_desc,
            path_A=json.dumps(get_ancestor_path(node_a), ensure_ascii=False),
            depth_A=node_a.depth,
            concept_B=node_b.concept,
            description_B=node_b.concept_desc,
            path_B=json.dumps(get_ancestor_path(node_b), ensure_ascii=False),
            depth_B=node_b.depth,
        )

    async def _check_pair(self, node_a, node_b) -> dict | None:
        async with self.semaphore:
            prompt = self._prepare_prompt(node_a, node_b)
            messages = [[{"role": "user", "content": prompt}]]
            response, _ = await self.generator.predict(messages)
            parsed = extract_json_object(response[0])
            if parsed is None:
                print(f"Failed to parse response for: {node_a.concept!r} / {node_b.concept!r}")
            return {
                "node_A": {
                    "uuid": node_a.uuid,
                    "concept": node_a.concept,
                    "concept_desc": node_a.concept_desc,
                    "depth": node_a.depth,
                    "path": get_ancestor_path(node_a),
                },
                "node_B": {
                    "uuid": node_b.uuid,
                    "concept": node_b.concept,
                    "concept_desc": node_b.concept_desc,
                    "depth": node_b.depth,
                    "path": get_ancestor_path(node_b),
                },
                "is_synonym": bool(parsed.get("is_synonym", False)) if parsed else None,
                "keep": parsed.get("keep", "NA") if parsed else "NA",
                "explanation": parsed.get("explanation", "NA") if parsed else "NA",
                "raw_response": response[0] if parsed is None else None,
            }

    async def run(self, nodes: list) -> list:
        def is_ancestor_of(anc, node):
            cur = node.parent
            while cur is not None:
                if cur.uuid == anc.uuid:
                    return True
                cur = cur.parent
            return False

        pairs = [
            (a, b) for a, b in combinations(nodes, 2)
            if not is_ancestor_of(a, b) and not is_ancestor_of(b, a)
        ]
        print(f"Nodes collected: {len(nodes)}, Pairs to check (excl. ancestor-descendant): {len(pairs)}")

        tasks = [self._check_pair(a, b) for a, b in pairs]
        results = await tqdm.gather(*tasks, desc="Checking pairs")

        results = [r for r in results if r is not None]
        results.sort(key=lambda x: (x["is_synonym"] is not True, x["is_synonym"] is None, x["node_A"]["concept"]))
        return results


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Filter near-synonym nodes in taxonomy")
    parser.add_argument("--input_pkl", type=str, required=True, help="Input taxonomy pickle path")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path")
    parser.add_argument("--min_depth", type=int, default=2, help="Minimum node depth to include (default: 2)")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent LLM calls (default: 8)")
    parser.add_argument("--model_name", type=str, default=None, help="Override model name")
    parser.add_argument("--port", type=int, default=None, help="Override model port")
    return parser.parse_args()


async def main():
    args = parse_args()
    cfg = load_configs(["configs/agents.yaml", "configs/taxonomy_generation.yaml"])

    if args.model_name is not None:
        cfg.agents.taxonomy_generator.model_name = args.model_name
    if args.port is not None:
        cfg.agents.taxonomy_generator.port = args.port

    with open(args.input_pkl, "rb") as f:
        root = pickle.load(f)

    nodes = get_all_nodes(root, min_depth=args.min_depth)
    print(f"Nodes at depth >= {args.min_depth}: {len(nodes)}")

    prompt_template = load_prompt_template("prompts/hallucination_taxonomy/similar_filter.txt")
    generator = load_agent(agent="taxonomy_generator", cfg=cfg)

    pipeline = SimilarFilterPipeline(
        generator=generator,
        prompt_template=prompt_template,
        concurrency=args.concurrency,
    )

    results = await pipeline.run(nodes)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    synonym_count = sum(1 for r in results if r["is_synonym"])
    print(f"Done. {synonym_count}/{len(results)} synonym pairs. Saved to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(main())
