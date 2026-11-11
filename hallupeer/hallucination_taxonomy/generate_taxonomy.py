"""
Hierarchical Generating Hallucination Taxonomy

This module contains classes and functions for generating hierarchical
hallucination taxonomy trees using LLM-based generation.

Usage:
    python -m hallupeer.hallucination_taxonomy.generate_taxonomy \
        --model_name MODEL_NAME \
        --port PORT \
        --max_width MAX_WIDTH \
        --max_depth MAX_DEPTH \
        --resume \
        --output_path OUTPUT_PATH \
        --usage_path USAGE_PATH
"""

import argparse
import asyncio
import json
import os
import pickle
import re

from tqdm.asyncio import tqdm

from hallupeer.agents import load_agent
from hallupeer.hallucination_taxonomy import HallucinationTaxonomyTreeNode
from hallupeer.utils import load_configs, load_json_configs, load_prompt_template

# ============================================================================
# Tree Generation Functions
# ============================================================================


class TaxonomyGenerationPipeline:
    def __init__(self, generator, prompt, max_width, max_depth, resume=False, **kwargs):
        self.generator = generator
        self.prompt = prompt
        self.max_width = max_width
        self.max_depth = max_depth

        if resume:
            # Load existing tree from pickle
            with open(kwargs.taxonomy_generation.resume_from_pickle, "rb") as f:
                self.root_node = pickle.load(f)
            print(
                f"Resumed taxonomy tree from pickle: {kwargs.taxonomy_generation.resume_from_pickle}"
            )
        else:
            # Load predefined nodes
            self.root_node = self.initialize_predefined_nodes(
                file_path=kwargs["taxonomy_generation"]["predefined_nodes_path"]
            )
            self.root_node.print_tree()

        # Load prompt template
        self.prompt_template = load_prompt_template(
            kwargs["taxonomy_generation"]["prompt_template_path"]
        )

        # generate subtree usage tracking
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "max_total_tokens": 0,
        }

        # Add progress tracking
        self.processed_nodes = 0

    def initialize_predefined_nodes(self, file_path: str):
        """Initialize predefined nodes for the taxonomy tree."""

        predefined_nodes = load_json_configs(file_path)
        root_concept_data = predefined_nodes["root"]
        first_level_datas = predefined_nodes["first_level"]

        root_node = HallucinationTaxonomyTreeNode(
            concept=root_concept_data["concept"],
            concept_desc=root_concept_data["concept_desc"],
            depth=0,
        )

        for first_level_data in first_level_datas:
            child_node = HallucinationTaxonomyTreeNode(
                concept=first_level_data["concept"],
                concept_desc=first_level_data["concept_desc"],
                depth=1,
                parent=root_node,
            )
            root_node.add_child(child_node)

        root_node.mark_processed()
        return root_node

    def _get_ancestor_path(self, node):
        path = []
        current = node
        while current is not None:
            path.append({"concept": current.concept, "description": current.concept_desc})
            current = current.parent
        return list(reversed(path))

    def _prepare_prompt(self, node):
        ancestor_path = self._get_ancestor_path(node)
        existing_children = [
            {"concept": c.concept, "description": c.concept_desc}
            for c in node.children
        ]
        prompt_filled = self.prompt_template.format(
            ancestor_path=json.dumps(ancestor_path, ensure_ascii=False),
            parent_concept=json.dumps(node.concept, ensure_ascii=False),
            parent_concept_desc=json.dumps(node.concept_desc, ensure_ascii=False),
            current_depth=node.depth,
            sibling_concepts=json.dumps(existing_children, ensure_ascii=False),
            max_width=self.max_width,
        )
        return prompt_filled

    async def generate_subtree(self, node):
        """
        Recursively generate subtree for a given node.

        Args:
            node: The node to generate subtree for
            generator: HallucinationTaxonomyGenerator instance
            prompt: Prompt template
            max_width: Maximum width of each node
            max_depth: Maximum depth of the tree
        """

        print(f"Generating subtree for root concept: {node.concept}, Depth: {node.depth}")

        if node.depth >= self.max_depth:
            return
        else:

            if node._processed:
                for child in node.children:
                    await self.generate_subtree(child)
            else:

                prompt_filled = self._prepare_prompt(node)
                messages = [[{"role": "user", "content": prompt_filled}]]
                response, usage = await self.generator.predict(messages)
                self._usage_tracking(usage)

                try:
                    # Extract JSON array from response
                    clean_json = self._extract_json_array(response[0])

                    if clean_json is None:
                        print(
                            f"Failed to extract valid JSON from response for {node.concept}"
                        )
                        print(f"Response: {response[0][:200]}...")
                        node.mark_processed()
                        return

                    response_nodes = json.loads(clean_json)  # DFS only get one response

                    if response_nodes:
                        node.mark_processed()

                        c_nodes = []
                        for c in response_nodes:
                            concept_val = c.get("concept")
                            if concept_val is None:
                                for k in c:
                                    if "concept" in k.lower():
                                        concept_val = c[k]
                                        break
                            if concept_val is None:
                                print(f"Skipping node with no concept key: {c}")
                                continue
                            c_nodes.append(
                                HallucinationTaxonomyTreeNode(
                                    concept=concept_val,
                                    concept_desc=c.get("description", ""),
                                    depth=node.depth + 1,
                                    parent=node,
                                )
                            )

                        for c_node in c_nodes:
                            node.add_child(c_node)
                            await self.generate_subtree(c_node)
                    else:
                        # Empty list returned, mark as processed and return
                        print(
                            f"Empty list for {node.concept}, returning to parent {node.parent.concept if node.parent else 'Root'}"
                        )
                        node.mark_processed()
                        return
                except Exception as e:
                    print("Failed to parse JSON response:")
                    print(response)
                    print(f"Error: {e}")
                    node.mark_processed()

    def _usage_tracking(self, usage):
        """Track token usage."""
        if usage:
            self.total_usage["prompt_tokens"] += usage["prompt_tokens"]
            self.total_usage["completion_tokens"] += usage["completion_tokens"]
            self.total_usage["total_tokens"] += usage["total_tokens"]
            self.total_usage["max_total_tokens"] = max(
                self.total_usage["max_total_tokens"], usage["total_tokens"]
            )

    def _extract_json_array(self, text: str):
        """Extract JSON array from text that may contain CoT reasoning before the array."""
        cleaned = re.sub(r"```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)

        # Strategy 1: find all [...] candidates containing at least one {}, pick last valid one.
        # Using last match because CoT reasoning may contain earlier [...] lists.
        array_pattern = r"\[\s*(?:\{.*?\}\s*,?\s*)+\]"
        matches = list(re.finditer(array_pattern, cleaned, re.DOTALL))
        for m in reversed(matches):
            try:
                json.loads(m.group())
                return m.group()
            except json.JSONDecodeError:
                continue

        # Strategy 2: empty array — explicit [] anywhere (atomic node signal)
        if re.search(r"\[\s*\]", cleaned):
            return "[]"

        # Strategy 3: last [...] span (from last '[' that starts an array to matching ']')
        pos = len(cleaned) - 1
        while pos >= 0:
            if cleaned[pos] == "]":
                # find matching opening bracket
                depth, i = 0, pos
                while i >= 0:
                    if cleaned[i] == "]":
                        depth += 1
                    elif cleaned[i] == "[":
                        depth -= 1
                        if depth == 0:
                            candidate = cleaned[i : pos + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                break
                    i -= 1
            pos -= 1

        return None


def save_node_to_pickle(node, filename="node.pkl"):
    """Save a node to pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(node, f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate hallucination taxonomy tree")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to use for generation (overrides config)",
    )
    parser.add_argument(
        "--port", type=int, help="Port for model inference (overrides config)"
    )
    parser.add_argument("--max_width", type=int, help="Maximum width of each node")
    parser.add_argument("--max_depth", type=int, help="Maximum depth of the tree")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing pickle file"
    )
    parser.add_argument(
        "--output_path", type=str, help="Output path for taxonomy pickle"
    )
    parser.add_argument(
        "--usage_path", type=str, help="Output path for usage statistics"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation (overrides config)",
    )
    return parser.parse_args()


# ============================================================================
# Main execution
# ============================================================================


# python -m hallupeer.hallucination_taxonomy.generate_taxonomy
async def main():
    """Main execution function."""

    args = parse_args()
    cfg = load_configs(["configs/agents.yaml", "configs/taxonomy_generation.yaml"])

    # Override config with command line arguments if provided
    if args.max_width is not None:
        cfg.taxonomy_generation.max_width = args.max_width
    if args.max_depth is not None:
        cfg.taxonomy_generation.max_depth = args.max_depth
    if args.resume:
        cfg.taxonomy_generation.resume = True
    if args.model_name is not None:
        cfg.agents.taxonomy_generator.model_name = args.model_name
    if args.port is not None:
        cfg.agents.taxonomy_generator.port = args.port

    print(
        "Final configuration:"
        f"\nModel Name: {cfg.agents.taxonomy_generator.model_name}"
        f"\nPort: {cfg.agents.taxonomy_generator.port}"
        f"\nMax Width: {cfg.taxonomy_generation.max_width}"
        f"\nMax Depth: {cfg.taxonomy_generation.max_depth}"
        f"\nResume: {cfg.taxonomy_generation.resume}"
    )
    model_name_suffix = cfg.agents.taxonomy_generator.model_name.split("/")[-1]

    # Determine output paths
    output_taxonomy_path = (
        args.output_path
        if args.output_path
        else cfg.taxonomy_generation.output_taxonomy_path.format(model_name_suffix)
    )
    output_usage_path = (
        args.usage_path
        if args.usage_path
        else cfg.taxonomy_generation.output_usage_path.format(model_name_suffix)
    )

    # Check if output taxonomy pickle already exists
    if os.path.exists(output_taxonomy_path):
        raise FileExistsError(
            f"Output taxonomy pickle already exists: {output_taxonomy_path}. Please remove it or set resume=True to continue from existing tree."
        )

    # Load prompt
    prompt = load_prompt_template(cfg.taxonomy_generation.prompt_template_path)

    # Initialize generator
    generator = load_agent(agent="taxonomy_generator", cfg=cfg)

    # Get temperature: command line > config
    if args.temperature is not None:
        temperature = args.temperature
    else:
        temperature = getattr(cfg.agents.taxonomy_generator, "temperature", None)

    # Get seed: command line > config (if config has it)
    seed = args.seed
    if seed is None:
        # Optionally support seed in config
        seed = getattr(cfg.agents.taxonomy_generator, "seed", None)

    # Initialize pipeline
    pipeline = TaxonomyGenerationPipeline(
        generator=generator,
        prompt=prompt,
        max_width=cfg.taxonomy_generation.max_width,
        max_depth=cfg.taxonomy_generation.max_depth,
        resume=args.resume,
        temperature=temperature,
        seed=seed,
        **cfg,
    )

    # Generate subtree
    await pipeline.generate_subtree(pipeline.root_node)
    pipeline.root_node.print_tree()

    # Save results
    save_node_to_pickle(pipeline.root_node, filename=output_taxonomy_path)
    with open(output_usage_path, "w") as f:
        json.dump(pipeline.total_usage, f, indent=4)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
