"""
Hallucination injection template construction.

For every paper, combine the review sentences with the hallucination taxonomy to produce
injection templates. Each template records:
    - hallucination_instruction
    - hallucination_labels (taxonomy path)
    - injection_aspect (aspect of the target sentence)
    - injection_sentence (target sentence)
Candidate (sentence, taxonomy leaf) pairs are pre-filtered and then checked for feasibility
by an LLM; only the feasible templates are written out.

Prerequisites:
    1. Review aspect tagging has been run (results at the configured path).
    2. The final taxonomy tree (JSON or pickle) is available (`taxonomy_path` in the config).
    3. An OpenAI-compatible LLM server (e.g. vLLM) is running.

Usage:
    python -m hallupeer.dataset_construction.hallucination_injection.construct_template \\
        --dataset_dir ../datasets/openreview_data/ICLR2019 \\
        --port 8000 \\
        --resume \\
        --max_papers 1200

Arguments:
    --dataset_dir                   dataset directory, e.g. ../datasets/openreview_data/ICLR2019
    --port                          port of the LLM server
    --type_of_aspect_tagging_label  aspect label type, coarse or fine (default: from config)
    --max_papers                    maximum number of papers to process (default 1200)
    --template_sample_size          maximum number of templates per review (default None = all)
    --resume                        resume from an existing output file

Output:
    JSONL file, one line per paper, with fields:
        PaperID, ReviewerIDs, HallucinationInjectionTemplates
"""

import argparse
import asyncio
import copy
import json
import os
import random
import re

import nltk
from tqdm.asyncio import tqdm as tqdm_asyncio

nltk.download("punkt_tab")

from tqdm import tqdm

from hallupeer.agents import load_agent
from hallupeer.dataset import load_dataset
from hallupeer.hallucination_taxonomy import HallucinationTaxonomyTreeNode
from hallupeer.hallucination_taxonomy.hallucination_tree import load_taxonomy
from hallupeer.utils import load_configs, load_json_configs, load_prompt_template


class HallucinationTemplateConstructionPipeline:
    def __init__(
        self,
        dataloader,
        constructor,
        prompt,
        taxonomy_tree,
        prefilter_prompt=None,
        max_papers=None,
        template_sample_size=None,
        resume=False,
        **kwargs,
    ):
        self.dataloader = dataloader
        self.constructor = constructor
        self.prompt = prompt
        self.prefilter_prompt = prefilter_prompt
        self.taxonomy_tree = taxonomy_tree

        self.sent_tokenize = nltk.tokenize.sent_tokenize
        self.batch_size_fc = kwargs["template_construction"][
            "batch_size_feasible_check"
        ]
        self.batch_size_prefilter = kwargs["template_construction"].get(
            "batch_size_prefilter", 32
        )

        # Prepare taxonomy tree and leaf nodes
        self.leaf_nodes = taxonomy_tree.get_leaf_nodes()
        print(f"Taxonomy tree size: {self.taxonomy_tree.get_size()} nodes")
        print(f"Taxonomy tree leaf size: {len(self.leaf_nodes)} nodes")

        # Build depth-1 concept → leaf nodes mapping for pre-filter stage
        self.depth1_to_leaf_nodes = {}
        for leaf in self.leaf_nodes:
            path = leaf.get_path_to_root()  # [root, depth1, ..., leaf]
            if len(path) >= 2:
                depth1_concept = path[1]
                self.depth1_to_leaf_nodes.setdefault(depth1_concept, []).append(leaf)
        print(f"Depth-1 categories: {list(self.depth1_to_leaf_nodes.keys())}")

        # Read aspect tagging results
        path_aspect_tagging_results = kwargs["template_construction"][
            "review_aspect_tagging_results_path"
        ].format(
            type_of_aspect_tagging_label=kwargs["template_construction"][
                "type_of_aspect_tagging_label"
            ],
            venue=kwargs["dataset"]["venue"],
        )
        self.aspect_tagging_results = self._read_aspect_tagging_results(
            path_aspect_tagging_results
        )
        print(
            f"Loaded aspect tagging results from {path_aspect_tagging_results}, total {len(self.aspect_tagging_results)} records."
        )

        # Output paths
        self.output_path = kwargs["template_construction"]["output_path"].format(
            venue=kwargs["dataset"]["venue"],
            template_sample_size=template_sample_size,
        )
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))

        # Resume mechanism
        self.resume = resume
        self.processed_paper_ids = set()
        if self.resume and os.path.exists(self.output_path):
            self._load_processed_papers()
            print(
                f"Resume mode: Found {len(self.processed_paper_ids)} already processed papers"
            )
        # elif not resume and os.path.exists(self.output_path):
        # os.remove(self.output_path)

        # Maximum papers to process
        self.max_papers = max_papers
        
        # Template sample size per review
        self.template_sample_size = template_sample_size

    def _load_processed_papers(self):
        """Load already processed paper IDs from existing output file."""
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        self.processed_paper_ids.add(record["PaperID"])
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse line: {line[:100]}...")
                        continue
        except Exception as e:
            print(f"Error loading processed papers: {e}")
            self.processed_paper_ids = set()

    async def construct_template(self):
        """Construct hallucination injection templates for the dataset."""
        # Count total papers and papers still to process
        all_paper_ids = []
        for batch in self.dataloader:
            all_paper_ids.extend(batch["PaperID"])

        total_papers = len(all_paper_ids)
        papers_to_process = [
            pid for pid in all_paper_ids if pid not in self.processed_paper_ids
        ]

        # Limit the number of papers
        if self.max_papers is not None:
            remaining_quota = self.max_papers - len(self.processed_paper_ids)
            if remaining_quota <= 0:
                print(
                    f"Already processed {len(self.processed_paper_ids)} papers, reaching max limit {self.max_papers}"
                )
                return
            papers_to_process = papers_to_process[:remaining_quota]

        print(f"Total papers in dataset: {total_papers}")
        print(f"Already processed: {len(self.processed_paper_ids)}")
        print(f"To process: {len(papers_to_process)}")
        if self.max_papers is not None:
            print(f"Max papers limit: {self.max_papers}")

        # Generate predictions
        pbar = tqdm(total=len(papers_to_process), desc="Constructing templates")
        processed_count = 0
        total_pairs_sum = 0

        for _, batch in enumerate(self.dataloader):
            paper_ids = batch["PaperID"]
            reviews = batch["Reviews"]

            for paper_id, review in zip(paper_ids, reviews):
                # Stop once the paper limit is reached
                if (
                    self.max_papers is not None
                    and len(self.processed_paper_ids) >= self.max_papers
                ):
                    print(f"\nReached max papers limit: {self.max_papers}")
                    pbar.close()
                    return

                # Skip if already processed
                if paper_id in self.processed_paper_ids:
                    continue
                if paper_id not in self.aspect_tagging_results:
                    print(
                        f"\nWarning: No aspect tagging results for paper {paper_id}, skipping."
                    )
                    continue

                try:
                    reviewer_ids = [r["ReviewerID"] for r in review]
                    review_texts = [r["Review"] for r in review]
                    reviews_sents_list = [
                        self.sent_tokenize(text) for text in review_texts
                    ]

                    aspect_tags_paper = self.aspect_tagging_results[paper_id]
                    aspect_tags_list = [
                        aspect_tags_paper[reviewer_id] for reviewer_id in reviewer_ids
                    ]
                    assert len(reviews_sents_list) == len(aspect_tags_list)

                    # Stage 1: Pre-filter sentences to select relevant depth-1 categories
                    if self.prefilter_prompt is not None:
                        prefilter_results = await self._batch_prefilter_sentences(
                            reviews_sents_list
                        )
                        total_before = sum(len(s) for s in reviews_sents_list) * len(self.leaf_nodes)
                        total_after = sum(
                            len(self.depth1_to_leaf_nodes.get(cat, []))
                            for i, sents in enumerate(reviews_sents_list)
                            for j in range(len(sents))
                            for cat in prefilter_results[i][j]
                        )
                        reduction = 100 * (1 - total_after / total_before) if total_before > 0 else 0
                        print(f"\n[Pre-filter] {total_before} → {total_after} pairs ({reduction:.1f}% reduction)")
                    else:
                        prefilter_results = None

                    # Construct templates (Stage 1.5: cartesian product with filtered leaf nodes)
                    templates = self._construct_template_single_paper(
                        reviews_sents_list, aspect_tags_list, prefilter_results=prefilter_results
                    )

                    # Random sample templates for each review
                    if self.template_sample_size is not None:
                        templates_sampled = [
                            (
                                random.sample(t, min(self.template_sample_size, len(t)))
                                if len(t) > self.template_sample_size
                                else t
                            )  # sample per review
                            for t in templates
                        ]
                    else:
                        templates_sampled = templates  # No sampling for now

                    # Check feasible template construction
                    feasible_templates = await self._batch_feasible_template_check(
                        templates_sampled
                    )

                    # Masked feasible templates if false
                    checked_feasible_templates = []
                    for template_sampled, feasible_template in zip(
                        templates_sampled, feasible_templates
                    ):
                        checked_feasible_templates.append(
                            [
                                template_sampled[idx]
                                for idx in range(len(template_sampled))
                                if feasible_template[idx]
                            ]
                        )

                    total_feasible = sum(len(t) for t in checked_feasible_templates)
                    total_before_fc = sum(len(t) for t in templates_sampled)
                    print(f"\n[Stage-2] {total_before_fc} → {total_feasible} pairs after feasibility check")

                    output = {
                        "PaperID": paper_id,
                        "ReviewerIDs": reviewer_ids,
                        "HallucinationInjectionTemplates": checked_feasible_templates,
                    }

                    # Write to file
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(output, ensure_ascii=False) + "\n")

                    # Update processed set
                    self.processed_paper_ids.add(paper_id)
                    processed_count += 1
                    total_pairs_sum += total_feasible
                    pbar.set_postfix(avg_pairs=f"{total_pairs_sum / processed_count:.1f}")
                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing paper {paper_id}: {e}")
                    # Optional: write an error log
                    error_log_path = self.output_path.replace(".jsonl", "_errors.log")
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{paper_id}: {str(e)}\n")
                    continue

        pbar.close()
        print(f"\nCompleted! Processed {processed_count} new papers.")
        print(f"Total processed papers: {len(self.processed_paper_ids)}")

    def _construct_template_single_paper(self, reviews_sents_list, aspect_tags_list, prefilter_results=None):
        """Construct hallucination templates for a single paper given its reviews and aspect tags."""
        all_templates = []

        for review_idx, (review_sents, aspect_tags) in enumerate(zip(
            reviews_sents_list, aspect_tags_list
        )):  # per review
            review_templates = []
            for sent_idx, (review_sent, aspect_tag) in enumerate(zip(
                review_sents, aspect_tags
            )):  # per sentence
                # Stage 1: use pre-filtered leaf nodes if available
                if prefilter_results is not None:
                    selected_categories = prefilter_results[review_idx][sent_idx]
                    leaf_nodes = [
                        leaf
                        for cat in selected_categories
                        for leaf in self.depth1_to_leaf_nodes.get(cat, [])
                    ]
                else:
                    leaf_nodes = self.leaf_nodes

                for leaf_node in leaf_nodes:  # per hallucination type
                    template = {
                        "hallucination_instruction": leaf_node.concept_desc,
                        "hallucination_labels": " > ".join(leaf_node.get_path_to_root()),
                        "injection_aspect": ", ".join(aspect_tag),
                        "injection_sentence": review_sent,
                    }
                    review_templates.append(template)
            all_templates.append(review_templates)

        return all_templates

    async def _batch_prefilter_sentences(self, reviews_sents_list):
        """Stage 1: for each sentence, select compatible depth-1 categories via LLM."""
        semaphore = asyncio.Semaphore(self.batch_size_prefilter)

        async def prefilter_with_semaphore(review_idx, sent_idx, sent):
            async with semaphore:
                result = await self._prefilter_single_sentence(sent)
                return review_idx, sent_idx, result

        all_tasks = [
            prefilter_with_semaphore(review_idx, sent_idx, sent)
            for review_idx, sents in enumerate(reviews_sents_list)
            for sent_idx, sent in enumerate(sents)
        ]

        raw_results = await tqdm_asyncio.gather(*all_tasks, desc="Pre-filtering sentences")

        # Reconstruct per-review, per-sentence structure
        results = [[None] * len(sents) for sents in reviews_sents_list]
        for review_idx, sent_idx, categories in raw_results:
            results[review_idx][sent_idx] = categories

        return results

    async def _prefilter_single_sentence(self, sentence):
        """Call LLM to select compatible depth-1 category names for a sentence."""
        try:
            messages = [[{
                "role": "user",
                "content": self.prefilter_prompt.format(injection_sentence=sentence),
            }]]
            response, _ = await self.constructor.predict(messages)
            categories = json.loads(response[0].strip())
            if isinstance(categories, list):
                return [c for c in categories if c in self.depth1_to_leaf_nodes]
        except Exception:
            pass
        # Fallback: include all categories (conservative, no false negatives)
        return list(self.depth1_to_leaf_nodes.keys())

    async def _batch_feasible_template_check(self, templates):
        """Check feasible templates with controlled concurrency."""

        semaphore = asyncio.Semaphore(self.batch_size_fc)

        async def check_with_semaphore(review_idx, template_idx, template):
            async with semaphore:
                result = await self._check_single_template(template)
                return review_idx, template_idx, result

        all_tasks = [
            check_with_semaphore(review_idx, template_idx, template)
            for review_idx, review_templates in enumerate(templates)
            for template_idx, template in enumerate(review_templates)
        ]

        all_results = await tqdm_asyncio.gather(*all_tasks, desc="Checking templates")

        feasible_templates = [[False] * len(review_templates) for review_templates in templates]
        for review_idx, template_idx, result in all_results:
            if result is not None and isinstance(result, list) and len(result) > 0:
                feasible_templates[review_idx][template_idx] = result[0]

        return feasible_templates

    async def _check_single_template(self, template):
        """Check a single template feasibility."""
        try:
            messages = [
                [
                    {
                        "role": "user",
                        "content": self.prompt.format(
                            hallucination_instruction=template["hallucination_instruction"],
                            injection_sentence=template["injection_sentence"],
                            hallucination_labels=template["hallucination_labels"],
                            injection_aspect=template["injection_aspect"],
                        ),
                    }
                ]
            ]
            response, _ = await self.constructor.predict(messages)
            return [r.strip().lower().startswith("true") for r in response]
        except Exception as e:
            print(f"Feasibility check failed: {e}")
            return [False]

    def _read_aspect_tagging_results(self, filepath):
        """Read aspect tagging results from a JSON file."""
        records = []
        with open(filepath, "r") as f:
            for line in f:
                record = json.loads(line)
                records.append(record)

        aspect_tagging_results = {}
        for record in records:
            paper_id = record["PaperID"]
            reviewer_ids = record["ReviewerIDs"]
            aspect_tags = record["ReviewAspectTags"]
            aspect_tagging_results[paper_id] = {
                reviewer_id: aspect_tag
                for reviewer_id, aspect_tag in zip(reviewer_ids, aspect_tags)
            }
        return aspect_tagging_results


def clean_json_response(response):
    """Clean JSON response from the tagger."""
    # Remove markdown code blocks, leading [' and trailing ']
    clean_json = response.strip()
    # Remove markdown code fences (```json or ```)
    clean_json = re.sub(r"```(?:json)?\s*", "", clean_json)
    try:
        json.loads(clean_json)
        return clean_json
    except json.JSONDecodeError:
        pass
    # Escape unescaped backslashes
    clean_json = re.sub(r"(?<!\\)\\(?![\\\"/bfnrtu])", r"\\\\", clean_json)
    # Remove leading [' and trailing ']
    clean_json = re.sub(r"^\[\s*'|'\s*\]$", "", clean_json)
    # # Replace single quotes with double quotes
    try:
        json.loads(clean_json)
        return clean_json
    except json.JSONDecodeError:
        print("Failed to clean JSON response:")
        print(clean_json)
        exit()


async def main():
    """Main execution function."""

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset_dir", type=str, default=None, help="Path to the dataset directory"
    )
    argparser.add_argument(
        "--port", type=int, default=None, help="Port number for the tagger agent"
    )
    argparser.add_argument(
        "--type_of_aspect_tagging_label",
        type=str,
        default=None,
        help="Type of aspect tagging label to use (coarse or fine)",
    )
    argparser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from existing output file",
    )
    argparser.add_argument(
        "--max_papers",
        type=int,
        default=1200,
        help="Maximum number of papers to process (default: 1200)",
    )
    argparser.add_argument(
        "--template_sample_size",
        type=int,
        default=None,
        help="Number of templates to sample per review (default: None, meaning no sampling)",
    )
    argparser.add_argument(
        "--prompt_template_path",
        type=str,
        default=None,
        help="Override cfg.template_construction.prompt_template_path",
    )
    argparser.add_argument(
        "--prefilter_prompt_template_path",
        type=str,
        default=None,
        help="Override cfg.template_construction.prefilter_prompt_template_path (set to empty string to disable prefilter)",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Override cfg.template_construction.output_path",
    )
    argparser.add_argument(
        "--output_usage_path",
        type=str,
        default=None,
        help="Override cfg.template_construction.output_usage_path",
    )
    argparser.add_argument(
        "--batch_size_feasible_check",
        type=int,
        default=None,
        help="Override cfg.template_construction.batch_size_feasible_check (semaphore concurrency for Stage-2)",
    )
    argparser.add_argument(
        "--batch_size_prefilter",
        type=int,
        default=None,
        help="Override cfg.template_construction.batch_size_prefilter (semaphore concurrency for Stage-1)",
    )
    args = argparser.parse_args()

    cfg = load_configs(
        [
            "configs/agents.yaml",
            "configs/dataset.yaml",
            "configs/template_construction.yaml",
        ]
    )
    print(cfg)

    # Override cfg fields if provided via command line arguments
    if args.dataset_dir is not None:
        cfg.dataset.dataset_dir = args.dataset_dir
    if args.port is not None:
        cfg.agents.templates_constructor.port = args.port
    if args.type_of_aspect_tagging_label is not None:
        cfg.template_construction.type_of_aspect_tagging_label = (
            args.type_of_aspect_tagging_label
        )
    if args.prompt_template_path is not None:
        cfg.template_construction.prompt_template_path = args.prompt_template_path
    if args.prefilter_prompt_template_path is not None:
        # Empty string → disable prefilter
        cfg.template_construction.prefilter_prompt_template_path = (
            args.prefilter_prompt_template_path or None
        )
    if args.output_path is not None:
        cfg.template_construction.output_path = args.output_path
    if args.output_usage_path is not None:
        cfg.template_construction.output_usage_path = args.output_usage_path
    if args.batch_size_feasible_check is not None:
        cfg.template_construction.batch_size_feasible_check = args.batch_size_feasible_check
    if args.batch_size_prefilter is not None:
        cfg.template_construction.batch_size_prefilter = args.batch_size_prefilter

    venue = cfg.dataset.dataset_dir.split("/")[-1]
    cfg.dataset.venue = venue

    # Load dataset
    dataloader = await load_dataset(
        dataset_dir=cfg.dataset.dataset_dir,
        batch_size=cfg.dataset.batch_size_template_construction,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        **cfg,
    )

    # Load prompts
    prompt = load_prompt_template(cfg.template_construction.prompt_template_path)
    prefilter_prompt_path = cfg.template_construction.get("prefilter_prompt_template_path", None)
    prefilter_prompt = load_prompt_template(prefilter_prompt_path) if prefilter_prompt_path else None

    # Load taxonomy tree
    taxonomy_tree = load_taxonomy(cfg.template_construction.taxonomy_path)
    # constraint the taxonomy tree to a certain depth
    taxonomy_tree = taxonomy_tree.clone_subtree(
        max_depth=cfg.template_construction.max_depth_load
    )
    print(f"Loaded taxonomy tree from {cfg.template_construction.taxonomy_path} (depth≤{cfg.template_construction.max_depth_load})")
    
    # Initialize constructor
    constructor = load_agent(agent="templates_constructor", cfg=cfg)

    # Initialize pipeline
    pipeline = HallucinationTemplateConstructionPipeline(
        dataloader=dataloader,
        constructor=constructor,
        prompt=prompt,
        prefilter_prompt=prefilter_prompt,
        taxonomy_tree=taxonomy_tree,
        resume=args.resume,
        max_papers=args.max_papers,
        template_sample_size=args.template_sample_size,
        **cfg,
    )

    # Construct templates
    await pipeline.construct_template()


if __name__ == "__main__":
    asyncio.run(main())
