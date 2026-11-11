"""
Hallucination Template Injection Pipeline
==========================================

Injects hallucinated sentences into peer review text based on pre-constructed
hallucination templates (produced by the template construction stage).

Pipeline overview:
    1. Load template construction results (JSONL) keyed by PaperID → ReviewerID.
    2. For each review sentence matched to a template, call the injector LLM to
       generate a hallucinated rewrite of that sentence.
    3. Run a post-hoc semantic checker: if the hallucinated sentence is too similar
       to the original, retry up to ``max_retries`` times; mark as invalid on failure.
    4. Write per-paper output JSON to ``output_base_dir``.

Main class:
    HallucinationTemplateInjectionPipeline

Usage:
    python -m hallupeer.dataset_construction.hallucination_injection.inject_template \\
        --dataset_dir ../datasets/openreview_data/ICLR2019 \\
        --port 8003 \\
        [--post_check_port 8004] \\
        [--template_sample_size 300] \\
        [--max_papers 1200] \\
        [--resume]

Config files loaded:
    configs/agents.yaml
    configs/dataset.yaml
    configs/template_injection.yaml
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
from hallupeer.utils import load_configs, load_json_configs, load_prompt_template


class HallucinationTemplateInjectionPipeline:
    def __init__(
        self,
        dataloader,
        injector,
        prompt,
        post_checker=None,
        post_check_prompt=None,
        resume=False,
        max_papers=None,
        template_sample_size=None,
        max_concurrent=80,
        **kwargs,
    ):
        self.dataloader = dataloader
        self.injector = injector
        self.prompt = prompt
        self.post_checker = post_checker
        self.post_check_prompt = post_check_prompt
        self.max_concurrent = max_concurrent

        self.sent_tokenize = nltk.sent_tokenize
        
        # Read template construction results
        path_template_construction_results = kwargs["template_injection"][
            "template_construction_results_path"
        ].format(
            venue=kwargs["dataset"]["venue"],
            template_sample_size=template_sample_size,
        )
        self.template_construction_results = self._read_template_construction_results(
            path_template_construction_results
        )
        print(
            f"Loaded template construction results from {path_template_construction_results}, total {len(self.template_construction_results)} records."
        )

        # Output paths
        self.output_base_dir = kwargs["template_injection"]["output_base_dir"].format(
            venue=kwargs["dataset"]["venue"],
            template_sample_size=template_sample_size,
        )
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)

        # Resume mechanism
        self.resume = resume
        self.processed_paper_ids = set()
        if self.resume and os.path.exists(self.output_base_dir):
            self._load_processed_papers()
            print(
                f"Resume mode: Found {len(self.processed_paper_ids)} already processed papers"
            )

        self.max_papers = max_papers

    def _load_processed_papers(self):
        """Load already processed paper IDs from existing output file."""

        for file in os.listdir(self.output_base_dir):
            if file.endswith(".json"):
                self.processed_paper_ids.add(file.replace(".json", ""))

    async def inject_template(self):
        """Construct hallucination injection templates for the dataset."""

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
        pbar = tqdm(total=len(papers_to_process), desc="Injecting templates")
        processed_count = 0

        for _, batch in enumerate(self.dataloader):
            paper_ids = batch["PaperID"]
            contents = batch["Contents"]
            reviews = batch["Reviews"]

            for paper_id, content, review in zip(paper_ids, contents, reviews):
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
                if paper_id not in self.template_construction_results:
                    print(
                        f"\nWarning: No template construction results for paper {paper_id}, skipping."
                    )
                    continue

                # output format
                output = {
                    "paper_id": paper_id,
                    "paper_content": content,
                    "reviews": [],
                }

                try:
                    reviewer_ids = [r["ReviewerID"] for r in review]
                    review_texts = [r["Review"] for r in review]
                    reviews_sents_list = [
                        self.sent_tokenize(text) for text in review_texts
                    ]  # List of list of sentences [[sent1, sent2,...], [...], ...], per review

                    template_construction_results = self.template_construction_results[
                        paper_id
                    ]
                    templates_list = [
                        template_construction_results[reviewer_id]
                        for reviewer_id in reviewer_ids
                    ]  # List of list of templates [[template1, template2,...], [...], ...], per review

                    # Inject templates
                    injection_results = await self._batch_template_injection(
                        templates_list
                    )
                    assert (
                        len(reviewer_ids)
                        == len(templates_list)
                        == len(injection_results)
                    )

                    # add injection results into templates
                    for review_templates, review_injection_results in zip(
                        templates_list, injection_results
                    ):
                        assert len(review_templates) == len(review_injection_results)
                        for template, injection_result in zip(
                            review_templates, review_injection_results
                        ):
                            if injection_result is None:
                                # Mark template as invalid by adding a flag
                                template["_invalid"] = True
                            else:
                                template["hallucination_sentence"] = injection_result

                    # Result
                    for review_id, review_sents, templates in zip(
                        reviewer_ids, reviews_sents_list, templates_list
                    ):
                        # output format per review
                        output_review = {
                            "review_id": review_id,
                            "review_sentences": [],
                        }

                        # match review sent with template
                        for review_sent_idx, review_sent in enumerate(review_sents):
                            sent_id = f"{paper_id}_{review_id}_s{review_sent_idx}"
                            matched = []
                            t_idx = 0
                            for template in templates:
                                # Skip invalid templates (those with None hallucination_sentence)
                                if template.get("_invalid", False):
                                    continue

                                temp = copy.deepcopy(template)
                                if review_sent == temp["injection_sentence"]:
                                    temp.pop("injection_sentence")  # remove redundant field
                                    temp = {"template_id": f"{sent_id}_t{t_idx}", **temp}
                                    matched.append(temp)
                                    t_idx += 1

                            output_review["review_sentences"].append(
                                {
                                    "sent_id": sent_id,
                                    "review_sentence": review_sent,
                                    "hallucination_injection_templates": matched,
                                }
                            )

                        output["reviews"].append(output_review)

                    # Write to file
                    with open(
                        f"{self.output_base_dir}/{paper_id}.json", "w", encoding="utf-8"
                    ) as f:
                        f.write(json.dumps(output, ensure_ascii=False, indent=4))

                    # Update processed set
                    self.processed_paper_ids.add(paper_id)
                    processed_count += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing paper {paper_id}: {e}")
                    exit()
                    continue

        pbar.close()
        print(f"\nCompleted! Processed {processed_count} new papers.")
        print(f"Total processed papers: {len(self.processed_paper_ids)}")

    async def _batch_template_injection(self, templates):
        """Inject templates in batch for multiple reviews."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def inject_with_semaphore(template):
            async with semaphore:
                return await self._inject_single_template(template)

        tasks = []
        review_sizes = []
        for review_templates in templates:
            review_sizes.append(len(review_templates))
            tasks.extend(
                inject_with_semaphore(template) for template in review_templates
            )

        all_injection_results = await tqdm_asyncio.gather(
            *tasks, desc="Injecting templates"
        )

        reshaped_results = []
        index = 0
        for size in review_sizes:
            reshaped_results.append(all_injection_results[index : index + size])
            index += size

        return reshaped_results

    async def _inject_single_template(self, template):
        """Inject a single template with post-hoc semantic validation."""

        # Prepare messages for the injector
        messages = [
            [
                {
                    "role": "user",
                    "content": self.prompt.format(
                        hallucination_instruction=template["hallucination_instruction"],
                        injection_sentence=template["injection_sentence"],
                    ),
                }
            ]
        ]

        # Get predictions from the injector with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            response, _ = await self.injector.predict(messages)
            hallucination_sentence = response[0].strip()

            # Post-hoc semantic check
            if self.post_checker and self.post_check_prompt:
                is_different = await self._check_semantic_difference(
                    template["injection_sentence"],
                    hallucination_sentence,
                )

                if is_different:
                    return hallucination_sentence
                else:
                    # Log the retry attempt
                    if attempt < max_retries - 1:
                        tqdm.write(
                            f"\nSemantic check failed (attempt {attempt + 1}/{max_retries}). Regenerating hallucination sentence..."
                        )
                        # Continue to next iteration to regenerate
                        continue
                    else:
                        # Max retries reached, return None to indicate failure
                        tqdm.write(
                            f"\nWarning: Max retries ({max_retries}) reached. Semantic similarity still detected. Returning None."
                        )
                        return None
            else:
                # No post-checker configured, return the first result
                return hallucination_sentence


    async def _check_semantic_difference(
        self, original_sentence, hallucination_sentence
    ):
        """Check if hallucination sentence is semantically different from original sentence."""

        check_messages = [
            [
                {
                    "role": "user",
                    "content": self.post_check_prompt.format(
                        REVIEW_SENTENCE_1=original_sentence,
                        REVIEW_SENTENCE_2=hallucination_sentence,
                    ),
                }
            ]
        ]

        response, _ = await self.post_checker.predict(check_messages)
        result = response[0].strip().lower()

        # Expected response: "0" (different) or "1" (similar)
        return result == "0" or result.startswith("0")

    def _read_template_construction_results(self, filepath):
        """Read template construction results from a JSON file."""
        records = []
        with open(filepath, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line: {line}")
                    continue

        template_construction_results = {}
        for record in records:
            paper_id = record["PaperID"]
            reviewer_ids = record["ReviewerIDs"]
            injection_templates = record["HallucinationInjectionTemplates"]
            template_construction_results[paper_id] = {
                reviewer_id: injection_template
                for reviewer_id, injection_template in zip(
                    reviewer_ids, injection_templates
                )
            }
        return template_construction_results


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
    try:
        json.loads(clean_json)
        return clean_json
    except json.JSONDecodeError:
        print("Failed to clean JSON response:")
        print(clean_json)
        exit()


# python -m hallupeer.dataset_construction.hallucination_injection.inject_template --dataset_dir ../datasets/openreview_data/ICLR2019 --port 8003 --resume
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
        "--post_check_port",
        type=int,
        default=None,
        help="Port number for the post checker agent",
    )
    argparser.add_argument(
        "--template_sample_size",
        type=int,
        default=None,
        help="For loading constructed template, number of templates to sample per review (default: None, use all)",
    )
    argparser.add_argument(
        "--template_construction_results_path",
        type=str,
        default=None,
        help="Path to template construction results JSONL (overrides config; supports {venue} and {template_sample_size} placeholders)",
    )
    argparser.add_argument(
        "--max_papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: None, process all)",
    )
    argparser.add_argument(
        "--max_concurrent",
        type=int,
        default=80,
        help="Max concurrent LLM requests via semaphore (default: 80)",
    )
    argparser.add_argument(
        "--injector_model_name",
        type=str,
        default=None,
        help="Override templates_injector model_name from agents.yaml",
    )
    argparser.add_argument(
        "--post_checker_model_name",
        type=str,
        default=None,
        help="Override post_checker model_name from agents.yaml",
    )
    argparser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Override output_base_dir from template_injection.yaml",
    )
    argparser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from existing output file",
    )
    args = argparser.parse_args()

    cfg = load_configs(
        [
            "configs/agents.yaml",
            "configs/dataset.yaml",
            "configs/template_injection.yaml",
        ]
    )
    print(cfg)

    # Override dataset_dir and port if provided via command line arguments
    if args.dataset_dir is not None:
        cfg.dataset.dataset_dir = args.dataset_dir
    if args.port is not None:
        cfg.agents.templates_injector.port = args.port
    if args.post_check_port is not None:
        cfg.agents.post_checker.port = args.post_check_port
    if args.max_papers is not None:
        cfg.template_injection.max_papers = args.max_papers
    if args.template_construction_results_path is not None:
        cfg.template_injection.template_construction_results_path = args.template_construction_results_path
    if args.injector_model_name is not None:
        cfg.agents.templates_injector.model_name = args.injector_model_name
    if args.post_checker_model_name is not None:
        cfg.agents.post_checker.model_name = args.post_checker_model_name
    if args.output_base_dir is not None:
        cfg.template_injection.output_base_dir = args.output_base_dir

    venue = cfg.dataset.dataset_dir.split("/")[-1]
    cfg.dataset.venue = venue

    # Load dataset
    dataloader = await load_dataset(
        dataset_dir=cfg.dataset.dataset_dir,
        batch_size=cfg.dataset.batch_size_template_injection,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        **cfg,
    )

    # Load prompts
    prompt = load_prompt_template(cfg.template_injection.prompt_template_path)
    post_check_prompt = load_prompt_template(
        cfg.template_injection.post_check_prompt_path
    )

    # Initialize agents
    injector = load_agent(agent="templates_injector", cfg=cfg)
    post_checker = load_agent(agent="post_checker", cfg=cfg)

    # Initialize pipeline
    pipeline = HallucinationTemplateInjectionPipeline(
        dataloader=dataloader,
        injector=injector,
        prompt=prompt,
        post_checker=post_checker,
        post_check_prompt=post_check_prompt,
        resume=args.resume,
        max_papers=cfg.template_injection.get("max_papers", None),
        template_sample_size=args.template_sample_size,
        max_concurrent=args.max_concurrent,
        **cfg,
    )

    # Inject templates
    await pipeline.inject_template()


if __name__ == "__main__":
    asyncio.run(main())
