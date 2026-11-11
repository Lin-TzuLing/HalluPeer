import asyncio
import argparse
import json
import re
import os
import numpy as np
import pandas as pd

import nltk
nltk.download('punkt')

from collections import defaultdict
from tqdm import tqdm
from string import Template


from resources.review_aspect.utils import *
from hallupeer.utils import load_configs, load_prompt_template, load_json_configs
from hallupeer.agents import load_agent
from hallupeer.dataset import load_dataset

# ============================================================================
# Review Aspect Prediction Functions
# ============================================================================

class ReviewAspectTaggingPipeline:
    def __init__(self, dataloader, tagger, prompt_template, type_of_labels="fine", resume=False, **kwargs):
        self.dataloader = dataloader
        self.tagger = tagger
        self.type_of_labels = type_of_labels  # 'fine' or 'coarse'
        self.column_name = 'FINE' if type_of_labels == 'fine' else 'COARSE'
        self.resume = resume
        
        self.sent_tokenize = nltk.tokenize.sent_tokenize
        
        # Prepare aspect categories
        categories = self._prepare_label(
            path_category_aspect_mappings=kwargs["review_aspect_tagging"]["path_category_aspect_mappings"].format(self.type_of_labels),
        )
        
        # Format system prompt with aspect labels
        prompt_template = Template(prompt_template)
        categories_for_prompt = sorted(c for c in categories)
        self.prompt = prompt_template.safe_substitute(aspect_label_set=", ".join(categories_for_prompt))
        
        # Generate aspect tagging usage tracking
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, 
                            "total_tokens": 0, "max_total_tokens": 0}
        
        # Output path
        self.output_path = kwargs["review_aspect_tagging"]["output_path"].format(label=self.type_of_labels, dataset_venue=kwargs["dataset"]["venue"])
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        if not resume and os.path.exists(self.output_path):
            os.remove(self.output_path)

        # Load processed paper IDs if resuming
        self.processed_paper_ids = set()
        if resume and os.path.exists(self.output_path):
            self._load_processed_papers()
    
    
    def _load_processed_papers(self):
        """Load already processed paper IDs from output file."""
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self.processed_paper_ids.add(data['PaperID'])
                    except json.JSONDecodeError:
                        continue
            print(f"Resuming: Found {len(self.processed_paper_ids)} already processed papers, will skip them.")
        except Exception as e:
            print(f"Warning: Could not load processed papers: {e}")
            self.processed_paper_ids = set()
            
    
    async def aspect_tagging(self):
        """Tag aspects in reviews using the tagger agent."""
        
        # Generate predictions
        for _, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Tagging review aspects"):    
            paper_ids = batch['PaperID']
            reviews = batch['Reviews'] # ReviewerID, Review, Rating, Additional_Assessment
            
            for paper_id, review in tqdm(zip(paper_ids, reviews), total=len(paper_ids), desc="Processing batch reviews"):
                # Skip if already processed and resuming
                if self.resume and paper_id in self.processed_paper_ids:
                    continue
                
                reviewer_ids = [r['ReviewerID'] for r in review] # per paper
                review_texts = [r['Review'] for r in review] # per paper
                
                # Split reviews into sentences
                reviews_sents_list = [self.sent_tokenize(text) for text in review_texts]
                
                try:
                    review_tags = await self.aspect_tagging_by_sentences(reviews_sents_list)
                except Exception as e:
                    print("Error during aspect tagging:")
                    print(f"PaperID: {paper_id}")
                    print(f"ReviewerIDs: {reviewer_ids}")
                    print(f"Review Texts: {review_texts}")
                    print(f"Reviews Sents List: {reviews_sents_list}")
                    print(f"Error: {e}")
                    exit()
                
                output = {
                    "PaperID": paper_id,
                    "ReviewerIDs": reviewer_ids,
                    "ReviewAspectTags": review_tags,
                }
                
                # Save intermediate results per paper
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')

                # Add to processed set
                self.processed_paper_ids.add(paper_id)
 
    async def aspect_tagging_by_sentences(self, reviews_sents_list):
        """Tag aspects in reviews using the tagger agent, processing one sentence each request."""
        
        review_tags = []
        for review_sents in reviews_sents_list: # per review
            # Prepare messages for the tagger
            messages = [[
                {"role": "user", "content": self.prompt.format(review_sent=review_sent)}
            ] for review_sent in review_sents]
        
            # Get predictions from the tagger
            response, usage = await self.tagger.predict(messages)
            self._usage_tracking(usage)
                
            try:
                clean_json = [clean_json_response(r) for r in response] 
                prediction_per_review = [] # per review
                for cj in clean_json:
                    try:
                        prediction_per_review.append(json.loads(cj)['Aspects'])
                    except:
                        prediction_per_review.append(['-'])                      
                review_tags.append(prediction_per_review)
            except Exception as e:
                print("Failed to parse JSON response:")
                print(response)
                print(f"Error: {e}")

        return review_tags

        
    def _usage_tracking(self, usage):
        """Track token usage."""
        if usage:
            self.total_usage["prompt_tokens"] += usage["prompt_tokens"]
            self.total_usage["completion_tokens"] += usage["completion_tokens"]
            self.total_usage["total_tokens"] += usage["total_tokens"]
            self.total_usage["max_total_tokens"] = max(self.total_usage["max_total_tokens"], usage["total_tokens"])
       
    
    def _prepare_label(self, path_category_aspect_mappings):
        # Load category to aspect mappings
        category_to_aspect = pd.read_csv(path_category_aspect_mappings)
        aspect_to_category = defaultdict(set)
        for i in range(len(category_to_aspect)):
            aspect_to_category[category_to_aspect['LLM annotation'].to_list()[i]].add(
                category_to_aspect[self.column_name].to_list()[i]
            )

        # Create categories list
        categories = ['-']
        for _, category in aspect_to_category.items():
            categories.extend(list(category))
        categories = list(set(categories))
        
        return categories


def clean_json_response(response):
    """Clean JSON response from the tagger."""
    # Remove markdown code blocks, leading [' and trailing ']
    clean_json = response.strip()
    # Remove markdown code fences (```json or ```)
    clean_json = re.sub(r"```(?:json)?\s*", "", clean_json)
    # Remove leading [' and trailing ']
    clean_json = re.sub(r"^\[\s*'|'\s*\]$", "", clean_json)     
    return clean_json
    
    
# python -m hallupeer.dataset_construction.review_aspect_tagging.tag_review_aspect --dataset_dir ../datasets/openreview_data/ICLR2020 --port 8002 --resume
async def main():
    """Main execution function."""
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_dir", type=str, default=None, help="Path to the dataset directory")
    argparser.add_argument("--port", type=int, default=None, help="Port number for the tagger agent")
    argparser.add_argument("--resume", action="store_true", help="Whether to resume from existing output file")
    args = argparser.parse_args()
    
    cfg = load_configs(["configs/dataset.yaml", 
                        "configs/agents.yaml", 
                        "configs/review_aspect_tagging.yaml"])
    
    # Override dataset_dir and port if provided via command line arguments
    if args.dataset_dir is not None:
        cfg.dataset.dataset_dir = args.dataset_dir
    if args.port is not None:
        cfg.agents.review_aspect_tagger.port = args.port
        
    venue = cfg.dataset.dataset_dir.split("/")[-1]
    cfg.dataset.venue = venue
    
    # Load dataset
    dataloader = await load_dataset(
        dataset_dir=cfg.dataset.dataset_dir,
        batch_size=cfg.dataset.batch_size_review_aspect_tagging,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        **cfg
    )

    # Load prompt
    prompt_template = load_prompt_template(cfg.review_aspect_tagging.prompt_template_path)
    
    # Initialize generator
    tagger = load_agent(agent="review_aspect_tagger", cfg=cfg)
    
    # Initilize pipeline
    pipeline = ReviewAspectTaggingPipeline(
        dataloader=dataloader,
        tagger=tagger,
        prompt_template=prompt_template,
        type_of_labels=cfg.review_aspect_tagging.type_of_labels,
        resume=args.resume,
        **cfg
    )
    
    # tag review aspects
    await pipeline.aspect_tagging()
    
    # Save usage results
    usage_path = cfg.review_aspect_tagging.output_usage_path.format(label=pipeline.type_of_labels, dataset_venue=venue)
    with open(usage_path, "w") as f:
        json.dump(pipeline.total_usage, f, indent=4)
    
  
if __name__ == "__main__":
    # Run async main
    asyncio.run(main())