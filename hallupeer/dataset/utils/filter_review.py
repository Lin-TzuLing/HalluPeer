import asyncio
import re
import json

from tqdm import tqdm

from hallupeer.agents import load_agent
from hallupeer.utils import load_prompt_template, kwargs_to_edict

class ReviewFilterPipeline:
    def __init__(self, path_filter_list, **kwargs):
        
        self.path_filter_list = path_filter_list
        
        cfg = kwargs_to_edict(**kwargs)
        self.filter = load_agent(agent="review_filter", cfg=cfg)
        self.prompt = load_prompt_template(cfg.dataset.prompt_template_path)
        self.batch_size = cfg.dataset.batch_size_review_filtering
        
        # generate aspect tagging usage tracking
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, 
                            "total_tokens": 0, "max_total_tokens": 0}

        print(f"Filter list path: {self.path_filter_list}")
    
    
    async def batch_filter_review(self, datas: list):
        
        filter_list = []
        total = len(datas)
        for start in tqdm(range(0, total, self.batch_size)):
            batch = datas[start:start + self.batch_size]
            tasks = []
            paper_ids = []
            for data in batch:
                paper_id = data['PaperID']
                reviews = data['Reviews']
                meta_reviews = data['MetaReviews']

                reviewer_reviews = [r['Review'] for r in reviews]
                meta_review_text = "\n".join([mr['MetaReview'] for mr in meta_reviews]) if len(meta_reviews) > 0 else ""

                tasks.append(self.filter_review_single_paper(
                    reviewer_reviews=reviewer_reviews,
                    meta_review=meta_review_text,
                ))
                paper_ids.append(paper_id)

            results = await asyncio.gather(*tasks)

            with open(self.path_filter_list, 'a', encoding='utf-8') as f:
                for paper_id, sorted_json in zip(paper_ids, results):
                    output = {
                        "PaperID": paper_id,
                        "Filtered_Review": sorted_json
                    }
                    filter_list.append(output)
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
        
        return filter_list
    
    
    async def filter_review_single_paper(self, reviewer_reviews: list[str], 
                                         meta_review:str):
        
        # Prepare messages for the checker
        messages = [[
            {"role": "user", "content": self.prompt.format(reviewer_review=reviewer_review,
                                                           meta_review=meta_review)}]
            for reviewer_review in reviewer_reviews]
        
        
        # Get predictions from the filter
        response, usage = await self.filter.predict(messages)
        self._usage_tracking(usage)

        try:
            clean_json = [clean_json_response(r) for r in response] 
            response_json = [json.loads(cj) for cj in clean_json] 
             
            conf_order = {"high": 0, "medium": 1, "low": 2}
            sorted_json = sorted(response_json, 
                                 key=lambda x: (
                                     not x["adopted"], # True (0) > False (1) 
                                     conf_order[x["confidence"]] # high (0) > medium (1) > low (2)
                                     )
                                 )
        except Exception as e:
            print("Failed to parse JSON response:")
            print(response)
            print(f"Error: {e}")
            
        return sorted_json
    
    
    async def filter_review(self, datas: list):
        
        filter_list = []
        c = 0
        for data in tqdm(datas):
            paper_id = data['PaperID']
            reviews = data['Reviews']  # dict_keys(['ReviewerID', 'Review', 'Rating', 'Addtional_Assessment'])
            meta_reviews = data['MetaReviews']  # dict_keys(['MetaReviewerID', 'MetaReview', 'MetaDecision', 'Addtional_Assessment'])
            
            reviewer_reviews = [r['Review'] for r in reviews]
            meta_review_text = "\n".join([mr['MetaReview'] for mr in meta_reviews]) if len(meta_reviews)>0 else ""

            # Get filter response for single paper
            sorted_json = await self.filter_review_single_paper(
                reviewer_reviews=reviewer_reviews,
                meta_review=meta_review_text,
            )
            
            # Generate filter list for the paper
            output = { 
                "PaperID": paper_id,
                "Filtered_Review": sorted_json
            }
            filter_list.append(output)
            
            # Save immediate filter list to file
            with open(self.path_filter_list, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
    
    
    def _usage_tracking(self, usage):
        """Track token usage."""
        if usage:
            self.total_usage["prompt_tokens"] += usage["prompt_tokens"]
            self.total_usage["completion_tokens"] += usage["completion_tokens"]
            self.total_usage["total_tokens"] += usage["total_tokens"]
            self.total_usage["max_total_tokens"] = max(self.total_usage["max_total_tokens"], usage["total_tokens"])


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
    clean_json = re.sub(r'(?<!\\)\\(?![\\\"/bfnrtu])',r'\\\\',clean_json)
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
        # pass
        # # Escape unescaped backslashes
        # return clean_json
    
 