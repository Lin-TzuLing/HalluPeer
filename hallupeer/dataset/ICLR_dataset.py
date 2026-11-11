import glob
import os 
import json

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset

from hallupeer.dataset.utils.filter_review import ReviewFilterPipeline 


class ICLRDataset(Dataset):
    """
    ICLR 2019~2024 dataset.

    Args:
        dataset_dir (str): Path to the dataset directory.
    """
    
    def __init__(self, dataset_dir: str):
        
        self.dataset_dir = dataset_dir
        self.venue = self.dataset_dir.split("/")[-1]  # e.g., ICLR2019
        
        # ====================================Dataset path==========================================

        # Get paths of parsed pdfs and notes 
        parsed_pdf_path_list, notes_path_list = self._get_papers_and_notes(
            dataset_base_path = self.dataset_dir,
            )

        parsed_pdf_path_list = natsorted(parsed_pdf_path_list)
        notes_path_list = natsorted(notes_path_list)
        
        # ==========================================================================================  
        
        print("="*50)     
        print(f"Loading {self.venue} dataset from {self.dataset_dir}...")
        self.datas = self._read_data(parsed_pdf_path_list, notes_path_list)
        print(f"    - Total number of papers: {len(self.datas)}")
        print(f"    - Total number of paper-review pairs: {self.__len_pairs__()}")
        print("="*50)


    @classmethod
    async def create(cls, dataset_dir: str, **kwargs):
        instance = cls(dataset_dir)
        await instance.filter_data_pairs(**kwargs)
        return instance


    async def filter_data_pairs(self, **kwargs):
        print(f"Filtering {self.venue} dataset...")
        
        # set filter list path
        path_filter_list = kwargs["dataset"]["path_filter_list"].format(venue=self.venue)
          
        # Load existing filter records
        existing_records = self._load_filter_records(path_filter_list)
        pending_datas = [d for d in self.datas if d["PaperID"] not in existing_records]

        if pending_datas:
            print(f"Need to filter {len(pending_datas)} remaining papers.")
            Path(path_filter_list).parent.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(path_filter_list):
                open(path_filter_list, "w").close()

            filter_pipeline = ReviewFilterPipeline(path_filter_list, **kwargs)
            await filter_pipeline.batch_filter_review(pending_datas)

            usage_path = kwargs['dataset']['usage_path'].format(venue=self.venue)
            with open(usage_path, "w") as f:
                json.dump(filter_pipeline.total_usage, f, indent=4)

            existing_records.update(self._load_filter_records(path_filter_list)) # update with new records
        else:
            print("Filter list already complete.")
        
        # Load existing filter list
        print(f"Loading existing filter list from {path_filter_list}...")
        filter_dict = existing_records
        
        min_count, max_count = float('inf'), float('-inf')
        self.datas = [  # only keep filtered reviews
            {
                "PaperID": data["PaperID"],
                "Contents": data["Contents"],
                "Reviews": [review for (review, filter) in zip(data["Reviews"], filter_dict[data["PaperID"]]) if filter["adopted"]],
                "MetaReviews": data["MetaReviews"],
            }
            for data in self.datas
        ]
        
        self.datas = [data for data in self.datas if len(data["Reviews"]) > 0]  # remove papers with no reviews after filtering
        min_count = min(min_count, min(len(data["Reviews"]) for data in self.datas))
        max_count = max(max_count, max(len(data["Reviews"]) for data in self.datas))
        print(f"    - Total number of filtered papers: {len(self.datas)}")
        print(f"    - Total number of filtered paper-review pairs: {self.__len_pairs__()}")
        print(f"    - Min number of reviews per paper after filtering: {min_count}")
        print(f"    - Max number of reviews per paper after filtering: {max_count}") 


    def _load_filter_records(self, path_filter_list: str) -> dict:
        records = {}
        if not os.path.exists(path_filter_list):
            return records
        with open(path_filter_list, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records[record["PaperID"]] = record["Filtered_Review"]
                except json.JSONDecodeError:
                    continue
        return records
    
    
    def __len__(self):
        return len(self.datas)
    
    
    def __len_pairs__(self):
        count_pairs = 0
        for data in self.datas:
            count_pairs += len(data['Reviews'])
        return count_pairs
    
    
    def __getitem__(self, index):
        return self.datas[index]
      
      
    def _read_data(self, parsed_pdf_path_list, notes_path_list):
        """
        Read data from the raw dataset.
        
        Args:
            parsed_pdf_path_list (:obj:`list`): the list of parsed pdf path. 
            notes_path_list (:obj:`list`): the list of notes path.
            
        Returns:
            datas (list): the data of all papers.
        """  
        
        # read paper files
        data_paper = {}
        for file in parsed_pdf_path_list:
            paper_id = self.venue + "_" +os.path.basename(file).split('.json')[0]
            with open(file, 'r', encoding='utf-8') as f:
                data_paper[paper_id] = json.loads(f.read())
        
        # read review files  
        data_review = {}
        for file in notes_path_list:
            paper_id = self.venue + "_" + os.path.basename(file).split('.json')[0]
            if paper_id not in data_paper:
                continue  # skip if no corresponding parsed paper
            with open(file, 'r', encoding='utf-8') as f:
                data_review[paper_id] = json.loads(f.read())

        assert data_paper.keys()==data_review.keys(), "Number of papers and reviews do not match."

        # merge data
        datas = []
        
        # use tqdm for progress bar
        from tqdm import tqdm
        for id in tqdm(data_paper.keys(), desc="Processing papers and reviews"):
            paper = data_paper[id]         
            review = data_review[id] 
            
            contents, reviews, meta_reviews = self._construct_paper_and_reviews(paper, review)
            
            if contents is None and reviews is None and meta_reviews is None:
                continue  # skip withdrawn or desk rejected papers
            
            data = {'PaperID': id, 
                    'Contents': contents, 
                    'Reviews': reviews,
                    'MetaReviews': meta_reviews,}
            
            datas.append(data)  # merge two dict and keep initial contents
        
        return datas


    def _construct_paper_and_reviews(self, paper, review):
        """
        Construct paper contents and reviews.
        
        Args:
            paper (dict): the parsed paper data. 
            review (dict): the review data.
        """
        
        contents_dict = {}
        reviews_list = []
        meta_reviews_list = []
        
        # construct paper contents
        if self.venue in ['ICLR2019', 'ICLR2020', 'ICLR2021', 'ICLR2022', 'ICLR2023']:
            contents_dict['Title'] = review['content']['title'] 
            contents_dict['Abstract'] = review['content']['abstract']
        elif self.venue in ['ICLR2024', 'ICLR2025']:
            contents_dict['Title'] = review['content']['title']['value']
            contents_dict['Abstract'] = review['content']['abstract']['value']
        
        # construct sections
        for sec in paper['metadata']['section']:
            contents_dict[f"Section {sec['number']}. {sec['heading'].capitalize()}"] = sec['text'] 
                
        # construct reviews based on different years' formats
        if self.venue in ['ICLR2019']:
            for reply in review['details']['directReplies']:
                if 'Official_Review' in reply['invitation']:
                    signature = reply['signatures'][0].split('/')[-1]
                    reviews_list.append({
                        'ReviewerID': signature,
                        'Review': reply['content']['review'],
                        'Rating': reply['content']['rating'],
                        'Addtional_Assessment':{
                            'confidence': reply['content']['confidence'],
                        }
                    })  
                if 'Meta_Review' in reply['invitation']:
                    meta_signature = reply['signatures'][0].split('/')[-1]
                    meta_reviews_list.append({
                        'MetaReviewerID': meta_signature,
                        'MetaReview': reply['content']['metareview'],
                        'MetaDecision': reply['content']['recommendation'],
                        'Addtional_Assessment':{
                            'meta_confidence': reply['content']['confidence'],
                        }
                    })                    
        elif self.venue in ['ICLR2020']:
            for reply in review['details']['directReplies']:
                if 'Official_Review' in reply['invitation']:
                    signature = reply['signatures'][0].split('/')[-1]
                    reviews_list.append({
                        'ReviewerID': signature,
                        'Review': reply['content']['review'],
                        'Rating': reply['content']['rating'],
                        'Addtional_Assessment':{
                            'experience_assessment': reply['content']['experience_assessment'],
                            'review_assessment:_checking_correctness_of_derivations_and_theory': reply['content']['review_assessment:_checking_correctness_of_derivations_and_theory'],
                            'review_assessment:_checking_correctness_of_experiments': reply['content']['review_assessment:_checking_correctness_of_experiments'],
                            'review_assessment:_thoroughness_in_paper_reading': reply['content']['review_assessment:_thoroughness_in_paper_reading'],
                        }
                    })   
                if 'Decision' in reply['invitation']:
                    meta_signature = reply['signatures'][0].split('/')[-1]
                    meta_reviews_list.append({
                        'MetaReviewerID': meta_signature,
                        'MetaReview': reply['content']['comment'],
                        'MetaDecision': reply['content']['decision'],
                        'Addtional_Assessment': None
                    })  
        elif self.venue in ['ICLR2021']:
            for reply in review['details']['directReplies']:
                if 'Official_Review' in reply['invitation']:
                    signature = reply['signatures'][0].split('/')[-1]
                    reviews_list.append({
                        'ReviewerID': signature,
                        'Review': reply['content']['review'],
                        'Rating': reply['content']['rating'],
                        'Addtional_Assessment':{
                            'confidence': reply['content']['confidence'],
                        }
                    })  
                if 'Decision' in reply['invitation']:
                    meta_signature = reply['signatures'][0].split('/')[-1]
                    meta_reviews_list.append({
                        'MetaReviewerID': meta_signature,
                        'MetaReview': reply['content']['comment'],
                        'MetaDecision': reply['content']['decision'],
                        'Addtional_Assessment': None
                    })              
        elif self.venue in ['ICLR2022']:
            for reply in review['details']['directReplies']:
                if 'Official_Review' in reply['invitation']:
                    signature = reply['signatures'][0].split('/')[-1]
                    # the review content is split into three parts
                    summary_of_the_paper = f"Summary of the paper:\n{reply['content']['summary_of_the_paper']}"
                    main_review = f"Review:\n{reply['content']['main_review']}"
                    summary_of_the_review = f"Summary of the review:\n{reply['content']['summary_of_the_review']}"
                    review_concatenated = summary_of_the_paper + "\n" + main_review + "\n" + summary_of_the_review
                    reviews_list.append({
                        'ReviewerID': signature,
                        'Review': review_concatenated,
                        'Rating': reply['content']['recommendation'],
                        'Addtional_Assessment':{
                            'confidence': reply['content']['confidence'],
                            'correctness': reply['content']['correctness'],
                            'technical_novelty_and_significance': reply['content']['technical_novelty_and_significance'],
                            'empirical_novelty_and_significance': reply['content']['empirical_novelty_and_significance'],
                        }
                    })
                if 'Decision' in reply['invitation']:
                    meta_signature = reply['signatures'][0].split('/')[-1]
                    meta_reviews_list.append({
                        'MetaReviewerID': meta_signature,
                        'MetaReview': reply['content']['comment'],
                        'MetaDecision': reply['content']['decision'],
                        'Addtional_Assessment': None
                    })   
        elif self.venue in ['ICLR2023']:
            for reply in review['details']['directReplies']:
                if 'Official_Review' in reply['invitation']:
                    signature = reply['signatures'][0].split('/')[-1]
                    # the review content is split into four parts
                    summary_of_the_paper = f"Summary of the paper:\n{reply['content']['summary_of_the_paper']}"
                    strengths_and_weaknesses = f"Strength and Weaknesses:\n{reply['content']['strength_and_weaknesses']}"
                    clarity_quality_novelty_and_reproducibility = f"Clarity, Quality, Novelty, and Reproducibility:\n{reply['content']['clarity,_quality,_novelty_and_reproducibility']}"
                    summary_of_the_review = f"Summary of the review:\n{reply['content']['summary_of_the_review']}"
                    review_concatenated = summary_of_the_paper + "\n" + strengths_and_weaknesses + "\n" + clarity_quality_novelty_and_reproducibility + "\n" + summary_of_the_review
                    reviews_list.append({
                        'ReviewerID': signature,
                        'Review': review_concatenated,
                        'Rating': reply['content']['recommendation'],
                        'Addtional_Assessment':{
                            'confidence': reply['content']['confidence'],
                            'correctness': reply['content']['correctness'],
                            'technical_novelty_and_significance': reply['content']['technical_novelty_and_significance'],
                            'empirical_novelty_and_significance': reply['content']['empirical_novelty_and_significance'],
                        }
                    })
                if 'Decision' in reply['invitation']:
                    meta_signature = reply['signatures'][0].split('/')[-1]
                    summary_strengths_weaknesses = f"Summary, Strengths and Weaknesses:\n{reply['content']['metareview:_summary,_strengths_and_weaknesses']}"
                    justification_for_why_not_higher_score = f"Justification for why not higher score:\n{reply['content']['justification_for_why_not_higher_score']}"
                    justification_for_why_not_lower_score = f"Justification for why not lower score:\n{reply['content']['justification_for_why_not_lower_score']}"
                    meta_review_concatenated = summary_strengths_weaknesses + "\n" + justification_for_why_not_higher_score + "\n" + justification_for_why_not_lower_score
                    meta_reviews_list.append({
                        'MetaReviewerID': meta_signature,
                        'MetaReview': meta_review_concatenated,
                        'MetaDecision': reply['content']['decision'],
                        'Addtional_Assessment': None
                    })   
        # elif self.venue in ['ICLR2024', 'ICLR2025']:
        elif self.venue in ['ICLR2024']:
            if "Withdrawn" in review['content']['venue']['value'] or "Desk Rejected" in review['content']['venue']['value']:
                return None, None, None  # skip withdrawn or desk rejected papers
            meta_review_flag = False
            meta_decision_flag = False
            for reply in review['details']['directReplies']:
                if 'Official_Review' in reply['invitations'][0]:
                    signature = reply['signatures'][0].split('/')[-1]
                    # the review content is split into four parts
                    summary = f"Summary:\n{reply['content']['summary']['value']}"
                    strengths = f"Strengths:\n{reply['content']['strengths']['value']}"
                    weaknesses = f"Weaknesses:\n{reply['content']['weaknesses']['value']}"
                    questions = f"Questions:\n{reply['content']['questions']['value']}"
                    review_concatenated = summary + "\n" + strengths + "\n" + weaknesses + "\n" + questions
                    reviews_list.append({
                        'ReviewerID': signature,
                        'Review': review_concatenated,
                        'Rating': reply['content']['rating']['value'],
                        'Addtional_Assessment':{
                            'confidence': reply['content']['confidence']['value'],
                            'soundness': reply['content']['soundness']['value'],
                            'presentation': reply['content']['presentation']['value'],
                            'contribution': reply['content']['contribution']['value'],     
                            'code_of_conduct': reply['content']['code_of_conduct']['value'],
                        }
                    })
                if 'Meta_Review' in reply['invitations'][0]:
                    meta_signature = reply['signatures'][0].split('/')[-1]
                    # the review content is split into four parts
                    metareview = f"Meta Review:\n{reply['content']['metareview']['value']}"
                    justification_for_why_not_higher_score = f"Justification for why not higher score:\n{reply['content']['justification_for_why_not_higher_score']['value']}"
                    justification_for_why_not_lower_score = f"Justification for why not lower score:\n{reply['content']['justification_for_why_not_lower_score']['value']}"
                    meta_review_concatenated = metareview + "\n" + justification_for_why_not_higher_score + "\n" + justification_for_why_not_lower_score
                    meta_review_flag = True
                if 'Decision' in reply['invitations'][0]:
                    decision = reply['content']['decision']['value']
                    meta_decision_flag = True
            if meta_review_flag and meta_decision_flag:
                meta_reviews_list.append({
                    'MetaReviewerID': meta_signature,
                    'MetaReview': meta_review_concatenated,
                    'MetaDecision': decision,
                    'Addtional_Assessment': None
                })                      
        else:
            raise ValueError(f"Unsupported venue: {self.venue}")
                  
        return contents_dict, reviews_list, meta_reviews_list
    
    
    def _get_papers_and_notes(self, dataset_base_path: str):
        """
        Get the paths of parsed pdfs and notes based on the dataset base path and regex patterns
        Args:
            dataset_base_path (str): Base path of the dataset.
        """
        
        parsed_pdf_path_list = glob.glob(f"{dataset_base_path}/parsed_pdfs/*.json")
        notes_path_list = glob.glob(f"{dataset_base_path}/notes/*.json")
    
        return parsed_pdf_path_list, notes_path_list


def parse_rating(rating_str):
    """
    Parse rating string to extract numerical value.
    Examples: "8 (good)" -> 8, "8: good" -> 8, "8" -> 8
    
    Args:
        rating_str: Rating string or number
        
    Returns:
        int or float: Numerical rating value
    """
    if isinstance(rating_str, (int, float)):
        return rating_str
    
    if isinstance(rating_str, str):
        # Remove text in parentheses and extract first number
        import re
        # Try to find first number (integer or float)
        match = re.search(r'[-+]?\d*\.?\d+', rating_str.strip())
        if match:
            num_str = match.group()
            return int(num_str) if '.' not in num_str else float(num_str)
    
    return None
