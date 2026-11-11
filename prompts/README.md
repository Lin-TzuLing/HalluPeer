# prompts — Prompt templates for dataset construction

Loaded with `hallupeer.utils.load_prompt_template(<path relative to prompts/>)`.

| Folder | File | Step |
|---|---|---|
| `review_filtering/` | `review_filtering.txt` | Keep reviews whose content is consistent with the meta-review |
| `review_aspect_tagging/` | `aspect_tagging.txt` | Tag each review sentence with an aspect |
| `hallucination_taxonomy/` | `hallucination_taxonomy_desc.txt` | Expand / describe taxonomy nodes |
| | `similar_filter.txt` | Judge whether two taxonomy nodes are synonyms |
| `template_construction/` | `taxonomy_prefilter.txt` | Stage 1: first-level concept applicability |
| | `template_construction.txt` | Stage 2: template feasibility check |
| `template_injection/` | `template_injection.txt` | Rewrite a sentence according to a template |
| | `post_check.txt` | Semantic verification of the rewritten sentence |

Evaluation prompts (Tasks 1–3) are in `evaluation/prompts/`.
