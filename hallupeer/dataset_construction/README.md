# dataset_construction — Aspect tagging, template construction, hallucination injection

## review_aspect_tagging/

`tag_review_aspect.py` assigns each review sentence an aspect label (e.g. Method, Figure, Result) with an LLM, using the aspect inventory in `resources/review_aspect/` (fine / coarse levels).
Aspects later restrict which taxonomy leaves may be injected into a sentence.

```bash
python -m hallupeer.dataset_construction.review_aspect_tagging.tag_review_aspect \
    --dataset_dir ../datasets/openreview_data/NeurIPS2024 --port 8001 --resume
```
Config: `configs/review_aspect_tagging.yaml`; prompt: `prompts/review_aspect_tagging/aspect_tagging.txt`.

## hallucination_injection/

### `construct_template.py` — Template construction

For every review sentence `s` and every taxonomy leaf `v`, build a template `T_{s,v} = (s, a_s, δ_v, π_v)` and keep it only if it passes a two-stage LLM check:
1. **Prefilter** (`prompts/template_construction/taxonomy_prefilter.txt`) — drop leaves whose first-level concept cannot apply to the sentence.
2. **Feasibility check** (`prompts/template_construction/template_construction.txt`) — the instruction can be applied to the sentence and the label path matches it.

Output: `hallucination_templates_{venue}.jsonl` with `hallucination_instruction`, `hallucination_labels` (taxonomy path), `injection_aspect`, `injection_sentence`.

### `inject_template.py` — Injection + semantic verification

Rewrites the sentence according to each template (`prompts/template_injection/template_injection.txt`), then a post-checker LLM (`prompts/template_injection/post_check.txt`) verifies that no opinions/facts are added or omitted and that emphasis, scope and certainty are unchanged.
Output: one JSON per paper (`{paper_id}.json`) with `paper_content` and, for every review sentence, its list of `hallucination_injection_templates` (including `hallucination_sentence`).

Ready-to-run launchers (vLLM server + pipeline) are in `scripts/construct/` and `scripts/inject/`.
Configs: `configs/template_construction.yaml`, `configs/template_injection.yaml`.
